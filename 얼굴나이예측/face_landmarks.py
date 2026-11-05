"""얼굴 좌표만으로 비교하는 공통 추론 코드. 이 파일을 불러와도 학습하지 않습니다.

강아지: DogFLW 기반 localizer + 46점 모델 (CC BY-NC 4.0).
사람: MediaPipe Face Landmarker. 웹/검색 준비 코드가 함께 사용하는 순수 추론 모듈입니다.
"""
from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT / 'models/face_landmarks'
SCHEMA = 'face_geometry_v1'
FEATURE_NAMES = ['눈의 세로/가로 비율','눈꼬리 기울기','눈 사이 간격/눈 너비',
                 '코 너비/눈 사이 거리','눈~코 길이/눈 사이 거리',
                 '코~입 길이/눈 사이 거리','입 너비/눈 사이 거리','하관 길이/눈 사이 거리']
FEATURE_WEIGHTS = np.array([.20,.15,.15,.10,.10,.10,.10,.10],dtype=np.float32)
GROUPS = [('눈매',[0,1]),('눈 사이 간격',[2]),('코 비율',[3,4]),('입·하관 비율',[5,6,7])]


class LiteModel:
    def __init__(self, name, threads=1):
        from ai_edge_litert.interpreter import Interpreter
        # Windows 한글 경로도 지원하도록 파일 내용으로 전달합니다.
        self.model = Interpreter(model_content=(MODEL_DIR/name).read_bytes(), num_threads=threads)
        self.model.allocate_tensors()
        self.input = self.model.get_input_details()[0]
        self.output = self.model.get_output_details()[0]

    def __call__(self, image):
        size = tuple(int(x) for x in self.input['shape'][1:3][::-1])
        value = np.asarray(image.convert('RGB').resize(size, Image.Resampling.BILINEAR), dtype=np.float32)/255.0
        self.model.set_tensor(self.input['index'], value[None])
        self.model.invoke()
        return self.model.get_tensor(self.output['index'])[0].copy()


class DogLandmarker:
    def __init__(self, threads=1):
        self.localizer = LiteModel('dog_face_localizer.tflite', threads)
        self.landmarker = LiteModel('dog_face_landmarks_full.tflite', threads)

    def detect(self, image):
        image = image.convert('RGB')
        # localizer 입력은 정사각형 letterbox. 빈 공간은 검은색입니다.
        scale = 224 / max(image.size)
        width, height = max(1, round(image.width*scale)), max(1, round(image.height*scale))
        pad_x, pad_y = (224-width)//2, (224-height)//2
        square = Image.new('RGB', (224,224))
        square.paste(image.resize((width,height), Image.Resampling.BILINEAR), (pad_x,pad_y))
        raw = np.clip(self.localizer(square),0,1)*224
        # 공개 추론 코드처럼 두 모서리를 정렬합니다. 원시 출력의 순서는 뒤집힐 수 있습니다.
        raw = np.array([min(raw[0],raw[2]),min(raw[1],raw[3]),max(raw[0],raw[2]),max(raw[1],raw[3])])
        xy = (raw-np.array([pad_x,pad_y,pad_x,pad_y]))/np.array([width/image.width,height/image.height]*2)
        x0,y0,x1,y1 = xy
        if not np.isfinite(xy).all() or x1-x0<12 or y1-y0<12:
            raise ValueError('face_box_too_small')
        # 공개 모델의 입력 계약: 예측 얼굴 상자의 각 방향에 10% 여백을 추가합니다.
        margin_x, margin_y = (x1-x0)*.1, (y1-y0)*.1
        box = (max(0,math.floor(x0-margin_x)),max(0,math.floor(y0-margin_y)),
               min(image.width,math.ceil(x1+margin_x)),min(image.height,math.ceil(y1+margin_y)))
        if box[2]-box[0]<24 or box[3]-box[1]<24:
            raise ValueError('face_crop_too_small')
        points = self.landmarker(image.crop(box)).reshape(46,2)
        points = points*np.array([box[2]-box[0],box[3]-box[1]])+np.array(box[:2])
        return {'points':points,'box':box}


class HumanLandmarker:
    def __init__(self):
        import mediapipe as mp
        self.mp=mp
        options=mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=(MODEL_DIR/'face_landmarker.task').read_bytes()),
            num_faces=2, min_face_detection_confidence=.5, min_face_presence_confidence=.5)
        self.model=mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self,image):
        rgb=np.asarray(image.convert('RGB'))
        result=self.model.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=rgb))
        if not result.face_landmarks:
            raise ValueError('얼굴을 찾지 못했어요. 정면 얼굴이 크게 보이도록 잘라 주세요.')
        if len(result.face_landmarks)>1:
            raise ValueError('얼굴이 여러 개 보여요. 한 사람의 얼굴만 선택해 주세요.')
        points=np.array([[p.x*image.width,p.y*image.height] for p in result.face_landmarks[0]],dtype=np.float32)
        common=points[[33,159,133,145,263,386,362,374,98,327,1,61,291,0,152]]
        features=describe_face(common)
        # 출력 사진도 얼굴 위주로 보여 주되, 사용자가 선택한 원본은 수정하지 않습니다.
        face=points[:468]
        lo=face.min(axis=0);hi=face.max(axis=0);margin=(hi-lo)*.12
        box=(max(0,int(lo[0]-margin[0])),max(0,int(lo[1]-margin[1])),
             min(image.width,int(hi[0]+margin[0])),min(image.height,int(hi[1]+margin[1])))
        return {'points':points,'common':common,'features':features,'box':box}

    def close(self):
        self.model.close()


def dog_common(points):
    # 실제 DogFLW 좌표와 추론 그림을 확인한 대응표입니다.
    # 눈: [바깥, 위, 안쪽, 아래], 코 양끝·코끝, 입 양끝·입 중앙, 턱끝.
    # 개의 귀/털 윤곽은 사람과 대응하지 않으므로 점수에 넣지 않습니다.
    return np.asarray(points)[[18,20,16,22,19,21,17,23,26,27,25,39,40,32,42]]


def describe_face(common):
    """15개 공통 지점 → 8개 비율. RGB/옷/배경은 이 계산에 전달하지 않습니다."""
    p=np.asarray(common,dtype=np.float64).copy()
    if p.shape!=(15,2) or not np.isfinite(p).all():
        raise ValueError('invalid_landmarks')
    eyes=[p[:4].copy(),p[4:8].copy()]
    eyes.sort(key=lambda e:float(e[[0,2],0].mean()))
    centers=np.array([e[[0,2]].mean(axis=0) for e in eyes])
    axis=centers[1]-centers[0];iod=np.linalg.norm(axis)
    if iod<10: raise ValueError('얼굴의 눈 사이 거리가 너무 작아요. 얼굴을 더 크게 선택해 주세요.')
    angle=math.atan2(axis[1],axis[0])
    if abs(angle)>math.radians(35): raise ValueError('얼굴의 기울기가 커요. 정면 사진으로 다시 시도해 주세요.')
    c,s=math.cos(angle),math.sin(angle)
    rotation=np.array([[c,-s],[s,c]])
    origin=centers.mean(axis=0)
    q=(p-origin)@rotation/iod
    aligned=[(e-origin)@rotation/iod for e in eyes]
    widths=np.array([np.linalg.norm(e[0]-e[2]) for e in aligned])
    heights=np.array([np.linalg.norm(e[1]-e[3]) for e in aligned])
    openness=heights/np.maximum(widths,1e-6)
    if widths.min()<.12 or widths.min()/widths.max()<.5 or np.any((openness<.045)|(openness>1.0)):
        raise ValueError('양쪽 눈을 비교하기 어려워요. 눈이 잘 보이는 정면 사진을 사용해 주세요.')
    nose=q[10];mouth=q[13];chin=q[14]
    if abs(nose[0])>.35 or nose[1]<.12 or mouth[1]-nose[1]<.015 or chin[1]-mouth[1]<.04:
        raise ValueError('정면 얼굴의 눈·코·입 위치를 안정적으로 비교하기 어려워요.')
    # 양쪽 눈의 바깥쪽/안쪽을 사진 좌우 기준으로 재지정해서 좌우 반전에 동일하게 계산합니다.
    tilts=[]
    for side,e in enumerate(aligned):
        corners=e[[0,2]]
        outer=corners[np.argmin(corners[:,0]) if side==0 else np.argmax(corners[:,0])]
        inner=corners[np.argmax(corners[:,0]) if side==0 else np.argmin(corners[:,0])]
        tilts.append((inner[1]-outer[1])/max(np.linalg.norm(inner-outer),1e-6))
    features=np.array([openness.mean(),np.mean(tilts),1/widths.mean(),
        np.linalg.norm(q[8]-q[9]),nose[1],mouth[1]-nose[1],
        np.linalg.norm(q[11]-q[12]),chin[1]],dtype=np.float32)
    if not np.isfinite(features).all(): raise ValueError('invalid_geometry')
    return features


def reference_stats(features):
    """사람·개의 주둥이 길이 차이를 줄이기 위해 종별 중앙값과 산포를 구합니다."""
    values=np.asarray(features,dtype=np.float32)
    center=np.median(values,axis=0)
    q1,q3=np.percentile(values,[25,75],axis=0)
    scale=np.maximum((q3-q1)/1.349,1e-3)
    return {'center':center.tolist(),'scale':scale.tolist(),'count':len(values)}


def standardize(features,stats):
    # 각각의 종 안에서 '평균보다 눈이 큰 편인가' 등을 맞춰 비교합니다.
    return np.clip((np.asarray(features)-np.asarray(stats['center']))/np.asarray(stats['scale']),-3,3).astype(np.float32)


def rank_faces(query,dog_features,human_stats,dog_stats):
    q=standardize(query,human_stats)
    candidates=standardize(dog_features,dog_stats)
    difference=np.abs(candidates-q)
    distance=(difference*FEATURE_WEIGHTS).sum(axis=1)
    # 프로젝트에서 정한 점수식입니다. 닮았을 확률이나 검증 정확도가 아닙니다.
    scores=100/(1+distance)
    return np.argsort(-scores,kind='stable'),scores,difference


def draw_landmarks(image, points, box=None, numbers=False):
    """추출 위치를 직접 확인하는 용도. 점과 선은 검색 점수에 영향을 주지 않습니다."""
    canvas=image.convert('RGB').copy()
    if box is None: box=(0,0,*canvas.size)
    canvas=canvas.crop(tuple(int(v) for v in box))
    scale=min(1.0,640/max(canvas.size))
    canvas=canvas.resize((max(1,round(canvas.width*scale)),max(1,round(canvas.height*scale))))
    coords=(np.asarray(points)-np.array(box[:2]))*scale
    draw=ImageDraw.Draw(canvas)
    if len(coords)==15:
        for indices in [[0,1,2,3,0],[4,5,6,7,4],[8,10,9],[11,13,12],[13,14]]:
            draw.line([tuple(coords[i]) for i in indices],fill='#65ffca',width=2)
    for i,(x,y) in enumerate(coords):
        color='#65ffca' if i<24 else '#ffd56a'
        draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
        if numbers: draw.text((x+3,y-7),str(i),fill='#ff3030',stroke_width=1,stroke_fill='white')
    return canvas
