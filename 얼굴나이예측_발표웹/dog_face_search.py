"""얼굴 구조 검색 연결. 나이 학습 코드와 별개로 저장된 좌표 목록만 읽습니다."""
from pathlib import Path
from threading import RLock
import importlib.util
import json
import numpy as np
from PIL import Image, ImageOps

TRAINING_PATH=Path(__file__).resolve().parent.parent/'얼굴나이예측'
DOG_ROOT=Path(r'C:\Users\sangn\Desktop\mbc\dog_images')
INDEX_PATH=TRAINING_PATH/'results/dog_face_geometry/index.npz'

# 학습을 실행하지 않는 공통 모델 추론/좌표 계산 파일을 사용합니다.
spec=importlib.util.spec_from_file_location('faceage_geometry',TRAINING_PATH/'face_landmarks.py')
geometry=importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


class FaceEncoder:
    def __init__(self):
        self.model=geometry.HumanLandmarker()
        # Streamlit 캐시 객체는 여러 세션에서 공유될 수 있으므로 추론을 잠급니다.
        self.lock=RLock()

    def detect(self,image):
        with self.lock:return self.model.detect(image)


def load_search():
    with np.load(INDEX_PATH,allow_pickle=False) as saved:
        index={k:saved[k].copy() for k in ['features','points','boxes','paths','breeds']}
        index['meta']=json.loads(str(saved['metadata']))
    if index['meta']['schema']!=geometry.SCHEMA:
        raise RuntimeError('얼굴 특징 검색 목록과 코드 버전이 맞지 않습니다.')
    return FaceEncoder(),index,'cpu'


def find_similar(image,encoder,index,device=None):
    human=encoder.detect(image)
    meta=index['meta']
    order,scores,difference=geometry.rank_faces(human['features'],index['features'],meta['human_stats'],meta['dog_stats'])
    position=int(order[0])
    components=[]
    for name,columns in geometry.GROUPS:
        weight=float(geometry.FEATURE_WEIGHTS[columns].sum())
        distance=float(np.average(difference[position,columns],weights=geometry.FEATURE_WEIGHTS[columns]))
        components.append({'name':name,'weight':weight,'score':100/(1+distance)})
    with Image.open(DOG_ROOT/str(index['paths'][position])) as source:
        dog=ImageOps.exif_transpose(source).convert('RGB')
    box=index['boxes'][position].tolist()
    return {'path':str(DOG_ROOT/str(index['paths'][position])),
        'breed':str(index['breeds'][position]),'similarity':float(scores[position]),
        'box':box,'searched_count':len(index['paths']),'source_count':meta['source_count'],
        'components':components,'human_features':human['features'].tolist(),
        'dog_features':index['features'][position].tolist(),'feature_names':geometry.FEATURE_NAMES,
        'human_landmarks':geometry.draw_landmarks(image,human['common'],human['box']),
        'dog_landmarks':geometry.draw_landmarks(dog,geometry.dog_common(index['points'][position]),box),
        'model_name':meta['dog_model'],'human_model':meta['human_model'],
        'alternatives':[{'path':str(DOG_ROOT/str(index['paths'][int(i)])),
            'box':index['boxes'][int(i)].tolist(),'score':float(scores[int(i)])} for i in order[1:4]]}
