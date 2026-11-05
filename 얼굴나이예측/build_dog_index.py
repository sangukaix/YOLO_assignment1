"""강아지 얼굴 랜드마크 검색 준비. 나이 모델을 다시 학습하지 않습니다.

실행: 전이학습/.venv/Scripts/python.exe build_dog_index.py
중간 결과를 64장 단위로 저장하므로 종료 후 같은 명령으로 이어서 실행할 수 있습니다.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import argparse
import hashlib
import json
import time
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from face_landmarks import (PROJECT, MODEL_DIR, SCHEMA, DogLandmarker, HumanLandmarker,
                            dog_common, describe_face, reference_stats, FEATURE_NAMES,
                            FEATURE_WEIGHTS, draw_landmarks)

ROOT=Path(r'C:\Users\sangn\Desktop\mbc\dog_images')
OUTPUT=PROJECT/'results/dog_face_geometry'
_DOG=None


def init_worker():
    global _DOG
    _DOG=DogLandmarker(threads=1)


def process_chunk(samples):
    records=[]
    for relative,body_box in samples:
        record={'path':relative,'breed':Path(relative).parent.name}
        try:
            with Image.open(ROOT/relative) as source:
                image=ImageOps.exif_transpose(source).convert('RGB')
            # 기존 몸통 탐지 영역이 있으면 얼굴을 찾을 범위만 좁힙니다.
            # 최종 검색 벡터에는 몸통/배경 이미지 특징이 들어가지 않습니다.
            crop_box=(0,0,image.width,image.height)
            if body_box:
                x0,y0,x1,y1=body_box;mx=(x1-x0)*.08;my=(y1-y0)*.08
                crop_box=(max(0,int(x0-mx)),max(0,int(y0-my)),min(image.width,int(x1+mx)),min(image.height,int(y1+my)))
            result=_DOG.detect(image.crop(crop_box))
            offset=np.array(crop_box[:2])
            points=result['points']+offset
            local_box=result['box']
            box=np.asarray(local_box)+np.tile(offset,2)
            record.update(points=points.tolist(),box=box.astype(int).tolist())
        except (OSError,ValueError) as error:
            record['error']=str(error)
        records.append(record)
    return records


def prepare_human_reference():
    """나이 정답을 학습하는 과정이 아닙니다. 사람 얼굴 비율의 기준 분포만 구합니다."""
    import csv
    data=PROJECT/'data/age_merged'
    with (data/'labels.csv').open(encoding='utf-8-sig',newline='') as file:
        rows=list(csv.DictReader(file))
    train=[r for r in rows if r['split']=='train']
    rng=np.random.default_rng(42)
    order=rng.permutation(len(train))
    model=HumanLandmarker();features=[];used=[];attempts=0
    for position in order:
        row=train[position];attempts+=1
        try:
            with Image.open(data/row['file']) as source:image=source.convert('RGB')
            result=model.detect(image)
            features.append(result['features']);used.append(row['file'])
        except (OSError,ValueError): pass
        if len(features)>=512 or attempts>=3000:break
    if len(features)<100:raise RuntimeError('사람 얼굴 기준 사진이 100장 미만입니다.')
    # 예측값으로 고르지 않고, 얼굴 비교가 가능한 첫 성인 검증 사진을 체험용으로 씁니다.
    demo=None
    for row in rows:
        if row['split']!='validation' or not 20<=float(row['age'])<=40:continue
        try:
            with Image.open(data/row['file']) as source:image=source.convert('RGB')
            result=model.detect(image)
            demo={'file':row['file'],'label_age':float(row['age']),'box':result['box'],
                  'reason':'first adult validation photo passing face geometry checks; not selected by prediction'}
            break
        except (OSError,ValueError):continue
    model.close()
    return reference_stats(features),{'attempted':attempts,'accepted':len(features),'files':used},demo


def main(workers=8):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    body_boxes=json.loads((OUTPUT/'source_boxes.json').read_text(encoding='utf-8'))
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'})
    samples=[(str(p.relative_to(ROOT)),body_boxes.get(str(p.relative_to(ROOT)))) for p in files]
    fingerprint=hashlib.sha256((''.join(f'{p.relative_to(ROOT)}:{p.stat().st_size}:{p.stat().st_mtime_ns}\n' for p in files)
                +(MODEL_DIR/'manifest.json').read_text(encoding='utf-8')).encode()).hexdigest()[:16]
    cache=OUTPUT/('cache_'+fingerprint);cache.mkdir(exist_ok=True)
    chunks=[samples[i:i+64] for i in range(0,len(samples),64)]
    pending=[i for i in range(len(chunks)) if not (cache/f'{i:04}.json').is_file()]
    started=time.time();done=len(chunks)-len(pending)
    print(f'원본 {len(files)}장 · 완료된 묶음 {done}/{len(chunks)} · CPU 작업자 {workers}',flush=True)
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker) as pool:
        futures={pool.submit(process_chunk,chunks[i]):i for i in pending}
        for future in as_completed(futures):
            i=futures[future];records=future.result()
            temp=cache/f'{i:04}.tmp';temp.write_text(json.dumps(records,ensure_ascii=False),encoding='utf-8')
            temp.replace(cache/f'{i:04}.json');done+=1
            if done%8==0 or done==len(chunks):
                print(f'얼굴 좌표 추출 {min(done*64,len(samples)):,}/{len(samples):,}장 · {time.time()-started:.0f}초',flush=True)
    accepted=[];excluded=Counter()
    for i in range(len(chunks)):
        for row in json.loads((cache/f'{i:04}.json').read_text(encoding='utf-8')):
            try:
                if row.get('error'):raise ValueError(row['error'])
                row['features']=describe_face(dog_common(row['points']))
                accepted.append(row)
            except ValueError as error:excluded[str(error)]+=1
    if not accepted:raise RuntimeError('품질 조건을 통과한 강아지 얼굴이 없습니다.')
    print('사람 얼굴 기준 분포를 계산합니다.',flush=True)
    human_stats,human_audit,demo=prepare_human_reference()
    features=np.stack([x['features'] for x in accepted]);dog_stats=reference_stats(features)
    meta={'schema':SCHEMA,'source_count':len(files),'image_count':len(accepted),'excluded_count':len(files)-len(accepted),
          'breed_count':len({x['breed'] for x in accepted}),'feature_names':FEATURE_NAMES,'feature_weights':FEATURE_WEIGHTS.tolist(),
          'human_stats':human_stats,'dog_stats':dog_stats,'human_reference':human_audit,
          'excluded_reasons':dict(excluded),'fingerprint':fingerprint,'dog_model_revision':json.loads((MODEL_DIR/'manifest.json').read_text(encoding='utf-8'))['dog_revision'],
          'dog_model':'DogFLW · MobileNetV3-Large · 46 points','localizer':'EfficientNetB2 · 224px',
          'human_model':'MediaPipe Face Landmarker','runtime':'LiteRT CPU; age model CUDA unchanged',
          'score':'100 / (1 + weighted absolute difference of species-standardized geometry); not probability',
          'quality_filter':'heuristic eye visibility, eye size, frontal symmetry and landmark ordering; not a calibrated face confidence',
          'validation':'No labeled human-dog resemblance benchmark. Ranking weights are manually designed.'}
    with (OUTPUT/'index.tmp').open('wb') as file:
        np.savez_compressed(file,features=features,points=np.array([x['points'] for x in accepted],dtype=np.float32),
            boxes=np.array([x['box'] for x in accepted],dtype=np.int32),paths=np.array([x['path'] for x in accepted]),
            breeds=np.array([x['breed'] for x in accepted]),metadata=np.array(json.dumps(meta,ensure_ascii=False)))
    (OUTPUT/'index.tmp').replace(OUTPUT/'index.npz')
    (OUTPUT/'info.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    if demo:(OUTPUT/'demo_example.json').write_text(json.dumps(demo,ensure_ascii=False,indent=2),encoding='utf-8')
    # 품종·점수를 보고 선별하지 않은 고정 난수 예시로 위치 추출 품질을 확인합니다.
    positions=np.random.default_rng(42).choice(len(accepted),size=min(16,len(accepted)),replace=False)
    sheet=Image.new('RGB',(1000,1000),'#eeeeff')
    for j,position in enumerate(positions):
        row=accepted[position]
        with Image.open(ROOT/row['path']) as source:im=ImageOps.exif_transpose(source).convert('RGB')
        pic=draw_landmarks(im,row['points'],row['box']);pic.thumbnail((240,220))
        sheet.paste(pic,((j%4)*250,(j//4)*250))
        ImageDraw.Draw(sheet).text(((j%4)*250,(j//4)*250+225),row['breed'],fill='black')
    sheet.save(OUTPUT/'quality_examples.jpg')
    print(json.dumps({k:meta[k] for k in ['source_count','image_count','excluded_count','breed_count']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8)
    main(parser.parse_args().workers)
