"""저장된 출처/체크섬과 대조하고, 없는 공개 모델 파일만 내려받습니다."""
from pathlib import Path
import hashlib
import json
import urllib.request


def prepare():
    folder=Path(__file__).resolve().parent/'models/face_landmarks'
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    names=['dog_face_localizer.tflite','dog_face_landmarks_full.tflite','face_landmarker.task',
           'DOG_MODEL_LICENSE.txt','DOG_MODEL_CARD.md']
    for name in names:
        record=manifest['files'][name];path=folder/name
        if not path.is_file():
            temp=path.with_suffix(path.suffix+'.download')
            urllib.request.urlretrieve(record['url'],temp)
            if hashlib.sha256(temp.read_bytes()).hexdigest()!=record['sha256']:
                raise RuntimeError(f'다운로드 체크섬 불일치: {name}')
            temp.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise RuntimeError(f'기존 파일 체크섬 불일치: {name}')
        print('OK',name)


if __name__=='__main__':prepare()
