FaceAge 모델 학습

기본 모델: EfficientNetV2-S
훈련 10,190장 / 검증 2,546장
최종 저장본: 9회차, 검증 MAE 4.99세

핵심 코드
train.py                 나이 예측 모델 학습
evaluate.py              저장된 예측 기록의 성능 평가
prepare_data.py          숫자 나이 라벨 통합 및 중복 제거
face_landmarks.py        사람·강아지 얼굴 좌표와 비율 계산
prepare_models.py        얼굴 특징점 모델 파일 확인 및 준비
build_dog_index.py       강아지 얼굴 검색 목록 생성
show_training_log.py     최종 모델의 저장된 학습 로그 확인
experiments/             모델·데이터 수·미세조정 비교 실험

데이터와 결과
data/age_merged/          통합 학습 데이터와 labels.csv
data/age_regression/      기존 3,137장 자료
models/face_landmarks/    공개 특징점 모델 및 출처·라이선스
results/                 최고 모델, 학습 기록, 검증 예측값과 비교 결과

실행 환경
D:\mbca_home\YOLO\전이학습\.venv\Scripts\python.exe

학습 실행: 위 파이썬으로 train.py 실행
학습 코드는 결과 폴더에 모델과 기록을 저장하므로, 재학습 전 기존 결과를 별도로 보관할 것.
웹 실행: 상위 폴더의 FaceAge_웹실행.bat
강아지 원본 사진: C:\Users\sangn\Desktop\mbc\dog_images
환경 의존성은 requirements-lock.txt에 기록됨.
