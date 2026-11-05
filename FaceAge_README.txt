FaceAge

FaceAge_웹실행.bat 실행 후 http://127.0.0.1:8502 에 접속.

얼굴나이예측/         학습 코드, 데이터, 모델, 검증 기록
얼굴나이예측_발표웹/  Streamlit 화면과 예측·검색 코드
FaceAge.code-workspace  VS Code에서 두 폴더 함께 열기

최종 모델: EfficientNetV2-S / 마지막 특징 모듈 2개 + FC / 9회차 저장본
검증: MAE 4.99세, ±5세 64.96%, ±10세 85.15%

공유 실행 환경: 전이학습/.venv
강아지 사진: C:\Users\sangn\Desktop\mbc\dog_images
