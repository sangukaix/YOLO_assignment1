FaceAge 웹

얼굴 사진에서 나이를 예측하고 얼굴 비율이 가까운 강아지를 검색하는 Streamlit 서비스.

실행: run_web.bat
접속: http://127.0.0.1:8502

구성
app.py                  화면, 사진 자르기, 분석 결과 표시
age_inference.py        나이 모델 불러오기와 예측
dog_face_search.py      얼굴 비율을 이용한 강아지 검색
reports.py              모델 선택 목록과 비교 기록 읽기
validation_metrics.py   MAE 등 검증 지표 계산
assets/                 화면 스타일, 모델 정보, 품종 한글명
.streamlit/             Streamlit 실행 설정
.runtime/               실행 중 생성되는 로그와 프로세스 번호

학습 코드와 데이터는 옆의 얼굴나이예측 폴더에서 관리함.
파이썬 실행 환경은 상위 전이학습/.venv를 사용함.
모델 가중치, 검증 지표, 검색 후보 목록은 저장된 결과를 그대로 읽음.
강아지 사진 원본 경로: C:\Users\sangn\Desktop\mbc\dog_images

웹페이지_실행.ps1       백그라운드 실행
웹페이지_종료.ps1       로컬 웹 종료
공유링크_종료.bat       임시 외부 공유 연결 종료
외부 공유 주소는 공유 연결이 실행 중일 때만 유효함.
