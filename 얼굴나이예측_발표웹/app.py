"""발표용 FaceAge. 학습은 옆의 학습 프로젝트에서, 화면과 추론은 이 폴더에서 처리합니다."""
from pathlib import Path
from io import BytesIO
import base64
import hashlib
import html
import json
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from streamlit_cropper import st_cropper
from age_inference import TRAINING_PATH, load_model, predict
from dog_face_search import INDEX_PATH, load_search, find_similar
from reports import read_json, architecture_results, model_choices
from validation_metrics import metrics_for_csv, rounded_age

WEB = Path(__file__).resolve().parent
st.set_page_config(page_title='FaceAge · 얼굴 나이와 닮은 강아지', page_icon='◌', layout='wide')
st.markdown('<style>'+ (WEB/'assets/style.css').read_text(encoding='utf-8') +'</style>', unsafe_allow_html=True)


def image_url(image, size=640):
    copy=image.convert('RGB').copy()
    copy.thumbnail((size,size))
    stream=BytesIO();copy.save(stream,format='JPEG',quality=90)
    return 'data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode()


def table(headers, rows, best_index=None, compact=False):
    """정적 표에도 학습 중/미학습 상태를 구분하고, 가장 낮은 오차를 강조합니다."""
    head=''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)
    body=''
    for i,row in enumerate(rows):
        cells=''.join('<td>'+html.escape(str(x)).replace('\n','<br>')+'</td>' for x in row)
        body+=f'<tr class="{"best" if i==best_index else ""}">{cells}</tr>'
    css_class='report-table compact' if compact else 'report-table'
    st.markdown(f'<div class="table-wrap"><table class="{css_class}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>',unsafe_allow_html=True)


@st.cache_resource(max_entries=1)
def age_model(path, modified, schema=4):
    return load_model(path)


@st.cache_resource(max_entries=1)
def dog_search(modified):
    return load_search()


st.markdown('<div class="site-header"><div class="brand">FaceAge</div></div>',unsafe_allow_html=True)
st.markdown('''<div class="hero">
<h1>나와 닮은 친구를 만났어요</h1>
<p>얼굴 사진 한 장을 올려 보세요.<br>직접 학습한 모델이 나이를 예측하고 닮은 강아지를 찾아줘요.</p>
<div class="steps"><span><b>1</b>사진 올리기</span><i class="line"></i><span><b>2</b>영역 자르기</span><i class="line"></i><span><b>3</b>결과 확인</span></div>
</div>''',unsafe_allow_html=True)

choices=model_choices()
architecture_keys=[k for k in ['resnet18','resnet152','efficientnet_v2_s'] if k in choices]
default_key=min(architecture_keys,key=lambda k:choices[k]['mae'])
# 이전에 선택했던 항목이 목록에서 제외되면 기본 모델로 전환합니다.
if 'age_model_choice' in st.session_state and st.session_state['age_model_choice'] not in choices:
    st.session_state.pop('age_model_choice')
image=None;actual_age=None;image_name='';crop_image=None

with st.container(key='upload_card'):
    uploaded=st.file_uploader('얼굴 사진 올리기',type=['jpg','jpeg','png','webp'],
                              max_upload_size=20,label_visibility='collapsed',key='face_upload')
    if uploaded is None:
        if st.button('사진이 없나요? 예시 사진으로 체험하기',key='example_btn',width='stretch'):
            st.session_state['demo_photo']=True
    if uploaded is not None:
        try:
            with Image.open(BytesIO(uploaded.getvalue())) as source:
                image=ImageOps.exif_transpose(source).convert('RGB')
            image_name=uploaded.name
        except (OSError,ValueError,UnidentifiedImageError,Image.DecompressionBombError):
            st.error('사진을 읽을 수 없어요. JPG, PNG, WEBP 사진을 다시 선택해 주세요.')

    # 모델 선택은 발표 화면을 가리지 않도록 접힌 설정 안에 배치합니다.
    with st.expander('분석 모델 설정',expanded=False):
        model_key=st.selectbox('나이 예측에 사용할 모델',list(choices),
                              index=list(choices).index(default_key),
                              format_func=lambda k:choices[k]['label'],key='age_model_choice')
    choice=choices[model_key]

    if uploaded is None and st.session_state.get('demo_photo'):
        # 예측값을 보고 고르는 대신, 검증 목록에서 나이 라벨이 20~40인 첫 사진을 고릅니다.
        # 모델을 바꿔도 같은 예시 사진을 유지해야 예측값을 공정하게 비교할 수 있습니다.
        validation_csv=TRAINING_PATH/'results/merged/age_validation_predictions.csv'
        examples=pd.read_csv(validation_csv).to_dict('records') if validation_csv.is_file() else []
        if examples:
            # 예측 결과로 예시를 선별하지 않습니다. 첫 성인 나이 라벨 예시를 사용합니다.
            # 정면 얼굴 비교가 가능한 첫 성인 사진. 나이 예측 성능으로 고르지 않습니다.
            example=read_json(INDEX_PATH.parent/'demo_example.json') or next((x for x in examples if 20<=x['label_age']<=40),examples[0])
            with Image.open(TRAINING_PATH/'data/age_merged'/example['file']) as source:image=source.convert('RGB')
            actual_age=example['label_age'];image_name='검증 예시 사진'

    if image is not None:
        st.markdown(f'<div class="ready-photo"><img src="{image_url(image,180)}"><p>{html.escape(image_name)}</p><small>사진이 준비됐어요. 필요하면 얼굴 부분만 선택해 주세요.</small></div>',unsafe_allow_html=True)
        crop_image=image
        with st.expander('마우스로 사진 자르기',expanded=True):
            st.caption('모서리를 끌어 크기를 조절하고, 사각형 안쪽을 끌어 위치를 옮겨 주세요.')
            preview=image.copy();preview.thumbnail((300,340))
            crop_key=hashlib.sha256(image.tobytes()).hexdigest()[:16]
            rect=st_cropper(preview,realtime_update=True,box_color='#8b88f3',aspect_ratio=None,
                            default_coords=(0,preview.width,0,preview.height),
                            return_type='box',should_resize_image=False,key='crop_'+crop_key)
            # 화면의 좌표를 원본 사진 좌표로 환산합니다. 모델에는 원본의 선택 영역을 넣습니다.
            sx,sy=image.width/preview.width,image.height/preview.height
            x0=max(0,int(rect['left']*sx));y0=max(0,int(rect['top']*sy))
            x1=min(image.width,int((rect['left']+rect['width'])*sx))
            y1=min(image.height,int((rect['top']+rect['height'])*sy))
            if x1>x0 and y1>y0:crop_image=image.crop((x0,y0,x1,y1))
            st.caption(f'분석 영역 {crop_image.width} × {crop_image.height} 픽셀')

    # 다른 사진/영역/모델로 바꾸면 이전 분석 결과를 즉시 비웁니다.
    index_version=str(INDEX_PATH.stat().st_mtime_ns) if INDEX_PATH.is_file() else 'preparing'
    current_key=(hashlib.sha256(crop_image.tobytes()).hexdigest()+model_key+index_version) if crop_image is not None else ''
    if st.session_state.get('analysis_key')!=current_key:
        st.session_state.pop('faceage_result',None)
    clicked=st.button('나이와 닮은 강아지 분석하기',type='primary',width='stretch',disabled=image is None)
    if clicked:
        if not INDEX_PATH.is_file():
            st.info('강아지 사진 검색을 준비하고 있어요. 잠시 후 다시 눌러 주세요.')
        else:
            try:
                with st.spinner('사진 속 나이와 닮은 강아지를 찾고 있어요…'):
                    model,device,info=age_model(str(choice['checkpoint']),choice['checkpoint'].stat().st_mtime_ns)
                    info=dict(info)
                    saved_metrics=read_json(choice['folder']/'age_metrics.json')
                    info['train_count']=info.get('train_count') or saved_metrics.get('train_count')
                    info['validation_count']=info.get('validation_count') or saved_metrics.get('validation_count')
                    info['experiment_name']=choice['label']
                    info['validation_metrics']=metrics_for_csv(choice['folder']/'age_validation_predictions.csv')
                    age,elapsed=predict(crop_image,model,device,info['target_scale'])
                    encoder,index,dog_device=dog_search(INDEX_PATH.stat().st_mtime_ns)
                    # 얼굴 비교가 어려워도 나이 결과는 따로 보여 줍니다.
                    match=None;dog_url='';dog_error=''
                    try:
                        match=find_similar(crop_image,encoder,index,dog_device)
                        with Image.open(match['path']) as dog_image:
                            dog_image=ImageOps.exif_transpose(dog_image)
                            dog_url=image_url(dog_image.crop(match['box']))
                    except ValueError as error:dog_error=str(error)
                st.session_state['analysis_key']=current_key
                st.session_state['faceage_result']={'age':age,'elapsed':elapsed,'info':info,'match':match,
                    'face_url':image_url(crop_image),'dog_url':dog_url,'actual_age':actual_age,'dog_error':dog_error}
            except (OSError,RuntimeError) as error:
                st.error('분석 중 문제가 생겼어요. 사진이나 모델 파일을 확인해 주세요.')
                with st.expander('오류 내용 확인'):st.code(str(error))

result=st.session_state.get('faceage_result')
if result is not None:
    age=result['age'];match=result['match'];info=result['info']
    validation=info.get('validation_metrics') or metrics_for_csv(choice['folder']/'age_validation_predictions.csv')
    actual_note=f"검증 나이 라벨 {result['actual_age']:.0f}세 · 오차 {abs(age-result['actual_age']):.1f}세" if result['actual_age'] is not None else ''
    if match:
        # 영문 품종명과 한글 대응명을 두 줄로 보여 줍니다.
        breed_en=match['breed'].replace('_',' ').title()
        breed_ko=read_json(WEB/'assets/breed_names_ko.json').get(match['breed'], '')
        breed_label=html.escape(breed_en)+(f'<br>{html.escape(breed_ko)}' if breed_ko else '')
        dog_card=f'''<div class="result-card dog"><img class="photo" src="{result['dog_url']}" alt="얼굴 비율이 가장 가까운 강아지 사진">
        <div class="result-label"><span>{breed_label}</span></div><div class="result-number">{match['similarity']:.1f}<small>%</small></div>
        <div class="note">DogFLW · MobileNetV3-Large</div></div>'''
    else:
        dog_card=f'''<div class="result-card dog"><div class="result-label">강아지 얼굴 비교</div>
        <div class="note">{html.escape(result.get('dog_error','정면 얼굴 사진을 선택해 주세요.'))}</div></div>'''
    result_heading='나와 닮은 친구를 만났어요' if match else '나이 분석 결과'
    st.markdown(f'''<div class="result-heading"><h2>{result_heading}</h2></div>
    <div class="result-shell"><div class="result-grid">
    <div class="result-card"><img class="photo" src="{result['face_url']}" alt="분석한 얼굴 사진">
    <div class="result-label">사진 속 예측 나이</div><div class="result-number">{rounded_age(age)}<small>세</small></div><div class="note">{actual_note}</div></div>
    {dog_card}</div>
    <div class="result-meta"><span>{html.escape(choice['label'])}</span><span>검증 평균 절대 오차 (MAE) {info['validation_mae']:.2f}세 · {info.get('validation_count') or '—'}장</span><span>모델 예측 {result['elapsed']:.0f} ms</span></div>
    <div class="accuracy-grid">
    <div><small>나이 100% 일치율</small><span class="metric-alias">Accuracy · 반올림 기준</span><strong>{validation['exact_accuracy']:.2%}</strong><span>{validation['exact_count']:,} / {validation['count']:,}장</span></div>
    <div><small>나이 ±5세 이내 일치율</small><span class="metric-alias">Within ±5 years</span><strong>{validation['within_5_years']:.2%}</strong><span>{validation['within5_count']:,} / {validation['count']:,}장</span></div>
    <div><small>나이 ±10세 이내 일치율</small><span class="metric-alias">Within ±10 years</span><strong>{validation['within_10_years']:.2%}</strong><span>{validation['within10_count']:,} / {validation['count']:,}장</span></div>
    </div></div>''',unsafe_allow_html=True)
    if match:
        with st.expander('눈·코·입 비교 근거 보기',expanded=False):
            # 좁은 화면에서도 두 얼굴을 나란히 비교할 수 있게 표시합니다.
            st.markdown(f'''<div class="result-grid">
            <div class="result-card"><img class="photo" src="{image_url(match['human_landmarks'])}" alt="사람 얼굴 특징점">
            <div class="note">사람 얼굴 · 비교에 사용한 15개 지점</div></div>
            <div class="result-card"><img class="photo" src="{image_url(match['dog_landmarks'])}" alt="강아지 얼굴 특징점">
            <div class="note">강아지 얼굴 · 46개 중 공통 15개 지점</div></div></div>''',unsafe_allow_html=True)
            table(['비교 항목','거리 반영 비중','항목별 점수'],
                  [[c['name'],f"{c['weight']:.0%}",f"{c['score']:.1f}점"] for c in match['components']],compact=True)
            # 학습용 데이터셋, 딥러닝 모델, 실행 라이브러리, 검색용 사진을 구분합니다.
            dog_details=read_json(INDEX_PATH.parent/'info.json')
            st.markdown('**닮은 강아지를 찾는 데 사용한 기술**')
            with st.container(key='dog_technology'):
                table(['구분','이름과 역할'],[
                    ['모델 학습용 데이터셋','DogFLW: 강아지 얼굴에 눈·코·입·턱·귀 등 46개 기준점이 표시된 데이터셋. 공개된 강아지 특징점 모델의 학습에 사용됐어요.'],
                    ['강아지 얼굴 위치 모델',f"{dog_details.get('localizer','EfficientNetB2')}: 사진에서 강아지 얼굴 위치를 예측하는 딥러닝 모델. 찾은 얼굴을 잘라 다음 모델에 전달해요."],
                    ['강아지 특징점 모델','MobileNetV3-Large: DogFLW로 학습된 공개 딥러닝 모델. 강아지 얼굴의 특징점 46개를 예측해요.'],
                    ['사람 얼굴 분석','MediaPipe: 파이썬에서 사용하는 라이브러리. 딥러닝 기반 Face Landmarker로 사람의 눈·코·입 등 얼굴 좌표를 추출해요.'],
                    ['모델 실행 라이브러리','LiteRT (ai-edge-litert): 학습된 강아지 모델 파일(.tflite)을 내 컴퓨터에서 실행하는 파이썬 라이브러리예요.'],
                    ['특징점과 비교 방식','46포인트는 얼굴의 기준점 46개와 각 점의 (x, y) 좌표예요. 사람·강아지에서 공통 15개 지점을 골라 얼굴 비율 8개를 계산하고 비교해요.'],
                    ['검색용 원본 사진',f"dog_images 폴더의 {match['source_count']:,}장. DogFLW 학습용 데이터와 구분되는, 닮은 강아지 후보 사진이에요."],
                    ['최종 검색 목록',f"눈 간격·얼굴 기울기 등 좌표 품질 조건을 통과한 {match['searched_count']:,}장 · {dog_details['breed_count']:,}개 품종. 나머지 {match['source_count']-match['searched_count']:,}장은 검색에서 제외했어요."]
                ],compact=True)
            st.caption('[DogFLW 데이터셋](https://github.com/martvelge/DogFLW) · [강아지 얼굴 공개 모델](https://huggingface.co/hugocornellier/dog-face-landmarks)')
            with st.expander('실제로 계산한 얼굴 비율 8개'):
                table(['특징','사람','강아지'],[[n,f'{a:.3f}',f'{b:.3f}'] for n,a,b in zip(match['feature_names'],match['human_features'],match['dog_features'])],compact=True)
                st.caption('위 값은 표준화 전 비율이에요. 검색에는 종별 중앙값·산포로 표준화한 값의 가중 절대 거리와 100/(1+거리) 점수를 사용해요.')
            st.markdown('**다른 후보 사진**')
            for column,item in zip(st.columns(3),match['alternatives']):
                with column:
                    with Image.open(item['path']) as source:
                        candidate=ImageOps.exif_transpose(source).convert('RGB').crop(item['box'])
                    st.image(candidate,caption=f"얼굴 구조 {item['score']:.1f}점",width='stretch')

st.markdown('<div class="section-intro"><h2>모델 정보</h2></div>',unsafe_allow_html=True)
with st.container(key='reports_card'):
    # 가장 중요한 데이터 정보를 먼저 보여 줍니다.
    report=read_json(TRAINING_PATH/'results/merged/data_report.json')
    # 설정은 선택 모델의 기록을 읽습니다. FC 전용·마지막 3회 실험도 구분합니다.
    start_epoch=choice['finetune_start_epoch']
    if start_epoch is None:
        training_order=f"1~{choice['total_epochs']}회: FC 출력층만 학습"
    else:
        feature_scope='마지막 특징 모듈 2개' if choice['finetuned_feature_modules']==2 else 'layer4 단계'
        training_order=f"1~{start_epoch-1}회: FC 출력층만\n{start_epoch}~{choice['total_epochs']}회: {feature_scope} + FC 출력층"
    with st.container(key='training_summary'):
        table(['항목','내용'],[
            ['선택 모델 훈련 / 검증\n(train / validation)',f"{choice['train_count']:,}장 / {choice['validation_count']:,}장"],
            ['기존 나이 라벨 (label)','파일명 앞 숫자'],['추가 나이 라벨 (label)','폴더 이름의 숫자'],
            ['통합 데이터 (dataset)',f"{report['usable_images']:,}장"],
            ['중복 / 라벨 충돌 제외',f"{report['duplicate_extra_files']:,}개 / {report['conflicting_file_count']:,}개"],
            ['입력 사진 크기 (Resize)','224 × 224 픽셀'],
            ['배치 크기 (batch_size)','32장'],
            ['총 에포크 (epochs)',f"총 {choice['total_epochs']}회 실행\n최고 저장: {choice['saved_epoch']}회차 (검증 MAE 최저)"],
            ['가중치 학습 범위\n(requires_grad)',training_order],
            ['전체 파라미터 (parameters)',f"{choice['total_parameters']:,}개"],
            ['학습 대상 파라미터\n(requires_grad=True)',f"{choice['trainable_parameters']:,}개 · 저장 모델 기준"],
            ['고정 파라미터\n(requires_grad=False)',f"{choice['frozen_parameters']:,}개 · 저장 모델 기준"],
            ['최적화 방법 (optimizer)','Adam · optim.Adam'],
            ['출력층 학습률 (lr)','0.001'],
            ['특징층 미세조정 학습률 (lr)','0.0001' if start_epoch else '미세조정 없음'],
            ['드롭아웃 (Dropout)','실제 학습에서 미적용'],
            ['평균 절대 오차 (MAE)',f"{choice['mae']:.2f}세\n예측 나이와 정답 나이의 절대 차이를 검증 사진 전체에서 평균한 값"],
        ],compact=True)
    with st.expander('모델별 결과·비교 자세히 보기',expanded=False):
        st.markdown('**모델별 결과**')
        architectures = architecture_results()
        completed = sorted([r for r in architectures if r['ready']], key=lambda r: r['metrics']['validation_mae'])
        ordered = completed + [r for r in architectures if not r['ready']]
        if completed:
            st.markdown(f"**현재 검증 MAE가 가장 낮은 모델: {completed[0]['name']} · {completed[0]['metrics']['validation_mae']:.2f}세**")
        rows = []
        for r in ordered:
            m = r['metrics']
            rows.append([r['name'], r['status'], f"{r['total_parameters']:,}개", f"{m['validation_mae']:.2f}세" if r['ready'] else '—', f"{m['rmse']:.2f}세" if r['ready'] else '—', f"{m['within_5_years']:.2%}" if r['ready'] else '—', f"{m['saved_epoch']}회차" if r['ready'] else '—'])
        table(['모델', '상태', '전체 파라미터', '평균 절대 오차 MAE ↓', 'RMSE ↓', '±5세 이내 ↑', '선택 회차'], rows, 0 if completed else None)
        st.caption('세 모델 모두 같은 통합 데이터, 224×224 전처리, 10회 학습 조건이에요. 미완료 모델에는 성능을 표시하지 않아요.')
        st.markdown('**모델별 추가 지표와 그래프 보기**')
        table(['모델', '전체 가중치 수', 'FC층만 학습할 때', '미세조정 대상', '고정된 가중치'], [[r['name']] + [f"{r['parameter_info'].get(k, 0):,}개" for k in ['total_parameters', 'head_parameters', 'trainable_parameters', 'frozen_parameters']] for r in completed])
        st.caption('파라미터는 모델의 가중치 개수예요. 훈련·검증·테스트 사진 수와는 다른 개념이고, 검증 과정에서는 가중치를 수정하지 않아요.')
        table(['모델', 'MSE ↓ (세²)', '결정계수 R² ↑', '±10세 이내 ↑', '검증 사진 수'], [[r['name'], f"{r['metrics']['mse']:.2f}", f"{r['metrics']['r2']:.3f}", f"{r['metrics']['within_10_years']:.2%}", f"{r['metrics']['validation_count']:,}장"] for r in completed])
        st.caption('MSE는 오차 제곱의 평균, R²는 나이의 변동을 설명하는 정도예요. R²는 1에 가까울수록 좋아요.')
        curves = {'회차': list(range(1, 11))}
        for r in completed:
            curves[r['name']] = read_json(r['folder'] / 'age_history.json')['val_mae']
        st.line_chart(pd.DataFrame(curves), x='회차', y=list(curves)[1:], x_label='학습 회차', y_label='검증 MAE (세)')
        if completed:
            selected_report = st.selectbox('검증 그래프에서 볼 모델', completed, format_func=lambda r: r['name'])
            predictions = pd.read_csv(selected_report['folder'] / 'age_validation_predictions.csv')
            st.caption('가로축은 라벨 나이, 세로축은 예측 나이예요. 같은 나이 근처에 모일수록 좋아요.')
            st.scatter_chart(predictions.rename(columns={'label_age': '라벨 나이', 'predicted_age': '예측 나이'}), x='라벨 나이', y='예측 나이', size=12)
            groups = predictions.assign(나이대=(predictions.label_age // 10 * 10).astype(int)).groupby('나이대').agg(사진수=('label_age', 'size'), MAE=('absolute_error', 'mean')).reset_index()
            st.caption('나이대별 사진 수와 MAE도 함께 확인해 주세요. 사진 수가 적은 구간의 수치는 크게 흔들릴 수 있어요.')
            table(['나이대', '검증 사진 수', '평균 절대 오차 (MAE)'], [[f'{int(r.나이대)}~{int(r.나이대) + 9}세', f'{int(r.사진수):,}장', f'{r.MAE:.2f}세'] for r in groups.itertuples()])
        st.download_button('모델 비교표 내려받기', pd.DataFrame(rows, columns=['모델', '상태', '전체 파라미터', '평균 절대 오차 (MAE)', 'RMSE', '나이 ±5세 이내 일치율', '선택 회차']).to_csv(index=False).encode('utf-8-sig'), '모델별_학습결과.csv', 'text/csv')
        if st.button('최신 학습 결과 불러오기', key='refresh_reports'):
            st.rerun()
        evaluation = read_json(TRAINING_PATH / 'results/evaluation/validation_summary.json')
        accuracy_rows = [r for r in evaluation.get('records', []) if r['scope'] == 'common']
        if accuracy_rows:
            st.markdown('**같은 검증 2,546장에서 세 가지 기준으로 맞힌 비율**')
            table(['모델·훈련 자료', '나이 100% 일치율 (Accuracy·반올림)', '나이 ±5세 이내 (Within ±5 years)', '나이 ±10세 이내 (Within ±10 years)'], [[f"{r['model']} · {r['data_source']}", f"{r['exact_accuracy']:.2%}", f"{r['within_5_years']:.2%}", f"{r['within_10_years']:.2%}"] for r in accuracy_rows])
            st.caption('Accuracy는 예측 나이를 정수로 반올림한 뒤 라벨과 정확히 같은 사진 수 ÷ 전체 검증 사진 수예요. 분류 모델의 argmax 대신 반올림을 사용해요. Within ±5 / ±10 years는 반올림 전 절대 오차가 각각 5세 / 10세 이하인 사진의 비율이에요.')
            st.markdown('**다섯 모델의 MSE·RMSE·R² 보기**')
            table(['모델·자료', 'MSE (세²)', 'RMSE (세)', 'R²'], [[f"{r['model']} · {r['data_source']}", f"{r['mse']:.2f}", f"{r['rmse']:.2f}", f"{r['r2']:.3f}"] for r in accuracy_rows])
        # 미세조정 비교는 학습 방식·선택 회차·평균 오차만 보여 줍니다.
        st.markdown('**미세조정 비교**')
        summary=read_json(TRAINING_PATH/'results/finetune_comparison/summary.json')
        labels={'fc_only':'FC층만 학습','late3':'마지막 3회 미세조정','early8':'3회째부터 미세조정'}
        ranked=sorted(summary.get('results',[]),key=lambda r:r['mae'])
        if ranked:
            st.caption('ResNet18 · 훈련 10,190장 / 검증 2,546장 · MAE는 낮을수록 좋아요.')
            table(['학습 방식','선택 회차','MAE (세) ↓'],
                  [[labels[r['strategy']],f"{r['best_epoch']}회차",f"{r['mae']:.2f}"] for r in ranked],
                  best_index=0,compact=True)
            st.markdown(f"**최저 오차: {labels[ranked[0]['strategy']]} · {ranked[0]['mae']:.2f}세**")

        # 데이터 수 비교는 같은 검증 사진에서의 평균 오차 표 하나로 줄입니다.
        st.markdown('**데이터 수 비교**')
        comparison=read_json(TRAINING_PATH/'results/data_size_comparison/summary.json')
        compared=sorted(comparison.get('results',[]),
                        key=lambda r:(r['architecture']!='resnet18',r['train_count']))
        if compared:
            names={'resnet18':'ResNet18','efficientnet_v2_s':'EfficientNetV2-S'}
            best_index=min(range(len(compared)),key=lambda i:compared[i]['validation']['mae'])
            best=compared[best_index]
            st.caption('모든 모델을 같은 검증 사진 2,546장으로 비교했어요.')
            table(['모델','훈련 사진','MAE (세) ↓'],
                  [[names[r['architecture']],f"{r['train_count']:,}장",f"{r['validation']['mae']:.2f}"] for r in compared],
                  best_index=best_index,compact=True)
            st.markdown(f"**최저 오차: {names[best['architecture']]} · 훈련 {best['train_count']:,}장 · {best['validation']['mae']:.2f}세**")

st.markdown('<div class="footer-note">FaceAge · 전이학습으로 구현한 얼굴 나이 예측 프로젝트</div>',unsafe_allow_html=True)
