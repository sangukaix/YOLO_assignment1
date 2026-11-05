"""학습 폴더의 실제 결과만 읽습니다. 미완료 모델에는 성능 숫자를 붙이지 않습니다."""
from pathlib import Path
import csv
import json
import math
from functools import lru_cache

TRAINING = Path(__file__).resolve().parent.parent / '얼굴나이예측'
ARCHITECTURES = [('resnet18', 'ResNet18'), ('resnet152', 'ResNet152'),
                 ('efficientnet_v2_s', 'EfficientNetV2-S')]


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {} if default is None else default


@lru_cache(maxsize=16)
def regression_metrics(csv_path, modified):
    with open(csv_path, encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    true = [float(row['label_age']) for row in rows]
    pred = [float(row['predicted_age']) for row in rows]
    errors = [b-a for a,b in zip(true,pred)]
    mse = sum(e*e for e in errors)/len(errors)
    mean = sum(true)/len(true)
    variance = sum((x-mean)**2 for x in true)
    return {'mae':sum(abs(e) for e in errors)/len(errors), 'rmse':math.sqrt(mse),
            'mse':mse, 'r2':1-sum(e*e for e in errors)/variance if variance else None,
            'within_5_years':sum(abs(e)<=5 for e in errors)/len(errors),
            'within_10_years':sum(abs(e)<=10 for e in errors)/len(errors)}


def architecture_results():
    meta = read_json(Path(__file__).resolve().parent/'assets/model_meta.json')
    summaries={r['architecture']:r for r in read_json(TRAINING/'results/architectures/summary.json').get('results',[])}
    rows=[]
    for key,name in ARCHITECTURES:
        folder=TRAINING/'results/merged' if key=='resnet18' else TRAINING/f'results/architectures/{key}'
        ready=(folder/'ready.json').is_file()
        metrics=read_json(folder/'age_metrics.json') if ready else {}
        csv_file=folder/'age_validation_predictions.csv'
        if ready and csv_file.is_file():
            metrics.update(regression_metrics(str(csv_file),csv_file.stat().st_mtime_ns))
        history=read_json(folder/'age_history.json')
        rows.append({'key':key, 'name':name, 'folder':folder, 'ready':ready,
                     'status':'학습·검증 완료' if ready else f"학습 중 · {len(history.get('val_mae',[]))}/10회",
                     'metrics':metrics, 'total_parameters':meta.get(key,{}).get('total_parameters'),
                     'parameter_info':summaries.get(key,{})})
    return rows


def model_choices():
    choices={}
    architectures=architecture_results()
    parameter_info={row['key']:row['parameter_info'] for row in architectures}
    for row in architectures:
        if not row['ready']: continue
        checkpoint='age_regression_resnet18_best.pth' if row['key']=='resnet18' else 'age_regression_best.pth'
        choices[row['key']]={'label':row['name'], 'folder':row['folder'],
                              'checkpoint':row['folder']/checkpoint,
                              'data':TRAINING/'data/age_merged',
                              'mae':row['metrics']['validation_mae']}
    fine=read_json(TRAINING/'results/finetune_comparison/summary.json')
    for key in ['fc_only','late3']:
        folder=TRAINING/f'results/finetune_comparison/{key}'
        if (folder/'age_metrics.json').is_file():
            choices[key]={'label':'ResNet18','folder':folder,
                           'checkpoint':folder/'age_regression_resnet18_best.pth',
                           'data':TRAINING/'data/age_merged',
                           'mae':read_json(folder/'age_metrics.json')['validation_mae']}
    choices['original']={'label':'ResNet18', 'folder':TRAINING/'results',
                         'checkpoint':TRAINING/'results/age_regression_resnet18_best.pth',
                         'data':TRAINING/'data/age_regression', 'mae':7.841719519388542}
    # 모델명 옆 장수는 실제 훈련 + 검증 합계입니다. 추정한 숫자를 쓰지 않습니다.
    for key,choice in choices.items():
        metrics=read_json(choice['folder']/'age_metrics.json')
        choice['train_count']=metrics['train_count']
        choice['validation_count']=metrics['validation_count']
        choice['total_count']=choice['train_count']+choice['validation_count']
        # 계획한 10회가 아닌, 실제 선택된 저장 회차까지의 학습 방식을 표시합니다.
        # 학습 코드/체크포인트 확인: late3는 8회차부터, 나머지 미세조정 실험은 3회차부터입니다.
        saved_epoch=metrics['saved_epoch']
        start_epoch=None if key=='fc_only' else (8 if key=='late3' else 3)
        fine_epochs=max(0,saved_epoch-start_epoch+1) if start_epoch else 0
        if fine_epochs==0:
            description=f'FC층만 학습 {saved_epoch}회'
            feature_modules=0
        else:
            feature_modules=2 if key.startswith('efficientnet') else 1
            scope='마지막 특징 모듈 2개' if feature_modules==2 else 'layer4 단계 1개'
            description=f'FC + {scope} 미세조정 {fine_epochs}회'
        choice['saved_epoch']=saved_epoch
        # 실제 기록에서 총 회차를 읽고 선택 모델에 맞는 학습 설정을 연결합니다.
        choice['total_epochs']=len(read_json(choice['folder']/'age_history.json').get('val_mae',[]))
        choice['finetune_start_epoch']=start_epoch
        architecture=key if key in parameter_info else 'resnet18'
        params=parameter_info[architecture]
        choice['total_parameters']=params['total_parameters']
        choice['head_parameters']=params['head_parameters']
        choice['trainable_parameters']=params['head_parameters'] if fine_epochs==0 else params['trainable_parameters']
        choice['frozen_parameters']=choice['total_parameters']-choice['trainable_parameters']
        choice['actual_finetune_epochs']=fine_epochs
        choice['finetuned_feature_modules']=feature_modules
        choice['training_description']=description
        choice['label']=f"{choice['label']} ({choice['total_count']:,}장 데이터 사용 · {description})"
    return choices
