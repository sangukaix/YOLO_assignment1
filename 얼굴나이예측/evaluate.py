"""저장된 모델과 검증 기록으로 나이 예측 성능을 평가합니다."""
# 새로 학습하지 않고 저장된 모델과 검증 예측 기록을 평가합니다.
# %%
# 0. 필요한 모듈과 저장 경로
from pathlib import Path
import csv
import json
import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
if not (PROJECT/'data/age_merged/labels.csv').is_file() and (PROJECT/'얼굴나이예측').is_dir():
    PROJECT=PROJECT/'얼굴나이예측'  # ! VS Code 작업 폴더가 YOLO일 때의 노트북 경로
DATA=PROJECT/'data/age_merged'
OUTPUT=PROJECT/'results/evaluation'
OUTPUT.mkdir(parents=True,exist_ok=True)
with (DATA/'labels.csv').open(encoding='utf-8') as file:MANIFEST=list(csv.DictReader(file))
ORIGINAL_VAL={r['file'] for r in MANIFEST if r['split']=='validation' and json.loads(r['old_indices'])}
SPECS=[
 ('resnet18_original','ResNet18','기존',3137,2510,'results/data_size_comparison/predictions/resnet18_original_validation.csv'),
 ('resnet18_merged','ResNet18','통합',12736,10190,'results/data_size_comparison/predictions/resnet18_merged_validation.csv'),
 ('resnet152_merged','ResNet152','통합',12736,10190,'results/architectures/resnet152/age_validation_predictions.csv'),
 ('efficientnet_small','EfficientNetV2-S','기존',3137,2510,'results/data_size_comparison/predictions/efficientnet_small_validation.csv'),
 ('efficientnet_merged','EfficientNetV2-S','통합',12736,10190,'results/data_size_comparison/predictions/efficientnet_merged_validation.csv')]


# %%
# 1. 나이 회귀에 맞는 검증 지표
def regression_metrics(rows):
    actual=[float(r['label_age']) for r in rows]
    predicted=[float(r['predicted_age']) for r in rows]
    errors=[p-y for p,y in zip(predicted,actual)]
    count=len(rows);mean=sum(actual)/count
    sse=sum(e*e for e in errors);sst=sum((y-mean)**2 for y in actual)
    # 분류의 argmax 대신 숫자 나이를 반올림한 값과 라벨을 비교합니다.
    exact=sum(math.floor(p+0.5)==y for p,y in zip(predicted,actual))
    within5=sum(abs(e)<=5 for e in errors);within10=sum(abs(e)<=10 for e in errors)
    return {'count':count,'mae':sum(abs(e) for e in errors)/count,'mse':sse/count,
      'rmse':math.sqrt(sse/count),'r2':1-sse/sst if sst else None,
      'exact_count':exact,'exact_accuracy':exact/count,
      'within5_count':within5,'within_5_years':within5/count,
      'within10_count':within10,'within_10_years':within10/count}


# %%
# 2. 기존 저장본 5개의 검증 기록 집계
def collect_csv_metrics():
    records=[]
    for key,model,source,total,train,relative in SPECS:
        with (PROJECT/relative).open(encoding='utf-8-sig') as file:rows=list(csv.DictReader(file))
        assert len(rows)==2546 and all(float(r['label_age']).is_integer() for r in rows)
        for scope,subset in [('common',rows),('original',[r for r in rows if r['file'] in ORIGINAL_VAL])]:
            metrics=regression_metrics(subset)
            assert metrics['count']==(2546 if scope=='common' else 627)
            records.append({'key':key,'model':model,'data_source':source,'total_images':total,
              'train_count':train,'scope':scope,'source_file':relative,**metrics})
    result={'records':records,'definitions':{
      'exact_accuracy':'floor(predicted_age + 0.5) == integer age label',
      'within_5_years':'absolute error of unrounded prediction <= 5',
      'within_10_years':'absolute error of unrounded prediction <= 10',
      'r2':'1 - SSE/SST; baseline is mean label age of the same evaluation set'},
      'limitation':'Model-selection validation, not an independent final test; no person-ID split.'}
    (OUTPUT/'validation_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


# %%
class AgeDataset(Dataset):
    def __init__(self,split):
        self.rows=[r for r in MANIFEST if r['split']==split]
        self.transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(DATA/row['file']) as image:tensor=self.transform(image.convert('RGB'))
        return tensor,torch.tensor(float(row['age']))


# %%
# 4. 선택한 모델을 loss, 세 가지 accuracy, 사진 비교로 평가
def evaluate_best_model():
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved=torch.load(PROJECT/'results/architectures/efficientnet_v2_s/age_regression_best.pth',map_location='cpu',weights_only=True)
    model=models.efficientnet_v2_s(weights=None)
    model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)
    model.load_state_dict(saved['model_state_dict']);model=model.to(device)
    model.eval()  # 평가할 때는 Dropout과 BatchNorm을 평가 모드로 바꿉니다.
    criterion=nn.L1Loss()  # ? 숫자 나이 회귀이므로 CrossEntropyLoss 대신 사용합니다.
    report={'model':'EfficientNetV2-S','saved_epoch':saved['epoch'],'sets':{},
      'loss_definition':'L1Loss on raw model output versus age/100, before clipping; weighted sample mean.',
      'evaluation_type':'Saved best checkpoint re-evaluated; not online epoch training loss.'}
    for split in ['train','validation']:
        dataset=AgeDataset(split);loader=DataLoader(dataset,batch_size=32,shuffle=False,num_workers=0)
        total_loss=0;epoch_accuracy=0;within5=0;within10=0;predictions=[];labels_all=[]
        with torch.no_grad():  # 평가에서는 가중치의 기울기를 계산하지 않습니다.
            for images,labels in loader:
                images,labels=images.to(device),labels.to(device)
                outputs=model(images).squeeze(1)
                loss=criterion(outputs,labels/100)
                # 마지막 배치의 사진 수가 달라도 사진 한 장당 평균이 되도록 계산합니다.
                total_loss+=loss.item()*images.size(0)
                ages=(outputs*100).clamp(0,116)
                guessed=torch.floor(ages+0.5)
                epoch_accuracy+=(guessed==labels).sum().item()
                within5+=((ages-labels).abs()<=5).sum().item()
                within10+=((ages-labels).abs()<=10).sum().item()
                predictions.extend(ages.cpu().tolist());labels_all.extend(labels.cpu().tolist())
        metrics=regression_metrics([{'label_age':y,'predicted_age':p} for y,p in zip(labels_all,predictions)])
        assert metrics['exact_count']==epoch_accuracy and metrics['within5_count']==within5 and metrics['within10_count']==within10
        metrics['loss']=total_loss/len(dataset)
        report['sets'][split]=metrics
        print(split,json.dumps(metrics,ensure_ascii=False),flush=True)
    (OUTPUT/'best_model_evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    # 정답과 예측을 사진으로 확인합니다. 나이 구간별 첫 사진을 골라 잘 맞힌 사진만 선별하지 않습니다.
    with (PROJECT/'results/architectures/efficientnet_v2_s/age_validation_predictions.csv').open(encoding='utf-8-sig') as file:
        rows=list(csv.DictReader(file))
    samples=[next(r for r in rows if low<=float(r['label_age'])<high) for low,high in [(0,10),(10,20),(20,40),(40,60),(60,80),(80,117)]]
    plt.rcParams['font.family']='Malgun Gothic'
    fig,axes=plt.subplots(2,3,figsize=(10,7))
    for ax,row in zip(axes.flat,samples):
        with Image.open(DATA/row['file']) as image:ax.imshow(image.convert('RGB'))
        ax.set_title(f"라벨 {float(row['label_age']):.0f}세 / 예측 {float(row['predicted_age']):.1f}세",fontsize=11);ax.axis('off')
    fig.tight_layout();fig.savefig(OUTPUT/'검증사진_정답예측비교.png',dpi=140);plt.close(fig)
    (OUTPUT/'example_predictions.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2),encoding='utf-8')


# %%
# 5. 검증 실행 (모델의 가중치는 바꾸지 않습니다.)
if __name__=='__main__':
    collect_csv_metrics()
    evaluate_best_model()
