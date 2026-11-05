"""기존/통합 데이터 모델을 동일한 검증 사진으로 재평가합니다. 학습은 실행하지 않습니다."""
# 사진 수의 영향을 보기 위해 모델 구조가 같은 결과끼리 비교합니다.
from pathlib import Path
import csv
import gc
import hashlib
import json
import math
import statistics
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT=Path(__file__).resolve().parent.parent
DATA=PROJECT/'data/age_merged'
OUTPUT=PROJECT/'results/data_size_comparison'
SMALL=OUTPUT/'efficientnet_v2_s_small'
SPECS=[
    ('resnet18_original','ResNet18 · 기존 사진','resnet18',PROJECT/'results',2510,'age_regression_resnet18_best.pth'),
    ('resnet18_merged','ResNet18 · 통합 사진','resnet18',PROJECT/'results/merged',10190,'age_regression_resnet18_best.pth'),
    ('efficientnet_small','EfficientNetV2-S · 기존 사진','efficientnet_v2_s',SMALL,2510,'age_regression_best.pth'),
    ('efficientnet_merged','EfficientNetV2-S · 통합 사진','efficientnet_v2_s',PROJECT/'results/architectures/efficientnet_v2_s',10190,'age_regression_best.pth')]
TRANSFORM=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])


class EvaluationPhotos(Dataset):
    def __init__(self,rows,indices):self.rows=rows;self.indices=indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,position):
        index=self.indices[position]
        with Image.open(DATA/self.rows[index]['file']) as image: tensor=TRANSFORM(image.convert('RGB'))
        return tensor,index


def metrics(rows):
    errors=[r['predicted_age']-r['label_age'] for r in rows]
    actuals=[r['label_age'] for r in rows]
    mse=sum(e*e for e in errors)/len(errors)
    mean=sum(actuals)/len(actuals)
    variance=sum((a-mean)**2 for a in actuals)
    return {'count':len(rows),'mae':sum(abs(e) for e in errors)/len(errors),'rmse':math.sqrt(mse),
      'mse':mse,'median_absolute_error':statistics.median(abs(e) for e in errors),
      'r2':1-sum(e*e for e in errors)/variance if variance else None,
      'within_5_years':sum(abs(e)<=5 for e in errors)/len(errors),
      'within_10_years':sum(abs(e)<=10 for e in errors)/len(errors)}


def evaluate():
    torch.set_num_threads(4)
    assert (SMALL/'ready.json').is_file(),'기존 사진 EfficientNet 학습을 먼저 끝내세요.'
    with (DATA/'labels.csv').open(encoding='utf-8') as file:rows=list(csv.DictReader(file))
    validation=[i for i,r in enumerate(rows) if r['split']=='validation']
    original_train=[i for i,r in enumerate(rows) if r['split']=='train' and json.loads(r['old_indices'])]
    merged_train=[i for i,r in enumerate(rows) if r['split']=='train']
    original_validation=[i for i in validation if json.loads(rows[i]['old_indices'])]
    additional_validation=[i for i in validation if not json.loads(rows[i]['old_indices'])]
    assert (len(validation),len(original_train),len(merged_train),len(original_validation))==(2546,2510,10190,627)
    # 같은 이미지가 훈련과 검증에 들어가지 않았는지 확인합니다.
    assert {rows[i]['pixel_hash'] for i in merged_train}.isdisjoint({rows[i]['pixel_hash'] for i in validation})
    same_validation_hash=hashlib.sha256(json.dumps(validation).encode()).hexdigest()
    for folder in [SMALL,PROJECT/'results/architectures/efficientnet_v2_s']:
        assert json.loads((folder/'age_split.json').read_text())['validation']==validation
    demo_index=next(i for i in validation if 20<=float(rows[i]['age'])<=40)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (OUTPUT/'predictions').mkdir(parents=True,exist_ok=True)
    results=[]
    for key,name,architecture,folder,train_count,filename in SPECS:
        saved=torch.load(folder/filename,map_location='cpu',weights_only=True)
        # 오래된 저장본은 사진 수가 가중치 파일 대신 평가 JSON에 있습니다.
        saved_metrics=json.loads((folder/'age_metrics.json').read_text(encoding='utf-8'))
        assert saved['architecture']==architecture and saved.get('train_count',saved_metrics['train_count'])==train_count
        model=getattr(models,architecture)(weights=None)
        if architecture=='resnet18':model.fc=nn.Linear(model.fc.in_features,1)
        else:model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)
        model.load_state_dict(saved['model_state_dict']);model=model.to(device).eval()
        train_indices=original_train if train_count==2510 else merged_train
        evaluated=[]
        with torch.no_grad():
            loader=DataLoader(EvaluationPhotos(rows,train_indices+validation),batch_size=32,shuffle=False,num_workers=0)
            for images,indices in loader:
                guesses=(model(images.to(device)).squeeze(1)*saved['target_scale']).clamp(0,116).cpu().tolist()
                for index,prediction in zip(indices.tolist(),guesses):
                    label=float(rows[index]['age'])
                    evaluated.append({'index':index,'file':rows[index]['file'],'split':rows[index]['split'],
                      'original':bool(json.loads(rows[index]['old_indices'])),'label_age':label,
                      'predicted_age':prediction,'absolute_error':abs(prediction-label)})
        val=[r for r in evaluated if r['split']=='validation']
        train=[r for r in evaluated if r['split']=='train']
        original_val=[r for r in val if r['original']]
        original_train_rows=[r for r in train if r['original']]
        extra_val=[r for r in val if not r['original']]
        demo=next(r for r in val if r['index']==demo_index)
        record={'key':key,'name':name,'architecture':architecture,'training_source_total':3137 if train_count==2510 else 12736,
          'train_count':train_count,'checkpoint':str(folder/filename),'best_epoch':saved['epoch'],
          'selection_validation_count':saved.get('validation_count',saved_metrics['validation_count']),'validation_ids_sha256':same_validation_hash,
          'validation':metrics(val),'training':metrics(train),'original_validation':metrics(original_val),
          'original_training':metrics(original_train_rows),'additional_validation':metrics(extra_val),
          'example':{k:demo[k] for k in ['file','label_age','predicted_age','absolute_error']}}
        record['original_source_gap']=record['original_validation']['mae']-record['original_training']['mae']
        results.append(record)
        with (OUTPUT/'predictions'/f'{key}_validation.csv').open('w',newline='',encoding='utf-8-sig') as file:
            writer=csv.DictWriter(file,fieldnames=list(val[0]));writer.writeheader();writer.writerows(val)
        print(f"{name} | 훈련 {train_count:,}장 | 같은 검증 MAE {record['validation']['mae']:.2f}세 | 기존 627장 MAE {record['original_validation']['mae']:.2f}세",flush=True)
        del model,saved;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    summary={'validation_count':2546,'original_validation_count':627,'additional_validation_count':1919,
      'device':torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU','results':results,
      'conditions':'EfficientNet same ImageNet weights, seed42, batch32, 224x224, Adam, L1Loss, 10 epochs, fine-tuning from epoch3, Dropout off; train 2510 vs10190; selection validation same2546.',
      'limitations':['Model-selection validation; no independent final test.','Single seed; age/source distribution changes with added data.','No person-ID split; exact pixel duplicates excluded.','Old ResNet18 checkpoint selected using627; EfficientNet checkpoints both selected using2546.']}
    (OUTPUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    flat=[{'모델':r['name'],'훈련사진수':r['train_count'],'공통검증수':2546,'MAE':r['validation']['mae'],
           'RMSE':r['validation']['rmse'],'±5세이내':r['validation']['within_5_years'],
           '기존627장MAE':r['original_validation']['mae'],'기존훈련2510장MAE':r['original_training']['mae'],'최고회차':r['best_epoch']} for r in results]
    with (OUTPUT/'데이터수_비교결과.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    plt.rcParams['font.family']='Malgun Gothic'
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for ax,scope,title in zip(axes,['validation','original_validation'],['같은 검증 2,546장','그중 기존 검증 627장']):
        values=[r[scope]['mae'] for r in results]
        ax.bar(range(4),values,color=['#afb9ed','#788be2','#cfadeb','#8969c9'])
        ax.set_xticks(range(4),['ResNet18\n훈련 2,510장','ResNet18\n훈련 10,190장','EfficientNet\n훈련 2,510장','EfficientNet\n훈련 10,190장'])
        ax.set(title=title,ylabel='평균 오차 (세) ↓',ylim=(0,max(values)*1.18));ax.spines[['top','right']].set_visible(False)
        for i,value in enumerate(values):ax.text(i,value,f'{value:.2f}',ha='center',va='bottom')
    fig.suptitle('데이터 추가 전후 비교 — 낮을수록 좋음');fig.tight_layout()
    fig.savefig(OUTPUT/'데이터수_비교결과.png',dpi=150);plt.close(fig)
    (OUTPUT/'ready.json').write_text(json.dumps({'completed':True,'validation_count':2546}),encoding='utf-8')
    print('동일 사진 비교 완료',flush=True)

if __name__=='__main__':evaluate()
