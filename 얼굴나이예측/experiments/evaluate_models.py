"""세 모델의 저장된 검증 예측을 같은 기준으로 집계합니다. 재학습은 하지 않습니다."""
# 모델별 검증 결과와 비교 그래프를 생성합니다.
from pathlib import Path
import csv
import json
import math
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT/'results/architectures'
MODELS = [('resnet18','ResNet18',PROJECT/'results/merged','age_regression_resnet18_best.pth'),
          ('resnet152','ResNet152',OUTPUT/'resnet152','age_regression_best.pth'),
          ('efficientnet_v2_s','EfficientNetV2-S',OUTPUT/'efficientnet_v2_s','age_regression_best.pth')]


def summarize():
    rows=[]
    for key,name,folder,checkpoint in MODELS:
        assert (folder/'ready.json').is_file(), f'{name} 학습·평가 미완료'
        metrics=json.loads((folder/'age_metrics.json').read_text(encoding='utf-8'))
        with open(folder/'age_validation_predictions.csv',encoding='utf-8') as file:
            predictions=list(csv.DictReader(file))
        errors=[float(r['predicted_age'])-float(r['label_age']) for r in predictions]
        actuals=[float(r['label_age']) for r in predictions]
        mae=sum(abs(e) for e in errors)/len(errors)
        assert len(errors)==2546 and abs(mae-metrics['validation_mae'])<1e-5
        mse=sum(e*e for e in errors)/len(errors)
        mean=sum(actuals)/len(actuals)
        saved=torch.load(folder/checkpoint,map_location='cpu',weights_only=True)
        total=saved.get('total_parameters',11177025)
        trainable=saved.get('trainable_parameters',8394241)
        rows.append({'architecture':key,'name':name,'mae':mae,'rmse':math.sqrt(mse),'mse':mse,
          'r2':1-sum(e*e for e in errors)/sum((a-mean)**2 for a in actuals),
          'within_5_years':sum(abs(e)<=5 for e in errors)/len(errors),
          'within_10_years':sum(abs(e)<=10 for e in errors)/len(errors),
          'best_epoch':saved['epoch'],'actual_finetune_epochs':saved['epoch']-2,
          'total_parameters':total,'trainable_parameters':trainable,'frozen_parameters':total-trainable,
          'head_parameters':513 if key=='resnet18' else 2049 if key=='resnet152' else 1281,
          'train_count':10190,'validation_count':len(errors),'checkpoint':str(folder/checkpoint)})
        del saved
    rows.sort(key=lambda r:r['mae'])
    summary={'selected_architecture':rows[0]['architecture'],'results':rows,
      'conditions':'Same 10190/2546 split, seed42, 224x224, batch32, Adam, L1Loss, 10 epochs, last stage fine-tuned from epoch3.',
      'limitation':'Single run per architecture; validation used for checkpoint and model selection; no independent final test. Default ImageNet pretrained weights differ by architecture.'}
    (OUTPUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with open(OUTPUT/'모델별_학습결과.csv','w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.rcParams['font.family']='Malgun Gothic'
    fig,axes=plt.subplots(1,3,figsize=(13,4.2))
    names=[r['name'] for r in rows];colors=['#7a67ce','#91a3ef','#b7c7ed']
    for ax,metric,title in zip(axes,['mae','within_5_years','total_parameters'],['평균 오차 (세) ↓','±5세 이내 예측 (%) ↑','전체 파라미터 (백만 개)']):
        values=[r[metric]*(100 if metric=='within_5_years' else 1e-6 if metric=='total_parameters' else 1) for r in rows]
        ax.bar(names,values,color=colors);ax.set_title(title);ax.tick_params(axis='x',labelsize=9)
        ax.spines[['top','right']].set_visible(False)
        for i,value in enumerate(values):ax.text(i,value,f'{value:.2f}',ha='center',va='bottom')
        ax.set_ylim(0,max(values)*1.2)
    fig.suptitle('동일한 검증 사진 2,546장 · 10회 학습 중 최고 저장본',fontsize=12)
    fig.tight_layout();fig.savefig(OUTPUT/'모델별_학습결과.png',dpi=150);plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':summarize()
