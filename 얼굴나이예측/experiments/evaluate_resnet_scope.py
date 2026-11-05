"""같은 검증 사진으로 FC 전용 / 늦은 미세조정 / 기존 미세조정을 비교합니다.

학습은 train_resnet18_fc.py와 train_resnet18_late.py에서 실행합니다.
이 파일은 저장 모델을 평가하고 결과와 그래프만 만듭니다.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT / 'results/finetune_comparison'
DATA = PROJECT / 'data/age_merged'
torch.set_num_threads(4)
transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class AgeDataset(Dataset):
    def __init__(self):
        with open(DATA / 'labels.csv', encoding='utf-8') as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(DATA / row['file']) as image:
            image = transform(image.convert('RGB'))
        return image, float(row['age'])


def metrics(actual, predicted):
    errors = [abs(a-b) for a,b in zip(actual,predicted)]
    return {'mae': sum(errors)/len(errors),
            'rmse': math.sqrt(sum(e*e for e in errors)/len(errors)),
            'within_5_years': sum(e<=5 for e in errors)/len(errors),
            'within_10_years': sum(e<=10 for e in errors)/len(errors)}


def evaluate():
    config = json.loads((OUTPUT / 'config.json').read_text())
    assert hashlib.sha256((DATA/'labels.csv').read_bytes()).hexdigest()==config['dataset_csv_sha256']
    dataset = AgeDataset()
    train_indices = [i for i,r in enumerate(dataset.rows) if r['split']=='train']
    val_indices = [i for i,r in enumerate(dataset.rows) if r['split']=='validation']
    assert len(train_indices)==10190 and len(val_indices)==2546
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    strategies = [('fc_only', 'FC only', 0), ('late3', 'Late fine-tuning (max 3)', 3),
                  ('early8', 'Current fine-tuning (max 8)', 8)]
    base_state = models.resnet18(weights='DEFAULT').state_dict()
    summary=[]
    histories={}
    for key,label,planned_epochs in strategies:
        folder = PROJECT/'results/merged' if key=='early8' else OUTPUT/key
        checkpoint=torch.load(folder/'age_regression_resnet18_best.pth',map_location='cpu',weights_only=True)
        history=json.loads((folder/'age_history.json').read_text())
        assert len(history['val_mae'])==10
        model=models.resnet18(weights=None)
        model.fc=nn.Linear(model.fc.in_features,1)
        model.load_state_dict(checkpoint['model_state_dict'])
        # 고정했어야 하는 특징 블록과 BatchNorm 통계가 바뀌지 않았는지 확인합니다.
        for name,value in model.state_dict().items():
            if name.startswith('fc.') or (key!='fc_only' and name.startswith('layer4.')):
                continue
            assert torch.equal(value,base_state[name]), f'고정 가중치 변경: {key}/{name}'
        model=model.to(device)
        model.eval()
        predicted_by_split={}
        for split,indices in [('train',train_indices),('validation',val_indices)]:
            actual,predicted=[],[]
            loader=DataLoader(Subset(dataset,indices),batch_size=32,shuffle=False,num_workers=0)
            with torch.no_grad():
                for images,labels in loader:
                    guessed=(model(images.to(device)).squeeze(1)*100).clamp(0,116).cpu().tolist()
                    actual.extend(labels.tolist());predicted.extend(guessed)
            predicted_by_split[split]={'actual':actual,'predicted':predicted}
        val=predicted_by_split['validation'];train=predicted_by_split['train']
        result=metrics(val['actual'],val['predicted'])
        train_result=metrics(train['actual'],train['predicted'])
        actual_ft=0 if planned_epochs==0 else max(0,checkpoint['epoch']-(10-planned_epochs))
        result.update({'strategy':key,'label':label,'planned_finetune_epochs':planned_epochs,
                       'best_epoch':checkpoint['epoch'],'actual_finetune_epochs':actual_ft,
                       'train_eval_mae':train_result['mae'],
                       'generalization_gap':result['mae']-train_result['mae'],
                       'trainable_parameters_at_selected_epoch':513 if actual_ft==0 else 8394241,
                       'checkpoint':str(folder/'age_regression_resnet18_best.pth')})
        assert abs(result['mae']-checkpoint['validation_mae'])<0.01
        # 동일한 기존 627장에서도 비교합니다.
        old_positions=[pos for pos,i in enumerate(val_indices) if json.loads(dataset.rows[i]['old_indices'])]
        result['original_627']=metrics([val['actual'][i] for i in old_positions],
                                        [val['predicted'][i] for i in old_positions])
        summary.append(result);histories[key]=history
        (OUTPUT/f'{key}_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(key, json.dumps(result),flush=True)
        del model

    # 0.10세 이내 차이는 실용적으로 비슷한 결과로 보고 미세조정을 적게 한 쪽을 선택합니다.
    # 통계적 유의성 판정이나 새 인물의 성능 보장은 아닙니다.
    best=min(r['mae'] for r in summary)
    # 학습 방법이 달라지기 전까지 결과가 같은지 확인합니다.
    for metric_name in ['train_mae','val_mae']:
        assert all(abs(histories['fc_only'][metric_name][i]-histories['late3'][metric_name][i])<1e-5
                   for i in range(7)), '미세조정 전의 두 실험 결과가 다름'
        assert all(abs(histories['fc_only'][metric_name][i]-histories['early8'][metric_name][i])<1e-5
                   for i in range(2)), '기존 실험과 초기 조건이 다름'
    candidates=[r for r in summary if r['mae']<=best+config['practical_tie_years']]
    selected=min(candidates,key=lambda r:(r['actual_finetune_epochs'],r['planned_finetune_epochs'],r['mae']))
    report={'train_count':len(train_indices),'validation_count':len(val_indices),
            'selected_strategy':selected['strategy'],'practical_tie_years':0.10,
            'selection_rule':config['selection_rule'],'results':summary,
            'control_checks':{'same_initial_results':True,'frozen_weights_unchanged':True},
            'limitation':'Single seed, model-selection validation; no independent final test.'}
    (OUTPUT/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for key,label,_ in strategies:
        axes[0].plot(range(1,11),histories[key]['val_mae'],label=label)
        axes[1].plot(range(1,11),histories[key]['train_mae'],label=label)
    axes[0].set(title='Validation MAE by epoch',xlabel='Epoch',ylabel='MAE (years)')
    axes[1].set(title='Online training MAE by epoch',xlabel='Epoch',ylabel='MAE (years)')
    axes[0].legend(fontsize=8);axes[1].legend(fontsize=8)
    labels=['FC only','Late (max 3)','Current (max 8)']
    axes[2].bar(labels,[r['mae'] for r in summary],color=['#7B8B9E','#3989B3','#32A590'])
    for i,r in enumerate(summary):
        axes[2].text(i,r['mae']+0.1,f"{r['mae']:.2f}",ha='center')
    axes[2].set(title='Best-checkpoint validation MAE',ylabel='MAE (years)')
    plt.tight_layout();plt.savefig(OUTPUT/'comparison.png',dpi=140);plt.close(fig)
    (OUTPUT/'ready.json').write_text(json.dumps({'selected_strategy':selected['strategy']},indent=2),encoding='utf-8')
    print('Selected strategy:',selected['strategy'],flush=True)


if __name__=='__main__':
    evaluate()
