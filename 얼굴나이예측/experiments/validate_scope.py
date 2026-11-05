"""새 비교 모델 3개를 검증 사진 전체에 직접 입력하여 최종 평가함."""
from pathlib import Path
import csv,json,time,hashlib
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from torchvision import models
from compare_scope import Photos,metrics,write_json,ROOT,OUT

def main():
    torch.set_num_threads(4)
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    with (ROOT/'data/age_merged/labels.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    ids=[i for i,r in enumerate(rows) if r['split']=='validation']
    loader=DataLoader(Subset(Photos(rows),ids),batch_size=32,shuffle=False,num_workers=4,
        pin_memory=True,generator=torch.Generator().manual_seed(55))
    nets={};preds={}
    for r in summary['results']:
        saved=torch.load(r['checkpoint'],map_location='cpu',weights_only=True)
        net=models.efficientnet_v2_s(weights=None);net.classifier[-1]=nn.Linear(1280,1)
        net.load_state_dict(saved['model_state_dict']);net.to('cuda').eval()
        nets[r['strategy']]=net;preds[r['strategy']]=[]
    started=time.perf_counter();order=[]
    with torch.no_grad():
        for step,(idx,images) in enumerate(loader,1):
            images=images.to('cuda',non_blocking=True);order.extend(idx.tolist())
            for key,net in nets.items():preds[key].append((net(images).squeeze(1)*100).clamp(0,116).cpu())
            if step%20==0 or step==len(loader):print(f'FULL PHOTO CHECK {min(step*32,len(ids))}/{len(ids)} | {time.perf_counter()-started:.1f}s',flush=True)
    assert order==ids
    actual=torch.tensor([float(rows[i]['age']) for i in ids]);audit=[]
    for r in summary['results']:
        key=r['strategy'];pred=torch.cat(preds[key]);full=metrics(actual,pred)
        previous=OUT/key/'validation_predictions.csv'
        if not previous.is_file():previous=OUT/key/'full_validation_predictions.csv'
        with previous.open(encoding='utf-8-sig',newline='') as f:cached=list(csv.DictReader(f))
        cachedp=torch.tensor([float(x['predicted_age']) for x in cached])
        assert len(cached)==len(ids)
        audit.append({'strategy':key,'cached_mae':r['mae'],'full_mae':full['mae'],
            'max_prediction_difference_years':(pred-cachedp).abs().max().item(),
            'mean_prediction_difference_years':(pred-cachedp).abs().mean().item()})
        # 캐시 평가도 보관하고, 실제 사진을 직접 입력한 평가를 최종 보고값으로 저장함.
        write_json(OUT/key/'cached_metrics.json',r.copy())
        r.update(full);r['evaluation']='full original photos, complete model forward'
        with (OUT/key/'full_validation_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['file','label_age','predicted_age','absolute_error']);w.writeheader()
            for i,a,p in zip(ids,actual.tolist(),pred.tolist()):w.writerow({'file':rows[i]['file'],'label_age':a,'predicted_age':p,'absolute_error':abs(a-p)})
        write_json(OUT/key/'metrics.json',r)
    summary['full_photo_evaluation']=audit;summary['full_photo_evaluation_seconds']=time.perf_counter()-started
    summary['lowest_mae_strategy']=min(summary['results'],key=lambda r:r['mae'])['strategy']
    write_json(OUT/'summary.json',summary)
    print(json.dumps(audit),flush=True);print('FULL PHOTO EVALUATION COMPLETE',flush=True)

if __name__=='__main__':main()
