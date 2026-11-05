"""EfficientNetV2-S의 FC / 마지막 모듈 1개+FC / 마지막 모듈 2개+FC 비교.

기존 모델은 수정하지 않음. 고정된 특징 추출 결과를 캐시하여 계산량을 줄임.
처음 2회는 동일하게 학습한 뒤 상태를 복제하고, 세 방식으로 각각 8회 더 학습함.
검증 자료는 기존과 동일하며, 새로 분리한 최종 테스트 자료는 아님.
"""
from pathlib import Path
from copy import deepcopy
import csv, json, hashlib, math, random, time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from torchvision import models, transforms
from PIL import Image

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'results/efficientnet_scope_comparison/run_20261104'
DATA=ROOT/'data/age_merged'
TRANSFORM=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

def log(s): print(s,flush=True)
def write_json(p,obj):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def save(p,obj):
    tmp=p.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

class Photos(Dataset):
    def __init__(self,rows): self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        with Image.open(DATA/self.rows[i]['file']) as im:x=TRANSFORM(im.convert('RGB'))
        return i,x

def metrics(y,p):
    err=p.double()-y.double();ae=err.abs();mse=(err*err).mean().item()
    return {'count':len(y),'mae':ae.mean().item(),'rmse':math.sqrt(mse),'mse':mse,
        'r2':1-(err*err).sum().item()/((y.double()-y.double().mean())**2).sum().item(),
        'exact_accuracy':((p+.5).floor()==y).double().mean().item(),
        'within_5_years':(ae<=5).double().mean().item(),
        'within_10_years':(ae<=10).double().mean().item()}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'summary.json').exists():
        log('Comparison already complete: '+str(OUT/'summary.json'));return
    assert torch.cuda.is_available(),'GPU가 필요합니다.'
    torch.set_num_threads(4);torch.manual_seed(42);random.seed(42)
    device=torch.device('cuda')
    with (DATA/'labels.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    trainids=[i for i,r in enumerate(rows) if r['split']=='train']
    validids=[i for i,r in enumerate(rows) if r['split']=='validation']
    assert (len(trainids),len(validids))==(10190,2546)
    assert {rows[i]['pixel_hash'] for i in trainids}.isdisjoint({rows[i]['pixel_hash'] for i in validids})
    old=ROOT/'results/architectures/efficientnet_v2_s/age_regression_best.pth'
    old_hash=sha(old)
    m=models.efficientnet_v2_s(weights='DEFAULT')
    for param in m.parameters():param.requires_grad=False
    m.classifier[-1]=nn.Linear(m.classifier[-1].in_features,1)
    m.to(device).eval()
    initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    initial_rng=torch.get_rng_state().clone()
    initial_cuda=torch.cuda.get_rng_state().clone()
    y=torch.tensor([float(r['age']) for r in rows],device=device)
    ids=TensorDataset(torch.arange(len(rows)))
    train_loader=DataLoader(Subset(ids,trainids),batch_size=32,shuffle=True,num_workers=0)
    val_loader=DataLoader(Subset(ids,validids),batch_size=32,shuffle=False,num_workers=0)
    fixed_train_loader=DataLoader(Subset(ids,trainids),batch_size=32,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(777))
    config={'architecture':'efficientnet_v2_s','seed':42,'train_count':10190,'validation_count':2546,
        'epochs':10,'common_head_only_epochs':2,'batch_size':32,'fc_lr':.001,'features_lr':.0001,
        'criterion':'L1Loss(age/100)','image_size':[224,224],'batchnorm_statistics':'fixed','dropout':'off',
        'cache_dtype':'float32','mixed_precision':False,'device':torch.cuda.get_device_name(0),
        'labels_sha256':sha(DATA/'labels.csv'),'original_checkpoint_sha256':old_hash,
        'comparison':'same initialization, same shuffled batches; common 2 epochs then 3 branches',
        'limitation':'Single seed; validation is reused for checkpoint/model selection, not an independent final test.'}
    write_json(OUT/'config.json',config)
    log('GPU: '+config['device']);log('Data: train=10190 validation=2546; batch=32; 10 epochs per strategy')
    # 특징 캐시는 model.eval()인 고정 구간만 재사용함. 학습 구간은 매번 순전파/역전파함.
    cache_path=OUT/'features_through_5.pt'
    cache_started=time.perf_counter()
    if cache_path.exists():
        log('Loading preserved feature cache...')
        c5=torch.load(cache_path,map_location=device,weights_only=True)
        assert c5.shape==(len(rows),160,14,14)
    else:
        c5=torch.empty((len(rows),160,14,14),device=device)
        photos=DataLoader(Photos(rows),batch_size=32,shuffle=False,num_workers=4,pin_memory=True,
            generator=torch.Generator().manual_seed(123),persistent_workers=False)
        with torch.no_grad():
            for step,(idx,images) in enumerate(photos,1):
                c5[idx.to(device)]=m.features[:6](images.to(device,non_blocking=True))
                if step%40==0 or step==len(photos):
                    torch.cuda.synchronize();elapsed=time.perf_counter()-cache_started
                    log(f'CACHE {min(step*32,len(rows))}/{len(rows)} | elapsed {elapsed:.1f}s | ETA {elapsed/step*(len(photos)-step):.1f}s')
                    write_json(OUT/'status.json',{'phase':'cache','done':min(step*32,len(rows)),'total':len(rows),'elapsed_seconds':elapsed})
        save(cache_path,c5.cpu())
    c6=torch.empty((len(rows),256,7,7),device=device)
    pooled=torch.empty((len(rows),1280),device=device)
    with torch.no_grad():
        for start in range(0,len(rows),32):
            end=min(start+32,len(rows));c6[start:end]=m.features[6](c5[start:end])
            pooled[start:end]=m.avgpool(m.features[7](c6[start:end])).flatten(1)
        # 같은 배치 크기로 확인하여 GPU 합성곱 커널 선택에 따른 수치 차이를 피함.
        sample=torch.stack([Photos(rows)[i][1] for i in range(32)]).to(device)
        full=m(sample).squeeze(1)
        cached=m.classifier[-1](pooled[:32]).squeeze(1)
        maxdiff=(full-cached).abs().max().item()
    assert maxdiff<1e-5,('cached and original forward differ',maxdiff)
    cache_seconds=time.perf_counter()-cache_started
    write_json(OUT/'cache_check.json',{'max_prediction_difference_years':maxdiff*100,'cache_seconds':cache_seconds,'cache_size_bytes':c5.numel()*c5.element_size(),'prefix_cache_equivalent':True})
    log(f'Cache ready; forward max difference {maxdiff*100:.8f} years; VRAM {torch.cuda.memory_allocated()/1024**3:.2f} GiB')

    def forward(idx,scope):
        idx=idx.to(device)
        if scope=='fc_only':return m.classifier[-1](pooled[idx]).squeeze(1)
        x=c6[idx] if scope=='last1_fc' else m.features[6](c5[idx])
        return m.classifier[-1](m.avgpool(m.features[7](x)).flatten(1)).squeeze(1)

    def run_epoch(scope,optimizer):
        m.eval();m.classifier[-1].train();total_error=0.
        torch.cuda.synchronize();started=time.perf_counter()
        for (idx,) in train_loader:
            labels=y[idx.to(device)]
            optimizer.zero_grad();pred=forward(idx,scope)
            loss=nn.functional.l1_loss(pred,labels/100.)
            loss.backward();optimizer.step()
            total_error+=(pred.detach().mul(100).clamp(0,116)-labels).abs().sum().item()
        m.eval();v_err=0.
        with torch.no_grad():
            for (idx,) in val_loader:
                p=forward(idx,scope).mul(100).clamp(0,116)
                v_err+=(p-y[idx.to(device)]).abs().sum().item()
        torch.cuda.synchronize()
        return {'train_mae':total_error/len(trainids),'val_mae':v_err/len(validids),'seconds':time.perf_counter()-started}

    def rng_state():return {'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}
    def restore_rng(s):torch.set_rng_state(s['cpu'].cpu());torch.cuda.set_rng_state(s['cuda'].cpu())
    def checkpoint(epoch,mae,scope):
        return {'model_state_dict':m.state_dict(),'architecture':'efficientnet_v2_s','task':'age_regression','output_dim':1,
            'target_scale':100.,'epoch':epoch,'validation_mae':mae,'label_age_min':1.,'label_age_max':116.,
            'dataset':'merged','strategy':scope,'finetune_start_epoch':None if scope=='fc_only' else 3,
            'total_parameters':sum(p.numel() for p in m.parameters()),
            'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),
            'train_count':len(trainids),'validation_count':len(validids)}

    commonfile=OUT/'common_epoch2.pt'
    if not commonfile.exists():
        m.load_state_dict(initial);torch.set_rng_state(initial_rng);torch.cuda.set_rng_state(initial_cuda)
        opt=optim.Adam(m.classifier[-1].parameters(),lr=.001);history=[]
        for ep in range(1,3):
            h=run_epoch('fc_only',opt);history.append(h)
            log(f'COMMON [{ep}/10] train MAE={h["train_mae"]:.4f} val MAE={h["val_mae"]:.4f} | {h["seconds"]:.1f}s')
            for scope in ['fc_only','last1_fc','last2_fc']:
                folder=OUT/scope;folder.mkdir(exist_ok=True)
                if ep==1 or h['val_mae']<history[-2]['val_mae']:save(folder/'best.pth',checkpoint(ep,h['val_mae'],scope))
        save(commonfile,{'model_state_dict':m.state_dict(),'optimizer':opt.state_dict(),'history':history,'rng':rng_state()})
    common=torch.load(commonfile,map_location=device,weights_only=True)
    oldhistory=json.loads((ROOT/'results/architectures/efficientnet_v2_s/age_history.json').read_text())
    common_diff=max(abs(common['history'][i]['val_mae']-oldhistory['val_mae'][i]) for i in range(2))
    write_json(OUT/'common_check.json',{'new_val_mae':[h['val_mae'] for h in common['history']],
        'old_val_mae':oldhistory['val_mae'][:2],'max_difference_years':common_diff,
        'all_new_strategies_share_exact_common_state':True})
    results=[]
    for scope in ['fc_only','last1_fc','last2_fc']:
        folder=OUT/scope;folder.mkdir(exist_ok=True)
        for p in m.parameters():p.requires_grad=False
        for p in m.classifier[-1].parameters():p.requires_grad=True
        if scope!='fc_only':
            for p in m.features[7 if scope=='last1_fc' else 6:].parameters():p.requires_grad=True
        m.load_state_dict(common['model_state_dict']);restore_rng(common['rng'])
        if scope=='fc_only':
            opt=optim.Adam(m.classifier[-1].parameters(),lr=.001);opt.load_state_dict(common['optimizer'])
        else:opt=optim.Adam([{'params':m.features[7 if scope=='last1_fc' else 6:].parameters(),'lr':.0001},
                            {'params':m.classifier[-1].parameters(),'lr':.001}])
        history=deepcopy(common['history']);best=min(h['val_mae'] for h in history);start_epoch=3
        latest=folder/'latest.pt'
        if latest.exists():
            state=torch.load(latest,map_location=device,weights_only=True)
            m.load_state_dict(state['model_state_dict']);opt.load_state_dict(state['optimizer']);restore_rng(state['rng'])
            history=state['history'];best=state['best'];start_epoch=state['epoch']+1
        log(f'START {scope} | trainable {sum(p.numel() for p in m.parameters() if p.requires_grad):,}')
        for ep in range(start_epoch,11):
            h=run_epoch(scope,opt);history.append(h)
            if h['val_mae']<best:best=h['val_mae'];save(folder/'best.pth',checkpoint(ep,best,scope))
            write_json(folder/'history.json',history)
            save(latest,{'model_state_dict':m.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(),'history':history,'best':best,'epoch':ep})
            write_json(OUT/'status.json',{'phase':'training','strategy':scope,'epoch':ep,'best_validation_mae':best,'last_epoch_seconds':h['seconds']})
            log(f'{scope} [{ep}/10] train MAE={h["train_mae"]:.4f} val MAE={h["val_mae"]:.4f} | {h["seconds"]:.1f}s | best {best:.4f}')
        saved=torch.load(folder/'best.pth',map_location=device,weights_only=True);m.load_state_dict(saved['model_state_dict']);m.eval()
        pred=[]
        # 고정 순서 평가에는 독립 난수 생성기를 사용하여 비교용 난수 순서를 유지함.
        eval_gen=torch.Generator().manual_seed(888)
        eval_loader=DataLoader(Subset(ids,validids),batch_size=32,shuffle=False,num_workers=0,generator=eval_gen)
        with torch.no_grad():
            for (idx,) in eval_loader:pred.append(forward(idx,scope).mul(100).clamp(0,116).cpu())
        pred=torch.cat(pred);actual=y[validids].cpu();metric=metrics(actual,pred)
        assert abs(metric['mae']-saved['validation_mae'])<1e-4
        # 학습이 허용되지 않은 모든 파라미터와 BN 통계가 초기값 그대로인지 확인함.
        prefixes=['classifier.1.']+([] if scope=='fc_only' else ['features.7.']+(['features.6.'] if scope=='last2_fc' else []))
        frozen_ok=all(torch.equal(v.detach().cpu(),initial[k]) for k,v in m.state_dict().items()
            if not any(k.startswith(prefix) for prefix in prefixes))
        bn_ok=all(torch.equal(v.detach().cpu(),initial[k]) for k,v in m.state_dict().items()
            if k.endswith(('running_mean','running_var','num_batches_tracked')))
        assert frozen_ok and bn_ok
        total=sum(p.numel() for p in m.parameters());training=sum(p.numel() for p in m.parameters() if p.requires_grad)
        result={'strategy':scope,**metric,'best_epoch':saved['epoch'],'fine_tuning_epochs_at_best':max(0,saved['epoch']-2) if scope!='fc_only' else 0,
            'trainable_parameters_at_final_stage':training,'frozen_parameters_at_final_stage':total-training,
            'total_parameters':total,'training_seconds':sum(h['seconds'] for h in history),'checkpoint':str(folder/'best.pth'),
            'frozen_parameters_unchanged':frozen_ok,'batchnorm_statistics_unchanged':bn_ok}
        write_json(folder/'metrics.json',result)
        with (folder/'validation_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['file','label_age','predicted_age','absolute_error']);w.writeheader()
            for i,a,p in zip(validids,actual.tolist(),pred.tolist()):w.writerow({'file':rows[i]['file'],'label_age':a,'predicted_age':p,'absolute_error':abs(a-p)})
        results.append(result);write_json(OUT/'partial_results.json',results)
        log('RESULT '+json.dumps(result,ensure_ascii=False))
    assert sha(old)==old_hash,'Original model changed'
    lowest=min(results,key=lambda r:r['mae'])
    # 최저 MAE와 0.1세 이내이면 실용적 근소 차이로 보고 학습 대상이 적은 방식을 효율 후보로 둠.
    efficient=min([r for r in results if r['mae']<=lowest['mae']+.10],key=lambda r:r['trainable_parameters_at_final_stage'])
    summary={'config':config,'results':results,'lowest_mae_strategy':lowest['strategy'],
        'efficient_strategy_within_0_10_years':efficient['strategy'],'practical_tie_margin_years':.1,
        'cache_seconds':cache_seconds,'shared_common_epochs':2,'original_checkpoint_unchanged':True,
        'common_vs_old_max_mae_difference':common_diff,'old_reference_mae':4.988964487824118}
    write_json(OUT/'summary.json',summary);write_json(OUT/'status.json',{'phase':'complete','lowest_mae_strategy':lowest['strategy']})
    log('COMPLETE '+str(OUT/'summary.json'))

if __name__=='__main__':main()
