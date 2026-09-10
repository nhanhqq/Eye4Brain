import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets import load_cached_dataset, CanonicalSeedDataset, fit_train_normalization
from data.tensor_loader import TensorStore, TensorBatchLoader
from models import NeuroQMixer
from models.quantum_projector import FourQubitProjector
from training.trainer import run_target_selected_fold
from losses import CompositeLoss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--output', default='results/current')
    p.add_argument('--fold', type=int)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--quantum-depth', type=int, default=2)
    p.add_argument('--loader', choices=('cpu', 'tensor'), default='tensor')
    p.add_argument('--fusion-hidden', type=int, default=0)
    p.add_argument('--feature-hidden', type=int, default=0)
    p.add_argument('--dropout', type=float, default=.2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--quantum-lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--warmup', type=int, default=0)
    p.add_argument('--grad-clip', type=float, default=0.)
    p.add_argument('--eye-amplitude-residual', action='store_true')
    p.add_argument('--spec-amp', action='store_true')
    p.add_argument('--eeg-views',choices=('full','de'),default='full')
    p.add_argument('--target-adapt',action='store_true',help='sweep unlabeled target AdaNorm statistics during target-selected evaluation')
    for name in ('xcon', 'orth', 'relation', 'gate', 'aux', 'subject'):
        p.add_argument('--lambda-'+name, type=float, default=0.)
    a = p.parse_args()
    if min(a.epochs, a.batch_size, a.threads, a.quantum_depth)<1:
        p.error('epochs, batch-size, threads and quantum-depth must be positive')
    if not 0<=a.dropout<1 or not 0<=a.warmup<a.epochs or min(a.fusion_hidden,a.feature_hidden)<0:
        p.error('invalid dropout, warmup or fusion-hidden')
    if min(a.lr,a.quantum_lr)<=0 or min(a.grad_clip,a.weight_decay)<0:
        p.error('invalid learning rate, weight decay or gradient clip')
    if any(getattr(a,'lambda_'+k)<0 for k in ('xcon','orth','relation','gate','aux','subject')):
        p.error('loss weights must be nonnegative')
    return a


def source_manifest():
    root=Path(__file__).resolve().parent
    files=[root/'train_loso.py',root/'NeuroQ_Mixer_Implementation_Spec.md']
    for name in ('data','models','losses','training'):
        files.extend((root/name).rglob('*.py'))
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def main():
    a=parse_args()
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    views=('de','spec','conn') if a.eeg_views=='full' else ('de',)
    if a.eeg_views=='de' and a.loader!='tensor': raise ValueError('DE-only variant requires --loader tensor')
    arrays=load_cached_dataset(a.cache,required_views=views)
    subjects=np.unique(arrays['subject'])
    folds=[a.fold] if a.fold is not None else list(range(len(subjects)))
    eye_dims={n:arrays['eye_'+n].shape[-1] for n in ('pupil','gaze','fixation','saccade','blink')}
    classes=int(arrays['label'].max())+1
    weights={f'lambda_{k}':getattr(a,f'lambda_{k}') for k in ('xcon','orth','relation','gate','aux','subject')}
    model_config=dict(num_classes=classes,eye_dims=eye_dims,quantum_depth=a.quantum_depth,
                      eye_amplitude_residual=a.eye_amplitude_residual,fusion_hidden=a.fusion_hidden,dropout=a.dropout,eeg_views=views,feature_hidden=a.feature_hidden)
    cache=Path(a.cache).resolve()
    cache_meta=json.loads(cache.with_suffix('.json').read_text()) if cache.with_suffix('.json').exists() else None
    hashes=source_manifest()
    for fold in folds:
        if fold<0 or fold>=len(subjects): raise ValueError(f'invalid fold {fold}')
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
        out=Path(a.output)/a.dataset/f'runs/fold_{fold}_seed_{a.seed}'
        out.mkdir(parents=True,exist_ok=False)
        target=subjects[fold]
        train_i=np.flatnonzero(arrays['subject']!=target); test_i=np.flatnonzero(arrays['subject']==target)
        metadata=dict(dataset=a.dataset,fold=fold,target_subject=int(target),
                      train_subjects=[int(x) for x in subjects if x!=target],
                      normalization_fit_subjects=[int(x) for x in subjects if x!=target],
                      protocol='target_selected_transductive' if a.target_adapt else 'non_strict_target_selected',
                      selection='highest held-out target accuracy across epochs',
                      train_samples=len(train_i),target_samples=len(test_i),
                      loss_weights=weights,arguments=vars(a),model_config=model_config,
                      source_sha256=hashes,cache_path=str(cache),cache_bytes=cache.stat().st_size,
                      cache_mtime_ns=cache.stat().st_mtime_ns,cache_manifest=cache_meta)
        (out/'run.json').write_text(json.dumps(metadata,indent=2))
        normalization=fit_train_normalization(arrays,train_i)
        target_variants=[]
        if a.target_adapt:
            target_normalization=fit_train_normalization(arrays,test_i)
            for rho in (.25,.5,.75,1.0):
                mixed={key:((1-rho)*normalization[key][0]+rho*target_normalization[key][0],
                            ((1-rho)*normalization[key][1]**2+rho*target_normalization[key][1]**2).clip(1e-6)**.5)
                       for key in normalization}
                target_variants.append((f'adanorm_rho_{rho:g}',mixed))
        np.savez(out/'normalization.npz',**{f'{k}_{s}':v[i] for k,v in normalization.items() for i,s in enumerate(('mean','std'))})
        if a.loader=='tensor':
            store=TensorStore(arrays,normalization,device)
            train=TensorBatchLoader(store,train_i,a.batch_size,shuffle=True)
            test=TensorBatchLoader(store,test_i,a.batch_size)
            variants=[('source',test)]
            for name,mixed in target_variants:
                variants.append((name,TensorBatchLoader(TensorStore(arrays,mixed,device),test_i,a.batch_size)))
        else:
            kw=dict(batch_size=a.batch_size,num_workers=a.workers,pin_memory=device.type=='cuda')
            train=DataLoader(CanonicalSeedDataset(arrays,train_i,normalization),shuffle=True,**kw)
            test=DataLoader(CanonicalSeedDataset(arrays,test_i,normalization),shuffle=False,**kw)
            variants=[('source',test)]
            for name,mixed in target_variants:
                variants.append((name,DataLoader(CanonicalSeedDataset(arrays,test_i,mixed),shuffle=False,**kw)))
        model=NeuroQMixer(**model_config).to(device)
        model.spec_encoder.amp=a.spec_amp
        objective=CompositeLoss(**weights,num_classes=classes,source_subjects=metadata['train_subjects']).to(device) if any(weights.values()) else None
        quantum=[p for module in model.modules() if isinstance(module,FourQubitProjector) for p in module.parameters()]
        qids={id(x) for x in quantum}
        classical=[x for x in model.parameters() if id(x) not in qids]
        if objective is not None: classical+=list(objective.parameters())
        opt=torch.optim.AdamW([{'params':classical,'lr':a.lr},{'params':quantum,'lr':a.quantum_lr}],weight_decay=a.weight_decay)
        if a.warmup:
            sch=torch.optim.lr_scheduler.SequentialLR(opt,[
                torch.optim.lr_scheduler.LinearLR(opt,start_factor=1/a.warmup,total_iters=a.warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs-a.warmup)],milestones=[a.warmup])
        else: sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs)
        best=run_target_selected_fold(model,train,test,opt,sch,a.epochs,device,classes,out,metadata,objective=objective,test_variants=variants)
        print(json.dumps({'fold':fold,'target_subject':int(target),'best_target_accuracy':best}),flush=True)
        del train,test,model,opt,sch,objective
        if a.loader=='tensor': del store
        if device.type=='cuda': torch.cuda.empty_cache()


if __name__=='__main__':
    main()
