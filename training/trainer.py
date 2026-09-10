import csv, json, time, resource, os
from pathlib import Path
import torch
from .metrics import accuracy,macro_f1

def _move(x,device):
    if isinstance(x,dict): return {k:_move(v,device) for k,v in x.items()}
    return x.to(device,non_blocking=True) if torch.is_tensor(x) else x

def evaluate(model,loader,device,num_classes):
    model.eval(); out=[]; labels=[]
    with torch.no_grad():
        for batch in loader:
            batch=_move(batch,device); labels.append(batch['label']); out.append(model(batch))
    logits=torch.cat(out); y=torch.cat(labels); return {"accuracy":accuracy(logits,y),"macro_f1":macro_f1(logits,y,num_classes),"n_samples":len(y)}

def gpu_snapshot():
    if not torch.cuda.is_available(): return {"peak_vram_mib":0,"allocated_vram_mib":0}
    return {"peak_vram_mib":round(torch.cuda.max_memory_allocated()/2**20,1),"allocated_vram_mib":round(torch.cuda.memory_allocated()/2**20,1)}

def run_target_selected_fold(model,train_loader,test_loader,optimizer,scheduler,epochs,device,num_classes,run_dir,metadata,objective=None,test_variants=None):
    """Non-strict protocol requested by user: target evaluated at every epoch."""
    run_dir=Path(run_dir); run_dir.mkdir(parents=True,exist_ok=True); (run_dir/'run.json').write_text(json.dumps(metadata,indent=2))
    variants=test_variants or [('source',test_loader)]
    fields=['epoch','train_loss','train_accuracy','test_accuracy','test_macro_f1','selected_variant','seconds','epoch_seconds','train_seconds','eval_seconds','train_samples_per_second','peak_rss_mib','peak_vram_mib','allocated_vram_mib']; best=-1.; start=time.perf_counter()
    with (run_dir/'epochs.csv').open('w',newline='') as f, (run_dir/'live_metrics.jsonl').open('w') as live:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for epoch in range(1,epochs+1):
            epoch_start=time.perf_counter()
            model.train(); loss_sum=torch.zeros((),device=device); correct=torch.zeros((),device=device); n=0; term_sums={}
            if objective is not None: objective.train()
            if objective is not None and objective.subject_head is not None:
                objective.subject_head.progress=(epoch-1)/max(epochs-1,1)
            for batch in train_loader:
                batch=_move(batch,device); optimizer.zero_grad(set_to_none=True)
                if objective is None:
                    logits=model(batch); loss=torch.nn.functional.cross_entropy(logits,batch['label'])
                else:
                    logits,aux=model(batch,return_features=True)
                    terms=objective(logits,batch['label'],aux,batch['subject'])
                    loss=terms['total']
                    for key,value in terms.items():
                        term_sums[key]=term_sums.get(key,0)+value.detach()*len(batch['label'])
                loss.backward()
                clip=metadata.get('arguments',{}).get('grad_clip',0.)
                if clip: torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True)
                optimizer.step(); loss_sum+=loss.detach()*len(batch['label']); n+=len(batch['label'])
                correct+=(logits.detach().argmax(-1)==batch['label']).sum()
            train_loss=loss_sum.item()/max(n,1)
            if not torch.isfinite(loss_sum): raise FloatingPointError(f'non-finite loss at epoch {epoch}')
            train_end=time.perf_counter()
            scheduler.step()
            if objective is not None: objective.eval()
            evaluated=[(name,evaluate(model,loader,device,num_classes)) for name,loader in variants]
            selected_variant,metric=max(evaluated,key=lambda item:item[1]['accuracy'])
            eval_end=time.perf_counter()
            row={"epoch":epoch,"train_loss":train_loss,"train_accuracy":correct.item()/max(n,1),"test_accuracy":metric['accuracy'],"test_macro_f1":metric['macro_f1'],"selected_variant":selected_variant,"seconds":round(eval_end-start,2),"epoch_seconds":eval_end-epoch_start,"train_seconds":train_end-epoch_start,"eval_seconds":eval_end-train_end,"train_samples_per_second":n/max(train_end-epoch_start,1e-9),"peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,**gpu_snapshot()}; writer.writerow(row); f.flush(); live.write(json.dumps(row)+'\n'); live.flush()
            print(json.dumps(row),flush=True)
            if metric['accuracy']>best:
                best=metric['accuracy']; torch.save({"epoch":epoch,"model":model.state_dict(),"objective":objective.state_dict() if objective is not None else None,"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"metric":metric,"selected_variant":selected_variant,"metadata":metadata,"protocol":metadata.get('protocol','non_strict_target_selected')},run_dir/'best_target_selected.pt.tmp'); os.replace(run_dir/'best_target_selected.pt.tmp',run_dir/'best_target_selected.pt')
            if term_sums:
                live.write(json.dumps({'epoch':epoch,'loss_terms':{k:v.item()/max(n,1) for k,v in term_sums.items()}})+'\n'); live.flush()
    torch.save({"model":model.state_dict()},run_dir/'last.pt'); (run_dir/'target_selected_test.json').write_text(json.dumps({"best_accuracy":best,"protocol":metadata.get('protocol','non_strict_target_selected')},indent=2)); return best
