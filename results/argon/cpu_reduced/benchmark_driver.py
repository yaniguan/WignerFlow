import torch,time,json,yaml
from equitraj.dataset import TrajWindows,compute_stats
from equitraj.model import EquiTraj
from equitraj.train import training_loss
cfg=yaml.safe_load(open('configs/argon/B_S3a_seed1.yaml'))
ds=TrajWindows('data/argon','train',10,3,m=2)
p,v,b=ds[0];p=p[None];v=v[None];b=b[None]
records=[]
for threads in [1,2,4,8]:
 torch.set_num_threads(threads)
 model=EquiTraj(n_types=1,k=3,stride_fs=50,**cfg['model'])
 st=compute_stats(ds,n_samples=100)
 for key,sk in [('dx_std','dx_std'),('v_std','v_std'),('out_dx_std','res_dx_std'),('out_v_std','res_v_std')]:getattr(model,key).copy_(st[sk])
 opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
 for m in [1,2]:
  times=[]
  for i in range(4):
   t=time.perf_counter();opt.zero_grad(set_to_none=True)
   loss=training_loss(model,p,v,b,ds.types,ds.masses,m,.01,'truncated',2)
   loss.backward();opt.step()
   if i:times.append(time.perf_counter()-t)
  rec=dict(threads=threads,batch=1,m=m,train_s_per_step=sum(times)/len(times),loss=loss.item());records.append(rec);print(json.dumps(rec),flush=True)
 t=time.perf_counter()
 for _ in range(3):model.predict(p[:,:3],v[:,:3],ds.types,b,ds.masses,n_steps=10)
 rec=dict(threads=threads,rollout_s_per_step_K10=(time.perf_counter()-t)/3);records.append(rec);print(json.dumps(rec),flush=True)
json.dump(records,open('results/argon/cpu_reduced/benchmark.json','w'),indent=2)
