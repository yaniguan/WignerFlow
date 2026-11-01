import pathlib,time,subprocess,os,sys
while not pathlib.Path('results/argon/cpu_reduced/train/done.json').exists():
 time.sleep(5)
baseenv=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLCONFIGDIR='/tmp/wigner_mpl')
py='/workspace/venv311/bin/python';timed=[py,'/tmp/wigner_timed.py']
commands=[('export',[py,'scripts/export_ckpt.py','--run','results/argon/cpu_reduced/train','--ckpt','snapshots/step001000.pt','--out','checkpoints/argon_flow_v1_cpu_reduced.pt']),
 ('evaluation_v1',[py,'scripts/evaluate.py','--run','.','--ckpt','checkpoints/argon_flow_v1_cpu_reduced.pt','--data_dir','data/argon','--steps','100','--seeds','1','--starts','1','--sampling_steps','10','--device','cpu','--threads','2','--out','results/argon/cpu_reduced/v1']),
 ('equivariance',[py,'scripts/compare_ckpt_equivariance.py','--checkpoints','checkpoints/argon_flow_v0.pt','checkpoints/argon_flow_v1_cpu_reduced.pt','--trials','3','--sampling_steps','10','--threads','2'])]
for name,cmd in commands:
 print('starting',name,flush=True)
 with open(f'results/argon/cpu_reduced/{name}.log','w') as f:
  p=subprocess.run(timed+[name]+cmd,stdout=f,stderr=subprocess.STDOUT,env=baseenv)
 if p.returncode:sys.exit(p.returncode)
 print('finished',name,flush=True)
