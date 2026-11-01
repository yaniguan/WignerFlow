import sys,subprocess,time,json,os
name=sys.argv[1]; cmd=sys.argv[2:]; start=time.time()
p=subprocess.run(cmd)
record=dict(name=name,command=cmd,wall_s=time.time()-start,exit_code=p.returncode,start_unix=start,end_unix=time.time())
with open('results/argon/cpu_reduced/commands.jsonl','a') as f:f.write(json.dumps(record)+'\n')
print(json.dumps(record),flush=True)
sys.exit(p.returncode)
