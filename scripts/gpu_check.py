"""Quick GPU check: torch works on this GPU, plus OpenMM speed for argon and a ~1100-atom PME box."""

import json
import subprocess
import sys

import torch

x = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize()
print(json.dumps({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
                  "matmul_ok": bool(torch.isfinite((x @ x).sum()).item())}), flush=True)
for cfg in ["configs/data/argon.yaml", "configs/data/water_box_bench.yaml"]:
    for precision in ["mixed", "single"]:
        out = subprocess.run([sys.executable, "-c", f"""
import yaml, json
from equitraj.simulate import benchmark
cfg = yaml.safe_load(open('{cfg}')); cfg['platform'] = 'CUDA'; cfg['precision'] = '{precision}'
print(json.dumps({{'system': cfg['system'], **benchmark(cfg, n_steps=20000)}}))
"""], capture_output=True, text=True)
        print(out.stdout.strip() or out.stderr.strip()[-500:], flush=True)
