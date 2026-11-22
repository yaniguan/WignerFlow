"""Quick GPU check: torch works on this GPU, plus OpenMM speed for configured systems."""

import argparse
import json
import os
import subprocess
import sys

import torch


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="*", default=["configs/data/argon.yaml", "configs/data/argon_full.yaml"])
    ap.add_argument("--steps", type=int, default=20000)
    args = ap.parse_args(argv)

    x = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    print(json.dumps({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
                      "matmul_ok": bool(torch.isfinite((x @ x).sum()).item())}), flush=True)
    for cfg in args.configs:
        if not os.path.exists(cfg):
            print(json.dumps({"config": cfg, "skipped": "file not found"}), flush=True)
            continue
        for precision in ["mixed", "single"]:
            code = (
                "import json, sys, yaml\n"
                "from equitraj.simulate import benchmark\n"
                "with open(sys.argv[1], encoding='utf-8') as fh:\n"
                "    cfg = yaml.safe_load(fh)\n"
                "cfg['platform'] = 'CUDA'\n"
                "cfg['precision'] = sys.argv[2]\n"
                "print(json.dumps({'config': sys.argv[1], 'system': cfg['system'], **benchmark(cfg, n_steps=int(sys.argv[3]))}))\n"
            )
            out = subprocess.run([sys.executable, "-c", code, cfg, precision, str(args.steps)],
                                 capture_output=True, text=True)
            print(out.stdout.strip() or out.stderr.strip()[-500:], flush=True)


if __name__ == "__main__":
    main()
