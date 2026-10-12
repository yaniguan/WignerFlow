# v0 checkpoints

| File | System | Head | Train step | Size |
|---|---|---|---|---|
| `argon_flow_v0.pt` | LJ argon, 256 atoms, stride 50 fs | flow matching (B) | 10000 (picked by validation rollouts) | 7.9 MB |

Each file holds the weights, the model config, atom types and masses, one 3-frame starting window, and the ground-truth reference RDF / VACF / temperature. It loads with `torch.load(..., weights_only=True)`. Check the download with `sha256sum -c argon_flow_v0.pt.sha256`.

```bash
python scripts/demo.py --ckpt checkpoints/argon_flow_v0.pt        # no OpenMM, no data files needed
```

Rebuild a file from a training run:

```bash
python scripts/export_ckpt.py --run results/argon/B_S3a_seed1 --ckpt snapshots/best_step10000.pt --out checkpoints/argon_flow_v0.pt
```
