# Checkpoints

Small `.pt` checkpoint files (~8 MB) and their `.sha256` checksums are tracked in git under `checkpoints/`.

| File | System | Head | Train step | Size |
|---|---|---|---|---|
| `argon_flow_v0.pt` | LJ argon, 256 atoms, stride 50 fs | flow matching (B) | 10000 (picked by validation rollouts; v0 release) | 7.9 MB |
| `argon_flow_v1_cpu_reduced.pt` | LJ argon, 256 atoms, stride 50 fs | flow matching (B) | 1000 (reduced CPU test run, not a release; see [`docs/RETRAIN_NOTES_oct29.md`](../docs/RETRAIN_NOTES_oct29.md)) | 7.9 MB |

Each file holds the weights, the model config, atom types and masses, one 3-frame starting window, and the ground-truth reference RDF / VACF / temperature. It loads with `torch.load(..., weights_only=True)`. Check the download with `sha256sum -c argon_flow_v0.pt.sha256`.

```bash
python scripts/demo.py --ckpt checkpoints/argon_flow_v0.pt        # no OpenMM, no data files needed
```

Rebuild a file from a training run:

```bash
python scripts/export_ckpt.py --run results/argon/B_S3a_seed1 --ckpt snapshots/best_step10000.pt --out checkpoints/argon_flow_v0.pt --attention-version v0
python scripts/export_ckpt.py --run results/argon/cpu_reduced/train --ckpt snapshots/step001000.pt --out checkpoints/argon_flow_v1_cpu_reduced.pt --attention-version v1
```

`--attention-version` is needed because these training runs do not store `attention_version`. Runs from the current `equitraj/train.py` store it, and the export uses that value.

Note: `argon_flow_v0.pt` was trained with the `v0` graph attention from tag `v0.1.0` (neighbor softmax followed by post-multiplication by the smooth cutoff envelope). On `main`, `EquiTraj` and all checkpoint loaders (`scripts/demo.py`, `equitraj/api.py`, `lab/surrogate_query.py`, `scripts/evaluate.py`, and `equitraj/pocket.py`) automatically run `v0` checkpoints in `attention_version="v0"` mode directly on `main` (no tag switch needed), while `v1` checkpoints (`argon_flow_v1_cpu_reduced.pt` and new training runs) use `attention_version="v1"` (log-envelope inside `segment_softmax` with a zero-logit sink).
