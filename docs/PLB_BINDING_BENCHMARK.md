# Pocket MD test on 40 protein-ligand X-ray poses

`equitraj/plb_bench.py` and `scripts/bench_plb_binding.py` run the CPU pocket MD engine
(`PocketMicroEngine`) on real crystal poses from the OpenForceField
[protein-ligand-benchmark](https://github.com/openforcefield/protein-ligand-benchmark)
(targets `CDK2`, `MCL1`, `Thrombin`, `TYK2`; 5 ligands each). The data is not stored in this
repo; clone that repository and pass its `data` folder.

For each ligand we run two poses:

- the **native** crystal pose, and
- a **decoy**: the same ligand rotated 180 degrees about the z axis and shifted by about 1.3 Å.

Each pose gets a short pocket MD run, a `CompScore` (pose drift minus contact survival, in the
style of OpenBPMD), and a short tau-RAMD style random-push run that measures how long the
ligand stays in the pocket.

## Results

From `results/binding_plb/off_plb_40complex_report.json` (40 poses: 20 native, 20 decoys;
all per-pose records are in the file):

- Native and decoy poses are fully separated by `CompScore`: AUROC `1.000`
  (mean `-3.087` for native vs. `-0.764` for decoys).
- Mean time to leave the pocket under random pushes is `1.109 ps` for native poses vs.
  `0.719 ps` for decoys (about 54% longer).

These decoys are easy: they have about 35 atom clashes on average vs. 0.3 for native poses,
so a clash count alone would also separate them. This test shows the pipeline runs on real
structures. It does not show that the method can rank close poses or predict binding
affinity. CPU time per pose depends on the machine (it is stored in the JSON for the run above).

## Reproduce

```bash
git clone https://github.com/openforcefield/protein-ligand-benchmark.git
python scripts/bench_plb_binding.py --plb-root protein-ligand-benchmark/data --json-out results/binding_plb/off_plb_40complex_report.json
```

RDKit is used to read the ligand SDF files when it is installed; otherwise a simple built-in
V2000 SDF reader is used.
