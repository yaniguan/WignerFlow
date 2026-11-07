import importlib.util, json, os, pathlib

_p = pathlib.Path(__file__).resolve().parents[1] / 'scripts' / 'analyze_profiling.py'
spec = importlib.util.spec_from_file_location('analyze_profiling', _p)
ap = importlib.util.module_from_spec(spec); spec.loader.exec_module(ap)

TABLE = """-------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                              aten::bmm         0.55%      46.319ms         0.75%      62.672ms      55.857us        1.318s        32.32%        1.318s       1.175ms          1122
                                       cudaLaunchKernel         7.85%     441.574ms         7.94%     446.506ms      11.628us       0.000us         0.00%       1.079ms       0.028us         38398
-------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 8.382s
Self CUDA time total: 4.079s
"""


def test_to_ms():
    assert ap.to_ms('1.5s') == 1500 and ap.to_ms('250us') == 0.25 and ap.to_ms('3ms') == 3


def test_parse_key_averages():
    r = ap.parse_key_averages(TABLE)
    assert r['self_cpu_total_ms'] == 8382 and r['self_cuda_total_ms'] == 4079
    assert r['top_cuda'][0]['name'] == 'aten::bmm' and r['top_cuda'][0]['self_cuda_ms'] == 1318
    assert r['cudaLaunchKernel_calls'] == 38398


def test_trace_busy(tmp_path):
    ev = [{'ph': 'X', 'cat': 'kernel', 'name': 'k', 'ts': 0, 'dur': 10},
          {'ph': 'X', 'cat': 'kernel', 'name': 'k', 'ts': 5, 'dur': 10},
          {'ph': 'X', 'cat': 'kernel', 'name': 'k2', 'ts': 30, 'dur': 10}]
    p = tmp_path / 't.json'; p.write_text(json.dumps({'traceEvents': ev}))
    r = ap.trace_gpu_busy(str(p))
    assert abs(r['busy_frac'] - 25 / 40) < 1e-9 and r['n_kernels'] == 3


def test_verdicts():
    cpu_bound = {'gpu_util': {'mean': 30}, 'key_averages': {'self_cpu_total_ms': 5000, 'self_cuda_total_ms': 800, 'top_cuda': []}}
    assert ap.verdict(cpu_bound)[0].startswith('CPU')
    data_bound = {'phases_ms_per_step': {'data': 50, 'forward': 20, 'backward': 20}}
    assert ap.verdict(data_bound)[0].startswith('data')
    bw = {'gpu_util': {'mean': 80}, 'key_averages': {'self_cpu_total_ms': 100, 'self_cuda_total_ms': 100,
          'top_cuda': [{'name': 'aten::add_', 'self_cuda_ms': 60}]}}
    assert 'bandwidth' in ap.verdict(bw)[0]
    assert ap.verdict({'gpu_util': {'mean': 90}})[0] == 'GPU compute bound'
