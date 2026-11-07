#!/usr/bin/env python3
"""Reproducible GPU-profiling analysis for WignerFlow / ouroboros-ocsr / YLM Colab runs.

Inputs are the artifacts the repos' profilers write (summary.json, key_averages.txt,
nvidia-smi CSV / gpu_utilization.jsonl / gpu_samples.csv, optional chrome trace.json).
Usage:
    python scripts/analyze_profiling.py --raw RAW_DIR --out OUT_DIR [--manifest manifest.json]
The manifest maps run names to files inside RAW_DIR (see DEFAULT_MANIFEST).
Only numbers found in the files are reported; missing items become null.
"""
import argparse, csv, json, os, re, statistics, collections

UNIT = {'us': 1e-3, 'ms': 1.0, 's': 1e3}


def to_ms(tok):
    m = re.fullmatch(r'(-?[\d.]+)(us|ms|s)', tok.strip())
    return round(float(m.group(1)) * UNIT[m.group(2)], 6) if m else None


def parse_key_averages(text, top=10):
    """Parse a torch.profiler key_averages().table() dump.

    Returns totals (ms) and the top rows by Self CUDA / Self CPU time."""
    lines = text.splitlines()
    out = {'self_cpu_total_ms': None, 'self_cuda_total_ms': None, 'top_cuda': [], 'top_cpu': []}
    for ln in lines:
        m = re.match(r'Self (CPU|CUDA) time total:\s*(\S+)', ln)
        if m:
            out['self_%s_total_ms' % m.group(1).lower()] = to_ms(m.group(2))
    header = next((i for i, l in enumerate(lines) if l.strip().startswith('Name')), None)
    if header is None:
        return out
    cols = [c.strip() for c in re.split(r'\s{2,}', lines[header].strip())]
    rows = []
    for ln in lines[header + 2:]:
        if ln.startswith('---') or not ln.strip():
            break
        parts = re.split(r'\s{2,}', ln.strip())
        if len(parts) < len(cols):
            continue
        name = ' '.join(parts[:len(parts) - len(cols) + 1])
        rec = dict(zip(cols[1:], parts[len(parts) - len(cols) + 1:]))
        rows.append({'name': name, 'self_cpu_ms': to_ms(rec.get('Self CPU', '')),
                     'self_cuda_ms': to_ms(rec.get('Self CUDA', '')),
                     'calls': int(rec.get('# of Calls', '0') or 0)})
    real = [r for r in rows if not r['name'].startswith('profile_step') and not r['name'].startswith('ProfilerStep')]
    out['top_cuda'] = sorted([r for r in real if r['self_cuda_ms']], key=lambda r: -r['self_cuda_ms'])[:top]
    out['top_cpu'] = sorted([r for r in real if r['self_cpu_ms']], key=lambda r: -r['self_cpu_ms'])[:top]
    launch = [r for r in rows if r['name'].startswith('cudaLaunchKernel')]
    out['cudaLaunchKernel_calls'] = sum(r['calls'] for r in launch) or None
    return out


def util_series(path):
    """Return a list of GPU utilization % samples from any of the three telemetry formats."""
    if path.endswith('.jsonl'):
        return [json.loads(l)['utilization_percent'] for l in open(path) if l.strip()]
    with open(path) as f:
        rd = csv.reader(f)
        head = [h.strip() for h in next(rd)]
        idx = next(i for i, h in enumerate(head) if h.startswith('util'))
        vals = []
        for r in rd:
            try:
                vals.append(float(r[idx].replace('%', '').strip()))
            except (ValueError, IndexError):
                pass
        return vals


def trace_gpu_busy(path):
    """Union of GPU kernel/memcpy intervals vs. trace span -> GPU-busy fraction."""
    d = json.load(open(path))
    ev = [e for e in d['traceEvents'] if e.get('ph') == 'X']
    k = sorted((e['ts'], e['ts'] + e['dur'], e['name']) for e in ev if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset'))
    if not k:
        return None
    busy, (cs, ce) = 0.0, k[0][:2]
    for s, e, _ in k[1:]:
        if s > ce:
            busy += ce - cs; cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    span = max(e for _, e, _ in k) - k[0][0]
    by = collections.Counter()
    for s, e, n in k:
        by[n[:80]] += e - s
    rt = collections.Counter()
    for e in ev:
        if e.get('cat') == 'cuda_runtime':
            rt[e['name']] += e['dur']
    return {'span_ms': span / 1e3, 'busy_ms': busy / 1e3, 'busy_frac': busy / span, 'n_kernels': len(k),
            'mean_kernel_us': sum(e - s for s, e, _ in k) / len(k),
            'top_kernels_ms': [[n, v / 1e3] for n, v in by.most_common(8)],
            'cuda_runtime_ms': {n: v / 1e3 for n, v in rt.most_common(5)}}


def stats(v):
    return {'mean': statistics.mean(v), 'median': statistics.median(v), 'max': max(v), 'n': len(v),
            'frac_below_10pct': sum(x < 10 for x in v) / len(v)} if v else None


ELEMENTWISE = {'aten::add_', 'aten::fill_', 'aten::copy_', 'aten::mul', 'aten::add', 'aten::zero_'}


def verdict(run):
    """Rule-based bound classification from the extracted evidence."""
    ev, util = [], run.get('gpu_util', {}) or {}
    u = util.get('mean')
    ka = run.get('key_averages') or {}
    cpu, gpu = ka.get('self_cpu_total_ms'), ka.get('self_cuda_total_ms')
    ph = run.get('phases_ms_per_step') or {}
    tot = sum(ph.values()) or None
    if tot and ph.get('data', 0) / tot > 0.2:
        return 'data-loader / input bound', ['data phase %.0f%% of step' % (100 * ph['data'] / tot)]
    if cpu and gpu and cpu / gpu > 2 and (u is None or u < 50):
        ev.append('self CPU %.0f ms vs self CUDA %.0f ms (ratio %.1f)' % (cpu, gpu, cpu / gpu))
        if u is not None: ev.append('mean GPU util %.0f%%' % u)
        return 'CPU / kernel-launch (Python overhead) bound', ev
    if gpu and ka.get('top_cuda'):
        ew = sum(r['self_cuda_ms'] for r in ka['top_cuda'] if r['name'] in ELEMENTWISE)
        if ew / gpu > 0.4 and (u is None or u >= 60):
            return 'GPU memory-bandwidth bound (elementwise ops)', ['elementwise add_/fill_/copy_/mul = %.0f%% of self CUDA time' % (100 * ew / gpu), 'mean GPU util %.0f%%' % u]
    if run.get('trace', {}) and run['trace'].get('busy_frac', 1) < 0.6:
        t = run['trace']
        return 'kernel-launch / small-kernel bound', ['GPU busy %.0f%% of trace span, %d kernels, mean %.0f us' % (100 * t['busy_frac'], t['n_kernels'], t['mean_kernel_us'])]
    if u is not None and u < 50:
        return 'likely CPU / launch bound (GPU idle >50%, no op table for this segment)', ['mean GPU util %.0f%%' % u]
    if u is not None and u >= 75:
        return 'GPU compute bound', ['mean GPU util %.0f%%' % u]
    return 'mixed (GPU partially idle)', ['mean GPU util %s' % (None if u is None else round(u))]


def load(raw, name):
    p = os.path.join(raw, name)
    return p if os.path.exists(p) else None


DEFAULT_MANIFEST = json.load(open(os.path.join(os.path.dirname(__file__), 'manifest.json'))) if os.path.exists(os.path.join(os.path.dirname(__file__), 'manifest.json')) else {}


def build(raw, manifest):
    data = {}
    for rid, m in manifest.items():
        run = {k: m[k] for k in ('repo', 'label', 'gpu', 'gpu_mem_gb', 'notes') if k in m}
        if m.get('summary') and load(raw, m['summary']):
            s = json.load(open(load(raw, m['summary'])))
            for path in m.get('summary_path', []):
                s = s[path]
            run['summary'] = {k: v for k, v in s.items() if not k.endswith('samples')} if isinstance(s, dict) else s
            if 'phase_seconds_per_step' in s:
                run['phases_ms_per_step'] = {k: v * 1e3 for k, v in s['phase_seconds_per_step'].items()}
            if 'phases' in s and isinstance(s['phases'], dict):
                run['phases_ms_per_step'] = {k: v['mean_ms'] for k, v in s['phases'].items()}
            if 'gpu_utilization_samples' in s:
                run['gpu_util'] = stats(s['gpu_utilization_samples'])
        if m.get('key_averages') and load(raw, m['key_averages']):
            run['key_averages'] = parse_key_averages(open(load(raw, m['key_averages'])).read())
        if m.get('gpu_csv') and load(raw, m['gpu_csv']):
            run['gpu_util'] = stats(util_series(load(raw, m['gpu_csv'])))
        if m.get('trace') and load(raw, m['trace']):
            run['trace'] = trace_gpu_busy(load(raw, m['trace']))
        run['bound'], run['evidence'] = verdict(run)
        data[rid] = run
    return data


def plot(data, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    paths = []
    colors = {'data': '#8dd3c7', 'data_loading': '#8dd3c7', 'host_to_device': '#ffffb3', 'neighbor_list': '#bebada',
              'forward': '#80b1d3', 'backward': '#fb8072', 'optimizer': '#fdb462'}
    for repo in sorted({r['repo'] for r in data.values()}):
        runs = {k: v for k, v in data.items() if v['repo'] == repo and v.get('phases_ms_per_step')}
        if runs:
            fig, ax = plt.subplots(figsize=(9, 0.6 * len(runs) + 1.8))
            for i, (k, v) in enumerate(runs.items()):
                left = 0
                for ph, ms in v['phases_ms_per_step'].items():
                    ax.barh(i, ms, left=left, color=colors.get(ph, '#ccc'), label=ph if i == 0 or ph not in ax.get_legend_handles_labels()[1] else None)
                    left += ms
                ax.text(left, i, ' %.0f ms' % left, va='center', fontsize=8)
            ax.set_yticks(range(len(runs))); ax.set_yticklabels([v['label'] for v in runs.values()], fontsize=8)
            ax.set_xlabel('ms per step (CUDA-synchronized phase timers)'); ax.set_title(f'{repo}: step-time breakdown')
            h, l = ax.get_legend_handles_labels(); ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=8, loc='lower right')
            fig.tight_layout(); p = os.path.join(out, f'{repo}_step_breakdown.png'); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)
        kas = {k: v for k, v in data.items() if v['repo'] == repo and v.get('key_averages', {}).get('top_cuda')}
        for k, v in kas.items():
            top = v['key_averages']['top_cuda'][:10][::-1]
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.barh([t['name'][:55] for t in top], [t['self_cuda_ms'] for t in top], color='#80b1d3')
            ax.set_xlabel('Self CUDA time (ms, profiled window)'); ax.set_title(f'{v["label"]}: top CUDA ops/kernels')
            ax.tick_params(axis='y', labelsize=7); fig.tight_layout()
            p = os.path.join(out, f'{k}_top_cuda.png'); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)
        for k, v in data.items():
            if v['repo'] == repo and v.get('trace'):
                top = v['trace']['top_kernels_ms'][::-1]
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.barh([n[:55] for n, _ in top], [ms for _, ms in top], color='#fb8072')
                ax.set_xlabel('GPU time in trace window (ms)'); ax.tick_params(axis='y', labelsize=7)
                ax.set_title('%s: top kernels (GPU busy %.0f%% of %.0f ms span)' % (v['label'][:40], 100 * v['trace']['busy_frac'], v['trace']['span_ms']))
                fig.tight_layout(); p = os.path.join(out, f'{k}_trace_kernels.png'); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)
    # cross-repo: GPU util and CPU/GPU time ratio
    runs = [(k, v) for k, v in data.items() if v.get('gpu_util')]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(len(runs)), [v['gpu_util']['mean'] for _, v in runs], color=['#80b1d3' if v['repo'] == 'WignerFlow' else '#fb8072' if v['repo'] == 'ouroboros-ocsr' else '#b3de69' for _, v in runs])
    ax.set_xticks(range(len(runs))); ax.set_xticklabels([v['label'] for _, v in runs], rotation=40, ha='right', fontsize=7)
    ax.set_ylabel('mean GPU util % (nvidia-smi)'); ax.axhline(75, ls='--', c='gray'); ax.set_title('GPU utilization by run (blue=WignerFlow, red=ocsr, green=YLM)')
    fig.tight_layout(); p = os.path.join(out, 'cross_gpu_util.png'); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)
    runs = [(k, v) for k, v in data.items() if v.get('key_averages', {}).get('self_cuda_total_ms')]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(runs))
    ax.bar([i - 0.2 for i in x], [v['key_averages']['self_cpu_total_ms'] / 1e3 for _, v in runs], 0.4, label='self CPU time (s)')
    ax.bar([i + 0.2 for i in x], [v['key_averages']['self_cuda_total_ms'] / 1e3 for _, v in runs], 0.4, label='self CUDA time (s)')
    ax.set_xticks(list(x)); ax.set_xticklabels([v['label'] for _, v in runs], rotation=20, ha='right', fontsize=8); ax.legend()
    ax.set_title('torch.profiler window: host (CPU) vs device (CUDA) time'); fig.tight_layout()
    p = os.path.join(out, 'cross_cpu_vs_cuda.png'); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)
    runs = [(k, v) for k, v in data.items() if v.get('peak_mem_gb')]
    if runs:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh([v['label'] for _, v in runs], [v['peak_mem_gb'] for _, v in runs], color='#bebada')
        for i, (_, v) in enumerate(runs):
            if v.get('gpu_mem_gb'): ax.plot([v['gpu_mem_gb']] * 2, [i - 0.4, i + 0.4], 'k--')
        ax.set_xlabel('peak torch.cuda.max_memory_allocated (GB); dashed = card capacity'); ax.tick_params(axis='y', labelsize=7)
        fig.tight_layout(); p = os.path.join(out, 'cross_peak_memory.png'); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--manifest', default=None)
    a = ap.parse_args(argv)
    manifest = json.load(open(a.manifest)) if a.manifest else DEFAULT_MANIFEST
    data = build(a.raw, manifest)
    for k, v in data.items():
        s = v.get('summary') or {}
        pk = s.get('cuda_peak_bytes') or s.get('peak_mem_bytes') or (s.get('peak_memory') or {}).get('allocated_bytes')
        if pk: v['peak_mem_gb'] = pk / 1e9
    os.makedirs(a.out, exist_ok=True)
    json.dump(data, open(os.path.join(a.out, 'data.json'), 'w'), indent=1, default=str)
    for p in plot(data, os.path.join(a.out, 'charts')):
        print(p)


if __name__ == '__main__':
    main()
