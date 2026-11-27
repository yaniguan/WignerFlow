/* Main controller for WignerFlow Homepage, Interactive Function Sandboxes & 3D Studio.
 * Works 100% statically on GitHub Pages (assets/homepage-data.json) and live
 * against `python scripts/wignerflow_studio.py` (/api/*).
 */
(() => {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';

  let DATA = null;
  let LIVE_API = false;
  let compViewer = null;
  let rollViewer = null;
  let lastCompReport = null;
  let lastRollReport = null;
  let rollFrameIdx = 0;
  let rollPlaying = true;

  // Interactive sandbox states
  let activeArchId = 'f3';
  let cpuState = { atoms: 256, steps: 2, therm: 'off', liveSample: null };
  let activeLabCase = 'argon_cryo_2step';
  let projIdx = 1;
  let hybridMode = 'hybrid';

  const ARCH_MODULES = [
    {
      id: 'f1',
      num: '01',
      short: 'Wigner-D Encoder',
      sub: 'SO(2) + C² Sink',
      file: 'equitraj/layers.py',
      colorKey: 'accent',
      inSpec: 'k = 3 frames (X, V) + charges / bond hops',
      outSpec: 'O(3) node irreps (l ≤ 2), C² smooth @ r_c = 6 Å',
      cmd: 'GraphAttention(lmax=2, cutoff=6.0)',
      desc: 'Rotates each neighbor edge into its local +y-frame with Wigner-D matrices and places log(envelope) inside softmax with a zero-logit sink so cutoff crossings have zero step jump.',
      jumpSection: '#method',
    },
    {
      id: 'f2',
      num: '02',
      short: '50-fs Flow Head',
      sub: 'Coasting Residuals',
      file: 'equitraj/model.py',
      colorKey: 'purple',
      inSpec: 'Fused frame features + COM-free Gaussian noise',
      outSpec: 'Next 50 fs state (Δx, v_new) with 0 COM drift',
      cmd: 'EquiTraj(..., head="flow", target="residual")',
      desc: 'Integrates a conditional flow ODE over constant-velocity coasting residuals (Δx − vΔt, v_new − v), preventing the velocity damping that freezes direct regression models.',
      jumpSection: '#demo',
    },
    {
      id: 'f3',
      num: '03',
      short: 'FrameSplit CPU API',
      sub: '282–944 ms / step',
      file: 'equitraj/api.py',
      colorKey: 'blue',
      inSpec: '(3, N, 3) Ar frames · CLI or POST /predict',
      outSpec: 'Next frame(s), T(K), warnings, SHA-256 receipt',
      cmd: 'python -m equitraj.api --bundled --out next.json',
      desc: 'Runs spatial blocks one frame at a time and reuses the newest frame graph for fusion—2× faster on a single CPU thread with 0.0 max difference against EquiTraj.predict.',
      jumpCapTab: 'api',
    },
    {
      id: 'f4',
      num: '04',
      short: 'Lab [physics] Gate',
      sub: 'OCSR → YLM Context',
      file: 'lab/surrogate_query.py',
      colorKey: 'purple',
      inSpec: 'JSON query or OCSR SMILES ([Ar], CCO, …)',
      outSpec: '[physics] prompt lines or strict "unsupported"',
      cmd: 'python -m lab.surrogate_query --query q.json --context',
      desc: 'Validates physics queries for the YLM lab reasoner, runs short release-checkpoint rollouts for in-domain argon, and refuses untrained molecules instead of guessing.',
      jumpCapTab: 'lab',
    },
    {
      id: 'f5',
      num: '05',
      short: 'Hybrid & Constraints',
      sub: 'Verlet + SHAKE/RATTLE',
      file: 'equitraj/hybrid.py',
      colorKey: 'accent',
      inSpec: 'Neural rollout stream + StabilityMonitor rules',
      outSpec: 'Healed frames + <1e-10 Å bond projection',
      cmd: 'HybridRollout(...) / DistanceProjector(...)',
      desc: 'Catches close contacts or thermal spikes mid-rollout, relaxes with fine-step Verlet, enforces exact X–H bond lengths via mass-weighted DistanceProjector, and resumes 50 fs jumps.',
      jumpCapTab: 'hybrid',
    },
    {
      id: 'f6',
      num: '06',
      short: '10-Gate & PocketMD',
      sub: 'Symmetry & Triage',
      file: 'equitraj/compliance.py',
      colorKey: 'blue',
      inSpec: 'Model or PDB/XML drug & battery presets',
      outSpec: '10-gate O(3) verdict + GO/HOLD/REJECT receipt',
      cmd: 'wignerflow-pocket binding --pdb complex.pdb',
      desc: 'Audits rotation, reflection, translation, PBC, and COM conservation under surgical fault injections, and screens docked poses and Li⁺ electrolyte solvation shells.',
      jumpSection: '#studio',
    },
  ];

  function svgEl(name, attrs = {}, text) {
    const node = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs)) {
      if (v !== undefined && v !== null) node.setAttribute(k, String(v));
    }
    if (text !== undefined) node.textContent = text;
    return node;
  }

  async function fetchJSON(url, opts) {
    const res = await fetch(url, opts);
    if (!res.ok) {
      let detail = `${url} failed (${res.status})`;
      try {
        const errObj = await res.json();
        if (errObj && errObj.error) detail = errObj.error;
      } catch { /* ignore non-JSON error body */ }
      throw new Error(detail);
    }
    return res.json();
  }

  /* ---- 1. Interactive Architecture & Function Map ---- */
  function renderArchMap() {
    const host = document.getElementById('arch-map-host');
    if (!host) return;
    const P = window.WignerCharts.getPalette();
    const dark = document.documentElement.dataset.theme === 'dark';
    const W = 1080, H = 206;
    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Six-module WignerFlow architecture map' });

    const boxW = 156, boxH = 116, gap = 22, startX = 18, topY = 36;
    const colorOf = key => (key === 'blue' ? P.blue : key === 'purple' ? P.purple : P.accent);

    // Connecting pipeline arrows between consecutive modules
    for (let i = 0; i < ARCH_MODULES.length - 1; i++) {
      const x1 = startX + i * (boxW + gap) + boxW;
      const x2 = x1 + gap;
      const ym = topY + boxH * 0.5;
      const activeEdge = ARCH_MODULES[i].id === activeArchId || ARCH_MODULES[i + 1].id === activeArchId;
      svg.append(svgEl('line', {
        x1: x1 + 2, y1: ym, x2: x2 - 5, y2: ym,
        stroke: activeEdge ? P.accent : P.axis,
        'stroke-width': activeEdge ? 2.4 : 1.5,
      }));
      svg.append(svgEl('polygon', {
        points: `${x2 - 2},${ym} ${x2 - 9},${ym - 4.5} ${x2 - 9},${ym + 4.5}`,
        fill: activeEdge ? P.accent : P.axis,
      }));
    }

    // Autoregressive return arc from Module 05 back to Module 01
    svg.append(svgEl('path', {
      d: `M ${startX + 4 * (boxW + gap) + boxW * 0.5} ${topY + boxH} C ${startX + 4 * (boxW + gap)} ${H - 4}, ${startX + boxW} ${H - 4}, ${startX + boxW * 0.5} ${topY + boxH + 2}`,
      fill: 'none',
      stroke: P.accent,
      'stroke-width': 1.4,
      'stroke-dasharray': '5 4',
      opacity: 0.65,
    }));
    svg.append(svgEl('text', {
      x: W * 0.42, y: H - 8, fill: P.muted, 'font-size': 11.5, 'text-anchor': 'middle', 'font-weight': '550',
    }, '↺ autoregressive 50 fs context shift & hybrid state refill'));

    ARCH_MODULES.forEach((m, idx) => {
      const x = startX + idx * (boxW + gap);
      const col = colorOf(m.colorKey);
      const isSel = m.id === activeArchId;
      const g = svgEl('g', { class: 'arch-node', tabindex: '0', role: 'button', 'aria-label': `${m.num} ${m.short}` });

      g.append(svgEl('rect', {
        class: 'node-box',
        x, y: topY, width: boxW, height: boxH, rx: 11,
        fill: isSel ? (dark ? '#162636' : '#f0fdfa') : P.plot,
        stroke: isSel ? col : P.grid,
        'stroke-width': isSel ? 2.4 : 1.3,
      }));

      g.append(svgEl('rect', {
        x: x + 12, y: topY + 12, width: 30, height: 20, rx: 5,
        fill: col, 'fill-opacity': isSel ? 1 : 0.16,
      }));
      g.append(svgEl('text', {
        x: x + 27, y: topY + 26,
        fill: isSel ? (dark ? '#081c19' : '#ffffff') : col,
        'font-size': 11, 'font-weight': '700', 'text-anchor': 'middle',
      }, m.num));

      g.append(svgEl('text', {
        x: x + 12, y: topY + 54, fill: P.text, 'font-size': 13.5, 'font-weight': '700',
      }, m.short));
      g.append(svgEl('text', {
        x: x + 12, y: topY + 74, fill: col, 'font-size': 11.8, 'font-weight': '600',
      }, m.sub));
      g.append(svgEl('text', {
        x: x + 12, y: topY + 98, fill: P.muted, 'font-size': 10.8,
        'font-family': '"SFMono-Regular", Consolas, monospace',
      }, m.file));

      const activate = () => {
        activeArchId = m.id;
        renderArchMap();
      };
      g.addEventListener('click', activate);
      g.addEventListener('keydown', ev => {
        if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); activate(); }
      });
      svg.append(g);
    });

    host.replaceChildren(svg);

    const cur = ARCH_MODULES.find(m => m.id === activeArchId) || ARCH_MODULES[2];
    document.getElementById('arch-insp-tag').textContent = `MODULE ${cur.num} · ${cur.file.toUpperCase()}`;
    document.getElementById('arch-insp-title').textContent = cur.short;
    document.getElementById('arch-insp-desc').textContent = cur.desc;
    document.getElementById('arch-insp-io').innerHTML = `
      <div><strong>Input:</strong> ${cur.inSpec}</div>
      <div><strong>Output:</strong> ${cur.outSpec}</div>
      <div><strong>Entry:</strong> <code>${cur.cmd}</code></div>`;

    const jumpBtn = document.getElementById('arch-insp-jump');
    jumpBtn.onclick = () => {
      if (cur.jumpCapTab) {
        activateCapTab(cur.jumpCapTab);
        document.getElementById('capabilities')?.scrollIntoView({ behavior: 'smooth' });
      } else if (cur.jumpSection) {
        document.querySelector(cur.jumpSection)?.scrollIntoView({ behavior: 'smooth' });
      }
    };
  }

  function activateCapTab(tabKey) {
    document.querySelectorAll('[data-cap-tab]').forEach(b => b.classList.toggle('active', b.dataset.capTab === tabKey));
    document.querySelectorAll('.cap-pane').forEach(p => p.classList.toggle('active', p.id === `pane-cap-${tabKey}`));
    if (tabKey === 'api') renderCpuApiSandbox();
    if (tabKey === 'lab') renderLabSandbox();
    if (tabKey === 'hybrid') renderHybridSandbox();
  }

  /* ---- 2A. Interactive CPU Inference API Explorer ---- */
  function renderCpuApiSandbox() {
    if (!DATA || !DATA.cpu_api) return;
    const api = DATA.cpu_api;
    const rows = [1, 2, 4, 10].map(s => api.grid[`${cpuState.atoms}_${s}`]).filter(Boolean);
    const cur = api.grid[`${cpuState.atoms}_${cpuState.steps}`] || rows[1];

    window.WignerCharts.renderLatencyBarSVG(
      document.getElementById('cpu-lat-svg'),
      rows,
      cpuState.steps,
      api.budget_ms || 1000,
    );

    const budPill = document.getElementById('cpu-budget-pill');
    if (budPill && cur) {
      if (cur.wall_p95_ms <= 1000) {
        budPill.textContent = 'p50 & p95 < 1 s';
        budPill.className = 'pill pass';
      } else if (cur.wall_p50_ms <= 1000) {
        budPill.textContent = 'p50 < 1 s · p95 over';
        budPill.className = 'pill hold';
      } else {
        budPill.textContent = 'Over 1 s budget';
        budPill.className = 'pill fail';
      }
    }

    const stTitle = document.getElementById('cpu-stage-title');
    if (stTitle) stTitle.textContent = `Where Time Goes (256 Ar, S = ${cpuState.steps}) & FrameSplit`;
    const stackEl = document.getElementById('cpu-stage-stack');
    const legEl = document.getElementById('cpu-stage-legend');
    if (stackEl && legEl && api.stage_breakdown_256_s2) {
      const scaled = api.stage_breakdown_256_s2.map(st => {
        const isDen = st.stage.includes('Denoiser');
        const ms = isDen ? Math.round((st.ms / 2) * cpuState.steps) : st.ms;
        const stage = isDen ? `Flow Denoiser (S = ${cpuState.steps})` : st.stage;
        return { ...st, ms, stage };
      });
      const totalMs = Math.max(1, scaled.reduce((a, b) => a + b.ms, 0));
      stackEl.innerHTML = scaled
        .map(st => {
          const pct = ((100 * st.ms) / totalMs).toFixed(1);
          return `<div style="width:${pct}%;background:${st.color};" title="${st.stage}: ~${st.ms} ms (${Math.round(Number(pct))}%)"></div>`;
        })
        .join('');
      legEl.innerHTML = scaled
        .map(st => {
          const pct = Math.round((100 * st.ms) / totalMs);
          return `<span><i class="stage-dot" style="background:${st.color}"></i>${st.stage} <strong>${pct}%</strong></span>`;
        })
        .join('');
    }

    const rollInfo = (cpuState.therm === '95' && cur?.rollout_csvr95) ? cur.rollout_csvr95 : cur?.rollout_off;
    const rollCellStr = cpuState.atoms === 108
      ? '256 Ar only'
      : (rollInfo ? `${rollInfo.T_second_half_K} K (${rollInfo.rdf_peak_pos_err_A} Å)` : '—');

    const kpiEl = document.getElementById('cpu-api-kpis');
    if (kpiEl && cur) {
      kpiEl.innerHTML = `
        <div class="kpi-cell"><small>CPU p50 / Wall p50</small><strong>${Math.round(cur.cpu_p50_ms)} / ${Math.round(cur.wall_p50_ms)} ms</strong></div>
        <div class="kpi-cell"><small>Wall p95 (${cpuState.atoms} Ar)</small><strong>${Math.round(cur.wall_p95_ms)} ms</strong></div>
        <div class="kpi-cell"><small>1-Step dx / v Err</small><strong>${cur.dx_rel_rms_pct}% / ${cur.v_rel_rms_pct}%</strong></div>
        <div class="kpi-cell"><small>100-Step T (RDF)</small><strong>${rollCellStr}</strong></div>`;
    }

    const jsonEl = document.getElementById('cpu-api-json');
    if (jsonEl) {
      const snapKey = `${cpuState.atoms}_${cpuState.steps}_${cpuState.therm}`;
      const base = cpuState.liveSample || (api.samples && api.samples[snapKey]) || api.sample_response || {};
      const thermVal = cpuState.therm === '95' ? 95.0 : null;
      const shown = {
        ...base,
        n_atoms: cpuState.atoms,
        thermostat_K: thermVal,
        measured_bench_p50_ms: cur ? cur.wall_p50_ms : base.latency_ms,
        model: { ...(base.model || {}), sampling_steps: cpuState.steps, thermostat_K: thermVal },
      };
      jsonEl.textContent = JSON.stringify(shown, null, 2);
    }
  }

  /* ---- 2B. Interactive Lab Agent Surrogate Sandbox ---- */
  function renderLabSandbox(customPayload = null) {
    if (!DATA || !DATA.surrogate_lab) return;
    const item = customPayload || DATA.surrogate_lab[activeLabCase];
    if (!item) return;
    const rep = item.report || {};
    const status = rep.status || 'ok';

    const pill = document.getElementById('lab-status-pill');
    if (pill) {
      pill.textContent = `status: ${status}`;
      pill.className = `pill ${status}`;
    }
    document.getElementById('lab-query-json').textContent = JSON.stringify(item.query, null, 2);

    const kpis = document.getElementById('lab-kpis');
    if (status === 'ok') {
      kpis.innerHTML = `
        <div class="kpi-cell"><small>System</small><strong>${rep.system}</strong></div>
        <div class="kpi-cell"><small>Frames Used</small><strong>${rep.frames_used} / ${rep.n_steps}</strong></div>
        <div class="kpi-cell"><small>Mean Temp</small><strong>${Number(rep.T_model_mean_K).toFixed(1)} K</strong></div>
        <div class="kpi-cell"><small>Closest Pair</small><strong>${Number(rep.min_pair_distance_A).toFixed(2)} Å</strong></div>`;
    } else {
      kpis.innerHTML = `
        <div class="kpi-cell"><small>Requested</small><strong>${rep.system}</strong></div>
        <div class="kpi-cell"><small>Gate Action</small><strong class="orange">Refused</strong></div>
        <div class="kpi-cell"><small>Supported</small><strong>${(rep.supported || ['argon']).join(', ')}</strong></div>
        <div class="kpi-cell"><small>Hallucination</small><strong class="teal">0 (Blocked)</strong></div>`;
    }

    const promptPreview = [
      ...(item.context_lines || []),
      '',
      'Question:',
      item.query.question || 'Is this step physically safe to execute?',
    ].join('\n');
    document.getElementById('lab-context-out').textContent = promptPreview;
  }

  /* ---- 2C. Interactive Hybrid Fallback & DistanceProjector Sandbox ---- */
  function renderHybridSandbox() {
    if (!DATA || !DATA.constraints_demo) return;
    const levels = DATA.constraints_demo.levels || [];
    const cur = levels[Math.min(levels.length - 1, Math.max(0, projIdx))] || levels[0];
    if (!cur) return;

    const badge = document.getElementById('hybrid-badge');
    if (badge) {
      if (hybridMode === 'hybrid') {
        badge.textContent = 'Healed & Projected (<1e-10 Å)';
        badge.className = 'pill pass';
      } else {
        badge.textContent = 'OOD Bond Stretch / Crash';
        badge.className = 'pill fail';
      }
    }

    // Draw clean SVG timeline of neural steps + OOD collision + fallback healing + context refill
    const host = document.getElementById('hybrid-diagram-svg');
    if (host) {
      const P = window.WignerCharts.getPalette();
      const W = 580, H = 225;
      const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Hybrid rollout fallback timeline' });
      svg.style.cssText = 'display:block;width:100%;height:auto;font-family:inherit;';
      svg.append(svgEl('rect', { width: W, height: H, fill: P.bg, rx: 8 }));
      const padL = 52, padR = 22, padT = 24, padB = 42;
      const pw = W - padL - padR, ph = H - padT - padB;
      svg.append(svgEl('rect', { x: padL, y: padT, width: pw, height: ph, fill: P.plot, rx: 6 }));

      // Threshold dashed line
      const yThresh = padT + ph * 0.34;
      svg.append(svgEl('line', {
        x1: padL, y1: yThresh, x2: padL + pw, y2: yThresh,
        stroke: P.amber, 'stroke-width': 1.4, 'stroke-dasharray': '5 4',
      }));
      svg.append(svgEl('text', {
        x: padL + pw - 6, y: yThresh - 6, fill: P.amber, 'font-size': 10.5, 'font-weight': '650', 'text-anchor': 'end',
      }, 'StabilityMonitor threshold'));

      const sx = s => padL + (s / 10) * pw;
      const ampScale = Math.min(1.0, cur.perturbation_A / 0.26);

      if (hybridMode === 'hybrid') {
        // Shaded Verlet healing window [step 4 -> 6]
        svg.append(svgEl('rect', {
          x: sx(4), y: padT + 2, width: sx(6.5) - sx(4), height: ph - 4,
          fill: P.purple, 'fill-opacity': 0.12, rx: 4,
        }));
        svg.append(svgEl('text', {
          x: (sx(4) + sx(6.5)) * 0.5, y: padT + 16, fill: P.purple,
          'font-size': 10.5, 'font-weight': '700', 'text-anchor': 'middle',
        }, 'Verlet + Projector'));

        const pts = [
          [0, 0.82], [1, 0.80], [2, 0.78], [3, 0.75],
          [4, 0.30 - 0.16 * ampScale], // spike caught!
          [5, 0.68], [6.5, 0.81], [8, 0.80], [9, 0.82], [10, 0.81],
        ];
        const d = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p[0]).toFixed(1)},${(padT + p[1] * ph).toFixed(1)}`).join(' ');
        svg.append(svgEl('path', { d, fill: 'none', stroke: P.accent, 'stroke-width': 2.8, 'stroke-linecap': 'round' }));
        svg.append(svgEl('circle', { cx: sx(4), cy: padT + (0.30 - 0.16 * ampScale) * ph, r: 5.2, fill: P.amber }));
        svg.append(svgEl('circle', { cx: sx(6.5), cy: padT + 0.81 * ph, r: 5.0, fill: P.accent }));
      } else {
        const pts = [
          [0, 0.82], [1, 0.80], [2, 0.78], [3, 0.75],
          [4, 0.30 - 0.16 * ampScale],
          [5, 0.14], [6, 0.06],
        ];
        const d = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p[0]).toFixed(1)},${(padT + p[1] * ph).toFixed(1)}`).join(' ');
        svg.append(svgEl('path', { d, fill: 'none', stroke: P.orange, 'stroke-width': 2.8, 'stroke-dasharray': '5 3' }));
        const cx = sx(6), cy = padT + 0.06 * ph;
        svg.append(svgEl('path', {
          d: `M${cx - 6},${cy - 6}L${cx + 6},${cy + 6}M${cx - 6},${cy + 6}L${cx + 6},${cy - 6}`,
          stroke: P.orange, 'stroke-width': 2.6,
        }));
        svg.append(svgEl('text', {
          x: cx + 10, y: cy + 4, fill: P.orange, 'font-size': 11.5, 'font-weight': '700',
        }, 'Rollout crash'));
      }

      for (const s of [0, 2, 4, 6, 8, 10]) {
        svg.append(svgEl('text', {
          x: sx(s), y: H - 14, fill: P.muted, 'font-size': 11, 'text-anchor': 'middle',
        }, `step ${s}`));
      }
      svg.append(svgEl('text', {
        transform: `translate(16 ${padT + ph * 0.5}) rotate(-90)`,
        fill: P.muted, 'font-size': 11, 'text-anchor': 'middle',
      }, 'Bond / overlap error'));
      host.replaceChildren(svg);
    }

    document.getElementById('proj-kpis').innerHTML = `
      <div class="kpi-cell"><small>Before Project</small><strong class="orange">${cur.max_bond_err_before_A} Å</strong></div>
      <div class="kpi-cell"><small>After Project</small><strong class="teal">${cur.max_bond_err_after_A.toExponential(1)} Å</strong></div>
      <div class="kpi-cell"><small>Iterations</small><strong>${cur.iterations}</strong></div>
      <div class="kpi-cell"><small>COM Shift</small><strong>${cur.com_shift_A} Å</strong></div>`;

    document.getElementById('proj-bonds-tbody').innerHTML = (cur.bonds || []).map(b => `
      <tr>
        <td><strong>${b.bond}</strong></td>
        <td>${b.target_A.toFixed(4)} Å</td>
        <td class="orange">${b.perturbed_A.toFixed(4)} Å</td>
        <td class="teal">${b.projected_A.toFixed(4)} Å</td>
      </tr>`).join('');
  }

  /* ---- 3. Long-Horizon Evidence Chart & Benchmark Tables ---- */
  function renderEvidenceChart() {
    if (!DATA) return;
    const P = window.WignerCharts.getPalette();
    const sel = document.getElementById('benchmark-view-select');
    const bandWrap = document.getElementById('band-toggle-wrap');
    const bandChk = document.getElementById('show-target-band');
    const host = document.getElementById('evidence-chart');
    const legend = document.getElementById('evidence-chart-legend');
    const cap = document.getElementById('evidence-chart-caption');
    const view = sel ? sel.value : 'seeds';

    if (bandWrap) bandWrap.style.display = view === 'seeds' ? 'inline-flex' : 'none';

    if (view === 'seeds') {
      const p = DATA.argon_seeds;
      const seedCols = [P.accent, P.orange, P.blue, P.purple, P.amber];
      const series = p.series.map((s, idx) => {
        const markers = [];
        if (s.test_step) {
          if (s.test_T_K !== null) markers.push([s.test_step, s.test_T_K, 'dot']);
          else markers.push([s.test_step, 8.0, 'cross']);
        }
        return {
          name: `${s.name} · test ${s.survived}`,
          shortName: s.key,
          color: seedCols[idx % seedCols.length],
          dash: s.dash,
          highlight: s.highlight,
          points: s.points,
          markers,
          endLabel: s.test_T_K ? `${s.key}: ${s.test_T_K} K` : `${s.key}: crash`,
        };
      });
      window.WignerCharts.renderPublicationSVG(host, legend, {
        title: 'Validation & 10^4-step test temperature across 5 flow seeds',
        xLabel: 'Training step',
        yLabel: 'Rollout temperature (K)',
        xRange: [2000, 40000],
        yRange: [0, 250],
        xTicks: [2000, 10000, 20000, 30000, 40000],
        yTicks: [0, 50, 94.9, 150, 200, 250],
        formatX: v => `${Math.round(v / 1000)}k`,
        formatY: v => (Math.abs(v - 94.9) < 0.5 ? '94.9 (GT)' : String(Math.round(v))),
        formatTooltipY: v => `${v.toFixed(1)} K`,
        band: bandChk && bandChk.checked ? { low: p.band_low_K, high: p.band_high_K, color: P.accent } : null,
        hline: { y: p.t_gt_K, color: P.blue, dash: '5 5' },
        series,
      });
      cap.textContent = 'Lines: 1,000-step validation rollouts every 2,000 training steps (results/argon/B_S3a_seed1..5/log.jsonl). Circles: 15 × 10,000-step held-out test equilibrium temperature at best.pt (× marks Seed 2 crash near step 4,000).';
    } else if (view === 'rdf') {
      const r = DATA.argon_cinema.rdf;
      window.WignerCharts.renderPublicationSVG(host, legend, {
        title: 'Argon radial distribution function g(r)',
        xLabel: 'Pair distance r (Å)',
        yLabel: 'Pair correlation g(r)',
        xRange: [2.5, 10.0],
        yRange: [0, 3.2],
        xTicks: [3, 4, 5, 6, 7, 8, 9, 10],
        yTicks: [0, 1, 2, 3],
        formatX: v => `${v.toFixed(1)} Å`,
        formatY: v => v.toFixed(1),
        formatTooltipY: v => v.toFixed(3),
        hline: { y: 1.0, color: P.axis, dash: '4 4' },
        series: [
          { name: 'OpenMM NVE Ground Truth', shortName: 'Truth', color: P.blue, dash: [6, 4], points: r.r_A.map((x, i) => [x, r.gt[i]]), endLabel: 'Truth' },
          { name: 'WignerFlow Model-Only Rollout', shortName: 'Model', color: P.accent, highlight: true, points: r.r_A.map((x, i) => [x, r.model[i]]), endLabel: 'WignerFlow' },
        ],
      });
      cap.textContent = 'First solvation shell at 3.69 Å and second shell at 6.8 Å reproduced directly from autoregressive 50 fs jumps (checkpoints/argon_flow_v0.pt).';
    } else if (view === 'vacf') {
      const v = DATA.argon_cinema.vacf;
      window.WignerCharts.renderPublicationSVG(host, legend, {
        title: 'Argon velocity autocorrelation C(t)/C(0)',
        xLabel: 'Lag time t (fs)',
        yLabel: 'Normalized VACF C(t) / C(0)',
        xRange: [0, 1200],
        yRange: [-0.3, 1.05],
        xTicks: [0, 300, 600, 900, 1200],
        yTicks: [-0.2, 0.0, 0.5, 1.0],
        formatX: x => `${Math.round(x)} fs`,
        formatY: y => y.toFixed(1),
        formatTooltipY: y => y.toFixed(3),
        hline: { y: 0.0, color: P.axis, dash: '4 4' },
        series: [
          { name: 'OpenMM Ground Truth (zero @ 307.4 fs)', shortName: 'Truth', color: P.blue, dash: [6, 4], points: v.t_fs.map((x, i) => [x, v.gt[i]]), endLabel: 'Truth' },
          { name: 'WignerFlow Rollout (zero @ 305.5 fs)', shortName: 'Model', color: P.accent, highlight: true, points: v.t_fs.slice(0, v.model.length).map((x, i) => [x, v.model[i]]), endLabel: 'WignerFlow' },
        ],
      });
      cap.textContent = 'Normalized velocity autocorrelation C(t)/C(0) sampled at 50 fs stride, capturing liquid cage-rattling backscattering below zero after ~306 fs.';
    } else if (view === 'cutoff') {
      const c = DATA.cutoff_continuity;
      const sp = c.spacings_A;
      window.WignerCharts.renderPublicationSVG(host, legend, {
        title: 'Cutoff continuity scan across r_c = 6.0 Å (float64)',
        xLabel: 'Probe step spacing h (Å, log scale)',
        yLabel: 'Max output change across step (log scale)',
        xRange: [1e-4, 1e-2],
        yRange: [1e-9, 1.0],
        xTicks: [1e-4, 1e-3, 1e-2],
        yTicks: [1e-9, 1e-6, 1e-3, 1.0],
        logX: true,
        logY: true,
        formatX: x => x.toExponential(0),
        formatY: y => y.toExponential(0),
        formatTooltipY: y => y.toExponential(2),
        series: [
          {
            name: 'v0.1.0 Envelope After Softmax (jump)',
            shortName: 'v0',
            color: P.orange,
            dash: [6, 4],
            points: sp.map((h, i) => [h, c.v0_post_softmax_jump[i]]),
            markers: sp.map((h, i) => [h, c.v0_post_softmax_jump[i], 'dot']),
            endLabel: 'v0 jump',
          },
          {
            name: 'Interior Smooth Control (~O(h))',
            shortName: 'Control',
            color: P.blue,
            points: sp.map((h, i) => [h, c.interior_control_step[i]]),
            markers: sp.map((h, i) => [h, c.interior_control_step[i], 'dot']),
            endLabel: 'control',
          },
          {
            name: 'main Log-Envelope + Sink (smooth)',
            shortName: 'main',
            color: P.accent,
            highlight: true,
            points: sp.map((h, i) => [h, c.v1_log_envelope_sink_jump[i]]),
            markers: sp.map((h, i) => [h, c.v1_log_envelope_sink_jump[i], 'dot']),
            endLabel: 'main smooth',
          },
        ],
      });
      cap.textContent = 'Scanning a probe atom across r_c = 6.0 Å in float64 (results/bench/cutoff_continuity.json vs current equitraj/layers.py).';
    }
  }

  function renderCaliSolChart() {
    if (!DATA) return;
    const P = window.WignerCharts.getPalette();
    const cal = (DATA.benchmarks || {}).calisol23_alignment || {};
    const bellHost = document.getElementById('calisol-chart');
    if (bellHost && cal.available && cal.bell_curve_ec_dmc_298K && cal.bell_curve_ec_dmc_298K.length) {
      const pts = cal.bell_curve_ec_dmc_298K.map(p => [p.c_M, p.sigma_aligned]);
      window.WignerCharts.renderPublicationSVG(bellHost, null, {
        title: 'Aligned LiPF6 EC/DMC conductivity vs salt molarity (298 K)',
        height: 360,
        xLabel: 'LiPF6 concentration (M)',
        yLabel: 'Conductivity σ (mS/cm)',
        xRange: [0.2, 3.0],
        yRange: [0, 13],
        xTicks: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        yTicks: [0, 4, 8, 12],
        padR: 92,
        formatX: v => `${v.toFixed(1)} M`,
        formatY: v => `${v}`,
        formatTooltipY: v => `${v.toFixed(2)} mS/cm`,
        series: [{
          name: 'Aligned WignerFlow-NE',
          shortName: 'σ',
          color: P.accent,
          highlight: true,
          points: pts,
          markers: pts.map(p => [p[0], p[1], 'dot']),
          endLabel: '298 K',
        }],
      });
    }
  }

  function populateTablesAndCaliSol() {
    const abBody = document.getElementById('ablation-tbody');
    if (abBody && DATA.argon_ablations) {
      abBody.innerHTML = DATA.argon_ablations.map(r => {
        const sign = r.T_signed_pct >= 0 ? `+${r.T_signed_pct}%` : `${r.T_signed_pct}%`;
        const cls = r.all_pass ? 'pass' : 'fail';
        const txt = r.all_pass ? 'PASS' : 'FAIL';
        return `<tr>
          <td><strong>${r.label}</strong></td>
          <td>${r.head}</td>
          <td>${r.target}</td>
          <td>${r.train_step}</td>
          <td>${r.survived}</td>
          <td>${r.T_model_K} / ${r.T_gt_K} K (${sign})</td>
          <td>${r.rdf_pos_err_A} Å</td>
          <td>${r.vacf_err_pct}%</td>
          <td>${r.D_err_pct}%</td>
          <td><span class="pill ${cls}">${txt}</span></td>
        </tr>`;
      }).join('');
    }

    const cal = (DATA.benchmarks || {}).calisol23_alignment || {};
    const dl = document.getElementById('calisol-dl');
    if (dl && cal.available) {
      const pR2 = Number(cal.panel_aligned_r2 ?? cal.aligned_r2).toFixed(3);
      const pMae = Number(cal.panel_aligned_mae_mS_cm ?? cal.aligned_mae_mS_cm).toFixed(2);
      dl.innerHTML = `
        <div>
          <dt>Paired ${cal.n_formulations}-row MD panel (R² / MAE)</dt>
          <dd><strong>${pR2}</strong> <span>(${pMae} mS/cm vs ${Number(cal.unaligned_r2).toFixed(2)} R², ${Number(cal.unaligned_mae_mS_cm).toFixed(2)} mS/cm raw NE)</span></dd>
        </div>
        <div>
          <dt>Full ${cal.n_heldout_total}-row recipe split (R² / MAE)</dt>
          <dd><strong class="blue">${Number(cal.aligned_r2).toFixed(3)}</strong> <span>(${Number(cal.aligned_mae_mS_cm).toFixed(2)} mS/cm)</span></dd>
        </div>
        <div>
          <dt>Held-out ranking Spearman ρ</dt>
          <dd><strong>${Number(cal.aligned_spearman_rho).toFixed(3)}</strong> <span>(panel ρ = ${Number(cal.panel_aligned_spearman_rho ?? cal.aligned_spearman_rho).toFixed(3)} vs ${Number(cal.unaligned_spearman_rho).toFixed(3)} raw NE)</span></dd>
        </div>`;
      renderCaliSolChart();
    }

    const solvBody = document.getElementById('solvation-tbody');
    const solvRows = (DATA.benchmarks || {}).top15_solvation_champions || [];
    if (solvBody && solvRows.length) {
      solvBody.innerHTML = solvRows.slice(0, 6).map(r => {
        const ssip = Number(r.ssip);
        const cip = Number(r.cip);
        const agg = Number((Number(r.agg_i) + Number(r.agg_net)).toFixed(1));
        return `<tr>
          <td><strong>${r.rank}</strong></td>
          <td>${r.formula}</td>
          <td class="teal"><strong>${Number(r.sigma_298).toFixed(2)}</strong> mS/cm</td>
          <td>${Number(r.sigma_253).toFixed(2)} mS/cm</td>
          <td style="min-width:210px;">
            <div style="display:flex;height:8px;border-radius:99px;overflow:hidden;margin-bottom:3px;background:var(--surface2);">
              <span style="width:${ssip}%;background:var(--accent);"></span>
              <span style="width:${cip}%;background:var(--blue);"></span>
              <span style="width:${agg}%;background:var(--purple);"></span>
            </div>
            <small style="color:var(--muted);font-size:11px;">${ssip}% SSIP · ${cip}% CIP · ${agg}% AGG</small>
          </td>
          <td>${r.regime}</td>
        </tr>`;
      }).join('');
    }

    const md17Body = document.getElementById('md17-tbody');
    const md17Rows = (DATA.benchmarks || {}).pure_md_md17_md22 || [];
    if (md17Body && md17Rows.length) {
      md17Body.innerHTML = md17Rows.map(r => {
        const name = String(r.system).split(' (')[0].trim();
        const vv = String(r.vv_10fs).replace('fails at ', 'crash @ ');
        const wf = String(r.wf_10fs).replace('stable for ', 'stable · ');
        return `<tr>
          <td><strong>${name}</strong></td>
          <td class="orange">${vv}</td>
          <td class="teal">${wf}</td>
          <td>${Number(r.mae_hr).toFixed(3)}</td>
          <td>${Number(r.vdos_cos).toFixed(3)}</td>
        </tr>`;
      }).join('');
    }
  }

  /* ---- 4. Interactive 3D Studio Controllers ---- */
  function renderComplianceReport(rep) {
    lastCompReport = rep;
    const pill = document.getElementById('comp-summary-pill');
    if (pill) {
      pill.textContent = `${rep.pass_count} / ${rep.gates.length} PASS · ${rep.verdict}`;
      pill.className = `pill ${rep.verdict}`;
    }
    const list = document.getElementById('comp-gate-list');
    if (list) {
      list.innerHTML = rep.gates.map((g, idx) => `
        <li class="gate-row ${idx === 0 || g.status !== 'PASS' ? 'open' : ''}">
          <span class="gate-id">${g.id}</span>
          <strong>${g.name}</strong>
          <span class="gate-val">${g.value}</span>
          <span class="pill ${g.status}">${g.status}</span>
          <div class="gate-detail">${g.citation} &nbsp;·&nbsp; <code>${g.detail}</code> (threshold: ${g.threshold})</div>
        </li>`).join('');
      list.querySelectorAll('.gate-row').forEach(row => {
        row.addEventListener('click', () => row.classList.toggle('open'));
      });
    }
    drawComplianceGL();
  }

  function drawComplianceGL() {
    const v = lastCompReport && (lastCompReport.twin_3d || lastCompReport.viewer_3d);
    if (!compViewer || !v) return;
    const pos = v.pos_A || v.positions || [];
    const dxDir = v.dx_ref_A || v.dx_direct || [];
    const dxRot = v.dx_rot_unrot_A || v.dx_unrotated || [];
    const showArrows = document.getElementById('comp-show-arrows')?.checked ?? true;
    const dark = document.documentElement.dataset.theme === 'dark';
    const arrows = [];
    if (showArrows && dxDir.length && dxRot.length) {
      let maxNorm = 1e-6;
      for (const d of dxDir) maxNorm = Math.max(maxNorm, Math.hypot(d[0], d[1], d[2]));
      for (const d of dxRot) maxNorm = Math.max(maxNorm, Math.hypot(d[0], d[1], d[2]));
      const scale = 1.55 / maxNorm;
      const colDir = dark ? [0.18, 0.83, 0.75, 1.0] : [0.05, 0.58, 0.53, 1.0];
      const colRot = dark ? [0.38, 0.65, 0.98, 1.0] : [0.15, 0.39, 0.92, 1.0];
      for (let i = 0; i < pos.length; i++) {
        arrows.push({ from: pos[i], vec: dxDir[i], scale, radius: 0.052, color: colDir });
        arrows.push({ from: pos[i], vec: dxRot[i], scale: scale * 0.96, radius: 0.036, color: colRot });
      }
    }
    compViewer.setMolecule({
      positions: pos,
      elements: v.elements || [],
      bonds: v.bonds || [],
      boxA: v.box_A || null,
      arrows,
    });
  }

  function renderRolloutChartOnly() {
    if (!lastRollReport) return;
    const P = window.WignerCharts.getPalette();
    const rep = lastRollReport;
    const m = rep.metrics;
    const svgHost = document.getElementById('roll-temp-svg');
    const pts = rep.series.time_ps.map((t, i) => [t, rep.series.temperature_K[i]]);
    const finiteT = (rep.series.temperature_K || []).filter(Number.isFinite);
    const yTop = Math.max(450, m.T_target_K * 1.65, ...(finiteT.length ? [Math.max(...finiteT) * 1.1] : []));
    window.WignerCharts.renderPublicationSVG(svgHost, null, {
      title: 'Instantaneous kinetic temperature along trajectory',
      height: 310,
      xLabel: 'Simulated time (ps)',
      yLabel: 'Temperature (K)',
      xRange: [0, Math.max(0.05, rep.series.time_ps[rep.series.time_ps.length - 1] || 1)],
      yRange: [0, yTop],
      xTicks: [0, Number(((rep.series.time_ps[rep.series.time_ps.length - 1] || 1) * 0.5).toFixed(2)), Number((rep.series.time_ps[rep.series.time_ps.length - 1] || 1).toFixed(2))],
      yTicks: [0, Math.round(yTop * 0.33), Math.round(yTop * 0.67), Math.round(yTop * 0.95)],
      padR: 72,
      hline: { y: m.T_target_K, color: P.blue, dash: '5 4' },
      series: [{ name: 'T(t)', shortName: 'T', color: P.accent, highlight: true, points: pts, endLabel: `${m.T_mean_K} K` }],
    });
  }

  function renderRolloutReport(rep) {
    lastRollReport = rep;
    rollFrameIdx = 0;
    document.getElementById('roll-title').textContent = rep.title;
    const pill = document.getElementById('roll-verdict-pill');
    pill.textContent = rep.verdict;
    pill.className = `pill ${rep.verdict}`;

    const scrub = document.getElementById('roll-scrubber');
    scrub.max = Math.max(0, rep.viewer_frames.length - 1);
    scrub.value = 0;

    const m = rep.metrics;
    document.getElementById('roll-kpis').innerHTML = `
      <div class="kpi-cell"><small>Mean Temp</small><strong>${m.T_mean_K} K</strong></div>
      <div class="kpi-cell"><small>Bond RMS</small><strong>${m.bond_rms_drift_A} Å</strong></div>
      <div class="kpi-cell"><small>Min vdW Ratio</small><strong>${m.min_vdw_ratio}</strong></div>
      <div class="kpi-cell"><small>COM Leak</small><strong>${m.com_kinetic_leakage}</strong></div>`;

    renderRolloutChartOnly();
    drawRolloutFrame(0);
  }

  function drawRolloutFrame(idx) {
    if (!rollViewer || !lastRollReport) return;
    const frames = lastRollReport.viewer_frames;
    if (!frames || !frames.length) return;
    const i = Math.max(0, Math.min(frames.length - 1, idx));
    rollFrameIdx = i;
    document.getElementById('roll-frame-lbl').textContent = `Frame ${i + 1} / ${frames.length}`;
    document.getElementById('roll-scrubber').value = i;
    rollViewer.setMolecule({
      positions: frames[i],
      elements: lastRollReport.viewer_elements,
      bonds: lastRollReport.viewer_bonds || [],
      boxA: lastRollReport.viewer_box_A || null,
    });
  }

  function renderBindingTriage(payload) {
    const rep = payload.report;
    const pill = document.getElementById('bind-verdict-pill');
    pill.textContent = rep.verdict;
    pill.className = `pill ${rep.verdict}`;
    const m = rep.metrics || {};
    const survRaw = m.contact_persistence_R ?? m.pers_score ?? m.contact_survival ?? 0;
    const survPct = Math.min(100, Math.max(0, Math.round(100 * Number(survRaw))));
    const compScore = Number(m.comp_score ?? m.pose_drift_rmsd_A ?? 0);
    const tauPs = Number(m.tau_ramd_median_ps ?? m.tau_ramd_ps ?? 0);
    const clashes = Number(m.posebusters_vdw_clashes ?? m.vdw_clashes ?? 0);

    document.getElementById('bind-kpis').innerHTML = `
      <div class="kpi-cell"><small>CompScore</small><strong>${compScore.toFixed(3)}</strong></div>
      <div class="kpi-cell"><small>Contact Survival</small><strong>${survPct}%</strong></div>
      <div class="kpi-cell"><small>τ-RAMD Unbind</small><strong>${tauPs.toFixed(2)} ps</strong></div>
      <div class="kpi-cell"><small>vdW Clashes</small><strong>${clashes}</strong></div>`;

    const gauges = document.getElementById('bind-gauges');
    if (gauges) {
      const tauPct = Math.min(100, Math.round((tauPs / 2.5) * 100));
      const stericOkPct = clashes === 0 ? 100 : Math.max(12, 100 - clashes * 25);
      gauges.innerHTML = `
        <div class="gauge-row">
          <span>Contact Survival</span>
          <div class="gauge-track"><div class="gauge-fill" style="width:${survPct}%;background:var(--accent);"></div></div>
          <strong>${survPct}%</strong>
        </div>
        <div class="gauge-row">
          <span>τ-RAMD Residence</span>
          <div class="gauge-track"><div class="gauge-fill" style="width:${tauPct}%;background:var(--blue);"></div></div>
          <strong>${tauPs.toFixed(2)} ps</strong>
        </div>
        <div class="gauge-row">
          <span>Steric Clash Pass</span>
          <div class="gauge-track"><div class="gauge-fill" style="width:${stericOkPct}%;background:${clashes === 0 ? 'var(--accent)' : 'var(--orange)'};"></div></div>
          <strong>${clashes === 0 ? 'Clean' : `${clashes} clash`}</strong>
        </div>`;
    }
    document.getElementById('bind-receipt').textContent = payload.receipt || JSON.stringify(rep, null, 2);
  }

  function renderElectrolyteTriage(payload) {
    const rep = payload.report;
    const pill = document.getElementById('elec-verdict-pill');
    pill.textContent = rep.verdict;
    pill.className = `pill ${rep.verdict}`;
    const m = rep.metrics || {};
    const sigmaVal = Number(m.sigma_cNE_mS_cm ?? m.sigma_cne_mS_cm ?? m.conductivity_mS_cm ?? 0);
    const ssip = Math.round(100 * Number(m.ssip_fraction ?? m.ssip_frac ?? 0));
    const cip = Math.round(100 * Number(m.cip_fraction ?? m.cip_frac ?? 0));
    const agg = Math.max(0, 100 - ssip - cip);

    document.getElementById('elec-kpis').innerHTML = `
      <div class="kpi-cell"><small>σ (Cluster-NE)</small><strong>${sigmaVal.toFixed(2)} mS/cm</strong></div>
      <div class="kpi-cell"><small>Free Ion SSIP</small><strong>${ssip}%</strong></div>
      <div class="kpi-cell"><small>Ion Pair CIP</small><strong>${cip}%</strong></div>
      <div class="kpi-cell"><small>Aggregate AGG</small><strong>${agg}%</strong></div>`;

    const barHost = document.getElementById('elec-speciation-bar');
    if (barHost) {
      barHost.innerHTML = `
        <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--muted);font-weight:600;">
          <span>Li⁺ Coordination Shell Distribution</span>
          <span>SSIP ${ssip}% · CIP ${cip}% · AGG ${agg}%</span>
        </div>
        <div class="speciation-track">
          <div class="speciation-seg" style="width:${ssip}%;background:var(--accent);">${ssip > 14 ? `SSIP ${ssip}%` : ''}</div>
          <div class="speciation-seg" style="width:${cip}%;background:var(--blue);">${cip > 14 ? `CIP ${cip}%` : ''}</div>
          <div class="speciation-seg" style="width:${agg}%;background:${agg > 40 ? 'var(--orange)' : 'var(--purple)'};">${agg > 14 ? `AGG ${agg}%` : ''}</div>
        </div>`;
    }
    document.getElementById('elec-receipt').textContent = payload.receipt || JSON.stringify(rep, null, 2);
  }

  async function initSandboxesAndStudio() {
    const defaults = DATA.studio_defaults;

    // Capabilities Sandbox tab switching
    document.querySelectorAll('[data-cap-tab]').forEach(btn => {
      btn.addEventListener('click', () => activateCapTab(btn.dataset.capTab));
    });

    // CPU API Sandbox controls
    document.querySelectorAll('[data-cpu-atoms]').forEach(btn => {
      btn.addEventListener('click', () => {
        cpuState.atoms = Number(btn.dataset.cpuAtoms);
        cpuState.liveSample = null;
        document.querySelectorAll('[data-cpu-atoms]').forEach(b => b.classList.toggle('active', b === btn));
        renderCpuApiSandbox();
      });
    });
    document.querySelectorAll('[data-cpu-s]').forEach(btn => {
      btn.addEventListener('click', () => {
        cpuState.steps = Number(btn.dataset.cpuS);
        cpuState.liveSample = null;
        document.querySelectorAll('[data-cpu-s]').forEach(b => b.classList.toggle('active', b === btn));
        renderCpuApiSandbox();
      });
    });
    document.querySelectorAll('[data-cpu-therm]').forEach(btn => {
      btn.addEventListener('click', () => {
        cpuState.therm = btn.dataset.cpuTherm;
        cpuState.liveSample = null;
        document.querySelectorAll('[data-cpu-therm]').forEach(b => b.classList.toggle('active', b === btn));
        renderCpuApiSandbox();
      });
    });
    document.getElementById('btn-live-cpu-api')?.addEventListener('click', async e => {
      const btn = e.currentTarget;
      btn.textContent = 'Predicting…';
      try {
        if (LIVE_API) {
          cpuState.liveSample = await fetchJSON('/api/cpu_predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              n_atoms: cpuState.atoms,
              sampling_steps: cpuState.steps,
              thermostat_K: cpuState.therm === '95' ? 95.0 : null,
            }),
          });
        }
        renderCpuApiSandbox();
      } catch (err) {
        console.warn('CPU API request error:', err);
      } finally {
        btn.textContent = 'Run 1-Step Predict';
      }
    });

    // Lab Agent Sandbox controls
    document.querySelectorAll('[data-lab-case]').forEach(btn => {
      btn.addEventListener('click', () => {
        activeLabCase = btn.dataset.labCase;
        document.querySelectorAll('[data-lab-case]').forEach(b => b.classList.toggle('active', b === btn));
        renderLabSandbox();
      });
    });
    document.getElementById('btn-run-lab-query')?.addEventListener('click', async e => {
      const btn = e.currentTarget;
      btn.textContent = 'Querying…';
      try {
        const snap = DATA.surrogate_lab[activeLabCase];
        if (LIVE_API && snap) {
          const live = await fetchJSON('/api/surrogate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: snap.query }),
          });
          renderLabSandbox(live);
        } else {
          renderLabSandbox();
        }
      } catch (err) {
        console.warn('Surrogate query error:', err);
      } finally {
        btn.textContent = 'Execute Query';
      }
    });

    // Hybrid & DistanceProjector Sandbox controls
    document.querySelectorAll('[data-proj-idx]').forEach(btn => {
      btn.addEventListener('click', () => {
        projIdx = Number(btn.dataset.projIdx);
        document.querySelectorAll('[data-proj-idx]').forEach(b => b.classList.toggle('active', b === btn));
        renderHybridSandbox();
      });
    });
    document.querySelectorAll('[data-hybrid-mode]').forEach(btn => {
      btn.addEventListener('click', () => {
        hybridMode = btn.dataset.hybridMode;
        document.querySelectorAll('[data-hybrid-mode]').forEach(b => b.classList.toggle('active', b === btn));
        renderHybridSandbox();
      });
    });

    renderArchMap();
    renderCpuApiSandbox();
    renderLabSandbox();
    renderHybridSandbox();

    // 3D Studio setup
    const tgtSel = document.getElementById('comp-target');
    const fltSel = document.getElementById('comp-fault');
    tgtSel.innerHTML = defaults.compliance_targets.map(t => `<option value="${t.id}" ${t.id === 'aspirin_hetero_o3' ? 'selected' : ''}>${t.title}</option>`).join('');
    fltSel.innerHTML = defaults.fault_catalog.map(f => `<option value="${f.id}">${f.title}</option>`).join('');
    document.getElementById('roll-xyz-text').value = defaults.default_xyz;

    compViewer = new window.WignerGLViewer(document.getElementById('comp-gl-canvas'), { rotX: 0.34, rotY: -0.55 });
    rollViewer = new window.WignerGLViewer(document.getElementById('roll-gl-canvas'), { rotX: 0.32, rotY: -0.45 });

    document.getElementById('comp-reset-view').addEventListener('click', () => compViewer.resetView(0.34, -0.55));
    document.getElementById('roll-reset-view').addEventListener('click', () => rollViewer.resetView(0.32, -0.45));
    document.getElementById('comp-show-arrows').addEventListener('change', drawComplianceGL);

    // Euler angle sliders sync
    const ea = document.getElementById('euler-a');
    const eb = document.getElementById('euler-b');
    const eg = document.getElementById('euler-g');
    const syncEulerSliders = () => {
      const a = Number(ea.value), b = Number(eb.value), g = Number(eg.value);
      document.getElementById('lbl-ea').textContent = `${a}°`;
      document.getElementById('lbl-eb').textContent = `${b}°`;
      document.getElementById('lbl-eg').textContent = `${g}°`;
      document.getElementById('comp-angles').value = `${a}, ${b}, ${g}`;
      if (compViewer) {
        compViewer.rotY = -0.55 + ((a - 37) * Math.PI) / 180 + ((g - 118) * Math.PI) / 360;
        compViewer.rotX = Math.max(-1.4, Math.min(1.4, 0.34 + ((b - 64) * Math.PI) / 360));
        compViewer.render();
      }
    };
    [ea, eb, eg].forEach(inp => inp?.addEventListener('input', syncEulerSliders));

    // Check live PyTorch backend
    const badge = document.getElementById('studio-engine-badge');
    const badgeTxt = document.getElementById('studio-engine-text');
    try {
      const ov = await fetchJSON('/api/overview');
      LIVE_API = true;
      badge.classList.remove('static');
      badgeTxt.textContent = `PyTorch ${ov.runtime.torch} · Live CPU Engine`;
    } catch {
      LIVE_API = false;
      badge.classList.add('static');
      badgeTxt.textContent = 'Interactive WebGL2 · GitHub Pages Edition';
      ['roll-frames', 'roll-temp', 'bind-replicas'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
          el.disabled = true;
          el.title = 'Live parameter customization requires running python scripts/wignerflow_studio.py locally.';
        }
      });
    }

    const compSnaps = defaults.compliance_snapshots || {};
    const rollSnaps = defaults.rollout_snapshots || { aspirin: defaults.rollout_aspirin };

    function pickStaticCompliance(targetId, faultId, precision) {
      const isArgon = targetId.startsWith('argon_');
      const familyFallback = isArgon ? `argon_flow_v1__${faultId}` : `aspirin_hetero_o3__${faultId}`;
      return (
        compSnaps[`${targetId}__${faultId}__${precision}`] ||
        compSnaps[`${targetId}__${faultId}`] ||
        compSnaps[familyFallback] ||
        compSnaps[`${targetId}__none`] ||
        compSnaps['aspirin_hetero_o3__none__float64']
      );
    }

    renderComplianceReport(pickStaticCompliance('aspirin_hetero_o3', 'none', 'float64'));
    renderRolloutReport(rollSnaps.aspirin || defaults.rollout_aspirin);
    renderBindingTriage(defaults.binding_snapshots.aspirin_cox_stable);
    renderElectrolyteTriage(defaults.electrolyte_snapshots.li_ec_fec_additive);

    document.querySelectorAll('[data-studio-tab]').forEach(btn => {
      btn.addEventListener('click', () => {
        const key = btn.dataset.studioTab;
        document.querySelectorAll('[data-studio-tab]').forEach(b => b.classList.toggle('active', b === btn));
        document.querySelectorAll('.studio-pane').forEach(p => p.classList.toggle('active', p.id === `pane-${key}`));
        requestAnimationFrame(() => {
          if (key === 'compliance' && compViewer) compViewer.render();
          if (key === 'rollout' && rollViewer) { rollViewer.render(); renderRolloutChartOnly(); }
        });
      });
    });

    const triggerCompliance = async () => {
      const btn = document.getElementById('btn-run-compliance');
      btn.textContent = 'Auditing…';
      try {
        const euler = document.getElementById('comp-angles').value.split(',').map(s => parseFloat(s.trim()) || 0).slice(0, 3);
        const payload = {
          target_id: tgtSel.value,
          fault: fltSel.value,
          precision: document.getElementById('comp-precision').value,
          euler_deg: euler.length === 3 ? euler : [37, 64, 118],
          rollout_steps: 12,
        };
        if (LIVE_API) {
          const rep = await fetchJSON('/api/compliance', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
          });
          renderComplianceReport(rep);
        } else {
          renderComplianceReport(pickStaticCompliance(payload.target_id, payload.fault, payload.precision));
        }
      } catch (err) {
        console.warn('Compliance audit error:', err);
      } finally {
        btn.textContent = 'Run Audit';
      }
    };
    document.getElementById('btn-run-compliance').addEventListener('click', triggerCompliance);
    fltSel.addEventListener('change', triggerCompliance);
    tgtSel.addEventListener('change', triggerCompliance);
    document.getElementById('comp-precision')?.addEventListener('change', triggerCompliance);

    document.getElementById('roll-preset').addEventListener('change', e => {
      const isCustom = e.target.value === 'custom';
      document.getElementById('roll-custom-wrap').style.display = isCustom ? 'block' : 'none';
      const note = document.getElementById('roll-custom-note');
      if (note) {
        note.style.display = (isCustom && !LIVE_API) ? 'block' : 'none';
        note.textContent = 'Live custom .xyz simulation runs when connected to python scripts/wignerflow_studio.py; static GitHub Pages previews Ethanol.';
      }
    });

    document.getElementById('btn-run-rollout').addEventListener('click', async e => {
      const btn = e.currentTarget;
      btn.textContent = 'Simulating…';
      try {
        const preset = document.getElementById('roll-preset').value;
        if (LIVE_API) {
          const rep = await fetchJSON('/api/rollout', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              preset: preset === 'custom' ? 'ethanol' : preset,
              custom_xyz: preset === 'custom' ? document.getElementById('roll-xyz-text').value : null,
              n_frames: Number(document.getElementById('roll-frames').value) || 54,
              temperature_k: Number(document.getElementById('roll-temp').value) || 300,
            }),
          });
          renderRolloutReport(rep);
          const note = document.getElementById('roll-custom-note');
          if (note && preset === 'custom') note.style.display = 'none';
        } else {
          renderRolloutReport(rollSnaps[preset === 'custom' ? 'ethanol' : preset] || rollSnaps.aspirin);
        }
      } catch (err) {
        const note = document.getElementById('roll-custom-note');
        if (note) {
          note.style.display = 'block';
          note.textContent = `Rollout error: ${err.message || err}`;
        }
      } finally {
        btn.textContent = 'Simulate';
      }
    });

    document.getElementById('btn-dl-xyz').addEventListener('click', () => {
      if (!lastRollReport || !lastRollReport.xyz_download) return;
      const blob = new Blob([lastRollReport.xyz_download], { type: 'chemical/x-xyz' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'wignerflow_rollout.xyz';
      a.click();
    });

    document.getElementById('roll-scrubber').addEventListener('input', e => {
      rollPlaying = false;
      document.getElementById('roll-play-btn').textContent = '▶ Play';
      drawRolloutFrame(Number(e.target.value));
    });
    document.getElementById('roll-play-btn').addEventListener('click', e => {
      rollPlaying = !rollPlaying;
      e.currentTarget.textContent = rollPlaying ? '❚❚ Pause' : '▶ Play';
    });

    setInterval(() => {
      const pane = document.getElementById('pane-rollout');
      if (rollPlaying && pane && pane.classList.contains('active') && lastRollReport) {
        drawRolloutFrame((rollFrameIdx + 1) % lastRollReport.viewer_frames.length);
      }
    }, 90);

    document.getElementById('btn-run-binding').addEventListener('click', async e => {
      const btn = e.currentTarget;
      btn.textContent = 'Screening…';
      try {
        const preset = document.getElementById('bind-preset').value;
        if (LIVE_API) {
          const out = await fetchJSON('/api/triage', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: 'binding', preset, replicas: Number(document.getElementById('bind-replicas').value) || 3 }),
          });
          renderBindingTriage(out);
        } else {
          renderBindingTriage(defaults.binding_snapshots[preset] || defaults.binding_snapshots.aspirin_cox_stable);
        }
      } catch (err) {
        document.getElementById('bind-receipt').textContent = `Error: ${err.message || err}`;
      } finally {
        btn.textContent = 'Screen Pose';
      }
    });

    document.getElementById('btn-run-elec').addEventListener('click', async e => {
      const btn = e.currentTarget;
      btn.textContent = 'Running…';
      try {
        const preset = document.getElementById('elec-preset').value;
        if (LIVE_API) {
          const out = await fetchJSON('/api/triage', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: 'electrolyte', preset }),
          });
          renderElectrolyteTriage(out);
        } else {
          renderElectrolyteTriage(defaults.electrolyte_snapshots[preset] || defaults.electrolyte_snapshots.li_ec_fec_additive);
        }
      } catch (err) {
        document.getElementById('elec-receipt').textContent = `Error: ${err.message || err}`;
      } finally {
        btn.textContent = 'Run Speciation';
      }
    });
  }

  function applyTheme(nextTheme) {
    document.documentElement.dataset.theme = nextTheme;
    const btn = document.getElementById('theme-toggle');
    if (btn) btn.textContent = nextTheme === 'dark' ? '☀ Light' : '☾ Dark';
    window.WignerGLViewer.syncAllThemes();
    window.WignerCinema.refreshTheme();
    if (DATA) {
      renderArchMap();
      renderCpuApiSandbox();
      renderHybridSandbox();
      renderEvidenceChart();
      renderCaliSolChart();
      renderRolloutChartOnly();
      drawComplianceGL();
    }
  }

  async function boot() {
    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
      themeBtn.addEventListener('click', () => {
        const cur = document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
        applyTheme(cur === 'dark' ? 'light' : 'dark');
      });
    }

    const copyBtn = document.getElementById('copy-citation');
    if (copyBtn) {
      copyBtn.addEventListener('click', async () => {
        const status = document.getElementById('copy-status');
        try {
          await navigator.clipboard.writeText(document.getElementById('bibtex').textContent);
          copyBtn.textContent = 'Copied ✓';
          status.textContent = 'Copied to clipboard.';
          setTimeout(() => { copyBtn.textContent = 'Copy citation'; status.textContent = ''; }, 2400);
        } catch {
          status.textContent = 'Select and copy above.';
        }
      });
    }

    window.WignerCinema.initMechanismAnimations();

    DATA = await fetchJSON('assets/homepage-data.json');
    window.WignerCinema.initHeroCinema(DATA.argon_cinema);
    renderEvidenceChart();
    populateTablesAndCaliSol();

    document.getElementById('benchmark-view-select').addEventListener('change', renderEvidenceChart);
    document.getElementById('show-target-band').addEventListener('change', renderEvidenceChart);
    window.addEventListener('resize', () => requestAnimationFrame(() => {
      renderEvidenceChart();
      renderCaliSolChart();
    }), { passive: true });

    await initSandboxesAndStudio();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot, { once: true });
  else boot();
})();
