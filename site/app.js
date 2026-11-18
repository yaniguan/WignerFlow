/* Main controller for WignerFlow Homepage & Interactive Studio.
 * Works both statically (GitHub Pages using assets/homepage-data.json)
 * and live against python scripts/wignerflow_studio.py (/api/*).
 */
(() => {
  'use strict';

  let DATA = null;
  let LIVE_API = false;
  let compViewer = null;
  let rollViewer = null;
  let lastCompReport = null;
  let lastRollReport = null;
  let rollFrameIdx = 0;
  let rollPlaying = true;

  async function fetchJSON(url, opts) {
    const res = await fetch(url, opts);
    if (!res.ok) throw new Error(`${url} failed (${res.status})`);
    return res.json();
  }

  function renderEvidenceChart() {
    if (!DATA) return;
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
      const series = p.series.map(s => {
        const markers = [];
        if (s.test_step) {
          if (s.test_T_K !== null) markers.push([s.test_step, s.test_T_K, 'dot']);
          else markers.push([s.test_step, 8.0, 'cross']);
        }
        return {
          name: `${s.name} · test ${s.survived}`,
          shortName: s.key,
          color: s.color,
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
        band: bandChk && bandChk.checked ? { low: p.band_low_K, high: p.band_high_K, color: '#b5d88e' } : null,
        hline: { y: p.t_gt_K, color: '#91b6df', dash: '5 5' },
        series,
      });
      cap.textContent = 'Lines: 1,000-step validation rollouts every 2,000 training steps (results/argon/B_S3a_seed1..5/log.jsonl). Circles: 15 × 10,000-step held-out test rollout equilibrium temperature at best.pt (× marks Seed 2, which crashes around step 4,000). Shaded band: ±5% around 94.9 K ground truth.';
    } else if (view === 'rdf') {
      const r = DATA.argon_cinema.rdf;
      const ptsGt = r.r_A.map((x, i) => [x, r.gt[i]]);
      const ptsMod = r.r_A.map((x, i) => [x, r.model[i]]);
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
        hline: { y: 1.0, color: '#5f7285', dash: '4 4' },
        series: [
          { name: 'OpenMM NVE Ground Truth', shortName: 'Truth', color: '#91b6df', dash: [6, 4], points: ptsGt, endLabel: 'Truth' },
          { name: 'WignerFlow Model-Only Rollout', shortName: 'Model', color: '#b5d88e', highlight: true, points: ptsMod, endLabel: 'WignerFlow' },
        ],
      });
      cap.textContent = 'First solvation shell at 3.69 Å and second shell at 6.8 Å reproduced directly from autoregressive 50 fs jumps with no force field (checkpoints/argon_flow_v0.pt).';
    } else if (view === 'vacf') {
      const v = DATA.argon_cinema.vacf;
      const ptsGt = v.t_fs.map((x, i) => [x, v.gt[i]]);
      const ptsMod = v.t_fs.slice(0, v.model.length).map((x, i) => [x, v.model[i]]);
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
        hline: { y: 0.0, color: '#6b7d90', dash: '4 4' },
        series: [
          { name: 'OpenMM Ground Truth (zero @ 307.4 fs)', shortName: 'Truth', color: '#91b6df', dash: [6, 4], points: ptsGt, endLabel: 'Truth' },
          { name: 'WignerFlow Rollout (10⁴-step zero @ 305.5 fs)', shortName: 'Model', color: '#b5d88e', highlight: true, points: ptsMod, endLabel: 'WignerFlow' },
        ],
      });
      cap.textContent = 'Normalized velocity autocorrelation C(t)/C(0) sampled at the 50 fs stride, capturing liquid cage-rattling backscattering below zero after ~306 fs.';
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
            name: 'v0.1.0 Envelope After Softmax (discontinuous)',
            shortName: 'v0',
            color: '#d97757',
            dash: [6, 4],
            points: sp.map((h, i) => [h, c.v0_post_softmax_jump[i]]),
            markers: sp.map((h, i) => [h, c.v0_post_softmax_jump[i], 'dot']),
            endLabel: 'v0 jump',
          },
          {
            name: 'Interior Smooth Control Step (~O(h))',
            shortName: 'Control',
            color: '#91b6df',
            points: sp.map((h, i) => [h, c.interior_control_step[i]]),
            markers: sp.map((h, i) => [h, c.interior_control_step[i], 'dot']),
            endLabel: 'control',
          },
          {
            name: 'main Log-Envelope + Zero-Logit Sink (continuous)',
            shortName: 'main',
            color: '#b5d88e',
            highlight: true,
            points: sp.map((h, i) => [h, c.v1_log_envelope_sink_jump[i]]),
            markers: sp.map((h, i) => [h, c.v1_log_envelope_sink_jump[i], 'dot']),
            endLabel: 'main smooth',
          },
        ],
      });
      cap.textContent = 'Scanning a probe atom across r_c = 6.0 Å in float64 (results/bench/cutoff_continuity.json vs current equitraj/layers.py). Moving log(envelope) inside softmax with a zero-logit sink eliminates the O(1) denominator jump.';
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
      dl.innerHTML = `
        <div>
          <dt>Held-out CaliSol-23 formulations (R²)</dt>
          <dd><strong>${Number(cal.aligned_r2).toFixed(3)}</strong> <span>vs ${Number(cal.unaligned_r2).toFixed(2)} raw Nernst–Einstein</span></dd>
        </div>
        <div>
          <dt>Held-out ranking Spearman ρ</dt>
          <dd><strong>${Number(cal.aligned_spearman_rho).toFixed(3)}</strong> <span>(${cal.n_heldout_total} held-out test formulations)</span></dd>
        </div>
        <div>
          <dt>Conductivity MAE (mS/cm)</dt>
          <dd><strong class="blue">${Number(cal.aligned_mae_mS_cm).toFixed(2)}</strong> <span>vs ${Number(cal.unaligned_mae_mS_cm).toFixed(2)} unaligned</span></dd>
        </div>`;
      const bellHost = document.getElementById('calisol-chart');
      if (bellHost && cal.bell_curve_ec_dmc_298K && cal.bell_curve_ec_dmc_298K.length) {
        const pts = cal.bell_curve_ec_dmc_298K.map(p => [p.c_M, p.sigma_aligned]);
        window.WignerCharts.renderPublicationSVG(bellHost, null, {
          title: 'Aligned LiPF6 EC/DMC conductivity vs salt molarity (298 K)',
          xLabel: 'LiPF6 concentration (M)',
          yLabel: 'Conductivity σ (mS/cm)',
          xRange: [0.2, 3.0],
          yRange: [0, 13],
          xTicks: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
          yTicks: [0, 4, 8, 12],
          padR: 96,
          formatX: v => `${v.toFixed(1)} M`,
          formatY: v => `${v}`,
          formatTooltipY: v => `${v.toFixed(2)} mS/cm`,
          series: [{
            name: 'Aligned WignerFlow-NE',
            shortName: 'σ',
            color: '#b5d88e',
            highlight: true,
            points: pts,
            markers: pts.map(p => [p[0], p[1], 'dot']),
            endLabel: '298 K',
          }],
        });
      }
    }

    const solvBody = document.getElementById('solvation-tbody');
    const solvRows = (DATA.benchmarks || {}).top15_solvation_champions || [];
    if (solvBody && solvRows.length) {
      solvBody.innerHTML = solvRows.slice(0, 7).map(r => `<tr>
        <td><strong>${r.rank}</strong></td>
        <td>${r.formula}</td>
        <td class="lime">${Number(r.sigma_298).toFixed(2)} mS/cm</td>
        <td>${Number(r.sigma_253).toFixed(2)} mS/cm</td>
        <td>${r.ssip}%</td>
        <td>${r.cip}%</td>
        <td>${(Number(r.agg_i) + Number(r.agg_net)).toFixed(1)}%</td>
        <td>${r.regime}</td>
      </tr>`).join('');
    }

    const md17Body = document.getElementById('md17-tbody');
    const md17Rows = (DATA.benchmarks || {}).pure_md_md17_md22 || [];
    if (md17Body && md17Rows.length) {
      md17Body.innerHTML = md17Rows.map(r => `<tr>
        <td><strong>${r.system}</strong></td>
        <td class="orange">${r.vv_10fs}</td>
        <td class="lime">${r.wf_10fs}</td>
        <td>${Number(r.mae_hr).toFixed(3)}</td>
        <td>${Number(r.vdos_cos).toFixed(3)}</td>
      </tr>`).join('');
    }
  }

  /* ---- Studio Section Interactive Controllers ---- */
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
    const arrows = [];
    if (showArrows && dxDir.length && dxRot.length) {
      let maxNorm = 1e-6;
      for (const d of dxDir) maxNorm = Math.max(maxNorm, Math.hypot(d[0], d[1], d[2]));
      for (const d of dxRot) maxNorm = Math.max(maxNorm, Math.hypot(d[0], d[1], d[2]));
      const scale = 1.55 / maxNorm;
      for (let i = 0; i < pos.length; i++) {
        arrows.push({
          from: pos[i],
          vec: dxDir[i],
          scale,
          radius: 0.052,
          color: [0.71, 0.85, 0.56, 1.0],
        });
        arrows.push({
          from: pos[i],
          vec: dxRot[i],
          scale: scale * 0.96,
          radius: 0.036,
          color: [0.57, 0.71, 0.88, 1.0],
        });
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

    const svgHost = document.getElementById('roll-temp-svg');
    const pts = rep.series.time_ps.map((t, i) => [t, rep.series.temperature_K[i]]);
    window.WignerCharts.renderPublicationSVG(svgHost, null, {
      title: 'Instantaneous kinetic temperature along trajectory',
      xLabel: 'Simulated time (ps)',
      yLabel: 'Temperature (K)',
      xRange: [0, Math.max(0.05, rep.series.time_ps[rep.series.time_ps.length - 1] || 1)],
      yRange: [0, Math.max(450, m.T_target_K * 1.65)],
      xTicks: [0, Number(((rep.series.time_ps[rep.series.time_ps.length - 1] || 1) * 0.5).toFixed(2)), Number((rep.series.time_ps[rep.series.time_ps.length - 1] || 1).toFixed(2))],
      yTicks: [0, Math.round(m.T_target_K * 0.5), Math.round(m.T_target_K), Math.round(m.T_target_K * 1.5)],
      padR: 72,
      hline: { y: m.T_target_K, color: '#91b6df', dash: '5 4' },
      series: [{ name: 'T(t)', shortName: 'T', color: '#b5d88e', highlight: true, points: pts, endLabel: `${m.T_mean_K} K` }],
    });
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
    document.getElementById('bind-kpis').innerHTML = `
      <div class="kpi-cell"><small>CompScore</small><strong>${Number(m.comp_score ?? m.pose_drift_rmsd_A ?? 0).toFixed(3)}</strong></div>
      <div class="kpi-cell"><small>Contact Survival</small><strong>${Math.round(100 * Number(m.contact_survival ?? 0))}%</strong></div>
      <div class="kpi-cell"><small>τ-RAMD Unbind</small><strong>${Number(m.tau_ramd_ps ?? 0).toFixed(2)} ps</strong></div>
      <div class="kpi-cell"><small>vdW Clashes</small><strong>${m.vdw_clashes ?? 0}</strong></div>`;
    document.getElementById('bind-receipt').textContent = payload.receipt || JSON.stringify(rep, null, 2);
  }

  function renderElectrolyteTriage(payload) {
    const rep = payload.report;
    const pill = document.getElementById('elec-verdict-pill');
    pill.textContent = rep.verdict;
    pill.className = `pill ${rep.verdict}`;
    const m = rep.metrics || {};
    document.getElementById('elec-kpis').innerHTML = `
      <div class="kpi-cell"><small>σ (Cluster-NE)</small><strong>${Number(m.sigma_cne_mS_cm ?? m.conductivity_mS_cm ?? 0).toFixed(2)} mS/cm</strong></div>
      <div class="kpi-cell"><small>Free Ion SSIP</small><strong>${Math.round(100 * Number(m.ssip_frac ?? 0))}%</strong></div>
      <div class="kpi-cell"><small>Ion Pair CIP</small><strong>${Math.round(100 * Number(m.cip_frac ?? 0))}%</strong></div>
      <div class="kpi-cell"><small>Aggregate AGG</small><strong>${Math.round(100 * Number(m.agg_frac ?? 0))}%</strong></div>`;
    document.getElementById('elec-receipt').textContent = payload.receipt || JSON.stringify(rep, null, 2);
  }

  async function initStudio() {
    const defaults = DATA.studio_defaults;
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
    }

    const compSnaps = defaults.compliance_snapshots || {};
    const rollSnaps = defaults.rollout_snapshots || { aspirin: defaults.rollout_aspirin };

    function pickStaticCompliance(targetId, faultId, precision) {
      return (
        compSnaps[`${targetId}__${faultId}__${precision}`] ||
        compSnaps[`${targetId}__${faultId}`] ||
        compSnaps[`aspirin_hetero_o3__${faultId}`] ||
        compSnaps[`${targetId}__none`] ||
        compSnaps['aspirin_hetero_o3__none__float64']
      );
    }

    // Initial instant render from real precomputed snapshots
    renderComplianceReport(pickStaticCompliance('aspirin_hetero_o3', 'none', 'float64'));
    renderRolloutReport(rollSnaps.aspirin || defaults.rollout_aspirin);
    renderBindingTriage(defaults.binding_snapshots.aspirin_cox_stable);
    renderElectrolyteTriage(defaults.electrolyte_snapshots.li_ec_fec_additive);

    // Tab switching
    document.querySelectorAll('[data-studio-tab]').forEach(btn => {
      btn.addEventListener('click', () => {
        const key = btn.dataset.studioTab;
        document.querySelectorAll('[data-studio-tab]').forEach(b => b.classList.toggle('active', b === btn));
        document.querySelectorAll('.studio-pane').forEach(p => p.classList.toggle('active', p.id === `pane-${key}`));
        requestAnimationFrame(() => {
          if (key === 'compliance' && compViewer) compViewer.render();
          if (key === 'rollout' && rollViewer) rollViewer.render();
        });
      });
    });

    // Live / Snapshot button handlers
    document.getElementById('btn-run-compliance').addEventListener('click', async e => {
      const btn = e.currentTarget;
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
      } finally {
        btn.textContent = 'Run Audit';
      }
    });

    document.getElementById('roll-preset').addEventListener('change', e => {
      document.getElementById('roll-custom-wrap').style.display = e.target.value === 'custom' ? 'block' : 'none';
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
        } else {
          renderRolloutReport(rollSnaps[preset === 'custom' ? 'ethanol' : preset] || rollSnaps.aspirin);
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
      } finally {
        btn.textContent = 'Run Speciation';
      }
    });
  }

  async function boot() {
    // BibTeX copy
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
    window.addEventListener('resize', () => requestAnimationFrame(renderEvidenceChart), { passive: true });

    await initStudio();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot, { once: true });
  else boot();
})();
