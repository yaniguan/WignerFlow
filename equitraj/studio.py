"""WignerFlow Studio: Minimalist web interface and HTTP API for equivariant MD evaluation."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import platform
import time

import torch

from equitraj import pocket
from equitraj.compliance import AUDIT_TARGETS, FAULT_CATALOG, run_compliance_audit
from equitraj.studio_data import BENCHMARK_SUITES, DEFAULT_ETHANOL_XYZ, run_studio_rollout

BINDING_PRESETS = ["aspirin_cox_stable", "aspirin_induced_fit", "aspirin_decoy_clash"]
ELECTROLYTE_PRESETS = ["li_ec_fec_additive", "li_ec_base", "li_ec_vc_additive", "li_high_salt_agg"]
ROLLOUT_PRESETS = ["aspirin", "ala2", "ethanol", "li_ec_pf6"]


def build_overview_payload() -> dict:
    return {
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu_threads": int(torch.get_num_threads()),
            "cuda": bool(torch.cuda.is_available()),
        },
        "compliance_targets": list(AUDIT_TARGETS.values()),
        "fault_catalog": FAULT_CATALOG,
        "binding_presets": BINDING_PRESETS,
        "electrolyte_presets": ELECTROLYTE_PRESETS,
        "rollout_presets": ROLLOUT_PRESETS,
        "default_xyz": DEFAULT_ETHANOL_XYZ,
        "datasets": pocket.DATASET_CATALOG,
        "benchmarks": BENCHMARK_SUITES,
    }


STUDIO_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WignerFlow Studio</title>
<style>
  :root {
    --bg: #FAF9F6;
    --surface: #FFFFFF;
    --well: #F3F2EE;
    --well-hover: #EAE8E2;
    --border: rgba(20, 20, 19, 0.08);
    --border-strong: rgba(20, 20, 19, 0.16);
    --ink: #141413;
    --ink-2: #5E5D59;
    --ink-3: #8C8B85;
    --pass: #15803D;
    --pass-bg: #EEFBF3;
    --hold: #B45309;
    --hold-bg: #FEF7EC;
    --fail: #B91C1C;
    --fail-bg: #FEF2F2;
    --radius: 16px;
    --radius-sm: 10px;
    --shadow: 0 1px 2px rgba(20, 20, 19, 0.03), 0 4px 16px rgba(20, 20, 19, 0.02);
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--ink); }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    font-size: 14px;
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
    font-variant-numeric: tabular-nums;
  }
  code, .mono {
    font-family: "SF Mono", "JetBrains Mono", Menlo, Consolas, ui-monospace, monospace;
    font-size: 12.5px;
  }

  /* Header */
  .nav {
    position: sticky;
    top: 0;
    z-index: 40;
    background: rgba(250, 249, 246, 0.86);
    backdrop-filter: blur(14px);
    border-bottom: 1px solid var(--border);
  }
  .nav-inner {
    max-width: 1160px;
    margin: 0 auto;
    height: 64px;
    padding: 0 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 600;
    font-size: 15px;
    letter-spacing: -0.015em;
  }
  .brand-mark {
    width: 26px;
    height: 26px;
    border-radius: 7px;
    background: var(--ink);
    color: #fff;
    display: grid;
    place-items: center;
  }
  .brand-sub { color: var(--ink-3); font-weight: 400; margin-left: 2px; }

  .seg-nav {
    display: inline-flex;
    background: var(--well);
    padding: 3px;
    border-radius: 999px;
    gap: 2px;
  }
  .seg-btn {
    border: 0;
    background: transparent;
    color: var(--ink-2);
    font-family: inherit;
    font-size: 13px;
    font-weight: 500;
    padding: 6px 15px;
    border-radius: 999px;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .seg-btn:hover { color: var(--ink); }
  .seg-btn.active {
    background: var(--surface);
    color: var(--ink);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
  }

  .nav-right {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .engine-status {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-size: 12.5px;
    color: var(--ink-2);
  }
  .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--pass);
    display: inline-block;
    flex-shrink: 0;
  }
  .dot.HOLD, .dot.WARN { background: var(--hold); }
  .dot.REJECT, .dot.FAIL { background: var(--fail); }

  /* Layout */
  .page {
    max-width: 1160px;
    margin: 0 auto;
    padding: 36px 32px 80px;
  }
  .view { display: none; }
  .view.active { display: block; animation: fadeIn 0.18s ease; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(3px); } to { opacity: 1; transform: translateY(0); } }

  .hero {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 24px;
    margin-bottom: 24px;
    flex-wrap: wrap;
  }
  .eyebrow {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ink-3);
    margin-bottom: 6px;
  }
  .hero h1 {
    margin: 0;
    font-size: 28px;
    font-weight: 600;
    letter-spacing: -0.028em;
    line-height: 1.18;
  }
  .hero p {
    margin: 6px 0 0;
    color: var(--ink-2);
    font-size: 14.5px;
    max-width: 600px;
  }

  /* Controls & Buttons */
  .toolbar {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 15px 18px;
    box-shadow: var(--shadow);
    display: flex;
    align-items: flex-end;
    gap: 14px;
    flex-wrap: wrap;
    margin-bottom: 24px;
  }
  .field { display: flex; flex-direction: column; gap: 6px; flex: 1; min-width: 150px; }
  .field.compact { flex: 0 0 auto; min-width: 108px; }
  .field label {
    font-size: 12px;
    font-weight: 500;
    color: var(--ink-2);
  }
  input[type="checkbox"] {
    accent-color: var(--ink);
    width: 14px;
    height: 14px;
    cursor: pointer;
  }
  select, input[type="number"], textarea {
    width: 100%;
    height: 36px;
    background-color: var(--bg);
    color: var(--ink);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-sm);
    padding: 0 11px;
    font-family: inherit;
    font-size: 13px;
    outline: none;
    transition: border-color 0.15s, background-color 0.15s;
  }
  select {
    appearance: none;
    -webkit-appearance: none;
    padding-right: 32px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12' fill='none'%3E%3Cpath d='M2.5 4.5L6 8L9.5 4.5' stroke='%235E5D59' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 11px center;
    cursor: pointer;
  }
  select:focus, input[type="number"]:focus, textarea:focus {
    border-color: var(--ink);
    background-color: var(--surface);
  }
  textarea {
    height: auto;
    padding: 10px 12px;
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 12px;
    line-height: 1.5;
  }

  .btn {
    height: 36px;
    padding: 0 18px;
    border-radius: 999px;
    border: 1px solid var(--ink);
    background: var(--ink);
    color: #fff;
    font-family: inherit;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 7px;
    white-space: nowrap;
    transition: opacity 0.15s, background 0.15s;
  }
  .btn:hover { opacity: 0.86; }
  .btn.ghost {
    background: var(--surface);
    color: var(--ink);
    border-color: var(--border-strong);
  }
  .btn.ghost:hover { background: var(--well); opacity: 1; }
  .btn.sm { height: 30px; padding: 0 12px; font-size: 12px; }

  /* Cards & Grids */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 22px 24px;
    box-shadow: var(--shadow);
  }
  .card-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 16px;
  }
  .card-title {
    font-size: 14.5px;
    font-weight: 600;
    letter-spacing: -0.01em;
    margin: 0;
  }
  .card-meta { font-size: 12.5px; color: var(--ink-3); }

  .split-physics {
    display: grid;
    grid-template-columns: 1.12fr 1fr;
    gap: 24px;
    align-items: start;
  }
  .split-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    align-items: start;
  }
  @media (max-width: 960px) {
    .split-physics, .split-2 { grid-template-columns: 1fr; }
    .nav-inner { height: auto; padding: 12px 20px; flex-wrap: wrap; }
    .page { padding: 24px 20px 60px; }
  }

  /* 3D Stage */
  .stage-wrap {
    position: relative;
    background: var(--well);
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border);
  }
  canvas.stage {
    width: 100%;
    height: 360px;
    display: block;
    cursor: grab;
  }
  canvas.stage:active { cursor: grabbing; }
  .stage-overlay {
    position: absolute;
    left: 14px;
    right: 14px;
    bottom: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    pointer-events: none;
  }
  .stage-pill {
    pointer-events: auto;
    background: rgba(255, 255, 255, 0.88);
    backdrop-filter: blur(8px);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 5px 12px;
    font-size: 12px;
    color: var(--ink-2);
    display: inline-flex;
    align-items: center;
    gap: 10px;
  }

  /* Status Badge */
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 11px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 500;
    background: var(--pass-bg);
    color: var(--pass);
  }
  .badge.HOLD, .badge.WARN { background: var(--hold-bg); color: var(--hold); }
  .badge.REJECT, .badge.FAIL { background: var(--fail-bg); color: var(--fail); }

  /* 10-Gate Minimalist Audit List */
  .gate-list {
    border-top: 1px solid var(--border);
  }
  .gate-item {
    border-bottom: 1px solid var(--border);
    padding: 11px 4px;
    cursor: pointer;
    transition: background 0.12s;
  }
  .gate-item:hover { background: rgba(20, 20, 19, 0.018); }
  .gate-line {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }
  .gate-left {
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
  }
  .gate-code {
    color: var(--ink-3);
    font-size: 12px;
    width: 26px;
    flex-shrink: 0;
  }
  .gate-name {
    font-weight: 500;
    font-size: 13.5px;
    color: var(--ink);
  }
  .gate-right {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-shrink: 0;
  }
  .gate-val { color: var(--ink-2); }
  .gate-state {
    font-size: 12px;
    font-weight: 500;
    width: 38px;
    text-align: right;
    color: var(--pass);
  }
  .gate-state.FAIL { color: var(--fail); }
  .gate-state.WARN { color: var(--hold); }
  .gate-drawer {
    display: none;
    padding: 8px 0 2px 36px;
    font-size: 12.5px;
    color: var(--ink-2);
    line-height: 1.5;
  }
  .gate-item.open .gate-drawer { display: block; }

  /* Metrics Strip */
  .stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
  }
  @media (max-width: 760px) { .stats-row { grid-template-columns: repeat(2, 1fr); } }
  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
    box-shadow: var(--shadow);
  }
  .stat-label { font-size: 12px; color: var(--ink-3); margin-bottom: 6px; }
  .stat-value { font-size: 22px; font-weight: 600; letter-spacing: -0.02em; }
  .stat-note { font-size: 12px; color: var(--ink-2); margin-top: 4px; }

  /* Minimalist Charts */
  canvas.plot {
    width: 100%;
    height: 180px;
    display: block;
  }

  /* Clean Editorial Tables */
  .table-wrap { overflow-x: auto; }
  table.clean {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  table.clean th {
    text-align: left;
    font-weight: 500;
    font-size: 12px;
    color: var(--ink-3);
    padding: 10px 12px;
    border-bottom: 1px solid var(--border-strong);
    white-space: nowrap;
  }
  table.clean td {
    padding: 12px;
    border-bottom: 1px solid var(--border);
    color: var(--ink);
    white-space: nowrap;
  }
  table.clean tr:last-child td { border-bottom: 0; }
  table.clean .num { text-align: right; font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 12.5px; }
  table.clean th.num { font-family: inherit; font-size: 12px; }
  .muted { color: var(--ink-2); }
  .subtle { color: var(--ink-3); }

  /* Segmented Bar for Solvation */
  .spec-bar {
    display: flex;
    height: 8px;
    border-radius: 999px;
    overflow: hidden;
    background: var(--well);
    gap: 2px;
  }
</style>
</head>
<body>

<header class="nav">
  <div class="nav-inner">
    <div class="brand">
      <div class="brand-mark">
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M2.5 11.5L5.5 4.5L8 9.5L10.5 4.5L13.5 11.5" stroke="white" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </div>
      <span>WignerFlow <span class="brand-sub">Studio</span></span>
    </div>

    <nav class="seg-nav">
      <button class="seg-btn active" onclick="openView('physics', this)">Physics</button>
      <button class="seg-btn" onclick="openView('simulate', this)">Simulate</button>
      <button class="seg-btn" onclick="openView('binding', this)">Drug Triage</button>
      <button class="seg-btn" onclick="openView('electrolyte', this)">Electrolytes</button>
      <button class="seg-btn" onclick="openView('benchmarks', this)">Benchmarks</button>
    </nav>

    <div class="nav-right">
      <span class="engine-status" id="runtimeBadge"><span class="dot"></span>Connecting...</span>
      <button class="btn ghost sm" onclick="exportSessionJson()">Export JSON</button>
    </div>
  </div>
</header>

<main class="page">

  <!-- ======================= 1. PHYSICS ======================= -->
  <section id="view-physics" class="view active">
    <div class="hero">
      <div>
        <div class="eyebrow">Equivariance &amp; Conservation</div>
        <h1>Physical Compliance</h1>
        <p>Ten live PyTorch tests verifying 3D rotation, mirror parity, momentum conservation, steric clearance, and rollout stability.</p>
      </div>
      <div id="physicsHeroBadge" class="badge"><span class="dot"></span>10 / 10 Passed</div>
    </div>

    <div class="toolbar">
      <div class="field" style="min-width:230px;">
        <label>Model</label>
        <select id="compTarget" onchange="runCompliance()">
          <option value="aspirin_hetero_o3" selected>Aspirin O(3) Flow (C, H, O + charges + bonds)</option>
          <option value="argon_flow_v1">Liquid Argon Flow v1 (256 atoms, trained)</option>
          <option value="aspirin_chiral_so3">Aspirin SO(3)-Only Control (breaks mirror parity)</option>
          <option value="argon_flow_v0">Liquid Argon Flow v0 (early baseline)</option>
        </select>
      </div>

      <div class="field compact">
        <label>Precision</label>
        <select id="compPrec" onchange="runCompliance()">
          <option value="float64" selected>FP64</option>
          <option value="float32">FP32</option>
        </select>
      </div>

      <div class="field" style="min-width:240px;">
        <label>Fault Injection (Falsifiability Test)</label>
        <select id="compFault" onchange="runCompliance()"></select>
      </div>

      <div class="field compact">
        <label>Rotation Test</label>
        <button class="btn ghost" onclick="randomizeEuler()" id="eulerBtn" title="Test a new random 3D rotation">37°, 64°, 118°</button>
      </div>

      <button class="btn" id="btnRunComp" onclick="runCompliance()">Run Check</button>
    </div>

    <div class="split-physics">
      <div style="display:flex;flex-direction:column;gap:20px;">
        <div class="card" style="padding:18px;">
          <div class="card-head" style="margin-bottom:12px;">
            <div>
              <h3 class="card-title">Equivariance Vector Overlay</h3>
              <div class="card-meta">Compares direct prediction f(X) against rotated twin R⁻¹f(RX)</div>
            </div>
            <label style="font-size:12.5px;color:var(--ink-2);display:inline-flex;align-items:center;gap:6px;cursor:pointer;">
              <input type="checkbox" id="chkTwinVec" checked onchange="drawComp3D()"> Twin arrows
            </label>
          </div>

          <div class="stage-wrap">
            <canvas id="canvasComp3D" class="stage" data-w="620" data-h="360"></canvas>
            <div class="stage-overlay">
              <div class="stage-pill">
                <span><b style="color:#141413;">—</b> Direct f(X)</span>
                <span><b style="color:#D97706;">- -</b> Rotated R⁻¹f(RX)</span>
              </div>
              <button class="stage-pill" style="cursor:pointer;" onclick="resetCam('comp', drawComp3D)">Reset view</button>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-head">
            <h3 class="card-title">Algebraic Precision (-log₁₀ error)</h3>
            <span class="card-meta">Digits of exact symmetry</span>
          </div>
          <canvas id="canvasDigits" class="plot" data-w="560" data-h="154" style="height:154px;"></canvas>
        </div>
      </div>

      <div class="card">
        <div class="card-head">
          <h3 class="card-title">Ten-Gate Audit</h3>
          <div style="display:flex;align-items:center;gap:12px;">
            <span class="card-meta" id="compTiming">—</span>
            <button class="btn ghost sm" onclick="toggleAllGates()" id="btnExpandAll">Details</button>
          </div>
        </div>
        <div class="gate-list" id="gatesContainer"></div>
      </div>
    </div>
  </section>

  <!-- ======================= 2. SIMULATE ======================= -->
  <section id="view-simulate" class="view">
    <div class="hero">
      <div>
        <div class="eyebrow">Interactive Trajectory</div>
        <h1>Molecular Rollout</h1>
        <p>Generate and inspect 3D Langevin trajectories on CPU from built-in systems or custom XYZ coordinates.</p>
      </div>
      <button class="btn ghost" onclick="downloadRolloutXyz()">Download .xyz</button>
    </div>

    <div class="toolbar">
      <div class="field" style="min-width:210px;">
        <label>System</label>
        <select id="rollPreset" onchange="onRollPresetChange()">
          <option value="aspirin">Aspirin (21 atoms)</option>
          <option value="ala2">Alanine Dipeptide (22 atoms)</option>
          <option value="ethanol">Ethanol (9 atoms)</option>
          <option value="li_ec_pf6">LiPF6 in EC (periodic cell)</option>
          <option value="custom">Custom XYZ coordinates...</option>
        </select>
      </div>
      <div class="field compact">
        <label>Frames</label>
        <input type="number" id="rollFrames" value="60" min="15" max="200">
      </div>
      <div class="field compact">
        <label>Step (fs)</label>
        <input type="number" id="rollDt" value="1.5" step="0.25" min="0.25" max="4.0">
      </div>
      <div class="field compact">
        <label>Temperature (K)</label>
        <input type="number" id="rollTemp" value="300" min="50" max="800">
      </div>
      <button class="btn" onclick="runRolloutLab()">Simulate</button>
    </div>

    <div id="xyzDrawer" class="card" style="display:none;margin-bottom:24px;">
      <div class="card-head">
        <h3 class="card-title">Custom XYZ Structure</h3>
        <span class="card-meta">Covalent bonds and angles are inferred automatically</span>
      </div>
      <textarea id="customXyzInput" rows="7"></textarea>
    </div>

    <div class="stats-row" id="rollStats"></div>

    <div class="split-physics">
      <div class="card" style="padding:18px;">
        <div class="card-head" style="margin-bottom:12px;">
          <h3 class="card-title" id="rollTitle">Aspirin</h3>
          <span class="card-meta">Drag to orbit · Scroll to zoom</span>
        </div>
        <div class="stage-wrap">
          <canvas id="canvasRoll3D" class="stage" width="640" height="360"></canvas>
          <div class="stage-overlay">
            <button class="stage-pill" style="cursor:pointer;" id="btnPlayRoll" onclick="togglePlay('roll')">Pause</button>
            <button class="stage-pill" style="cursor:pointer;" onclick="resetCam('roll', drawRoll3D)">Reset view</button>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-head">
          <h3 class="card-title">Thermal Stability T(t)</h3>
          <span class="card-meta">Instantaneous kinetic temperature (K)</span>
        </div>
        <canvas id="canvasRollChart" class="plot" width="480" height="330" style="height:330px;"></canvas>
      </div>
    </div>
  </section>

  <!-- ======================= 3. DRUG TRIAGE ======================= -->
  <section id="view-binding" class="view">
    <div class="hero">
      <div>
        <div class="eyebrow">Structure-Based Screening</div>
        <h1>Pocket Stability Triage</h1>
        <p>Filter docked ligand poses with a short CPU rollout before running expensive free-energy simulations.</p>
      </div>
      <div id="bindBadge" class="badge"><span class="dot"></span>Ready</div>
    </div>

    <div class="toolbar">
      <div class="field" style="min-width:280px;">
        <label>Candidate Pose</label>
        <select id="bindPreset" onchange="runBindingLab()">
          <option value="aspirin_cox_stable">COX-1 + Aspirin — Native crystal pose</option>
          <option value="aspirin_induced_fit">COX-1 + Aspirin — Tilted induced-fit pose</option>
          <option value="aspirin_decoy_clash">COX-1 + Aspirin — Flipped clash decoy</option>
        </select>
      </div>
      <div class="field compact">
        <label>Replicas</label>
        <input type="number" id="bindRep" value="3" min="1" max="6">
      </div>
      <div class="field compact">
        <label>Frames</label>
        <input type="number" id="bindFrames" value="70" min="30" max="140">
      </div>
      <button class="btn" onclick="runBindingLab()">Evaluate Pose</button>
    </div>

    <div class="stats-row" id="bindStats"></div>

    <div class="split-physics">
      <div class="card" style="padding:18px;">
        <div class="card-head" style="margin-bottom:12px;">
          <h3 class="card-title">Binding Pocket &amp; Ligand</h3>
          <span class="card-meta" id="bindSummaryText">—</span>
        </div>
        <div class="stage-wrap">
          <canvas id="canvasBind3D" class="stage" width="640" height="340"></canvas>
          <div class="stage-overlay">
            <div class="stage-pill">
              <span><b style="color:#15803D;">●</b> Ligand</span>
              <span><b style="color:#94A3B8;">●</b> Pocket</span>
            </div>
            <button class="stage-pill" style="cursor:pointer;" onclick="resetCam('bind', drawBind3D)">Reset view</button>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-head">
          <h3 class="card-title">Ligand Drift Across Replicas</h3>
          <span class="card-meta">RMSD (Å) over rollout</span>
        </div>
        <canvas id="canvasBindChart" class="plot" width="480" height="310" style="height:310px;"></canvas>
      </div>
    </div>
  </section>

  <!-- ======================= 4. ELECTROLYTES ======================= -->
  <section id="view-electrolyte" class="view">
    <div class="hero">
      <div>
        <div class="eyebrow">Battery Transport &amp; Solvation</div>
        <h1>Electrolyte Screening &amp; CALiSol-23</h1>
        <p>Separate current-carrying free ions from neutral ion pairs, and compare WignerFlow trajectories against experimental lab conductivity.</p>
      </div>
      <div id="elecBadge" class="badge"><span class="dot"></span>Ready</div>
    </div>

    <div class="toolbar">
      <div class="field" style="min-width:290px;">
        <label>Formulation</label>
        <select id="elecPreset" onchange="runElectrolyteLab()">
          <option value="li_ec_fec_additive">1.0 M LiPF6 in EC/DMC + 10% FEC</option>
          <option value="li_ec_base">1.0 M LiPF6 in EC baseline</option>
          <option value="li_ec_vc_additive">1.0 M LiPF6 in EC + high VC</option>
          <option value="li_high_salt_agg">High-pairing neutral cluster regime</option>
        </select>
      </div>
      <div class="field compact">
        <label>Frames</label>
        <input type="number" id="elecFrames" value="90" min="40" max="180">
      </div>
      <button class="btn" onclick="runElectrolyteLab()">Simulate Cell</button>
    </div>

    <div class="stats-row" id="elecStats"></div>

    <div class="split-2" style="margin-bottom:24px;">
      <div class="card" style="padding:18px;">
        <div class="card-head" style="margin-bottom:12px;">
          <h3 class="card-title">Periodic Solvation Cell</h3>
          <span class="card-meta">Li+ (violet) · PF6- (ochre) · Solvent (stone)</span>
        </div>
        <div class="stage-wrap">
          <canvas id="canvasElec3D" class="stage" width="520" height="290" style="height:290px;"></canvas>
        </div>
      </div>

      <div class="card">
        <div class="card-head">
          <div>
            <h3 class="card-title">Concentration Curve: LiPF6 in EC:DMC (298 K)</h3>
            <div class="card-meta" id="calisolCurveMeta">Saved WignerFlow-Align evaluation from results/calisol/</div>
          </div>
        </div>
        <canvas id="canvasParity" class="plot" data-w="500" data-h="245" style="height:245px;"></canvas>
        <div id="parityHover" class="card-meta" style="margin-top:10px;">Hover a point to inspect concentration vs. aligned conductivity.</div>
      </div>
    </div>

    <div class="card">
      <div class="card-head">
        <div>
          <h3 class="card-title">CALiSol-23 Post-Training Alignment Summary</h3>
          <div class="card-meta">Unseen salt–solvent combinations loaded from results/calisol/align_eval_seed42.json</div>
        </div>
      </div>
      <div class="table-wrap">
        <table class="clean" id="calisolParityTable"></table>
      </div>
    </div>
  </section>

  <!-- ======================= 5. BENCHMARKS ======================= -->
  <section id="view-benchmarks" class="view">
    <div class="hero">
      <div>
        <div class="eyebrow">Saved Evaluation Results</div>
        <h1>Benchmark Summary</h1>
        <p>All numbers below are loaded at start-up from committed result files in <code>results/</code>.</p>
      </div>
    </div>

    <div class="stats-row" id="benchStats"></div>

    <div class="card" style="margin-bottom:24px;">
      <div class="card-head">
        <h3 class="card-title">Large-Step Pair-Model Baseline (MD17 &amp; MD22 at 500 K)</h3>
        <span class="card-meta">Pair-model baseline · results/pure_md_md17/md17_md22_benchmark.json</span>
      </div>
      <div class="table-wrap">
        <table class="clean" id="md17Table"></table>
      </div>
    </div>

    <div class="card" style="margin-bottom:24px;">
      <div class="card-head">
        <h3 class="card-title">Measured CPU Periodic-Box Scaling</h3>
        <span class="card-meta">50 fs step timing on CPU · results/calisol/box_scaling_cpu.json</span>
      </div>
      <div class="table-wrap">
        <table class="clean" id="boxScalingTable"></table>
      </div>
    </div>

    <div class="card">
      <div class="card-head">
        <h3 class="card-title">Top 15 Candidates from 1,000,000-Recipe Screen</h3>
        <span class="card-meta">Toy-model screening output (not lab-tested) · Free SSIP · Paired CIP · Aggregate AGG</span>
      </div>
      <div class="table-wrap">
        <table class="clean" id="solvationTable"></table>
      </div>
    </div>
  </section>

</main>

<script>
const ATOM_STYLE = {
  H:  { fill: '#FFFFFF', rim: '#CBD5E1', r: 0.34 },
  C:  { fill: '#27272A', rim: '#18181B', r: 0.56 },
  N:  { fill: '#2563EB', rim: '#1D4ED8', r: 0.54 },
  O:  { fill: '#DC2626', rim: '#B91C1C', r: 0.53 },
  F:  { fill: '#0D9488', rim: '#0F766E', r: 0.50 },
  Li: { fill: '#7C3AED', rim: '#6D28D9', r: 0.62 },
  P:  { fill: '#D97706', rim: '#B45309', r: 0.66 },
  S:  { fill: '#CA8A04', rim: '#A16207', r: 0.62 },
  Ar: { fill: '#0284C7', rim: '#0369A1', r: 0.70 }
};

const STATE = {
  overview: null,
  euler: [37, 64, 118],
  openGate: null,
  expandAll: false,
  compData: null,
  rollData: null,
  bindData: null,
  elecData: null,
  parityPts: [],
  frameIdx: { comp: 0, roll: 0, bind: 0, elec: 0 },
  playing: { roll: true, bind: true, elec: true },
  cam: {
    comp: { rotX: 0.38, rotY: -0.52, zoom: 1.0 },
    roll: { rotX: 0.34, rotY: 0.58, zoom: 1.0 },
    bind: { rotX: 0.30, rotY: -0.44, zoom: 1.0 },
    elec: { rotX: 0.36, rotY: 0.50, zoom: 1.0 }
  }
};

function prepCanvas(cv) {
  const dpr = Math.min(2.5, window.devicePixelRatio || 1);
  const w = cv.clientWidth || Number(cv.dataset.w) || Number(cv.getAttribute('width')) || 560;
  const h = cv.clientHeight || Number(cv.dataset.h) || Number(cv.getAttribute('height')) || 320;
  const pw = Math.round(w * dpr), ph = Math.round(h * dpr);
  if (cv.width !== pw || cv.height !== ph) {
    cv.width = pw;
    cv.height = ph;
  }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function drawArrow(ctx, x0, y0, x1, y1, color, dashed) {
  const dx = x1 - x0, dy = y1 - y0;
  const len = Math.hypot(dx, dy);
  if (len < 1.5) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = dashed ? 1.8 : 2.1;
  ctx.lineCap = 'round';
  if (dashed) ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  ctx.setLineDash([]);
  const ang = Math.atan2(dy, dx);
  const head = Math.min(6.5, Math.max(4, len * 0.28));
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - head * Math.cos(ang - 0.45), y1 - head * Math.sin(ang - 0.45));
  ctx.lineTo(x1 - head * Math.cos(ang + 0.45), y1 - head * Math.sin(ang + 0.45));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function openView(name, btn) {
  document.querySelectorAll('.view').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.seg-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('view-' + name).classList.add('active');
  if (btn) btn.classList.add('active');
  if (name === 'physics' && STATE.compData) {
    drawComp3D();
    drawDigitsBar(STATE.compData.gates);
  }
  if (name === 'simulate') {
    if (!STATE.rollData) runRolloutLab();
    else {
      drawRoll3D();
      drawMinimalSeries('canvasRollChart', STATE.rollData.series.time_ps, [
        { label: 'Temperature (K)', data: STATE.rollData.series.temperature_K, color: '#141413' }
      ]);
    }
  }
  if (name === 'binding') {
    if (!STATE.bindData) runBindingLab();
    else {
      drawBind3D();
      drawMinimalSeries('canvasBindChart', STATE.bindData.series.time_ps, [
        { label: 'Ligand RMSD (Å)', data: STATE.bindData.series.ligand_rmsd_A, color: '#141413' },
        { label: 'COM Shift (Å)', data: STATE.bindData.series.com_displacement_A, color: '#8C8B85', dashed: true }
      ]);
    }
  }
  if (name === 'electrolyte') {
    if (!STATE.elecData) runElectrolyteLab();
    else drawElec3D();
    drawParityPlot();
  }
}

function resetCam(key, fn) {
  STATE.cam[key] = { rotX: 0.36, rotY: -0.48, zoom: 1.0 };
  fn();
}

function attachOrbit(canvasId, key, redraw) {
  const cv = document.getElementById(canvasId);
  let drag = false, lx = 0, ly = 0;
  cv.addEventListener('mousedown', e => { drag = true; lx = e.clientX; ly = e.clientY; });
  window.addEventListener('mouseup', () => { drag = false; });
  window.addEventListener('mousemove', e => {
    if (!drag) return;
    STATE.cam[key].rotY += (e.clientX - lx) * 0.009;
    STATE.cam[key].rotX += (e.clientY - ly) * 0.009;
    lx = e.clientX; ly = e.clientY;
    redraw();
  });
  cv.addEventListener('wheel', e => {
    e.preventDefault();
    STATE.cam[key].zoom = Math.max(0.5, Math.min(2.6, STATE.cam[key].zoom * (e.deltaY < 0 ? 1.07 : 0.93)));
    redraw();
  }, { passive: false });
}

function project(pt, center, scale, cam, w, h) {
  const x = pt[0] - center[0], y = pt[1] - center[1], z = pt[2] - center[2];
  const cy = Math.cos(cam.rotY), sy = Math.sin(cam.rotY);
  const x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
  const cx = Math.cos(cam.rotX), sx = Math.sin(cam.rotX);
  const y2 = y * cx - z1 * sx, z2 = y * sx + z1 * cx;
  const p = 440 / (440 + z2 * 8 + 1e-5);
  const s = scale * cam.zoom * p;
  return { sx: w * 0.5 + x1 * s, sy: h * 0.5 - y2 * s, z: z2, s };
}

function drawScene(canvasId, coords, elements, opts = {}) {
  const cv = document.getElementById(canvasId);
  if (!cv || !coords || !coords.length) return;
  const { ctx, w, h } = prepCanvas(cv);

  let cx = 0, cy = 0, cz = 0;
  coords.forEach(p => { cx += p[0]; cy += p[1]; cz += p[2]; });
  const center = opts.boxA ? [opts.boxA * 0.5, opts.boxA * 0.5, opts.boxA * 0.5] : [cx / coords.length, cy / coords.length, cz / coords.length];
  let maxR = opts.boxA ? opts.boxA * 0.66 : 2.4;
  coords.forEach(p => {
    const d = Math.hypot(p[0] - center[0], p[1] - center[1], p[2] - center[2]);
    if (d > maxR) maxR = d;
  });
  const scale = (Math.min(w, h) * 0.37) / maxR;
  const cam = STATE.cam[opts.camKey || 'comp'];

  if (opts.boxA) {
    const L = opts.boxA;
    const cs = [[0,0,0],[L,0,0],[L,L,0],[0,L,0],[0,0,L],[L,0,L],[L,L,L],[0,L,L]].map(c => project(c, center, scale, cam, w, h));
    ctx.strokeStyle = 'rgba(20, 20, 19, 0.14)';
    ctx.lineWidth = 1;
    [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]].forEach(([i, j]) => {
      ctx.beginPath(); ctx.moveTo(cs[i].sx, cs[i].sy); ctx.lineTo(cs[j].sx, cs[j].sy); ctx.stroke();
    });
  }

  const pts = coords.map((p, idx) => ({ ...project(p, center, scale, cam, w, h), idx }));

  if (opts.bonds && opts.bonds.length) {
    ctx.strokeStyle = '#A8A29E';
    ctx.lineWidth = 2.6;
    ctx.lineCap = 'round';
    opts.bonds.forEach(([i, j]) => {
      if (i < pts.length && j < pts.length) {
        const d3 = Math.hypot(coords[i][0]-coords[j][0], coords[i][1]-coords[j][1], coords[i][2]-coords[j][2]);
        if (!opts.boxA || d3 < opts.boxA * 0.45) {
          ctx.beginPath(); ctx.moveTo(pts[i].sx, pts[i].sy); ctx.lineTo(pts[j].sx, pts[j].sy); ctx.stroke();
        }
      }
    });
  }

  [...pts].sort((a, b) => b.z - a.z).forEach(item => {
    const i = item.idx;
    const el = (elements && elements[i]) || 'C';
    let st = ATOM_STYLE[el] || ATOM_STYLE.C;
    if (opts.roles && opts.roles[i]) {
      const rl = opts.roles[i];
      if (rl === 'ligand') st = { fill: '#15803D', rim: '#166534', r: 0.56 };
      else if (rl === 'pocket') st = { fill: '#CBD5E1', rim: '#94A3B8', r: 0.46 };
      else if (rl === 'cation') st = { fill: '#7C3AED', rim: '#5B21B6', r: 0.64 };
      else if (rl === 'anion') st = { fill: '#D97706', rim: '#92400E', r: 0.62 };
      else if (rl === 'additive') st = { fill: '#0D9488', rim: '#115E59', r: 0.54 };
      else if (rl === 'solvent') st = { fill: '#D6D3D1', rim: '#A8A29E', r: 0.42 };
    }
    const rad = Math.max(3.6, st.r * item.s * 0.42);
    const g = ctx.createRadialGradient(item.sx - rad * 0.28, item.sy - rad * 0.28, rad * 0.08, item.sx, item.sy, rad);
    g.addColorStop(0, '#FFFFFF');
    g.addColorStop(0.28, st.fill);
    g.addColorStop(1, st.rim);
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(item.sx, item.sy, rad, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'rgba(20,20,19,0.12)';
    ctx.lineWidth = 0.8;
    ctx.stroke();
  });

  if (opts.twinRef && opts.twinRot) {
    const n = Math.min(pts.length, opts.twinRef.length);
    for (let i = 0; i < n; i++) {
      const b = coords[i], vr = opts.twinRef[i], vt = opts.twinRot[i];
      const mag = Math.max(Math.hypot(vr[0], vr[1], vr[2]), Math.hypot(vt[0], vt[1], vt[2]), 1e-9);
      const sc = Math.min(650.0, 1.15 / mag);
      const p0 = pts[i];
      const p1 = project([b[0]+vr[0]*sc, b[1]+vr[1]*sc, b[2]+vr[2]*sc], center, scale, cam, w, h);
      const p2 = project([b[0]+vt[0]*sc, b[1]+vt[1]*sc, b[2]+vt[2]*sc], center, scale, cam, w, h);
      drawArrow(ctx, p0.sx, p0.sy, p1.sx, p1.sy, '#141413', false);
      drawArrow(ctx, p0.sx, p0.sy, p2.sx, p2.sy, '#D97706', true);
    }
  }
}

function drawMinimalSeries(canvasId, xArr, series) {
  const cv = document.getElementById(canvasId);
  if (!cv || !xArr || !xArr.length) return;
  const { ctx, w, h } = prepCanvas(cv);
  const pad = { l: 46, r: 16, t: 26, b: 28 };
  let ymin = Infinity, ymax = -Infinity;
  series.forEach(s => (s.data || []).forEach(v => {
    if (v !== null && isFinite(v)) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); }
  }));
  if (!isFinite(ymin)) { ymin = 0; ymax = 1; }
  if (Math.abs(ymax - ymin) < 1e-6) { ymax += 1; ymin -= 1; }
  const span = ymax - ymin; ymin -= 0.12 * span; ymax += 0.14 * span;

  ctx.strokeStyle = 'rgba(20, 20, 19, 0.07)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const yy = pad.t + (h - pad.t - pad.b) * (i / 3);
    ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(w - pad.r, yy); ctx.stroke();
    const val = ymax - (ymax - ymin) * (i / 3);
    ctx.fillStyle = '#8C8B85';
    ctx.font = '12px ui-monospace, monospace';
    ctx.fillText(val.toFixed(Math.abs(val) < 10 ? 2 : 0), 4, yy + 4);
  }

  series.forEach((s, idx) => {
    const arr = s.data || [];
    if (!arr.length) return;
    ctx.strokeStyle = s.color || '#141413';
    ctx.lineWidth = 2;
    if (s.dashed) ctx.setLineDash([4, 3]); else ctx.setLineDash([]);
    ctx.beginPath();
    arr.forEach((v, i) => {
      const xx = pad.l + (w - pad.l - pad.r) * (i / Math.max(1, arr.length - 1));
      const yy = pad.t + (h - pad.t - pad.b) * (1 - (v - ymin) / (ymax - ymin));
      if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
    });
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = s.color || '#141413';
    ctx.font = '500 12px -apple-system, sans-serif';
    ctx.fillText(s.label, pad.l + idx * 150, 14);
  });
}

function drawDigitsBar(gates) {
  const cv = document.getElementById('canvasDigits');
  const { ctx, w } = prepCanvas(cv);
  const g5 = gates.slice(0, 5);
  const labels = ['G1 Rotation', 'G2 Mirror', 'G3 Momentum', 'G4 Reorder', 'G5 Box Wrap'];
  const rowH = 27, top = 10, barLeft = 114, barMax = Math.max(80, w - barLeft - 58);

  g5.forEach((g, i) => {
    const y = top + i * rowH;
    const err = Math.max(1e-16, Number(g.value) || 1e-16);
    const digits = Math.max(0.3, Math.min(16, -Math.log10(err)));
    ctx.fillStyle = '#5E5D59';
    ctx.font = '12.5px -apple-system, sans-serif';
    ctx.fillText(labels[i], 4, y + 14);

    ctx.fillStyle = '#F3F2EE';
    ctx.beginPath();
    ctx.roundRect(barLeft, y + 4, barMax, 11, 5.5);
    ctx.fill();

    ctx.fillStyle = g.status === 'FAIL' ? '#B91C1C' : '#141413';
    ctx.beginPath();
    ctx.roundRect(barLeft, y + 4, barMax * (digits / 16), 11, 5.5);
    ctx.fill();

    ctx.fillStyle = '#8C8B85';
    ctx.font = '12px ui-monospace, monospace';
    ctx.fillText(digits.toFixed(1) + 'd', barLeft + barMax + 10, y + 14);
  });
}

function randomizeEuler() {
  STATE.euler = [0, 1, 2].map(() => Math.floor(Math.random() * 360));
  document.getElementById('eulerBtn').textContent = STATE.euler.map(v => v + '°').join(', ');
  runCompliance();
}

async function runCompliance() {
  const btn = document.getElementById('btnRunComp');
  btn.textContent = 'Checking...';
  const fault = document.getElementById('compFault').value || 'none';
  const r = await fetch('/api/compliance', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      target_id: document.getElementById('compTarget').value,
      fault,
      precision: document.getElementById('compPrec').value,
      euler_deg: STATE.euler,
      rollout_steps: 14
    })
  });
  const d = await r.json();
  btn.textContent = 'Run Check';
  STATE.compData = d;

  const hb = document.getElementById('physicsHeroBadge');
  hb.className = 'badge ' + d.verdict;
  hb.innerHTML = `<span class="dot ${d.verdict}"></span>${d.pass_count} / ${d.total_gates} Passed`;
  const precLabel = d.precision === 'float64' ? 'FP64' : 'FP32';
  document.getElementById('compTiming').textContent = `${d.wall_s}s · ${precLabel}`;

  const firstFail = d.gates.find(g => g.status === 'FAIL');
  STATE.openGate = firstFail ? firstFail.id : null;
  renderGateRows();
  drawDigitsBar(d.gates);
  drawComp3D();
}

function renderGateRows() {
  const d = STATE.compData;
  if (!d) return;
  document.getElementById('gatesContainer').innerHTML = d.gates.map(g => {
    const isOpen = (STATE.expandAll || STATE.openGate === g.id) ? ' open' : '';
    const stText = g.status === 'PASS' ? 'Pass' : (g.status === 'WARN' ? 'Hold' : 'Fail');
    return `<div class="gate-item${isOpen}" onclick="toggleGate('${g.id}')">
      <div class="gate-line">
        <div class="gate-left">
          <span class="dot ${g.status}"></span>
          <span class="gate-code mono">${g.id}</span>
          <span class="gate-name">${g.name}</span>
        </div>
        <div class="gate-right">
          <span class="gate-val mono">${g.value}</span>
          <span class="gate-state ${g.status}">${stText}</span>
        </div>
      </div>
      <div class="gate-drawer">
        ${g.citation}. <span class="mono subtle">Criterion: ${g.threshold} · ${g.detail}</span>
      </div>
    </div>`;
  }).join('');
}

function toggleGate(id) {
  if (STATE.expandAll) {
    STATE.expandAll = false;
    document.getElementById('btnExpandAll').textContent = 'Details';
  }
  STATE.openGate = STATE.openGate === id ? null : id;
  renderGateRows();
}

function toggleAllGates() {
  STATE.expandAll = !STATE.expandAll;
  document.getElementById('btnExpandAll').textContent = STATE.expandAll ? 'Collapse' : 'Details';
  renderGateRows();
}

function drawComp3D() {
  const d = STATE.compData;
  if (!d) return;
  const showTwin = document.getElementById('chkTwinVec').checked;
  drawScene('canvasComp3D', d.twin_3d.pos_A, d.twin_3d.elements, {
    bonds: d.twin_3d.bonds,
    boxA: d.twin_3d.box_A,
    twinRef: showTwin ? d.twin_3d.dx_ref_A : null,
    twinRot: showTwin ? d.twin_3d.dx_rot_unrot_A : null,
    camKey: 'comp'
  });
}

function onRollPresetChange() {
  const v = document.getElementById('rollPreset').value;
  document.getElementById('xyzDrawer').style.display = v === 'custom' ? 'block' : 'none';
  if (v !== 'custom') runRolloutLab();
}

async function runRolloutLab() {
  const presetVal = document.getElementById('rollPreset').value;
  const isCustom = presetVal === 'custom';
  const r = await fetch('/api/rollout', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      preset: isCustom ? 'ethanol' : presetVal,
      custom_xyz: isCustom ? document.getElementById('customXyzInput').value : null,
      n_frames: Number(document.getElementById('rollFrames').value),
      dt_fs: Number(document.getElementById('rollDt').value),
      temperature_k: Number(document.getElementById('rollTemp').value)
    })
  });
  const d = await r.json();
  STATE.rollData = d;
  STATE.frameIdx.roll = 0;
  document.getElementById('rollTitle').textContent = d.title;
  const m = d.metrics;
  document.getElementById('rollStats').innerHTML = [
    ['Throughput', `${d.ns_per_day_cpu} ns/day`, `${d.wall_s}s on CPU`],
    ['Mean Temperature', `${m.T_mean_K} K`, `Target ${m.T_target_K} K`],
    ['Bond Drift', `${m.bond_rms_drift_A} Å`, `${m.n_bonds} covalent bonds`],
    ['Steric Clearance', `${m.min_vdw_ratio}`, m.vdw_clashes === 0 ? 'Zero atom clashes' : `${m.vdw_clashes} clashes`]
  ].map(([l, v, n]) => `<div class="stat-card"><div class="stat-label">${l}</div><div class="stat-value">${v}</div><div class="stat-note">${n}</div></div>`).join('');

  drawMinimalSeries('canvasRollChart', d.series.time_ps, [
    { label: 'Temperature (K)', data: d.series.temperature_K, color: '#141413' }
  ]);
  drawRoll3D();
}

function drawRoll3D() {
  const d = STATE.rollData;
  if (!d || !d.viewer_frames.length) return;
  const fr = d.viewer_frames[STATE.frameIdx.roll % d.viewer_frames.length];
  drawScene('canvasRoll3D', fr, d.viewer_elements, {
    bonds: d.viewer_bonds, boxA: d.viewer_box_A, camKey: 'roll'
  });
}

function downloadRolloutXyz() {
  if (!STATE.rollData || !STATE.rollData.xyz_download) return;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([STATE.rollData.xyz_download], { type: 'text/plain' }));
  a.download = 'wignerflow_rollout.xyz';
  a.click();
}

async function runBindingLab() {
  const r = await fetch('/api/triage', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      mode: 'binding',
      preset: document.getElementById('bindPreset').value,
      replicas: Number(document.getElementById('bindRep').value),
      frames: Number(document.getElementById('bindFrames').value)
    })
  });
  const rep = (await r.json()).report;
  STATE.bindData = rep;
  STATE.frameIdx.bind = 0;
  const m = rep.metrics;

  const bb = document.getElementById('bindBadge');
  bb.className = 'badge ' + rep.verdict;
  bb.innerHTML = `<span class="dot ${rep.verdict}"></span>${rep.verdict}`;
  document.getElementById('bindSummaryText').textContent = rep.summary;

  document.getElementById('bindStats').innerHTML = [
    ['Stability Score', m.comp_score, 'OpenBPMD CompScore'],
    ['Ligand Drift', `${m.pose_score_A} Å`, 'Heavy-atom RMSD'],
    ['Unbinding Time', `${m.tau_ramd_median_ps} ps`, '50 fs random-push probe'],
    ['CPU Latency', `${rep.wall_s} s`, `Min vdW ratio ${m.posebusters_min_vdw_ratio}`]
  ].map(([l, v, n]) => `<div class="stat-card"><div class="stat-label">${l}</div><div class="stat-value">${v}</div><div class="stat-note">${n}</div></div>`).join('');

  drawMinimalSeries('canvasBindChart', rep.series.time_ps, [
    { label: 'Ligand RMSD (Å)', data: rep.series.ligand_rmsd_A, color: '#141413' },
    { label: 'COM Shift (Å)', data: rep.series.com_displacement_A, color: '#8C8B85', dashed: true }
  ]);
  drawBind3D();
}

function drawBind3D() {
  const d = STATE.bindData;
  if (!d || !d.viewer_frames.length) return;
  const fr = d.viewer_frames[STATE.frameIdx.bind % d.viewer_frames.length];
  drawScene('canvasBind3D', fr, null, { roles: d.viewer_roles, camKey: 'bind' });
}

async function runElectrolyteLab() {
  const r = await fetch('/api/triage', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      mode: 'electrolyte',
      preset: document.getElementById('elecPreset').value,
      frames: Number(document.getElementById('elecFrames').value)
    })
  });
  const rep = (await r.json()).report;
  STATE.elecData = rep;
  STATE.frameIdx.elec = 0;
  const m = rep.metrics;

  const eb = document.getElementById('elecBadge');
  eb.className = 'badge ' + rep.verdict;
  eb.innerHTML = `<span class="dot ${rep.verdict}"></span>${rep.verdict}`;

  document.getElementById('elecStats').innerHTML = [
    ['Cluster Conductivity', `${m.sigma_cNE_mS_cm} mS/cm`, 'Excludes neutral pairs'],
    ['Naive Conductivity', `${m.sigma_NE_mS_cm} mS/cm`, 'Counts all diffusing ions'],
    ['Free-Ion Fraction', `${(m.ssip_fraction * 100).toFixed(0)}%`, `${(m.cip_fraction * 100).toFixed(0)}% paired · ${(m.agg_fraction * 100).toFixed(0)}% agg`],
    ['Li+ Transference', m.transference_cNE, `Haven ionicity ${m.ionicity_haven_inv}`]
  ].map(([l, v, n]) => `<div class="stat-card"><div class="stat-label">${l}</div><div class="stat-value">${v}</div><div class="stat-note">${n}</div></div>`).join('');

  drawElec3D();
}

function drawElec3D() {
  const d = STATE.elecData;
  if (!d || !d.viewer_frames.length) return;
  const fr = d.viewer_frames[STATE.frameIdx.elec % d.viewer_frames.length];
  drawScene('canvasElec3D', fr, null, { roles: d.viewer_roles, boxA: d.box_A, camKey: 'elec' });
}

function drawParityPlot() {
  if (!STATE.overview) return;
  const bell = STATE.overview.benchmarks.calisol23_alignment.bell_curve_ec_dmc_298K || [];
  const cv = document.getElementById('canvasParity');
  if (!cv || !bell.length) return;
  const { ctx, w, h } = prepCanvas(cv);
  const pad = { l: 42, r: 18, t: 18, b: 28 };
  const maxX = 3.4, maxY = 14.0;
  const tx = c => pad.l + (w - pad.l - pad.r) * (c / maxX);
  const ty = s => h - pad.b - (h - pad.t - pad.b) * (s / maxY);

  ctx.strokeStyle = 'rgba(20,20,19,0.07)';
  [0, 4, 8, 12].forEach(s => {
    ctx.beginPath(); ctx.moveTo(pad.l, ty(s)); ctx.lineTo(w - pad.r, ty(s)); ctx.stroke();
    ctx.fillStyle = '#8C8B85';
    ctx.font = '12px ui-monospace, monospace';
    ctx.fillText(s, 8, ty(s) + 4);
  });
  [0, 1, 2, 3].forEach(c => {
    ctx.fillStyle = '#8C8B85';
    ctx.font = '12px ui-monospace, monospace';
    ctx.fillText(c + 'M', tx(c) - 8, h - 8);
  });

  ctx.strokeStyle = '#141413';
  ctx.lineWidth = 2;
  ctx.beginPath();
  bell.forEach((p, i) => {
    const x = tx(p.c_M), y = ty(p.sigma_aligned);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  STATE.parityPts = [];
  bell.forEach(p => {
    const x = tx(p.c_M), y = ty(p.sigma_aligned);
    ctx.fillStyle = '#141413';
    ctx.beginPath(); ctx.arc(x, y, 4.2, 0, Math.PI * 2); ctx.fill();
    STATE.parityPts.push({ x, y, p });
  });
}

function togglePlay(key) {
  STATE.playing[key] = !STATE.playing[key];
  if (key === 'roll') document.getElementById('btnPlayRoll').textContent = STATE.playing.roll ? 'Pause' : 'Play';
}

function populateTables(ov) {
  const b = ov.benchmarks;
  const cal = b.calisol23_alignment || {};
  const mdRows = b.pure_md_md17_md22 || [];
  const boxRows = cal.box_scaling || [];
  const fast256 = boxRows.find(r => r.atoms === 256 && String(r.config).includes('Fast')) || boxRows[0];
  const nStable = mdRows.filter(r => String(r.wf_20fs).startsWith('stable')).length;

  document.getElementById('benchStats').innerHTML = [
    ['20 fs Step Stability', mdRows.length ? `${nStable} / ${mdRows.length}` : '—', 'MD17/MD22 pair-model baseline'],
    ['CALiSol-23 Unseen R²', cal.available ? cal.aligned_r2.toFixed(3) : '—', cal.available ? `${cal.n_heldout_total.toLocaleString()} held-out rows` : 'No result file'],
    ['CALiSol-23 Spearman ρ', cal.available ? cal.aligned_spearman_rho.toFixed(3) : '—', cal.available ? `MAE ${cal.aligned_mae_mS_cm.toFixed(2)} mS/cm` : 'No result file'],
    ['256-Atom Cell (CPU)', fast256 ? `${fast256.ns_day.toFixed(1)} ns/day` : '—', fast256 ? `${fast256.cpu_ms_per_step.toFixed(0)} ms per 50 fs step` : 'No result file']
  ].map(([l, v, n]) => `<div class="stat-card"><div class="stat-label">${l}</div><div class="stat-value">${v}</div><div class="stat-note">${n}</div></div>`).join('');

  document.getElementById('calisolParityTable').innerHTML = cal.available ? `
    <thead>
      <tr>
        <th>Evaluation Stage</th>
        <th class="num">Unseen Rows</th>
        <th class="num">R²</th>
        <th class="num">Spearman ρ</th>
        <th class="num">MAE (mS/cm)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><b>Raw Unaligned MD</b> <span class="muted">(Nernst–Einstein)</span></td>
        <td class="num">${cal.n_formulations}</td>
        <td class="num muted">${cal.unaligned_r2.toFixed(3)}</td>
        <td class="num muted">${cal.unaligned_spearman_rho.toFixed(3)}</td>
        <td class="num muted">${cal.unaligned_mae_mS_cm.toFixed(2)}</td>
      </tr>
      <tr>
        <td><b>WignerFlow-Align</b> <span class="muted">(3D periodic-box panel)</span></td>
        <td class="num">${cal.n_formulations}</td>
        <td class="num" style="color:var(--pass);font-weight:600;">${cal.panel_aligned_r2.toFixed(3)}</td>
        <td class="num">${cal.panel_aligned_spearman_rho.toFixed(3)}</td>
        <td class="num">—</td>
      </tr>
      <tr>
        <td><b>WignerFlow-Align</b> <span class="muted">(full held-out test set)</span></td>
        <td class="num">${cal.n_heldout_total.toLocaleString()}</td>
        <td class="num" style="color:var(--pass);font-weight:600;">${cal.aligned_r2.toFixed(3)}</td>
        <td class="num" style="color:var(--pass);font-weight:600;">${cal.aligned_spearman_rho.toFixed(3)}</td>
        <td class="num" style="color:var(--pass);font-weight:600;">${cal.aligned_mae_mS_cm.toFixed(2)}</td>
      </tr>
    </tbody>` : '';

  document.getElementById('md17Table').innerHTML = `
    <thead>
      <tr>
        <th>System</th>
        <th class="num">Atoms</th>
        <th>Classical Verlet (10 fs)</th>
        <th>Pair Baseline (20 fs)</th>
        <th class="num">Pair Error</th>
        <th class="num">COM Drift (Å/fs)</th>
        <th class="num">CPU Speed</th>
      </tr>
    </thead>
    <tbody>${mdRows.map(r => `
      <tr>
        <td><b>${r.system}</b></td>
        <td class="num">${r.atoms}</td>
        <td class="muted">${r.vv_10fs}</td>
        <td style="color:var(--pass);font-weight:500;">${r.wf_20fs}</td>
        <td class="num">${r.mae_hr}</td>
        <td class="num">${r.com_drift}</td>
        <td class="num">${r.ns_day} ns/day</td>
      </tr>`).join('')}
    </tbody>`;

  document.getElementById('boxScalingTable').innerHTML = `
    <thead>
      <tr>
        <th>Configuration</th>
        <th class="num">Atoms</th>
        <th class="num">Box L (Å)</th>
        <th class="num">CPU ms / 50 fs Step</th>
        <th class="num">CPU Throughput</th>
      </tr>
    </thead>
    <tbody>${boxRows.map(r => `
      <tr>
        <td><b>${r.config}</b></td>
        <td class="num">${r.atoms.toLocaleString()}</td>
        <td class="num">${r.box_A.toFixed(2)}</td>
        <td class="num">${r.cpu_ms_per_step.toFixed(1)} ms</td>
        <td class="num">${r.ns_day.toFixed(3)} ns/day</td>
      </tr>`).join('')}
    </tbody>`;

  document.getElementById('solvationTable').innerHTML = `
    <thead>
      <tr>
        <th>Rank</th>
        <th>Formulation</th>
        <th class="num">σ (25 °C)</th>
        <th class="num">σ (-20 °C)</th>
        <th style="min-width:180px;">Li+ Shell Balance</th>
        <th class="num">Readiness</th>
      </tr>
    </thead>
    <tbody>${b.top15_solvation_champions.map(s => `
      <tr>
        <td class="muted">${s.rank}</td>
        <td><b>${s.formula}</b></td>
        <td class="num">${s.sigma_298.toFixed(2)} mS/cm</td>
        <td class="num">${s.sigma_253.toFixed(2)} mS/cm</td>
        <td>
          <div class="spec-bar">
            <div style="width:${s.ssip}%;background:#141413;"></div>
            <div style="width:${s.cip}%;background:#8C8B85;"></div>
            <div style="width:${s.agg_i + s.agg_net}%;background:#D6D3D1;"></div>
          </div>
          <div class="subtle" style="font-size:12px;margin-top:4px;">${s.ssip}% free · ${s.cip}% paired · ${(s.agg_i + s.agg_net).toFixed(1)}% agg</div>
        </td>
        <td class="num"><b>${s.readiness}</b> / 100</td>
      </tr>`).join('')}
    </tbody>`;
}

function exportSessionJson() {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify({
    exported_at: new Date().toISOString(),
    compliance: STATE.compData,
    rollout: STATE.rollData,
    binding: STATE.bindData,
    electrolyte: STATE.elecData
  }, null, 2)], { type: 'application/json' }));
  a.download = 'wignerflow_report.json';
  a.click();
}

async function init() {
  attachOrbit('canvasComp3D', 'comp', drawComp3D);
  attachOrbit('canvasRoll3D', 'roll', drawRoll3D);
  attachOrbit('canvasBind3D', 'bind', drawBind3D);
  attachOrbit('canvasElec3D', 'elec', drawElec3D);

  document.getElementById('canvasParity').addEventListener('mousemove', e => {
    const rect = e.target.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let best = null, bd = 999;
    STATE.parityPts.forEach(pt => {
      const d = Math.hypot(pt.x - mx, pt.y - my);
      if (d < bd) { bd = d; best = pt; }
    });
    if (best && bd < 20) {
      const p = best.p;
      document.getElementById('parityHover').textContent =
        `${p.c_M.toFixed(2)} M LiPF6 in EC:DMC (298 K) — Aligned conductivity: ${p.sigma_aligned.toFixed(2)} mS/cm`;
    }
  });

  const ov = await (await fetch('/api/overview')).json();
  STATE.overview = ov;
  document.getElementById('runtimeBadge').innerHTML =
    `<span class="dot"></span>PyTorch ${ov.runtime.torch.split('+')[0]} · Live`;
  document.getElementById('customXyzInput').value = ov.default_xyz;
  document.getElementById('compFault').innerHTML = ov.fault_catalog.map(f =>
    `<option value="${f.id}">${f.title}</option>`
  ).join('');

  populateTables(ov);
  await runCompliance();

  setInterval(() => {
    if (STATE.playing.roll && STATE.rollData) {
      STATE.frameIdx.roll = (STATE.frameIdx.roll + 1) % Math.max(1, STATE.rollData.viewer_frames.length);
      drawRoll3D();
    }
    if (STATE.playing.bind && STATE.bindData) {
      STATE.frameIdx.bind = (STATE.frameIdx.bind + 1) % Math.max(1, STATE.bindData.viewer_frames.length);
      drawBind3D();
    }
    if (STATE.playing.elec && STATE.elecData) {
      STATE.frameIdx.elec = (STATE.frameIdx.elec + 1) % Math.max(1, STATE.elecData.viewer_frames.length);
      drawElec3D();
    }
  }, 110);
}

window.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>
"""


class WignerFlowStudioHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _set_headers(self, content_type: str, length: int):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))

    def _send_json(self, code: int, payload: dict):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self._set_headers("application/json; charset=utf-8", len(raw))
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_headers("text/plain; charset=utf-8", 0)
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            raw = STUDIO_HTML.encode("utf-8")
            self.send_response(200)
            self._set_headers("text/html; charset=utf-8", len(raw))
            self.end_headers()
            self.wfile.write(raw)
        elif path == "/api/overview":
            self._send_json(200, build_overview_payload())
        elif path == "/api/catalog":
            self._send_json(200, {
                "binding_presets": BINDING_PRESETS,
                "electrolyte_presets": ELECTROLYTE_PRESETS,
                "datasets": pocket.DATASET_CATALOG,
            })
        elif path == "/api/health":
            self._send_json(200, {"status": "ok", "ts": time.time()})
        else:
            self._send_json(404, {"error": f"Unknown GET endpoint {path!r}"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        try:
            if path == "/api/compliance":
                euler = body.get("euler_deg", [37.0, 64.0, 118.0])
                rep = run_compliance_audit(
                    target_id=body.get("target_id", "argon_flow_v1"),
                    fault=body.get("fault", "none"),
                    precision=body.get("precision", "float32"),
                    euler_deg=tuple(euler) if euler else None,
                    rollout_steps=int(body.get("rollout_steps", 14)),
                    sampling_steps=int(body.get("sampling_steps", 3)),
                    seed=int(body.get("seed", 0)),
                )
                self._send_json(200, rep)
            elif path == "/api/rollout":
                rep = run_studio_rollout(
                    preset=body.get("preset", "aspirin"),
                    custom_xyz=body.get("custom_xyz"),
                    n_frames=int(body.get("n_frames", 60)),
                    substeps=int(body.get("substeps", 8)),
                    dt_fs=float(body.get("dt_fs", 1.5)),
                    temperature_k=float(body.get("temperature_k", 300.0)),
                    friction_per_ps=float(body.get("friction_per_ps", 2.0)),
                    seed=int(body.get("seed", 0)),
                )
                self._send_json(200, rep)
            elif path == "/api/triage":
                mode = body.get("mode", "binding")
                if mode == "binding":
                    rep = pocket.triage_binding(
                        preset=body.get("preset", "aspirin_cox_stable"),
                        pdb_path=body.get("pdb_path"),
                        xml_path=body.get("xml_path"),
                        n_frames=int(body.get("frames", 70)),
                        n_replicas=int(body.get("replicas", 3)),
                        seed=int(body.get("seed", 0)),
                    )
                elif mode == "electrolyte":
                    rep = pocket.triage_electrolyte(
                        preset=body.get("preset", "li_ec_fec_additive"),
                        n_frames=int(body.get("frames", 90)),
                        seed=int(body.get("seed", 0)),
                    )
                elif mode == "checkpoint":
                    rep = pocket.triage_checkpoint(
                        ckpt_path=body.get("ckpt_path"),
                        steps=int(body.get("steps", 24)),
                        sampling_steps=int(body.get("sampling_steps", 3)),
                        thermostat_on=bool(body.get("thermostat_on", False)),
                        seed=int(body.get("seed", 0)),
                    )
                else:
                    self._send_json(400, {"error": f"Unknown triage mode {mode!r}"})
                    return
                self._send_json(200, {"report": rep, "receipt": pocket.format_receipt(rep)})
            else:
                self._send_json(404, {"error": f"Unknown POST endpoint {path!r}"})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})


def create_studio_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, int(port)), WignerFlowStudioHandler)
