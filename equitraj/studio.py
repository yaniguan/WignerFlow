"""WignerFlow Studio: Interactive Web Evaluation Workbench & Equivariant Physics Compliance Lab.

Provides a zero-external-dependency HTTP/JSON server (`ThreadingHTTPServer`) paired with a rich
5-workspace browser application for interactively testing all WignerFlow capabilities and
verifying fundamental equivariant MD physical laws (`G1`–`G10` + `F1`–`F10` fault injection).
"""

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
<title>WignerFlow Studio - Physics Check & Molecule Simulator</title>
<style>
  :root {
    --bg: #0b1120;
    --panel: #111827;
    --card: #172033;
    --border: #283548;
    --fg: #f1f5f9;
    --muted: #94a3b8;
    --primary: #3b82f6;
    --cyan: #06b6d4;
    --emerald: #10b981;
    --amber: #f59e0b;
    --rose: #f43f5e;
    --purple: #a855f7;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--fg);
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 13px;
    line-height: 1.45;
  }
  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 12px 20px;
    background: #0f172a;
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 30;
    flex-wrap: wrap;
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .brand-logo {
    width: 34px; height: 34px; border-radius: 9px;
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 16px; color: #fff;
  }
  .brand h1 { margin: 0; font-size: 16px; font-weight: 700; letter-spacing: -0.01em; }
  .brand .sub { color: var(--muted); font-size: 12px; }
  .top-actions { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 11px; border-radius: 999px; font-weight: 700; font-size: 12px;
    border: 1px solid var(--border); background: var(--card);
  }
  .pill.GO, .pill.PASS { background: rgba(16,185,129,0.16); color: #6ee7b7; border-color: rgba(16,185,129,0.45); }
  .pill.HOLD, .pill.WARN { background: rgba(245,158,11,0.16); color: #fcd34d; border-color: rgba(245,158,11,0.45); }
  .pill.REJECT, .pill.FAIL { background: rgba(244,63,94,0.18); color: #fda4af; border-color: rgba(244,63,94,0.5); }
  .tabs {
    display: flex; gap: 6px; padding: 10px 20px; background: #0d1526;
    border-bottom: 1px solid var(--border); overflow-x: auto;
  }
  .tab-btn {
    background: transparent; color: var(--muted); border: 1px solid transparent;
    padding: 8px 14px; border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 13px;
    white-space: nowrap; transition: all 0.15s;
  }
  .tab-btn:hover { color: var(--fg); background: rgba(255,255,255,0.04); }
  .tab-btn.active {
    color: #fff; background: rgba(59,130,246,0.18); border-color: rgba(59,130,246,0.55);
  }
  .container { max-width: 1560px; margin: 0 auto; padding: 16px 20px 36px; }
  .workspace { display: none; }
  .workspace.active { display: block; }
  .grid-3 { display: grid; grid-template-columns: 330px minmax(420px, 1fr) 470px; gap: 14px; align-items: start; }
  .grid-2 { display: grid; grid-template-columns: 360px 1fr; gap: 14px; align-items: start; }
  @media (max-width: 1260px) {
    .grid-3, .grid-2 { grid-template-columns: 1fr; }
  }
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 14px; margin-bottom: 14px;
  }
  .card h3 { margin: 0 0 8px 0; font-size: 14px; font-weight: 700; display: flex; align-items: center; justify-content: space-between; }
  label { display: block; font-size: 12px; color: var(--muted); margin: 8px 0 4px; font-weight: 600; }
  select, input[type="number"], input[type="text"], textarea {
    width: 100%; background: #0b1120; color: var(--fg); border: 1px solid var(--border);
    border-radius: 7px; padding: 7px 9px; font-size: 12px; font-family: inherit;
  }
  textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; resize: vertical; }
  input[type="range"] { width: 100%; accent-color: var(--primary); }
  .btn {
    background: var(--primary); color: #fff; border: 0; border-radius: 8px;
    padding: 8px 13px; cursor: pointer; font-weight: 600; font-size: 12px;
    display: inline-flex; align-items: center; justify-content: center; gap: 6px;
  }
  .btn:hover { filter: brightness(1.08); }
  .btn.sec { background: #263349; color: #e2e8f0; border: 1px solid #364763; }
  .btn.full { width: 100%; margin-top: 10px; padding: 10px; font-size: 13px; }
  .fault-list { display: flex; flex-direction: column; gap: 5px; max-height: 330px; overflow-y: auto; padding-right: 3px; }
  .fault-item {
    padding: 7px 9px; border-radius: 8px; border: 1px solid var(--border);
    background: #0e1627; cursor: pointer; transition: all 0.12s;
  }
  .fault-item:hover { border-color: #475569; }
  .fault-item.active { border-color: var(--rose); background: rgba(244,63,94,0.13); }
  .fault-item.clean.active { border-color: var(--emerald); background: rgba(16,185,129,0.13); }
  .fault-title { font-weight: 700; font-size: 12px; }
  .fault-desc { font-size: 12px; color: var(--muted); margin-top: 2px; }
  canvas.viewport {
    width: 100%; height: 340px; background: radial-gradient(circle at 50% 42%, #17233a 0%, #090e1a 100%);
    border-radius: 10px; border: 1px solid var(--border); cursor: grab; display: block;
  }
  canvas.chart {
    width: 100%; height: 185px; background: #0b1120;
    border-radius: 9px; border: 1px solid var(--border); display: block;
  }
  .gate-row {
    padding: 9px 10px; border-radius: 9px; border: 1px solid var(--border);
    background: #0e1627; margin-bottom: 7px;
  }
  .gate-row.FAIL { border-color: rgba(244,63,94,0.65); background: rgba(127,29,29,0.18); }
  .gate-row.WARN { border-color: rgba(245,158,11,0.55); background: rgba(120,53,15,0.16); }
  .gate-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  .gate-id { font-family: ui-monospace, monospace; font-weight: 800; color: #93c5fd; }
  .gate-meta { font-size: 12px; color: var(--muted); margin-top: 4px; font-family: ui-monospace, monospace; }
  .gate-cite { font-size: 12px; color: #64748b; margin-top: 2px; }
  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: 8px; margin-bottom: 12px; }
  .kpi { background: #0e1627; border: 1px solid var(--border); border-radius: 9px; padding: 9px 11px; }
  .kpi .k { font-size: 12px; color: var(--muted); }
  .kpi .v { font-size: 16px; font-weight: 800; margin-top: 2px; font-family: ui-monospace, monospace; }
  table.bench { width: 100%; border-collapse: collapse; font-size: 12px; }
  table.bench th, table.bench td { padding: 8px 9px; border-bottom: 1px solid var(--border); text-align: left; }
  table.bench th { color: var(--muted); font-weight: 600; background: #0e1627; }
  .bar-stack { display: flex; height: 12px; border-radius: 6px; overflow: hidden; background: #0b1120; min-width: 140px; }
  .row-inline { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
</style>
</head>
<body>

<div class="topbar">
  <div class="brand">
    <div class="brand-logo">WF</div>
    <div>
      <h1>WignerFlow Studio · Physics Rules & Fast Molecule Simulator</h1>
      <div class="sub" id="runtimeInfo">Connecting to live PyTorch CPU engine...</div>
    </div>
  </div>
  <div class="top-actions">
    <span class="pill GO" id="globalVerdict">GO · 10/10 GATES PASS</span>
    <button class="btn sec" onclick="exportSessionJson()">Download Report (.json)</button>
  </div>
</div>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('compliance', this)">1. Physics Rules Check (G1-G10)</button>
  <button class="tab-btn" onclick="switchTab('rollout', this)">2. Molecule Simulator (3D & Custom .xyz)</button>
  <button class="tab-btn" onclick="switchTab('binding', this)">3. Drug Pocket Check</button>
  <button class="tab-btn" onclick="switchTab('electrolyte', this)">4. Battery Electrolytes & CALiSol-23</button>
  <button class="tab-btn" onclick="switchTab('benchmarks', this)">5. All Benchmark Numbers</button>
</div>

<div class="container">
  <!-- ======================= WORKSPACE 1: EQUIVARIANT & PHYSICS COMPLIANCE ======================= -->
  <div id="ws-compliance" class="workspace active">
    <div class="grid-3">
      <!-- Left: Controls + F1..F10 Chaos Drawer -->
      <div>
        <div class="card">
          <h3><span>Test Setup</span></h3>
          <label>Model to Check</label>
          <select id="compTarget">
            <option value="argon_flow_v1">Liquid Argon Flow v1 (Trained Model, 256 Atoms)</option>
            <option value="aspirin_hetero_o3">Aspirin O(3) Model (C/H/O + Charges + Bonds)</option>
            <option value="aspirin_chiral_so3">Aspirin Rotation-Only Control (parity=False, Fails G2 Mirror Flip!)</option>
            <option value="argon_flow_v0">Liquid Argon Flow v0 (Early Baseline)</option>
          </select>

          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
            <div>
              <label>Math Precision</label>
              <select id="compPrec">
                <option value="float32">float32 (Normal Speed)</option>
                <option value="float64">float64 (High Precision)</option>
              </select>
            </div>
            <div>
              <label>Rollout Steps</label>
              <input type="number" id="compSteps" value="14" min="8" max="40">
            </div>
          </div>

          <label>3D Rotation Angles (alpha=<span id="lblA">37</span>°, beta=<span id="lblB">64</span>°, gamma=<span id="lblG">118</span>°)</label>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;">
            <input type="range" id="angA" min="0" max="360" value="37" oninput="document.getElementById('lblA').textContent=this.value">
            <input type="range" id="angB" min="0" max="360" value="64" oninput="document.getElementById('lblB').textContent=this.value">
            <input type="range" id="angG" min="0" max="360" value="118" oninput="document.getElementById('lblG').textContent=this.value">
          </div>
          <div class="row-inline" style="margin-top:6px;">
            <button class="btn sec" style="flex:1;" onclick="randomizeEuler()">Pick Random 3D Rotation</button>
          </div>
          <button class="btn full" id="btnRunComp" onclick="runCompliance()">Run 10-Rule Physics Check</button>
        </div>

        <div class="card">
          <h3><span>Break-a-Rule Test Switches (F1-F10)</span></h3>
          <div style="color:var(--muted);font-size:12px;margin-bottom:8px;">Click any bug below to intentionally break one physics rule and watch the matching gate turn red:</div>
          <div class="fault-list" id="faultList"></div>
        </div>
      </div>

      <!-- Center: 3D Twin Symmetry & Rollout Viewport + Radar + Telemetry -->
      <div>
        <div class="card">
          <h3>
            <span>3D Rotation Arrow Check · <span id="comp3dTitle" style="color:var(--cyan);">Drag to Rotate / Scroll to Zoom</span></span>
            <span class="row-inline">
              <button class="btn sec" onclick="togglePlay('comp')" id="btnPlayComp">⏸ Play</button>
              <label style="margin:0;display:inline-flex;align-items:center;gap:4px;">
                <input type="checkbox" id="chkTwinVec" checked onchange="drawComp3D()"> Show Rotated vs Unrotated Arrows
              </label>
            </span>
          </h3>
          <canvas id="canvasComp3D" class="viewport" width="680" height="340"></canvas>
          <div class="row-inline" style="margin-top:8px;justify-content:space-between;">
            <span style="color:var(--muted);font-size:12px;">Cyan arrow: <b>Normal move</b> · Red dashed: <b>Rotated &amp; turned back (should match!)</b> · Gold +: <b>Center of mass</b></span>
            <input type="range" id="compFrameSlider" min="0" max="13" value="0" style="width:180px;" oninput="setCompFrame(this.value)">
          </div>
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
          <div class="card">
            <h3><span>Symmetry Digits of Accuracy (-log10 error)</span></h3>
            <canvas id="canvasRadar" class="chart" width="320" height="190"></canvas>
          </div>
          <div class="card">
            <h3><span>Temperature T(t) &amp; Closest Atom Ratio</span></h3>
            <canvas id="canvasCompTelemetry" class="chart" width="320" height="190"></canvas>
          </div>
        </div>
      </div>

      <!-- Right: 10-Gate Compliance Scorecard -->
      <div>
        <div class="card">
          <h3>
            <span>10 Physics Rules Scorecard (G1-G10)</span>
            <span id="compTiming" style="color:var(--muted);font-size:12px;">--</span>
          </h3>
          <div id="gatesContainer"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- ======================= WORKSPACE 2: STANDALONE & CUSTOM XYZ ROLLOUT ======================= -->
  <div id="ws-rollout" class="workspace">
    <div class="grid-2">
      <div class="card">
        <h3><span>Fast CPU Molecule Simulator</span></h3>
        <label>Built-in Molecule</label>
        <select id="rollPreset" onchange="if(this.value!=='custom'){document.getElementById('useCustomXyz').checked=false;}">
          <option value="aspirin">Aspirin (21 Atoms)</option>
          <option value="ala2">Alanine Dipeptide (22 Atoms)</option>
          <option value="ethanol">Ethanol CH3CH2OH (9 Atoms)</option>
          <option value="li_ec_pf6">LiPF6 in EC Battery Box</option>
        </select>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;">
          <div><label>Frames</label><input type="number" id="rollFrames" value="60" min="15" max="200"></div>
          <div><label>dt (fs)</label><input type="number" id="rollDt" value="1.5" step="0.25" min="0.25" max="4.0"></div>
          <div><label>Temp (K)</label><input type="number" id="rollTemp" value="300" min="50" max="800"></div>
        </div>
        <label style="margin-top:10px;">
          <input type="checkbox" id="useCustomXyz"> Use custom `.xyz` coordinates below (bonds are detected automatically)
        </label>
        <textarea id="customXyzInput" rows="10"></textarea>
        <button class="btn full" onclick="runRolloutLab()">Run Simulation on CPU</button>
        <button class="btn sec full" onclick="downloadRolloutXyz()">Download Trajectory (.xyz)</button>
      </div>

      <div>
        <div class="kpi-grid" id="rollKpis"></div>
        <div class="card">
          <h3>
            <span id="rollTitle">3D Molecule Motion</span>
            <button class="btn sec" onclick="togglePlay('roll')" id="btnPlayRoll">⏸ Play</button>
          </h3>
          <canvas id="canvasRoll3D" class="viewport" width="840" height="350"></canvas>
        </div>
        <div class="card">
          <h3><span>Temperature Over Time T(t) [K]</span></h3>
          <canvas id="canvasRollChart" class="chart" width="840" height="175"></canvas>
        </div>
      </div>
    </div>
  </div>

  <!-- ======================= WORKSPACE 3: DRUG BINDING TRIAGE ======================= -->
  <div id="ws-binding" class="workspace">
    <div class="grid-2">
      <div class="card">
        <h3><span>Fast Drug-in-Pocket Stability Check</span></h3>
        <p style="color:var(--muted);margin-top:0;">Before spending 50+ GPU hours on a heavy binding calculation, wiggle the drug inside the protein pocket for ~0.5 seconds on CPU to see if it stays put or pops out:</p>
        <label>Candidate Drug Pose</label>
        <select id="bindPreset">
          <option value="aspirin_cox_stable">COX-1 + Aspirin · Good Crystal Pose (Expected: GO)</option>
          <option value="aspirin_induced_fit">COX-1 + Aspirin · Slightly Tilted Pose (Expected: HOLD)</option>
          <option value="aspirin_decoy_clash">COX-1 + Aspirin · 180° Flipped Bad Pose (Expected: REJECT)</option>
        </select>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
          <div><label>Parallel Runs</label><input type="number" id="bindRep" value="3" min="1" max="6"></div>
          <div><label>Frames / Run</label><input type="number" id="bindFrames" value="70" min="30" max="140"></div>
        </div>
        <button class="btn full" onclick="runBindingLab()">Check Pocket Stability</button>
        <div id="bindSummaryBox" style="margin-top:12px;"></div>
      </div>

      <div>
        <div class="kpi-grid" id="bindKpis"></div>
        <div class="card">
          <h3><span>3D Protein Pocket &amp; Drug (Green = Drug, Gray = Protein Pocket)</span></h3>
          <canvas id="canvasBind3D" class="viewport" width="840" height="320"></canvas>
        </div>
        <div class="card">
          <h3><span>How Far the Drug Moves (RMSD in Å) &amp; Hydrogen Bonds</span></h3>
          <canvas id="canvasBindChart" class="chart" width="840" height="180"></canvas>
        </div>
      </div>
    </div>
  </div>

  <!-- ======================= WORKSPACE 4: ELECTROLYTE & 1000k SOLVATION ======================= -->
  <div id="ws-electrolyte" class="workspace">
    <div class="grid-2">
      <div class="card">
        <h3><span>Lithium Battery Liquid Check</span></h3>
        <p style="color:var(--muted);margin-top:0;">When Li+ and PF6- stick together as a neutral pair, they move without carrying electric current! Cluster conductivity (`sigma_cNE`) filters out neutral pairs so we measure real current:</p>
        <label>Battery Liquid Recipe</label>
        <select id="elecPreset">
          <option value="li_ec_fec_additive">1.0M LiPF6 in EC/DMC + 10% FEC (Fast Ions -> GO)</option>
          <option value="li_ec_base">1.0M LiPF6 in EC Baseline</option>
          <option value="li_ec_vc_additive">1.0M LiPF6 in EC + High VC (Slow &amp; Thick -> HOLD)</option>
          <option value="li_high_salt_agg">Stuck Neutral Ion Pairs (Fake Naive Conductivity -> REJECT)</option>
        </select>
        <label>Simulation Frames</label>
        <input type="number" id="elecFrames" value="90" min="40" max="180">
        <button class="btn full" onclick="runElectrolyteLab()">Run Battery Box Simulation</button>
        <div id="elecSummaryBox" style="margin-top:12px;"></div>
      </div>

      <div>
        <div class="kpi-grid" id="elecKpis"></div>
        <div style="display:grid;grid-template-columns:1.15fr 1fr;gap:14px;">
          <div class="card">
            <h3><span>3D Battery Box (Purple=Li+, Gold=PF6-, Cyan=Additive)</span></h3>
            <canvas id="canvasElec3D" class="viewport" width="460" height="270"></canvas>
          </div>
          <div class="card">
            <h3><span>Li+ Neighbor Shell Peak g(r)</span></h3>
            <canvas id="canvasElecChart" class="chart" width="400" height="270" style="height:270px;"></canvas>
          </div>
        </div>

        <div class="card">
          <h3><span>CALiSol-23 Real Lab Conductivity vs. WignerFlow MD Trajectory Conductivity (Unseen Recipes)</span></h3>
          <p style="color:var(--muted);margin-top:0;">Raw unaligned MD (`sigma_raw_ne`, R² = -0.61) fails because a small box drags ions down while neutral pairs get over-counted. Post-training + physics fixes (`WignerFlow-Align`, R² = 0.72 across 2,752 unseen recipes) recovers real lab conductivity:</p>
          <table class="bench" id="calisolParityTable"></table>
        </div>

        <div class="card">
          <h3><span>1,000,000-Recipe Screen · Top Candidates &amp; Li+ Shell Breakdown (Free Ions vs Stuck Pairs)</span></h3>
          <table class="bench" id="solvationTable"></table>
        </div>
      </div>
    </div>
  </div>

  <!-- ======================= WORKSPACE 5: UNIFIED BENCHMARKS ======================= -->
  <div id="ws-benchmarks" class="workspace">
    <div class="card">
      <h3><span>Big Time-Step Test (10 fs &amp; 20 fs Jumps) · Quantum Molecules &amp; 60-Atom DNA Pair at 500 K</span></h3>
      <table class="bench" id="md17Table"></table>
    </div>
    <div class="grid-2">
      <div class="card">
        <h3><span>CALiSol-23 Battery Conductivity Summary (72 3D Rollouts &amp; 2,752 Unseen Lab Measurements)</span></h3>
        <div class="kpi-grid" id="calisolKpis"></div>
        <table class="bench" id="calisolBoxTable"></table>
      </div>
      <div class="card">
        <h3><span>Public MD Datasets You Can Import in One Command (`scripts/import_md.py`)</span></h3>
        <table class="bench" id="datasetTable"></table>
      </div>
    </div>
  </div>
</div>

<script>
const ELEM_COLOR = {
  H: '#e2e8f0', C: '#38bdf8', N: '#818cf8', O: '#fb7185', F: '#34d399',
  Li: '#c084fc', P: '#fbbf24', S: '#facc15', Ar: '#22d3ee'
};
const ELEM_RADIUS = { H: 0.34, C: 0.56, N: 0.54, O: 0.53, F: 0.50, Li: 0.62, P: 0.68, Ar: 0.72 };

let STATE = {
  overview: null,
  selectedFault: 'none',
  compData: null,
  rollData: null,
  bindData: null,
  elecData: null,
  frameIdx: { comp: 0, roll: 0, bind: 0, elec: 0 },
  playing: { comp: true, roll: true, bind: true, elec: true },
  cam: {
    comp: { rotX: 0.42, rotY: -0.55, zoom: 1.0 },
    roll: { rotX: 0.35, rotY: 0.60, zoom: 1.0 },
    bind: { rotX: 0.32, rotY: -0.45, zoom: 1.0 },
    elec: { rotX: 0.38, rotY: 0.52, zoom: 1.0 }
  }
};

function switchTab(name, btn) {
  document.querySelectorAll('.workspace').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('ws-' + name).classList.add('active');
  if (btn) btn.classList.add('active');
  if (name === 'rollout' && !STATE.rollData) runRolloutLab();
  if (name === 'binding' && !STATE.bindData) runBindingLab();
  if (name === 'electrolyte' && !STATE.elecData) runElectrolyteLab();
}

function setupOrbit(canvasId, key, redrawFn) {
  const cv = document.getElementById(canvasId);
  let dragging = false, lx = 0, ly = 0;
  cv.addEventListener('mousedown', e => { dragging = true; lx = e.clientX; ly = e.clientY; });
  window.addEventListener('mouseup', () => { dragging = false; });
  window.addEventListener('mousemove', e => {
    if (!dragging) return;
    STATE.cam[key].rotY += (e.clientX - lx) * 0.01;
    STATE.cam[key].rotX += (e.clientY - ly) * 0.01;
    lx = e.clientX; ly = e.clientY;
    redrawFn();
  });
  cv.addEventListener('wheel', e => {
    e.preventDefault();
    STATE.cam[key].zoom = Math.max(0.45, Math.min(2.8, STATE.cam[key].zoom * (e.deltaY < 0 ? 1.08 : 0.92)));
    redrawFn();
  }, { passive: false });
}

function project3D(pt, center, scale, cam, w, h) {
  let x = pt[0] - center[0], y = pt[1] - center[1], z = pt[2] - center[2];
  const cy = Math.cos(cam.rotY), sy = Math.sin(cam.rotY);
  let x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
  const cx = Math.cos(cam.rotX), sx = Math.sin(cam.rotX);
  let y2 = y * cx - z1 * sx, z2 = y * sx + z1 * cx;
  const persp = 420 / (420 + z2 * 8 + 1e-5);
  const s = scale * cam.zoom * persp;
  return { sx: w * 0.5 + x1 * s, sy: h * 0.5 - y2 * s, z: z2, s };
}

function renderMoleculeCanvas(canvasId, coords, elements, opts = {}) {
  const cv = document.getElementById(canvasId);
  if (!cv || !coords || !coords.length) return;
  const ctx = cv.getContext('2d');
  const w = cv.width, h = cv.height;
  ctx.clearRect(0, 0, w, h);

  let cx = 0, cy = 0, cz = 0;
  coords.forEach(p => { cx += p[0]; cy += p[1]; cz += p[2]; });
  const center = opts.boxA ? [opts.boxA * 0.5, opts.boxA * 0.5, opts.boxA * 0.5] : [cx / coords.length, cy / coords.length, cz / coords.length];
  let maxR = opts.boxA ? opts.boxA * 0.65 : 2.5;
  coords.forEach(p => {
    const d = Math.hypot(p[0] - center[0], p[1] - center[1], p[2] - center[2]);
    if (d > maxR) maxR = d;
  });
  const scale = (Math.min(w, h) * 0.37) / maxR;
  const cam = STATE.cam[opts.camKey || 'comp'];

  // Draw periodic box wireframe if present
  if (opts.boxA) {
    const L = opts.boxA;
    const corners = [
      [0,0,0],[L,0,0],[L,L,0],[0,L,0],
      [0,0,L],[L,0,L],[L,L,L],[0,L,L]
    ].map(c => project3D(c, center, scale, cam, w, h));
    const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    ctx.strokeStyle = 'rgba(148,163,184,0.28)';
    ctx.lineWidth = 1;
    edges.forEach(([i,j]) => {
      ctx.beginPath(); ctx.moveTo(corners[i].sx, corners[i].sy); ctx.lineTo(corners[j].sx, corners[j].sy); ctx.stroke();
    });
  }

  const proj = coords.map((p, idx) => ({ ...project3D(p, center, scale, cam, w, h), idx, p }));

  // Draw covalent bonds
  if (opts.bonds && opts.bonds.length) {
    ctx.strokeStyle = 'rgba(203,213,225,0.48)';
    ctx.lineWidth = 2.2;
    opts.bonds.forEach(([i, j]) => {
      if (i < proj.length && j < proj.length) {
        const d3 = Math.hypot(coords[i][0]-coords[j][0], coords[i][1]-coords[j][1], coords[i][2]-coords[j][2]);
        if (!opts.boxA || d3 < opts.boxA * 0.45) {
          ctx.beginPath(); ctx.moveTo(proj[i].sx, proj[i].sy); ctx.lineTo(proj[j].sx, proj[j].sy); ctx.stroke();
        }
      }
    });
  }

  // Sort back-to-front
  const sorted = [...proj].sort((a, b) => b.z - a.z);
  sorted.forEach(item => {
    const idx = item.idx;
    const el = (elements && elements[idx]) || 'C';
    let col = ELEM_COLOR[el] || '#60a5fa';
    if (opts.roles && opts.roles[idx]) {
      const rl = opts.roles[idx];
      if (rl === 'ligand') col = '#34d399';
      else if (rl === 'pocket') col = '#64748b';
      else if (rl === 'cation') col = '#c084fc';
      else if (rl === 'anion') col = '#fbbf24';
      else if (rl === 'additive') col = '#22d3ee';
      else if (rl === 'solvent') col = '#475569';
    }
    const rad = Math.max(3.2, (ELEM_RADIUS[el] || 0.52) * item.s * 0.42);
    const grad = ctx.createRadialGradient(item.sx - rad * 0.3, item.sy - rad * 0.3, rad * 0.1, item.sx, item.sy, rad);
    grad.addColorStop(0, '#ffffff');
    grad.addColorStop(0.32, col);
    grad.addColorStop(1, '#090d16');
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.arc(item.sx, item.sy, rad, 0, Math.PI * 2); ctx.fill();
  });

  // Twin equivariance vectors overlay: f(X) vs R^-1 f(R X)
  if (opts.twinRef && opts.twinRot) {
    const nVec = Math.min(proj.length, opts.twinRef.length);
    for (let i = 0; i < nVec; i++) {
      const base = coords[i];
      const vRef = opts.twinRef[i];
      const vRot = opts.twinRot[i];
      const mag = Math.hypot(vRef[0], vRef[1], vRef[2]) + 1e-9;
      const vScale = Math.min(35.0, 1.35 / mag);
      const p0 = proj[i];
      const pRef = project3D([base[0]+vRef[0]*vScale, base[1]+vRef[1]*vScale, base[2]+vRef[2]*vScale], center, scale, cam, w, h);
      const pRot = project3D([base[0]+vRot[0]*vScale, base[1]+vRot[1]*vScale, base[2]+vRot[2]*vScale], center, scale, cam, w, h);

      ctx.strokeStyle = '#22d3ee'; ctx.lineWidth = 2.2;
      ctx.beginPath(); ctx.moveTo(p0.sx, p0.sy); ctx.lineTo(pRef.sx, pRef.sy); ctx.stroke();

      ctx.strokeStyle = '#f43f5e'; ctx.lineWidth = 1.6;
      ctx.setLineDash([3, 2]);
      ctx.beginPath(); ctx.moveTo(p0.sx, p0.sy); ctx.lineTo(pRot.sx, pRot.sy); ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // Golden COM crosshair
  const comP = project3D([cx / coords.length, cy / coords.length, cz / coords.length], center, scale, cam, w, h);
  ctx.strokeStyle = '#fbbf24'; ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(comP.sx - 7, comP.sy); ctx.lineTo(comP.sx + 7, comP.sy);
  ctx.moveTo(comP.sx, comP.sy - 7); ctx.lineTo(comP.sx, comP.sy + 7);
  ctx.stroke();
}

function drawLineChart(canvasId, xArr, seriesList, yLabel) {
  const cv = document.getElementById(canvasId);
  if (!cv || !xArr || !xArr.length) return;
  const ctx = cv.getContext('2d');
  const w = cv.width, h = cv.height;
  ctx.clearRect(0, 0, w, h);
  const pad = { l: 46, r: 14, t: 18, b: 26 };
  let ymin = Infinity, ymax = -Infinity;
  seriesList.forEach(s => (s.data || []).forEach(v => {
    if (v !== null && isFinite(v)) { if (v < ymin) ymin = v; if (v > ymax) ymax = v; }
  }));
  if (!isFinite(ymin)) { ymin = 0; ymax = 1; }
  if (Math.abs(ymax - ymin) < 1e-6) { ymax += 1; ymin -= 1; }
  const span = ymax - ymin; ymin -= 0.08 * span; ymax += 0.12 * span;

  ctx.strokeStyle = '#1e293b'; ctx.lineWidth = 1;
  for (let g = 0; g <= 3; g++) {
    const yy = pad.t + (h - pad.t - pad.b) * (g / 3);
    ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(w - pad.r, yy); ctx.stroke();
    const val = ymax - (ymax - ymin) * (g / 3);
    ctx.fillStyle = '#94a3b8'; ctx.font = '11px monospace';
    ctx.fillText(val.toFixed(Math.abs(val) < 10 ? 2 : 0), 4, yy + 4);
  }

  seriesList.forEach((s, sIdx) => {
    const arr = s.data || [];
    if (!arr.length) return;
    ctx.strokeStyle = s.color; ctx.lineWidth = 2;
    if (s.dashed) ctx.setLineDash([4, 3]); else ctx.setLineDash([]);
    ctx.beginPath();
    arr.forEach((v, i) => {
      const xx = pad.l + (w - pad.l - pad.r) * (i / Math.max(1, arr.length - 1));
      const yy = pad.t + (h - pad.t - pad.b) * (1 - (v - ymin) / (ymax - ymin));
      if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
    });
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = s.color; ctx.font = '11px sans-serif';
    ctx.fillText(s.label, pad.l + sIdx * 125, 13);
  });
}

function drawSymmetryRadar(gates) {
  const cv = document.getElementById('canvasRadar');
  const ctx = cv.getContext('2d');
  const w = cv.width, h = cv.height;
  ctx.clearRect(0, 0, w, h);
  const cx = w * 0.5, cy = h * 0.54, R = Math.min(w, h) * 0.36;
  const g5 = gates.slice(0, 5);
  const labels = ['G1 SO(3)', 'G2 O(3)', 'G3 T(3)+COM', 'G4 S_N', 'G5 PBC'];

  // Concentric rings (-log10 error: 0..16, higher = more exact!)
  ctx.strokeStyle = '#243247'; ctx.lineWidth = 1;
  [0.33, 0.66, 1.0].forEach(f => {
    ctx.beginPath();
    for (let i = 0; i < 5; i++) {
      const ang = -Math.PI / 2 + (i * 2 * Math.PI) / 5;
      const x = cx + Math.cos(ang) * R * f, y = cy + Math.sin(ang) * R * f;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.closePath(); ctx.stroke();
  });

  ctx.beginPath();
  g5.forEach((g, i) => {
    const err = Math.max(1e-16, Number(g.value) || 1e-16);
    const digits = Math.max(0.5, Math.min(16, -Math.log10(err)));
    const frac = digits / 16.0;
    const ang = -Math.PI / 2 + (i * 2 * Math.PI) / 5;
    const x = cx + Math.cos(ang) * R * frac, y = cy + Math.sin(ang) * R * frac;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.closePath();
  const anyFail = g5.some(g => g.status === 'FAIL');
  ctx.fillStyle = anyFail ? 'rgba(244,63,94,0.28)' : 'rgba(16,185,129,0.26)';
  ctx.strokeStyle = anyFail ? '#f43f5e' : '#10b981';
  ctx.lineWidth = 2;
  ctx.fill(); ctx.stroke();

  g5.forEach((g, i) => {
    const ang = -Math.PI / 2 + (i * 2 * Math.PI) / 5;
    const lx = cx + Math.cos(ang) * (R + 22), ly = cy + Math.sin(ang) * (R + 14);
    ctx.fillStyle = g.status === 'FAIL' ? '#fda4af' : '#cbd5e1';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(labels[i], lx, ly);
  });
  ctx.textAlign = 'left';
}

function renderFaultCatalog(faults) {
  const box = document.getElementById('faultList');
  box.innerHTML = '';
  faults.forEach(f => {
    const div = document.createElement('div');
    div.className = 'fault-item' + (f.id === 'none' ? ' clean' : '') + (STATE.selectedFault === f.id ? ' active' : '');
    div.innerHTML = `<div class="fault-title">${f.title}</div><div class="fault-desc">${f.description}</div>`;
    div.onclick = () => {
      STATE.selectedFault = f.id;
      renderFaultCatalog(faults);
      runCompliance();
    };
    box.appendChild(div);
  });
}

function randomizeEuler() {
  ['angA', 'angB', 'angG'].forEach((id, idx) => {
    const v = Math.floor(Math.random() * 360);
    document.getElementById(id).value = v;
    document.getElementById(['lblA', 'lblB', 'lblG'][idx]).textContent = v;
  });
  runCompliance();
}

async function runCompliance() {
  const btn = document.getElementById('btnRunComp');
  btn.textContent = '⏳ Auditing G1–G10 in PyTorch...';
  const body = {
    target_id: document.getElementById('compTarget').value,
    fault: STATE.selectedFault,
    precision: document.getElementById('compPrec').value,
    euler_deg: [
      Number(document.getElementById('angA').value),
      Number(document.getElementById('angB').value),
      Number(document.getElementById('angG').value)
    ],
    rollout_steps: Number(document.getElementById('compSteps').value)
  };
  const r = await fetch('/api/compliance', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
  });
  const data = await r.json();
  btn.textContent = 'Run 10-Rule Physics Check';
  STATE.compData = data;
  STATE.frameIdx.comp = 0;
  const sl = document.getElementById('compFrameSlider');
  sl.max = Math.max(0, (data.viewer_frames || []).length - 1);
  sl.value = 0;

  const badge = document.getElementById('globalVerdict');
  badge.className = 'pill ' + data.verdict;
  badge.textContent = `${data.verdict} · ${data.pass_count}/${data.total_gates} GATES PASS (${data.fault})`;
  document.getElementById('compTiming').textContent = `${data.wall_s}s on CPU · ${data.precision}`;

  const gc = document.getElementById('gatesContainer');
  gc.innerHTML = data.gates.map(g => `
    <div class="gate-row ${g.status}">
      <div class="gate-head">
        <span><span class="gate-id">${g.id}</span> · <b>${g.name}</b></span>
        <span class="pill ${g.status}">${g.status}</span>
      </div>
      <div class="gate-meta">measured = <b>${g.value}</b> &nbsp;|&nbsp; criterion: ${g.threshold}<br>${g.detail}</div>
      <div class="gate-cite">📚 ${g.citation}</div>
    </div>
  `).join('');

  drawSymmetryRadar(data.gates);
  drawLineChart('canvasCompTelemetry', data.rollout_series.time_ps, [
    { label: 'Temp (K)', data: data.rollout_series.temperature_K, color: '#38bdf8' },
    { label: '100 × min_vdW', data: data.rollout_series.min_vdw_ratio.map(v => v * 100), color: '#34d399', dashed: true }
  ]);
  drawComp3D();
}

function setCompFrame(v) {
  STATE.frameIdx.comp = Number(v);
  STATE.playing.comp = false;
  document.getElementById('btnPlayComp').textContent = '▶ Play';
  drawComp3D();
}

function drawComp3D() {
  const d = STATE.compData;
  if (!d) return;
  const showTwin = document.getElementById('chkTwinVec').checked;
  if (showTwin && STATE.frameIdx.comp === 0) {
    renderMoleculeCanvas('canvasComp3D', d.twin_3d.pos_A, d.twin_3d.elements, {
      bonds: d.twin_3d.bonds,
      boxA: d.twin_3d.box_A,
      twinRef: d.twin_3d.dx_ref_A,
      twinRot: d.twin_3d.dx_rot_unrot_A,
      camKey: 'comp'
    });
  } else {
    const frames = d.viewer_frames || [];
    const f = frames[STATE.frameIdx.comp % Math.max(1, frames.length)];
    renderMoleculeCanvas('canvasComp3D', f, d.viewer_elements, {
      bonds: d.twin_3d.bonds,
      boxA: d.viewer_box_A,
      camKey: 'comp'
    });
  }
}

async function runRolloutLab() {
  const useXyz = document.getElementById('useCustomXyz').checked;
  const body = {
    preset: document.getElementById('rollPreset').value,
    custom_xyz: useXyz ? document.getElementById('customXyzInput').value : null,
    n_frames: Number(document.getElementById('rollFrames').value),
    dt_fs: Number(document.getElementById('rollDt').value),
    temperature_k: Number(document.getElementById('rollTemp').value)
  };
  const r = await fetch('/api/rollout', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
  });
  const data = await r.json();
  STATE.rollData = data;
  STATE.frameIdx.roll = 0;
  document.getElementById('rollTitle').innerHTML = `${data.title} &nbsp;<span class="pill ${data.verdict}">${data.verdict}</span>`;
  const m = data.metrics;
  document.getElementById('rollKpis').innerHTML = [
    ['CPU Wall Time', `${data.wall_s} s`],
    ['Throughput', `${data.ns_per_day_cpu} ns/day`],
    ['T_mean / T_target', `${m.T_mean_K} / ${m.T_target_K} K`],
    ['Bond RMS Drift', `${m.bond_rms_drift_A} Å`],
    ['PoseBusters vdW', `${m.min_vdw_ratio}`],
    ['COM Leakage', `${m.com_kinetic_leakage}`]
  ].map(([k, v]) => `<div class="kpi"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
  drawLineChart('canvasRollChart', data.series.time_ps, [
    { label: 'Kinetic Temperature T(t) [K]', data: data.series.temperature_K, color: '#f59e0b' }
  ]);
  drawRoll3D();
}

function drawRoll3D() {
  const d = STATE.rollData;
  if (!d || !d.viewer_frames.length) return;
  const fr = d.viewer_frames[STATE.frameIdx.roll % d.viewer_frames.length];
  renderMoleculeCanvas('canvasRoll3D', fr, d.viewer_elements, {
    bonds: d.viewer_bonds, boxA: d.viewer_box_A, camKey: 'roll'
  });
}

function downloadRolloutXyz() {
  if (!STATE.rollData || !STATE.rollData.xyz_download) return;
  const blob = new Blob([STATE.rollData.xyz_download], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'wignerflow_studio_rollout.xyz';
  a.click();
}

async function runBindingLab() {
  const body = {
    mode: 'binding',
    preset: document.getElementById('bindPreset').value,
    replicas: Number(document.getElementById('bindRep').value),
    frames: Number(document.getElementById('bindFrames').value)
  };
  const r = await fetch('/api/triage', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
  });
  const res = await r.json();
  const rep = res.report;
  STATE.bindData = rep;
  STATE.frameIdx.bind = 0;
  const m = rep.metrics;
  document.getElementById('bindSummaryBox').innerHTML = `
    <div class="gate-row ${rep.verdict === 'GO' ? 'PASS' : (rep.verdict === 'HOLD' ? 'WARN' : 'FAIL')}">
      <div class="gate-head"><b>${rep.title}</b> <span class="pill ${rep.verdict}">${rep.verdict}</span></div>
      <div class="gate-meta">${rep.summary}</div>
    </div>`;
  document.getElementById('bindKpis').innerHTML = [
    ['OpenBPMD CompScore', m.comp_score],
    ['PoseScore RMSD', `${m.pose_score_A} Å`],
    ['Contact Persistence', m.pers_score],
    ['PoseBusters vdW', m.posebusters_min_vdw_ratio],
    ['50-fs τ-RAMD', `${m.tau_ramd_median_ps} ps`],
    ['CPU Time', `${rep.wall_s} s`]
  ].map(([k, v]) => `<div class="kpi"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
  drawLineChart('canvasBindChart', rep.series.time_ps, [
    { label: 'Ligand RMSD (Å)', data: rep.series.ligand_rmsd_A, color: '#34d399' },
    { label: 'COM Shift (Å)', data: rep.series.com_displacement_A, color: '#f43f5e', dashed: true },
    { label: 'H-Bonds', data: rep.series.hbond_count, color: '#38bdf8' }
  ]);
  drawBind3D();
}

function drawBind3D() {
  const d = STATE.bindData;
  if (!d || !d.viewer_frames.length) return;
  const fr = d.viewer_frames[STATE.frameIdx.bind % d.viewer_frames.length];
  renderMoleculeCanvas('canvasBind3D', fr, null, { roles: d.viewer_roles, camKey: 'bind' });
}

async function runElectrolyteLab() {
  const body = {
    mode: 'electrolyte',
    preset: document.getElementById('elecPreset').value,
    frames: Number(document.getElementById('elecFrames').value)
  };
  const r = await fetch('/api/triage', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
  });
  const res = await r.json();
  const rep = res.report;
  STATE.elecData = rep;
  STATE.frameIdx.elec = 0;
  const m = rep.metrics;
  document.getElementById('elecSummaryBox').innerHTML = `
    <div class="gate-row ${rep.verdict === 'GO' ? 'PASS' : (rep.verdict === 'HOLD' ? 'WARN' : 'FAIL')}">
      <div class="gate-head"><b>${rep.title}</b> <span class="pill ${rep.verdict}">${rep.verdict}</span></div>
      <div class="gate-meta">${rep.summary}</div>
    </div>`;
  document.getElementById('elecKpis').innerHTML = [
    ['Cluster σ_cNE', `${m.sigma_cNE_mS_cm} mS/cm`],
    ['Naive σ_NE', `${m.sigma_NE_mS_cm} mS/cm`],
    ['Haven Ionicity', m.ionicity_haven_inv],
    ['Cluster t+_cNE', m.transference_cNE],
    ['SSIP / CIP / AGG', `${(m.ssip_fraction*100).toFixed(0)}/${(m.cip_fraction*100).toFixed(0)}/${(m.agg_fraction*100).toFixed(0)}%`],
    ['Additive CAGS', m.cags_additive_score === null ? 'Base' : m.cags_additive_score]
  ].map(([k, v]) => `<div class="kpi"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
  drawLineChart('canvasElecChart', rep.series.rdf_r_A, [
    { label: 'RDF g_Li-O(r)', data: rep.series.rdf_g_li_o, color: '#c084fc' }
  ]);
  drawElec3D();
}

function drawElec3D() {
  const d = STATE.elecData;
  if (!d || !d.viewer_frames.length) return;
  const fr = d.viewer_frames[STATE.frameIdx.elec % d.viewer_frames.length];
  renderMoleculeCanvas('canvasElec3D', fr, null, { roles: d.viewer_roles, boxA: d.box_A, camKey: 'elec' });
}

function togglePlay(key) {
  STATE.playing[key] = !STATE.playing[key];
  const mapBtn = { comp: 'btnPlayComp', roll: 'btnPlayRoll' };
  if (mapBtn[key]) document.getElementById(mapBtn[key]).textContent = STATE.playing[key] ? '⏸ Play' : '▶ Play';
}

function populateBenchmarks(ov) {
  const b = ov.benchmarks;
  document.getElementById('md17Table').innerHTML = `
    <thead><tr><th>Molecule (Quantum Split)</th><th>Atoms</th><th>Naive VV 10 fs</th><th>WignerFlow 10 fs (20×)</th><th>WignerFlow 20 fs (40×)</th><th>MAE h(r)</th><th>VDOS Cos</th><th>COM Drift (Å/fs)</th><th>CPU Speed</th></tr></thead>
    <tbody>${b.pure_md_md17_md22.map(r => `
      <tr>
        <td><b>${r.system}</b></td><td>${r.atoms}</td>
        <td><span class="pill FAIL">${r.vv_10fs}</span></td>
        <td><span class="pill PASS">${r.wf_10fs}</span></td>
        <td><span class="pill PASS">${r.wf_20fs}</span></td>
        <td>${r.mae_hr}</td><td>${r.vdos_cos}</td><td><code>${r.com_drift}</code></td><td>${r.ns_day} ns/day</td>
      </tr>`).join('')}</tbody>`;

  const cal = b.calisol23_alignment;
  document.getElementById('calisolKpis').innerHTML = [
    ['Unseen Lab Tests', `${cal.n_formulations} 3D / ${cal.n_heldout_total} Total`],
    ['Rank Order (Before -> After)', `${cal.unaligned_spearman_rho} -> ${cal.aligned_spearman_rho}`],
    ['Avg Error MAE (Before -> After)', `${cal.unaligned_mae_mS_cm} -> ${cal.aligned_mae_mS_cm} mS/cm`],
    ['R² Fit (Raw -> Aligned)', `${cal.unaligned_r2} -> ${cal.aligned_r2}`]
  ].map(([k, v]) => `<div class="kpi"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');

  document.getElementById('calisolBoxTable').innerHTML = `
    <thead><tr><th>Box Atoms</th><th>Box Size L (Å)</th><th>CPU 50-Step Time</th><th>H200 200-ps Time</th><th>Speedup vs 12h H200 OpenMM</th></tr></thead>
    <tbody>${cal.box_scaling.map(x => `<tr><td>${x.atoms}</td><td>${x.box_A}</td><td>${x.cpu_wall_s} s</td><td><b>${x.h200_s} s</b></td><td><span class="pill PASS">${x.speedup_vs_h200_12h}</span></td></tr>`).join('')}</tbody>`;

  if (cal.parity_samples && document.getElementById('calisolParityTable')) {
    document.getElementById('calisolParityTable').innerHTML = `
      <thead><tr><th>Salt + Solvents</th><th>Conc (M)</th><th>Temp (K)</th><th>CALiSol-23 Lab (mS/cm)</th><th>Raw WignerFlow MD (mS/cm)</th><th>WignerFlow-Align MD (mS/cm)</th><th>Aligned Error</th></tr></thead>
      <tbody>${cal.parity_samples.map(p => {
        const err = (p.sigma_aligned - p.sigma_exp).toFixed(2);
        const sign = Number(err) >= 0 ? '+' : '';
        return `<tr>
          <td><b>${p.salt}</b> in ${p.solvents}</td>
          <td>${p.c_M.toFixed(2)}</td>
          <td>${p.T_K.toFixed(1)}</td>
          <td><b>${p.sigma_exp.toFixed(2)}</b></td>
          <td style="color:#fda4af;">${p.sigma_raw_ne.toFixed(2)}</td>
          <td style="color:#6ee7b7;font-weight:700;">${p.sigma_aligned.toFixed(2)}</td>
          <td><code>${sign}${err}</code></td>
        </tr>`;
      }).join('')}</tbody>`;
  }

  document.getElementById('solvationTable').innerHTML = `
    <thead><tr><th>Rank</th><th>Formulation</th><th>σ_298K</th><th>σ_253K</th><th>Speciation (SSIP / CIP / AGG-I / AGG-net)</th><th>τ_ex (ps)</th><th>Readiness</th><th>Regime</th></tr></thead>
    <tbody>${b.top15_solvation_champions.map(s => `
      <tr>
        <td><b>${s.rank}</b></td><td>${s.formula}</td><td>${s.sigma_298} mS/cm</td><td>${s.sigma_253} mS/cm</td>
        <td>
          <div class="bar-stack" title="SSIP ${s.ssip}% | CIP ${s.cip}% | AGG-I ${s.agg_i}% | AGG-net ${s.agg_net}%">
            <div style="width:${s.ssip}%;background:#38bdf8;"></div>
            <div style="width:${s.cip}%;background:#10b981;"></div>
            <div style="width:${s.agg_i}%;background:#f59e0b;"></div>
            <div style="width:${s.agg_net}%;background:#f43f5e;"></div>
          </div>
          <div style="font-size:11px;color:var(--muted);margin-top:2px;">${s.ssip}% SSIP · ${s.cip}% CIP · ${s.agg_net}% AGG-net</div>
        </td>
        <td>${s.tau_ex_ps}</td><td><b>${s.readiness}/100</b></td>
        <td><span class="pill ${s.regime.includes('GOLDILOCKS') ? 'PASS' : (s.regime.includes('RISK') ? 'FAIL' : 'WARN')}">${s.regime}</span></td>
      </tr>`).join('')}</tbody>`;

  document.getElementById('datasetTable').innerHTML = `
    <thead><tr><th>Domain</th><th>Dataset</th><th>One-Line Import Command</th></tr></thead>
    <tbody>${ov.datasets.map(d => `<tr><td><b>${d.domain}</b></td><td><a href="${d.url}" target="_blank" style="color:#60a5fa;">${d.title}</a></td><td><code>${d.import_hint}</code></td></tr>`).join('')}</tbody>`;
}

function exportSessionJson() {
  const payload = {
    exported_at: new Date().toISOString(),
    compliance: STATE.compData,
    rollout: STATE.rollData,
    binding: STATE.bindData,
    electrolyte: STATE.elecData
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'wignerflow_compliance_receipt.wignerstudio.json';
  a.click();
}

async function initStudio() {
  setupOrbit('canvasComp3D', 'comp', drawComp3D);
  setupOrbit('canvasRoll3D', 'roll', drawRoll3D);
  setupOrbit('canvasBind3D', 'bind', drawBind3D);
  setupOrbit('canvasElec3D', 'elec', drawElec3D);

  const r = await fetch('/api/overview');
  const ov = await r.json();
  STATE.overview = ov;
  document.getElementById('runtimeInfo').textContent =
    `Live Engine: Python ${ov.runtime.python} · PyTorch ${ov.runtime.torch} (${ov.runtime.cpu_threads} CPU threads)`;
  document.getElementById('customXyzInput').value = ov.default_xyz;
  renderFaultCatalog(ov.fault_catalog);
  populateBenchmarks(ov);
  await runCompliance();

  setInterval(() => {
    if (STATE.playing.comp && STATE.compData && STATE.compData.viewer_frames.length > 1 && !document.getElementById('chkTwinVec').checked) {
      STATE.frameIdx.comp = (STATE.frameIdx.comp + 1) % STATE.compData.viewer_frames.length;
      document.getElementById('compFrameSlider').value = STATE.frameIdx.comp;
      drawComp3D();
    }
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

window.addEventListener('DOMContentLoaded', initStudio);
</script>
</body>
</html>
"""


class WignerFlowStudioHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _set_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, code: int, payload: dict):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self._set_cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors()
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            raw = STUDIO_HTML.encode("utf-8")
            self.send_response(200)
            self._set_cors()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
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
