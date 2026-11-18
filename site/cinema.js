/* 60-fps Research Cinema & Animated Mechanism Diagrams for WignerFlow.
 * Replaces static Matplotlib plots with live WebGL2 3D liquid-box choreography
 * and crisp dark-mode vector mechanism animations (MolRSI style).
 */
(() => {
  'use strict';

  const CHAPTERS = [
    {
      t0: 0,
      t1: 9,
      tag: '01 · THE INTEGRATOR BOTTLENECK',
      headline: 'Classical MD inches 1–5 fs at a time. WignerFlow jumps 50 fs.',
      sub: 'Three context frames (t − 100 fs, t − 50 fs, t) directly predict frame t + 50 fs with zero force evaluations.',
    },
    {
      t0: 9,
      t1: 19,
      tag: '02 · SO(2) WIGNER-D EDGE ALIGNMENT',
      headline: 'Every neighbor pair rotates into its own equivariant edge frame.',
      sub: 'Spherical harmonics (l ≤ 2) align along r_ij via Wigner-D matrices; log(envelope) + zero-logit sink keeps cutoff crossings C² smooth.',
    },
    {
      t0: 19,
      t1: 30,
      tag: '03 · CONDITIONAL FLOW MATCHING',
      headline: 'Continuous ODE transport over residual displacements Δx − vΔt.',
      sub: 'Integrating s ∈ [0, 1] around constant-velocity coasting captures stochastic liquid cage rattling without damping kinetic energy.',
    },
    {
      t0: 30,
      t1: 40,
      tag: '04 · WHY RESIDUAL TARGETS MATTER',
      headline: 'Direct velocity targets freeze liquid argon. Residual flow stays liquid.',
      sub: 'Measured 10⁴-step rollouts (RESULTS.md): Direct Regression cools to 37.2 K (−60.8%); Residual Flow Seeds 3 & 4 hold 92.2–92.6 K vs 94.9 K truth.',
    },
    {
      t0: 40,
      t1: 50,
      tag: '05 · 10,000 STEPS WITHOUT AN INTEGRATOR',
      headline: '256 argon atoms · 0.5 ns model-only rollout · RDF & VACF preserved.',
      sub: '4 of 5 seeds finish 10⁴ autoregressive steps with zero crashes; RDF first-shell peak within 0.025 Å and VACF zero crossing within 0.6%.',
    },
  ];

  const TOTAL_DURATION = 50.0;

  function lerpFrame(fA, fB, alpha, boxA) {
    const out = new Array(fA.length);
    for (let i = 0; i < fA.length; i++) {
      let dx = fB[i][0] - fA[i][0];
      let dy = fB[i][1] - fA[i][1];
      let dz = fB[i][2] - fA[i][2];
      if (boxA) {
        dx -= boxA * Math.round(dx / boxA);
        dy -= boxA * Math.round(dy / boxA);
        dz -= boxA * Math.round(dz / boxA);
      }
      let x = fA[i][0] + dx * alpha;
      let y = fA[i][1] + dy * alpha;
      let z = fA[i][2] + dz * alpha;
      if (boxA) {
        x = ((x % boxA) + boxA) % boxA;
        y = ((y % boxA) + boxA) % boxA;
        z = ((z % boxA) + boxA) % boxA;
      }
      out[i] = [x, y, z];
    }
    return out;
  }

  function initHeroCinema(argonCinemaData) {
    const glCanvas = document.getElementById('cinema-gl-canvas');
    const hudCanvas = document.getElementById('cinema-hud-canvas');
    const toggleBtn = document.getElementById('cinema-play-toggle');
    const timeReadout = document.getElementById('cinema-time');
    const seekButtons = Array.from(document.querySelectorAll('[data-cinema-seek]'));
    if (!glCanvas || !hudCanvas || !argonCinemaData) return null;

    const viewer = new window.WignerGLViewer(glCanvas, {
      bg: [10 / 255, 15 / 255, 21 / 255],
      rotX: 0.30,
      rotY: -0.52,
      zoom: 1.06,
    });
    const ctx = hudCanvas.getContext('2d');
    const frames = argonCinemaData.frames;
    const elems = new Array(frames[0].length).fill('Ar');
    const boxA = argonCinemaData.box_A;
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');

    let playing = !reduced.matches;
    let tSec = 0.0;
    let lastTs = performance.now();

    function setChapterUI(t) {
      seekButtons.forEach((btn, idx) => {
        const ch = CHAPTERS[idx];
        const active = ch && t >= ch.t0 && t < ch.t1;
        btn.classList.toggle('active', !!active);
      });
      const mm = String(Math.floor(t / 60)).padStart(2, '0');
      const ss = String(Math.floor(t % 60)).padStart(2, '0');
      if (timeReadout) timeReadout.textContent = `${mm}:${ss} / 00:50`;
      if (toggleBtn) {
        toggleBtn.textContent = playing ? '❚❚ Pause' : '▶ Play';
        toggleBtn.setAttribute('aria-label', playing ? 'Pause cinema' : 'Play cinema');
      }
    }

    function drawMiniCurve(ctx2d, x, y, w, h, xs, ysGt, ysMod, label, unitLbl, progressFrac) {
      ctx2d.save();
      ctx2d.fillStyle = 'rgba(14, 22, 30, 0.88)';
      ctx2d.strokeStyle = '#263442';
      ctx2d.lineWidth = 1;
      ctx2d.beginPath();
      ctx2d.roundRect(x, y, w, h, 7);
      ctx2d.fill();
      ctx2d.stroke();

      ctx2d.fillStyle = '#aab6c4';
      ctx2d.font = '600 11px Inter, "Segoe UI", sans-serif';
      ctx2d.fillText(label, x + 12, y + 19);
      ctx2d.fillStyle = '#758698';
      ctx2d.font = '500 10.5px Inter, "Segoe UI", sans-serif';
      ctx2d.fillText(unitLbl, x + w - ctx2d.measureText(unitLbl).width - 12, y + 19);

      const px0 = x + 14, py0 = y + 30, pw = w - 28, ph = h - 42;
      let minY = Infinity, maxY = -Infinity;
      for (const v of ysGt) { if (v < minY) minY = v; if (v > maxY) maxY = v; }
      for (const v of ysMod) { if (v < minY) minY = v; if (v > maxY) maxY = v; }
      const span = Math.max(maxY - minY, 1e-4);

      const plotLine = (arr, color, width, dash, maxIdx) => {
        ctx2d.save();
        ctx2d.strokeStyle = color;
        ctx2d.lineWidth = width;
        ctx2d.setLineDash(dash);
        ctx2d.beginPath();
        const lim = Math.min(arr.length, maxIdx);
        for (let i = 0; i < lim; i++) {
          const ux = px0 + (i / Math.max(1, ysGt.length - 1)) * pw;
          const uy = py0 + ph - ((arr[i] - minY) / span) * ph;
          if (i === 0) ctx2d.moveTo(ux, uy); else ctx2d.lineTo(ux, uy);
        }
        ctx2d.stroke();
        ctx2d.restore();
      };

      plotLine(ysGt, '#91b6df', 1.7, [4, 3], ysGt.length);
      const revealCount = Math.max(2, Math.round(ysMod.length * Math.min(1, progressFrac)));
      plotLine(ysMod, '#b5d88e', 2.3, [], revealCount);
      ctx2d.restore();
    }

    function renderAt(t) {
      const chIdx = Math.min(CHAPTERS.length - 1, Math.max(0, CHAPTERS.findIndex(c => t >= c.t0 && t < c.t1)));
      const ch = CHAPTERS[chIdx === -1 ? 0 : chIdx];
      const local = (t - ch.t0) / Math.max(0.001, ch.t1 - ch.t0);

      // Slowly orbit camera unless user is dragging
      if (!viewer.dragging && playing) {
        viewer.rotY = -0.52 + t * 0.095;
        viewer.rotX = 0.28 + 0.08 * Math.sin(t * 0.35);
      }

      // Select molecular scene state per chapter
      const spanFrames = Math.max(1, frames.length - 1);
      const fFloat = ((t * 2.2) % spanFrames + spanFrames) % spanFrames;
      const fi = Math.min(frames.length - 2, Math.floor(fFloat));
      const frac = fFloat - fi;
      const curPos = lerpFrame(frames[fi], frames[fi + 1], frac, boxA);

      if (ch.t0 === 0) {
        // Chapter 1: 3-frame ghost context jumping 50 fs forward
        viewer.zoom = 1.06;
        const baseIdx = 2;
        const jumpAlpha = Math.min(1, (local * 1.6) % 1.0);
        const posNow = lerpFrame(frames[baseIdx], frames[baseIdx + 1], jumpAlpha, boxA);
        viewer.setMolecule({
          positions: posNow,
          elements: elems,
          boxA,
          ghostFrames: [frames[0], frames[1]],
          atomScale: 0.92,
        });
      } else if (ch.t0 === 9) {
        // Chapter 2: Highlight central atom #0 coordination shell inside r_c = 6.0 A
        viewer.zoom = 1.32;
        viewer.setMolecule({
          positions: curPos,
          elements: elems,
          boxA,
          highlightAtom: 0,
          cutoffA: 6.2,
          atomScale: 0.96,
        });
      } else if (ch.t0 === 19) {
        // Chapter 3: Flow ODE velocity arrows s in [0, 1]
        viewer.zoom = 1.16;
        const sOde = (local * 2.0) % 1.0;
        const fStart = frames[3], fEnd = frames[4];
        const odePos = lerpFrame(fStart, fEnd, sOde, boxA);
        const arrows = [];
        for (let i = 0; i < Math.min(36, fStart.length); i++) {
          let dx = fEnd[i][0] - fStart[i][0];
          let dy = fEnd[i][1] - fStart[i][1];
          let dz = fEnd[i][2] - fStart[i][2];
          dx -= boxA * Math.round(dx / boxA);
          dy -= boxA * Math.round(dy / boxA);
          dz -= boxA * Math.round(dz / boxA);
          arrows.push({
            from: odePos[i],
            vec: [dx, dy, dz],
            scale: 2.2,
            radius: 0.042,
            color: [0.71, 0.85, 0.56, 1.0],
          });
        }
        viewer.setMolecule({ positions: odePos, elements: elems, boxA, arrows, atomScale: 0.9 });
      } else {
        // Chapters 4 & 5: Continuous liquid argon rollout
        viewer.zoom = 1.05;
        viewer.setMolecule({ positions: curPos, elements: elems, boxA, atomScale: 0.94 });
      }

      // Render crisp vector-grade HUD overlay on hudCanvas
      const rect = hudCanvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const cw = Math.max(2, Math.round((rect.width || 960) * dpr));
      const chPx = Math.max(2, Math.round((rect.height || 540) * dpr));
      if (hudCanvas.width !== cw || hudCanvas.height !== chPx) {
        hudCanvas.width = cw;
        hudCanvas.height = chPx;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const w = cw / dpr, h = chPx / dpr;
      ctx.clearRect(0, 0, w, h);

      // Subtle left/bottom vignette so typography is always razor-sharp
      const grad = ctx.createLinearGradient(0, h * 0.52, 0, h);
      grad.addColorStop(0, 'rgba(10, 15, 21, 0.0)');
      grad.addColorStop(1, 'rgba(10, 15, 21, 0.88)');
      ctx.fillStyle = grad;
      ctx.fillRect(0, h * 0.52, w, h * 0.48);

      const topGrad = ctx.createLinearGradient(0, 0, 0, 96);
      topGrad.addColorStop(0, 'rgba(10, 15, 21, 0.78)');
      topGrad.addColorStop(1, 'rgba(10, 15, 21, 0.0)');
      ctx.fillStyle = topGrad;
      ctx.fillRect(0, 0, w, 96);

      // Top-left chapter pill
      ctx.fillStyle = '#b5d88e';
      ctx.font = '650 11.5px Inter, "Segoe UI", sans-serif';
      ctx.fillText(ch.tag, 26, 32);

      const badgeRight = '256 Ar · box 23.12 Å · Δτ = 50 fs · drag to orbit 3D';
      ctx.fillStyle = '#8b9cad';
      ctx.font = '500 11.5px Inter, "Segoe UI", sans-serif';
      if (w > 640) {
        ctx.fillText(badgeRight, w - ctx.measureText(badgeRight).width - 26, 32);
      }

      // Chapter-specific scientific HUD cards on right side (desktop)
      if (w > 720) {
        if (ch.t0 === 19) {
          const sOde = ((local * 2.0) % 1.0);
          ctx.fillStyle = 'rgba(14, 22, 30, 0.88)';
          ctx.strokeStyle = '#2b3947';
          ctx.beginPath(); ctx.roundRect(w - 280, 56, 254, 88, 8); ctx.fill(); ctx.stroke();
          ctx.fillStyle = '#b9a1de';
          ctx.font = '600 11px Inter, sans-serif';
          ctx.fillText('FLOW ODE INTEGRATION  s ∈ [0, 1]', w - 264, 78);
          ctx.fillStyle = '#edf1f7';
          ctx.font = '600 22px Inter, sans-serif';
          ctx.fillText(`s = ${sOde.toFixed(2)}`, w - 264, 106);
          ctx.fillStyle = '#22303d';
          ctx.fillRect(w - 264, 118, 222, 6);
          ctx.fillStyle = '#b5d88e';
          ctx.fillRect(w - 264, 118, 222 * sOde, 6);
        } else if (ch.t0 === 30) {
          ctx.fillStyle = 'rgba(14, 22, 30, 0.90)';
          ctx.strokeStyle = '#2b3947';
          ctx.beginPath(); ctx.roundRect(w - 306, 54, 280, 134, 8); ctx.fill(); ctx.stroke();
          ctx.fillStyle = '#aab6c4';
          ctx.font = '600 11px Inter, sans-serif';
          ctx.fillText('10⁴-STEP EQUILIBRIUM TEMPERATURE', w - 290, 76);

          ctx.fillStyle = '#b5d88e';
          ctx.font = '650 24px Inter, sans-serif';
          ctx.fillText('92.6 K', w - 290, 108);
          ctx.fillStyle = '#96a6b7';
          ctx.font = '500 12px Inter, sans-serif';
          ctx.fillText('Residual Flow (Seed 4 · −2.4% vs 94.9 K)', w - 204, 105);

          ctx.fillStyle = '#d97757';
          ctx.font = '650 24px Inter, sans-serif';
          ctx.fillText('37.2 K', w - 290, 146);
          ctx.fillStyle = '#96a6b7';
          ctx.font = '500 12px Inter, sans-serif';
          ctx.fillText('Direct Target (−60.8% systematic freeze)', w - 204, 143);
        } else if (ch.t0 === 40) {
          drawMiniCurve(
            ctx, w - 292, 50, 266, 106,
            argonCinemaData.rdf.r_A, argonCinemaData.rdf.gt, argonCinemaData.rdf.model,
            'RDF g(r)', 'truth (blue) vs model (lime)', Math.min(1, local * 1.4),
          );
          drawMiniCurve(
            ctx, w - 292, 166, 266, 106,
            argonCinemaData.vacf.t_fs, argonCinemaData.vacf.gt, argonCinemaData.vacf.model,
            'VACF C(t)/C(0)', 'zero @ 306 fs vs 307 fs', Math.min(1, local * 1.4),
          );
        }
      }

      // Bottom editorial caption bar inside the cinema frame
      const baseY = h - 56;
      ctx.fillStyle = '#edf1f7';
      ctx.font = `600 ${w < 600 ? 15 : 20}px Inter, "Segoe UI", sans-serif`;
      ctx.fillText(ch.headline, 26, baseY);
      ctx.fillStyle = '#aab6c4';
      ctx.font = `400 ${w < 600 ? 11.5 : 13.5}px Inter, "Segoe UI", sans-serif`;
      ctx.fillText(ch.sub, 26, baseY + 23, w - 52);

      // Thin progress bar along bottom edge
      ctx.fillStyle = 'rgba(255,255,255,0.09)';
      ctx.fillRect(0, h - 3, w, 3);
      ctx.fillStyle = '#b5d88e';
      ctx.fillRect(0, h - 3, w * (t / TOTAL_DURATION), 3);

      setChapterUI(t);
    }

    function tick(now) {
      const dt = Math.min(0.1, (now - lastTs) / 1000);
      lastTs = now;
      if (playing && !document.hidden) {
        tSec = (tSec + dt) % TOTAL_DURATION;
        renderAt(tSec);
      }
      requestAnimationFrame(tick);
    }

    if (toggleBtn) {
      toggleBtn.addEventListener('click', () => {
        playing = !playing;
        lastTs = performance.now();
        renderAt(tSec);
      });
    }
    seekButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        tSec = Number(btn.dataset.cinemaSeek || 0);
        lastTs = performance.now();
        renderAt(tSec);
      });
    });

    window.addEventListener('resize', () => renderAt(tSec), { passive: true });
    renderAt(0);
    requestAnimationFrame(tick);
    return { seek: s => { tSec = s; renderAt(tSec); }, viewer };
  }

  /**
   * Scroll-activated 16:9 mechanism animations in Section 01 / The Method
   * (matching MolRSI's dual-loop-memory & search-policy clean vector aesthetic).
   */
  function initMechanismAnimations() {
    const c1 = document.getElementById('mech-wigner-canvas');
    const c2 = document.getElementById('mech-flow-canvas');
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    const state = new Map();

    function setupCanvas(canvas, drawFn) {
      if (!canvas) return;
      const id = canvas.id;
      const btn = document.querySelector(`[data-mech-toggle="${id}"]`);
      const item = { canvas, ctx: canvas.getContext('2d'), playing: !reduced.matches, visible: true, t: 0, drawFn, btn };
      state.set(id, item);
      if (btn) {
        btn.addEventListener('click', () => {
          item.playing = !item.playing;
          btn.textContent = item.playing ? '❚❚' : '▶';
          btn.title = item.playing ? 'Pause animation' : 'Play animation';
        });
      }
      drawFn(item.ctx, canvas, 0.35);
    }

    // Mechanism 1: Edge-Aligned SO(2) Wigner-D + Smooth Cutoff Sink
    setupCanvas(c1, (ctx, canvas, t) => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const W = 960, H = 540;
      const pw = Math.max(2, Math.round((rect.width || 640) * dpr));
      const ph = Math.max(2, Math.round((rect.height || 360) * dpr));
      if (canvas.width !== pw || canvas.height !== ph) { canvas.width = pw; canvas.height = ph; }
      ctx.setTransform(pw / W, 0, 0, ph / H, 0, 0);

      ctx.fillStyle = '#0a0f15';
      ctx.fillRect(0, 0, W, H);

      // Header
      ctx.fillStyle = '#79c2bd';
      ctx.font = '650 13px Inter, sans-serif';
      ctx.fillText('WIGNERFLOW · SPATIAL GRAPH ATTENTION', 42, 46);
      ctx.fillStyle = '#edf1f7';
      ctx.font = '600 28px Inter, sans-serif';
      ctx.fillText('Wigner-D edge alignment + smooth cutoff sink', 42, 84);

      // Left orbit: Coordination shell r_c = 6.0 A
      const cx = 275, cy = 305, rc = 152;
      ctx.strokeStyle = '#2d3e4f';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 6]);
      ctx.beginPath(); ctx.arc(cx, cy, rc, 0, Math.PI * 2); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#8093a6';
      ctx.font = '500 13px Inter, sans-serif';
      ctx.fillText('cutoff r_c = 6.0 Å', cx - 52, cy - rc - 12);

      // Central destination atom i
      const g0 = ctx.createRadialGradient(cx - 6, cy - 6, 2, cx, cy, 22);
      g0.addColorStop(0, '#e5f5d3'); g0.addColorStop(0.55, '#b5d88e'); g0.addColorStop(1, '#435c2f');
      ctx.fillStyle = g0;
      ctx.beginPath(); ctx.arc(cx, cy, 22, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#0a0f15';
      ctx.font = '700 13px Inter, sans-serif';
      ctx.fillText('dst i', cx - 14, cy + 4);

      // 5 neighbor atoms + 1 crossing r_c smoothly
      const baseAngles = [-0.85, 0.35, 1.45, 2.45, -2.15];
      const baseR = [84, 108, 96, 122, 78];
      baseAngles.forEach((ang, idx) => {
        const a = ang + 0.14 * Math.sin(t * 0.9 + idx);
        const r = baseR[idx] + 6 * Math.cos(t * 1.1 + idx);
        const nx = cx + Math.cos(a) * r, ny = cy + Math.sin(a) * r;
        const env = Math.max(0, 1 - Math.pow(r / rc, 3));
        ctx.strokeStyle = `rgba(121, 194, 189, ${0.25 + 0.65 * env})`;
        ctx.lineWidth = 1.2 + 2.4 * env;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(nx, ny); ctx.stroke();

        // Traveling message packet along edge
        const pFrac = (t * 0.55 + idx * 0.21) % 1.0;
        const mx = nx + (cx - nx) * pFrac, my = ny + (cy - ny) * pFrac;
        ctx.fillStyle = '#b5d88e';
        ctx.beginPath(); ctx.arc(mx, my, 3.6, 0, Math.PI * 2); ctx.fill();

        ctx.fillStyle = '#91b6df';
        ctx.beginPath(); ctx.arc(nx, ny, 11, 0, Math.PI * 2); ctx.fill();
      });

      // Crossing atom oscillating across r_c = 6.0 A
      const rCross = rc * (0.80 + 0.25 * (0.5 + 0.5 * Math.sin(t * 1.25)));
      const aCross = 0.78;
      const kx = cx + Math.cos(aCross) * rCross, ky = cy + Math.sin(aCross) * rCross;
      const envCross = rCross < rc ? Math.pow(1 - rCross / rc, 2) : 0;
      if (envCross > 0.005) {
        ctx.strokeStyle = `rgba(185, 161, 222, ${envCross})`;
        ctx.lineWidth = 3.0 * envCross + 0.5;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(kx, ky); ctx.stroke();
      }
      ctx.fillStyle = envCross > 0.01 ? '#b9a1de' : '#435160';
      ctx.beginPath(); ctx.arc(kx, ky, 11, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#b9a1de';
      ctx.font = '500 12.5px Inter, sans-serif';
      ctx.fillText(`crossing pair · envelope = ${envCross.toFixed(2)}`, cx - 94, cy + rc + 28);

      // Right panel: Wigner-D l=0,1,2 + Softmax Sink Equation Card
      const rx = 500, ry = 132, rw = 420, rh = 344;
      ctx.fillStyle = '#101820';
      ctx.strokeStyle = '#263340';
      ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.roundRect(rx, ry, rw, rh, 10); ctx.fill(); ctx.stroke();

      ctx.fillStyle = '#b5d88e';
      ctx.font = '600 12px Inter, sans-serif';
      ctx.fillText('1 · ROTATE TO EDGE Z-AXIS:  D⁽ˡ⁾(α, β, γ)', rx + 22, ry + 34);

      // Draw l=0, l=1, l=2 harmonic lobes
      const lobes = [
        { lbl: 'l = 0 (scalar)', x: rx + 76, col: '#79c2bd', draw: (ox, oy) => { ctx.beginPath(); ctx.arc(ox, oy, 18, 0, Math.PI * 2); ctx.stroke(); } },
        { lbl: 'l = 1 (vector)', x: rx + 208, col: '#91b6df', draw: (ox, oy) => {
          ctx.beginPath(); ctx.ellipse(ox, oy - 11, 10, 13, 0, 0, Math.PI * 2); ctx.stroke();
          ctx.beginPath(); ctx.ellipse(ox, oy + 11, 10, 13, 0, 0, Math.PI * 2); ctx.stroke();
        } },
        { lbl: 'l = 2 (quadrupole)', x: rx + 340, col: '#b9a1de', draw: (ox, oy) => {
          for (let k = 0; k < 4; k++) {
            ctx.save(); ctx.translate(ox, oy); ctx.rotate(k * Math.PI * 0.5 + 0.25 * Math.sin(t));
            ctx.beginPath(); ctx.ellipse(11, 0, 11, 5.5, 0, 0, Math.PI * 2); ctx.stroke();
            ctx.restore();
          }
        } },
      ];
      for (const lb of lobes) {
        ctx.strokeStyle = lb.col;
        ctx.lineWidth = 2.0;
        lb.draw(lb.x, ry + 84);
        ctx.fillStyle = '#aab6c4';
        ctx.font = '500 12px Inter, sans-serif';
        ctx.fillText(lb.lbl, lb.x - ctx.measureText(lb.lbl).width * 0.5, ry + 132);
      }

      ctx.strokeStyle = '#22303c';
      ctx.beginPath(); ctx.moveTo(rx + 22, ry + 156); ctx.lineTo(rx + rw - 22, ry + 156); ctx.stroke();

      ctx.fillStyle = '#b5d88e';
      ctx.font = '600 12px Inter, sans-serif';
      ctx.fillText('2 · LOG-ENVELOPE + ZERO-LOGIT SINK (MAIN)', rx + 22, ry + 186);

      ctx.fillStyle = '#edf1f7';
      ctx.font = '500 13.2px "SFMono-Regular", Consolas, monospace';
      ctx.fillText('α_ij = env(d_ij)·exp(z_ij) / [1 + Σ_k env(d_ik)·exp(z_ik)]', rx + 22, ry + 216);

      // Smooth envelope curve bar
      const gx = rx + 24, gy = ry + 244, gw = rw - 48, gh = 68;
      ctx.fillStyle = '#0b1118';
      ctx.beginPath(); ctx.roundRect(gx, gy, gw, gh, 6); ctx.fill();
      ctx.strokeStyle = '#d97757';
      ctx.lineWidth = 1.6;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(gx + 14, gy + 18); ctx.lineTo(gx + gw * 0.76, gy + 22); ctx.lineTo(gx + gw * 0.76, gy + gh - 12); ctx.lineTo(gx + gw - 14, gy + gh - 12);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.strokeStyle = '#b5d88e';
      ctx.lineWidth = 2.6;
      ctx.beginPath();
      for (let s = 0; s <= 60; s++) {
        const u = s / 60;
        const px = gx + 14 + u * (gw - 28);
        const v = u < 0.76 ? Math.pow(1 - Math.pow(u / 0.76, 2), 3) : 0;
        const py = gy + gh - 12 - v * (gh - 30);
        if (s === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.fillStyle = '#b5d88e';
      ctx.font = '500 11.5px Inter, sans-serif';
      ctx.fillText('main: O(h³) smooth', gx + 20, gy + 54);
      ctx.fillStyle = '#d97757';
      ctx.fillText('v0: step jump @ r_c', gx + gw - 134, gy + 26);
    });

    // Mechanism 2: 3-Frame Context -> Residual Flow ODE -> Unroll Curriculum m: 2 -> 3 -> 4
    setupCanvas(c2, (ctx, canvas, t) => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const W = 960, H = 500;
      const pw = Math.max(2, Math.round((rect.width || 640) * dpr));
      const ph = Math.max(2, Math.round((rect.height || 340) * dpr));
      if (canvas.width !== pw || canvas.height !== ph) { canvas.width = pw; canvas.height = ph; }
      ctx.setTransform(pw / W, 0, 0, ph / H, 0, 0);

      ctx.fillStyle = '#0a0f15';
      ctx.fillRect(0, 0, W, H);

      ctx.fillStyle = '#b9a1de';
      ctx.font = '650 13px Inter, sans-serif';
      ctx.fillText('WIGNERFLOW · TEMPORAL FUSION & ANTI-DRIFT POST-TRAINING', 42, 44);
      ctx.fillStyle = '#edf1f7';
      ctx.font = '600 27px Inter, sans-serif';
      ctx.fillText('Residual flow matching + unroll curriculum (m: 2 → 3 → 4)', 42, 80);

      // Top pipeline cards
      const stages = [
        { x: 42, w: 196, col: '#91b6df', title: 'k = 3 Context Frames', sub1: 't − 100, t − 50, t fs', sub2: 'Positions X + Velocities V' },
        { x: 274, w: 204, col: '#79c2bd', title: 'Causal Attention', sub1: 'Invariant l = 0 weights', sub2: 'Equivariant l ≤ 2 values' },
        { x: 514, w: 204, col: '#b9a1de', title: 'Flow ODE  s ∈ [0, 1]', sub1: 'dz/ds = u_θ(z_s, s | c)', sub2: '10 Euler steps at test' },
        { x: 754, w: 164, col: '#b5d88e', title: 't + 50 fs Frame', sub1: 'Δx = v·Δt + a_θ', sub2: 'v_new = v + b_θ' },
      ];
      const cy = 122, ch = 120;
      stages.forEach((st, idx) => {
        ctx.fillStyle = '#101820';
        ctx.strokeStyle = st.col;
        ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.roundRect(st.x, cy, st.w, ch, 9); ctx.fill(); ctx.stroke();

        ctx.fillStyle = st.col;
        ctx.font = '650 14.5px Inter, sans-serif';
        ctx.fillText(st.title, st.x + 16, cy + 34);
        ctx.fillStyle = '#edf1f7';
        ctx.font = '500 13px Inter, sans-serif';
        ctx.fillText(st.sub1, st.x + 16, cy + 66);
        ctx.fillStyle = '#93a4b5';
        ctx.font = '400 12px Inter, sans-serif';
        ctx.fillText(st.sub2, st.x + 16, cy + 92);

        if (idx < stages.length - 1) {
          const ax0 = st.x + st.w + 4, ax1 = stages[idx + 1].x - 4, ay = cy + ch * 0.5;
          ctx.strokeStyle = '#47596b';
          ctx.lineWidth = 1.8;
          ctx.beginPath(); ctx.moveTo(ax0, ay); ctx.lineTo(ax1, ay); ctx.stroke();
          const pulse = (t * 0.7 - idx * 0.28 + 10) % 1.0;
          ctx.fillStyle = '#b5d88e';
          ctx.beginPath(); ctx.arc(ax0 + (ax1 - ax0) * pulse, ay, 4, 0, Math.PI * 2); ctx.fill();
        }
      });

      // Autoregressive feedback loop arc back to k=3 context
      ctx.strokeStyle = 'rgba(181, 216, 142, 0.48)';
      ctx.lineWidth = 1.6;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(836, cy + ch);
      ctx.bezierCurveTo(836, cy + ch + 30, 140, cy + ch + 30, 140, cy + ch);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#b5d88e';
      ctx.font = '500 12px Inter, sans-serif';
      ctx.fillText('autoregressive rollout feeds predicted frame back into context window', 284, cy + ch + 52);

      // Bottom stability alignment strip
      const by = 324, bh = 138;
      ctx.fillStyle = '#0e161e';
      ctx.strokeStyle = '#25323f';
      ctx.beginPath(); ctx.roundRect(42, by, 876, bh, 10); ctx.fill(); ctx.stroke();

      const pills = [
        ['Unroll 2 → 3 → 4', 'Gradual horizon switch + 1000-step LR re-warmup', '#b5d88e'],
        ['All-Step Loss [1,1,1,1]', 'Supervises every unrolled step, preventing m=4 jump', '#91b6df'],
        ['RepulsionPrior + Overlap', 'Soft steric barrier below 0.95 r_min stops clashes', '#79c2bd'],
        ['SpikeGuard + Micro-Batch', 'Skips >5× median loss spikes; accumulates m=4 grads', '#b9a1de'],
      ];
      pills.forEach((p, i) => {
        const px = 64 + i * 214;
        ctx.fillStyle = p[2];
        ctx.fillRect(px, by + 26, 3, 88);
        ctx.font = '650 14px Inter, sans-serif';
        ctx.fillText(p[0], px + 14, by + 46);
        ctx.fillStyle = '#9fb0c1';
        ctx.font = '400 12.5px Inter, sans-serif';
        const words = p[1].split(' ');
        let line = '', ly = by + 74;
        for (const w of words) {
          if ((line + w).length > 26) {
            ctx.fillText(line.trim(), px + 14, ly);
            line = w + ' '; ly += 20;
          } else { line += w + ' '; }
        }
        if (line.trim()) ctx.fillText(line.trim(), px + 14, ly);
      });
    });

    if ('IntersectionObserver' in window) {
      const obs = new IntersectionObserver(entries => {
        for (const e of entries) {
          const item = state.get(e.target.id);
          if (item) item.visible = e.isIntersecting;
        }
      }, { threshold: 0.15 });
      state.forEach(item => obs.observe(item.canvas));
    }

    let prev = performance.now();
    function loop(now) {
      const dt = Math.min(0.1, (now - prev) / 1000);
      prev = now;
      if (!document.hidden) {
        state.forEach(item => {
          if (item.playing && item.visible) {
            item.t += dt;
            item.drawFn(item.ctx, item.canvas, item.t);
          }
        });
      }
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
    window.addEventListener('resize', () => {
      state.forEach(item => item.drawFn(item.ctx, item.canvas, item.t));
    }, { passive: true });
  }

  window.WignerCinema = { initHeroCinema, initMechanismAnimations };
})();
