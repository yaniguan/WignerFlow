/* 60-fps Research Cinema & Animated Mechanism Diagrams for WignerFlow.
 * Theme-aware: renders crisp Porcelain Daylight visuals by default and adapts
 * cleanly when switched to Obsidian Dark Mode.
 */
(() => {
  'use strict';

  const CHAPTERS = [
    {
      t0: 0,
      t1: 10,
      tag: '01 · COARSE-STRIDE JUMP',
      headline: 'Three frames in. One 50 fs jump ahead.',
      sub: 'Context frames (t − 100, t − 50, t fs) predict frame t + 50 fs directly with zero force evaluations.',
    },
    {
      t0: 10,
      t1: 20,
      tag: '02 · SO(2) WIGNER-D SHELL',
      headline: 'Every neighbor pair aligns into its own equivariant edge frame.',
      sub: 'Spherical harmonics (l ≤ 2) rotate via Wigner-D matrices; log(envelope) + zero-logit sink keeps cutoffs smooth.',
    },
    {
      t0: 20,
      t1: 30,
      tag: '03 · RESIDUAL FLOW ODE',
      headline: 'Flow matching over coasting residuals Δx − vΔt.',
      sub: 'S = 2 Euler steps on CPU match S = 10 rollout structure while keeping 108-atom steps at 282 ms.',
    },
    {
      t0: 30,
      t1: 40,
      tag: '04 · WHY RESIDUAL TARGETS MATTER',
      headline: 'Direct targets freeze liquid argon. Residual flow stays liquid.',
      sub: '10⁴-step test rollouts: Direct Regression cools to 37.2 K (−60.8%); Residual Flow Seeds 3 & 4 hold 92.2–92.6 K.',
    },
    {
      t0: 40,
      t1: 50,
      tag: '05 · 10,000 STEPS WITHOUT AN INTEGRATOR',
      headline: '256 argon atoms · 0.5 ns model-only rollout · RDF & VACF preserved.',
      sub: '4 of 5 seeds survive all 10⁴ steps; RDF peak within 0.025 Å and VACF zero crossing within 0.6%.',
    },
  ];

  const TOTAL_DURATION = 50.0;

  function themeColors() {
    const dark = document.documentElement.dataset.theme === 'dark';
    if (dark) {
      return {
        dark: true,
        stageBg: '#0c131c',
        cardBg: 'rgba(15, 23, 34, 0.90)',
        cardSolid: '#111b27',
        cardInset: '#0b1119',
        border: '#253446',
        ink: '#f1f5f9',
        muted: '#94a3b8',
        subtle: '#64748b',
        accent: '#2dd4bf',
        blue: '#60a5fa',
        purple: '#a78bfa',
        orange: '#fb7185',
        topGrad0: 'rgba(11, 17, 24, 0.80)',
        topGrad1: 'rgba(11, 17, 24, 0.0)',
        botGrad0: 'rgba(11, 17, 24, 0.0)',
        botGrad1: 'rgba(11, 17, 24, 0.90)',
      };
    }
    return {
      dark: false,
      stageBg: '#f8fafc',
      cardBg: 'rgba(255, 255, 255, 0.92)',
      cardSolid: '#ffffff',
      cardInset: '#f1f5f9',
      border: '#cbd5e1',
      ink: '#0f172a',
      muted: '#475569',
      subtle: '#64748b',
      accent: '#0d9488',
      blue: '#2563eb',
      purple: '#7c3aed',
      orange: '#e11d48',
      topGrad0: 'rgba(243, 246, 251, 0.88)',
      topGrad1: 'rgba(243, 246, 251, 0.0)',
      botGrad0: 'rgba(243, 246, 251, 0.0)',
      botGrad1: 'rgba(243, 246, 251, 0.95)',
    };
  }

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

  let cinemaController = null;
  let mechRefreshAll = () => {};

  function initHeroCinema(argonCinemaData) {
    const glCanvas = document.getElementById('cinema-gl-canvas');
    const hudCanvas = document.getElementById('cinema-hud-canvas');
    const toggleBtn = document.getElementById('cinema-play-toggle');
    const timeReadout = document.getElementById('cinema-time');
    const seekButtons = Array.from(document.querySelectorAll('[data-cinema-seek]'));
    if (!glCanvas || !hudCanvas || !argonCinemaData) return null;

    const viewer = new window.WignerGLViewer(glCanvas, {
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

    function drawMiniCurve(ctx2d, C, x, y, w, h, ysGt, ysMod, label, unitLbl, progressFrac) {
      ctx2d.save();
      ctx2d.fillStyle = C.cardBg;
      ctx2d.strokeStyle = C.border;
      ctx2d.lineWidth = 1;
      ctx2d.beginPath();
      ctx2d.roundRect(x, y, w, h, 8);
      ctx2d.fill();
      ctx2d.stroke();

      ctx2d.fillStyle = C.ink;
      ctx2d.font = '650 11px Inter, "Segoe UI", sans-serif';
      ctx2d.fillText(label, x + 12, y + 19);
      ctx2d.fillStyle = C.muted;
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

      plotLine(ysGt, C.blue, 1.8, [4, 3], ysGt.length);
      const revealCount = Math.max(2, Math.round(ysMod.length * Math.min(1, progressFrac)));
      plotLine(ysMod, C.accent, 2.4, [], revealCount);
      ctx2d.restore();
    }

    function renderAt(t) {
      const C = themeColors();
      const chIdx = Math.min(CHAPTERS.length - 1, Math.max(0, CHAPTERS.findIndex(c => t >= c.t0 && t < c.t1)));
      const ch = CHAPTERS[chIdx === -1 ? 0 : chIdx];
      const local = (t - ch.t0) / Math.max(0.001, ch.t1 - ch.t0);

      if (!viewer.dragging && playing) {
        viewer.rotY = -0.52 + t * 0.095;
        viewer.rotX = 0.28 + 0.08 * Math.sin(t * 0.35);
      }

      const spanFrames = Math.max(1, frames.length - 1);
      const fFloat = ((t * 2.2) % spanFrames + spanFrames) % spanFrames;
      const fi = Math.min(frames.length - 2, Math.floor(fFloat));
      const frac = fFloat - fi;
      const curPos = lerpFrame(frames[fi], frames[fi + 1], frac, boxA);

      if (ch.t0 === 0) {
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
      } else if (ch.t0 === 10) {
        viewer.zoom = 1.32;
        viewer.setMolecule({
          positions: curPos,
          elements: elems,
          boxA,
          highlightAtom: 0,
          cutoffA: 6.2,
          atomScale: 0.96,
        });
      } else if (ch.t0 === 20) {
        viewer.zoom = 1.16;
        const sOde = (local * 2.0) % 1.0;
        const fStart = frames[3], fEnd = frames[4];
        const odePos = lerpFrame(fStart, fEnd, sOde, boxA);
        const arrows = [];
        const arrCol = C.dark ? [0.18, 0.83, 0.75, 1.0] : [0.05, 0.58, 0.53, 1.0];
        for (let i = 0; i < Math.min(36, fStart.length); i++) {
          let dx = fEnd[i][0] - fStart[i][0];
          let dy = fEnd[i][1] - fStart[i][1];
          let dz = fEnd[i][2] - fStart[i][2];
          dx -= boxA * Math.round(dx / boxA);
          dy -= boxA * Math.round(dy / boxA);
          dz -= boxA * Math.round(dz / boxA);
          arrows.push({ from: odePos[i], vec: [dx, dy, dz], scale: 2.2, radius: 0.044, color: arrCol });
        }
        viewer.setMolecule({ positions: odePos, elements: elems, boxA, arrows, atomScale: 0.9 });
      } else {
        viewer.zoom = 1.05;
        viewer.setMolecule({ positions: curPos, elements: elems, boxA, atomScale: 0.94 });
      }

      const rect = hudCanvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const cw = Math.max(2, Math.round((rect.width || 960) * dpr));
      const chPx = Math.max(2, Math.round((rect.height || 500) * dpr));
      if (hudCanvas.width !== cw || hudCanvas.height !== chPx) {
        hudCanvas.width = cw;
        hudCanvas.height = chPx;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const w = cw / dpr, h = chPx / dpr;
      ctx.clearRect(0, 0, w, h);

      const grad = ctx.createLinearGradient(0, h * 0.58, 0, h);
      grad.addColorStop(0, C.botGrad0);
      grad.addColorStop(1, C.botGrad1);
      ctx.fillStyle = grad;
      ctx.fillRect(0, h * 0.58, w, h * 0.42);

      const topGrad = ctx.createLinearGradient(0, 0, 0, 78);
      topGrad.addColorStop(0, C.topGrad0);
      topGrad.addColorStop(1, C.topGrad1);
      ctx.fillStyle = topGrad;
      ctx.fillRect(0, 0, w, 78);

      ctx.fillStyle = C.accent;
      ctx.font = '700 11.5px Inter, "Segoe UI", sans-serif';
      ctx.fillText(ch.tag, 24, 30);

      const badgeRight = '256 Ar · 23.12 Å box · Δτ = 50 fs · drag to orbit 3D';
      ctx.fillStyle = C.muted;
      ctx.font = '500 11.5px Inter, "Segoe UI", sans-serif';
      if (w > 640) {
        ctx.fillText(badgeRight, w - ctx.measureText(badgeRight).width - 24, 30);
      }

      if (w > 720) {
        if (ch.t0 === 20) {
          const sOde = ((local * 2.0) % 1.0);
          ctx.fillStyle = C.cardBg;
          ctx.strokeStyle = C.border;
          ctx.beginPath(); ctx.roundRect(w - 278, 50, 254, 86, 8); ctx.fill(); ctx.stroke();
          ctx.fillStyle = C.purple;
          ctx.font = '650 11px Inter, sans-serif';
          ctx.fillText('FLOW ODE TRANSPORT  s ∈ [0, 1]', w - 262, 72);
          ctx.fillStyle = C.ink;
          ctx.font = '650 21px Inter, sans-serif';
          ctx.fillText(`s = ${sOde.toFixed(2)}`, w - 262, 100);
          ctx.fillStyle = C.cardInset;
          ctx.fillRect(w - 262, 112, 222, 6);
          ctx.fillStyle = C.accent;
          ctx.fillRect(w - 262, 112, 222 * sOde, 6);
        } else if (ch.t0 === 30) {
          ctx.fillStyle = C.cardBg;
          ctx.strokeStyle = C.border;
          ctx.beginPath(); ctx.roundRect(w - 304, 48, 280, 128, 8); ctx.fill(); ctx.stroke();
          ctx.fillStyle = C.muted;
          ctx.font = '650 11px Inter, sans-serif';
          ctx.fillText('10⁴-STEP EQUILIBRIUM TEMPERATURE', w - 288, 70);

          ctx.fillStyle = C.accent;
          ctx.font = '700 23px Inter, sans-serif';
          ctx.fillText('92.6 K', w - 288, 102);
          ctx.fillStyle = C.muted;
          ctx.font = '500 12px Inter, sans-serif';
          ctx.fillText('Residual Flow (Seed 4 · −2.4%)', w - 202, 99);

          ctx.fillStyle = C.orange;
          ctx.font = '700 23px Inter, sans-serif';
          ctx.fillText('37.2 K', w - 288, 138);
          ctx.fillStyle = C.muted;
          ctx.font = '500 12px Inter, sans-serif';
          ctx.fillText('Direct Target (−60.8% freeze)', w - 202, 135);
        } else if (ch.t0 === 40) {
          drawMiniCurve(
            ctx, C, w - 290, 46, 266, 102,
            argonCinemaData.rdf.gt, argonCinemaData.rdf.model,
            'RDF g(r)', 'truth (blue) vs model (teal)', Math.min(1, local * 1.4),
          );
          drawMiniCurve(
            ctx, C, w - 290, 158, 266, 102,
            argonCinemaData.vacf.gt, argonCinemaData.vacf.model,
            'VACF C(t)/C(0)', 'zero @ 306 fs vs 307 fs', Math.min(1, local * 1.4),
          );
        }
      }

      const baseY = h - 50;
      ctx.fillStyle = C.ink;
      ctx.font = `650 ${w < 600 ? 15 : 19}px Inter, "Segoe UI", sans-serif`;
      ctx.fillText(ch.headline, 24, baseY);
      ctx.fillStyle = C.muted;
      ctx.font = `450 ${w < 600 ? 11.5 : 13}px Inter, "Segoe UI", sans-serif`;
      ctx.fillText(ch.sub, 24, baseY + 21, w - 48);

      ctx.fillStyle = C.border;
      ctx.fillRect(0, h - 3, w, 3);
      ctx.fillStyle = C.accent;
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
    cinemaController = { seek: s => { tSec = s; renderAt(tSec); }, redraw: () => renderAt(tSec), viewer };
    return cinemaController;
  }

  function initMechanismAnimations() {
    const c1 = document.getElementById('mech-wigner-canvas');
    const c2 = document.getElementById('mech-flow-canvas');
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    const state = new Map();

    function setupCanvas(canvas, drawFn) {
      if (!canvas) return;
      const id = canvas.id;
      const btn = document.querySelector(`[data-mech-toggle="${id}"]`);
      const item = { canvas, ctx: canvas.getContext('2d'), playing: !reduced.matches, visible: true, t: 0.35, drawFn, btn };
      state.set(id, item);
      if (btn) {
        btn.addEventListener('click', () => {
          item.playing = !item.playing;
          btn.textContent = item.playing ? '❚❚' : '▶';
          btn.title = item.playing ? 'Pause animation' : 'Play animation';
        });
      }
      drawFn(item.ctx, canvas, item.t);
    }

    setupCanvas(c1, (ctx, canvas, t) => {
      const C = themeColors();
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const W = 960, H = 520;
      const pw = Math.max(2, Math.round((rect.width || 640) * dpr));
      const ph = Math.max(2, Math.round((rect.height || 346) * dpr));
      if (canvas.width !== pw || canvas.height !== ph) { canvas.width = pw; canvas.height = ph; }
      ctx.setTransform(pw / W, 0, 0, ph / H, 0, 0);

      ctx.fillStyle = C.stageBg;
      ctx.fillRect(0, 0, W, H);

      ctx.fillStyle = C.accent;
      ctx.font = '700 12.5px Inter, sans-serif';
      ctx.fillText('SPATIAL GRAPH ATTENTION · EQUITRAJ/LAYERS.PY', 38, 42);
      ctx.fillStyle = C.ink;
      ctx.font = '650 26px Inter, sans-serif';
      ctx.fillText('Wigner-D edge alignment + C² cutoff sink', 38, 78);

      const cx = 268, cy = 296, rc = 148;
      ctx.strokeStyle = C.border;
      ctx.lineWidth = 1.6;
      ctx.setLineDash([6, 6]);
      ctx.beginPath(); ctx.arc(cx, cy, rc, 0, Math.PI * 2); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = C.subtle;
      ctx.font = '550 12.5px Inter, sans-serif';
      ctx.fillText('cutoff r_c = 6.0 Å', cx - 50, cy - rc - 12);

      ctx.fillStyle = C.accent;
      ctx.beginPath(); ctx.arc(cx, cy, 21, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#ffffff';
      ctx.font = '700 12.5px Inter, sans-serif';
      ctx.fillText('dst i', cx - 14, cy + 4);

      const baseAngles = [-0.85, 0.35, 1.45, 2.45, -2.15];
      const baseR = [82, 106, 94, 118, 76];
      baseAngles.forEach((ang, idx) => {
        const a = ang + 0.14 * Math.sin(t * 0.9 + idx);
        const r = baseR[idx] + 6 * Math.cos(t * 1.1 + idx);
        const nx = cx + Math.cos(a) * r, ny = cy + Math.sin(a) * r;
        const env = Math.max(0, 1 - Math.pow(r / rc, 3));
        ctx.strokeStyle = C.dark ? `rgba(45, 212, 191, ${0.25 + 0.65 * env})` : `rgba(13, 148, 136, ${0.22 + 0.68 * env})`;
        ctx.lineWidth = 1.2 + 2.3 * env;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(nx, ny); ctx.stroke();

        const pFrac = (t * 0.55 + idx * 0.21) % 1.0;
        const mx = nx + (cx - nx) * pFrac, my = ny + (cy - ny) * pFrac;
        ctx.fillStyle = C.accent;
        ctx.beginPath(); ctx.arc(mx, my, 3.6, 0, Math.PI * 2); ctx.fill();

        ctx.fillStyle = C.blue;
        ctx.beginPath(); ctx.arc(nx, ny, 10.5, 0, Math.PI * 2); ctx.fill();
      });

      const rCross = rc * (0.80 + 0.25 * (0.5 + 0.5 * Math.sin(t * 1.25)));
      const aCross = 0.78;
      const kx = cx + Math.cos(aCross) * rCross, ky = cy + Math.sin(aCross) * rCross;
      const envCross = rCross < rc ? Math.pow(1 - rCross / rc, 2) : 0;
      if (envCross > 0.005) {
        ctx.strokeStyle = C.dark ? `rgba(167, 139, 250, ${envCross})` : `rgba(124, 58, 237, ${envCross})`;
        ctx.lineWidth = 3.0 * envCross + 0.5;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(kx, ky); ctx.stroke();
      }
      ctx.fillStyle = envCross > 0.01 ? C.purple : C.subtle;
      ctx.beginPath(); ctx.arc(kx, ky, 10.5, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = C.purple;
      ctx.font = '550 12.5px Inter, sans-serif';
      ctx.fillText(`crossing pair · envelope = ${envCross.toFixed(2)}`, cx - 94, cy + rc + 28);

      const rx = 492, ry = 122, rw = 428, rh = 348;
      ctx.fillStyle = C.cardSolid;
      ctx.strokeStyle = C.border;
      ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.roundRect(rx, ry, rw, rh, 10); ctx.fill(); ctx.stroke();

      ctx.fillStyle = C.accent;
      ctx.font = '700 11.5px Inter, sans-serif';
      ctx.fillText('1 · ROTATE TO EDGE +Y AXIS:  D⁽ˡ⁾(α, β, γ)', rx + 22, ry + 32);

      const lobes = [
        { lbl: 'l = 0 (scalar)', x: rx + 76, col: C.accent, draw: (ox, oy) => { ctx.beginPath(); ctx.arc(ox, oy, 17, 0, Math.PI * 2); ctx.stroke(); } },
        { lbl: 'l = 1 (vector)', x: rx + 212, col: C.blue, draw: (ox, oy) => {
          ctx.beginPath(); ctx.ellipse(ox, oy - 11, 9.5, 12.5, 0, 0, Math.PI * 2); ctx.stroke();
          ctx.beginPath(); ctx.ellipse(ox, oy + 11, 9.5, 12.5, 0, 0, Math.PI * 2); ctx.stroke();
        } },
        { lbl: 'l = 2 (quadrupole)', x: rx + 348, col: C.purple, draw: (ox, oy) => {
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
        lb.draw(lb.x, ry + 82);
        ctx.fillStyle = C.muted;
        ctx.font = '550 11.5px Inter, sans-serif';
        ctx.fillText(lb.lbl, lb.x - ctx.measureText(lb.lbl).width * 0.5, ry + 128);
      }

      ctx.strokeStyle = C.border;
      ctx.beginPath(); ctx.moveTo(rx + 22, ry + 150); ctx.lineTo(rx + rw - 22, ry + 150); ctx.stroke();

      ctx.fillStyle = C.accent;
      ctx.font = '700 11.5px Inter, sans-serif';
      ctx.fillText('2 · LOG-ENVELOPE + ZERO-LOGIT SINK (MAIN)', rx + 22, ry + 180);

      ctx.fillStyle = C.ink;
      ctx.font = '550 13px "SFMono-Regular", Consolas, monospace';
      ctx.fillText('α_ij = env(d_ij)·exp(z_ij) / [1 + Σ_k env(d_ik)·exp(z_ik)]', rx + 22, ry + 210);

      const gx = rx + 22, gy = ry + 236, gw = rw - 44, gh = 82;
      ctx.fillStyle = C.cardInset;
      ctx.beginPath(); ctx.roundRect(gx, gy, gw, gh, 7); ctx.fill();
      ctx.strokeStyle = C.orange;
      ctx.lineWidth = 1.6;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(gx + 14, gy + 20); ctx.lineTo(gx + gw * 0.76, gy + 24); ctx.lineTo(gx + gw * 0.76, gy + gh - 14); ctx.lineTo(gx + gw - 14, gy + gh - 14);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.strokeStyle = C.accent;
      ctx.lineWidth = 2.6;
      ctx.beginPath();
      for (let s = 0; s <= 60; s++) {
        const u = s / 60;
        const px = gx + 14 + u * (gw - 28);
        const v = u < 0.76 ? Math.pow(1 - Math.pow(u / 0.76, 2), 3) : 0;
        const py = gy + gh - 14 - v * (gh - 34);
        if (s === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.fillStyle = C.accent;
      ctx.font = '600 11.5px Inter, sans-serif';
      ctx.fillText('main: smooth C² decay', gx + 18, gy + 64);
      ctx.fillStyle = C.orange;
      ctx.fillText('v0: step jump @ r_c', gx + gw - 132, gy + 28);
    });

    setupCanvas(c2, (ctx, canvas, t) => {
      const C = themeColors();
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const W = 960, H = 470;
      const pw = Math.max(2, Math.round((rect.width || 640) * dpr));
      const ph = Math.max(2, Math.round((rect.height || 320) * dpr));
      if (canvas.width !== pw || canvas.height !== ph) { canvas.width = pw; canvas.height = ph; }
      ctx.setTransform(pw / W, 0, 0, ph / H, 0, 0);

      ctx.fillStyle = C.stageBg;
      ctx.fillRect(0, 0, W, H);

      ctx.fillStyle = C.purple;
      ctx.font = '700 12.5px Inter, sans-serif';
      ctx.fillText('TEMPORAL FUSION & POST-TRAINING STABILITY · EQUITRAJ/TRAIN.PY', 38, 40);
      ctx.fillStyle = C.ink;
      ctx.font = '650 25px Inter, sans-serif';
      ctx.fillText('FrameSplit spatial encode → residual flow → unroll curriculum', 38, 74);

      const stages = [
        { x: 38, w: 202, col: C.blue, title: 'k = 3 Context Frames', sub1: 'FrameSplit: 1 frame/call', sub2: '2× faster on CPU (0 diff)' },
        { x: 274, w: 202, col: C.accent, title: 'Causal Fusion', sub1: 'Invariant l = 0 weights', sub2: 'Equivariant l ≤ 2 transport' },
        { x: 510, w: 204, col: C.purple, title: 'Flow ODE  s ∈ [0, 1]', sub1: 'S = 2 API / S = 10 eval', sub2: 'Coasting residual targets' },
        { x: 748, w: 174, col: C.accent, title: 't + 50 fs State', sub1: 'Δx = v·Δt + a_θ', sub2: 'COM drift = 0' },
      ];
      const cy = 112, ch = 114;
      stages.forEach((st, idx) => {
        ctx.fillStyle = C.cardSolid;
        ctx.strokeStyle = st.col;
        ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.roundRect(st.x, cy, st.w, ch, 9); ctx.fill(); ctx.stroke();

        ctx.fillStyle = st.col;
        ctx.font = '700 14px Inter, sans-serif';
        ctx.fillText(st.title, st.x + 15, cy + 32);
        ctx.fillStyle = C.ink;
        ctx.font = '550 12.8px Inter, sans-serif';
        ctx.fillText(st.sub1, st.x + 15, cy + 62);
        ctx.fillStyle = C.muted;
        ctx.font = '450 12px Inter, sans-serif';
        ctx.fillText(st.sub2, st.x + 15, cy + 88);

        if (idx < stages.length - 1) {
          const ax0 = st.x + st.w + 4, ax1 = stages[idx + 1].x - 4, ay = cy + ch * 0.5;
          ctx.strokeStyle = C.subtle;
          ctx.lineWidth = 1.6;
          ctx.beginPath(); ctx.moveTo(ax0, ay); ctx.lineTo(ax1, ay); ctx.stroke();
          const pulse = (t * 0.7 - idx * 0.28 + 10) % 1.0;
          ctx.fillStyle = C.accent;
          ctx.beginPath(); ctx.arc(ax0 + (ax1 - ax0) * pulse, ay, 4, 0, Math.PI * 2); ctx.fill();
        }
      });

      ctx.strokeStyle = C.accent;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(835, cy + ch);
      ctx.bezierCurveTo(835, cy + ch + 28, 138, cy + ch + 28, 138, cy + ch);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1.0;
      ctx.fillStyle = C.accent;
      ctx.font = '550 12px Inter, sans-serif';
      ctx.fillText('autoregressive loop shifts predicted t + 50 fs frame into context window', 280, cy + ch + 46);

      const by = 302, bh = 132;
      ctx.fillStyle = C.cardSolid;
      ctx.strokeStyle = C.border;
      ctx.beginPath(); ctx.roundRect(38, by, 884, bh, 10); ctx.fill(); ctx.stroke();

      const pills = [
        ['Unroll 2 → 3 → 4', 'Gradual horizon curriculum + LR re-warmup', C.accent],
        ['All-Step Loss [1,1,1,1]', 'Supervises every unrolled step cleanly', C.blue],
        ['RepulsionPrior + Overlap', 'Soft steric barrier prevents pair clashes', C.purple],
        ['SpikeGuard + Micro-Batch', 'Skips >5× loss spikes; fits m=4 in VRAM', C.orange],
      ];
      pills.forEach((p, i) => {
        const px = 58 + i * 216;
        ctx.fillStyle = p[2];
        ctx.fillRect(px, by + 24, 3, 84);
        ctx.font = '700 13.5px Inter, sans-serif';
        ctx.fillText(p[0], px + 13, by + 44);
        ctx.fillStyle = C.muted;
        ctx.font = '450 12.2px Inter, sans-serif';
        const words = p[1].split(' ');
        let line = '', ly = by + 70;
        for (const w of words) {
          if ((line + w).length > 26) {
            ctx.fillText(line.trim(), px + 13, ly);
            line = w + ' '; ly += 19;
          } else { line += w + ' '; }
        }
        if (line.trim()) ctx.fillText(line.trim(), px + 13, ly);
      });
    });

    mechRefreshAll = () => {
      state.forEach(item => item.drawFn(item.ctx, item.canvas, item.t));
    };

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
    window.addEventListener('resize', mechRefreshAll, { passive: true });
  }

  function refreshTheme() {
    if (cinemaController) cinemaController.redraw();
    mechRefreshAll();
  }

  window.WignerCinema = { initHeroCinema, initMechanismAnimations, refreshTheme };
})();
