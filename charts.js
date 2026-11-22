/* Dependency-free publication SVG charts for WignerFlow.
 * Automatically adapts to Porcelain Daylight ('light', default) and Obsidian ('dark').
 * Includes line/scatter charts, CPU latency budget bar charts, and bond-constraint charts.
 */
(() => {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';

  function getPalette() {
    const dark = document.documentElement.dataset.theme === 'dark';
    if (dark) {
      return {
        bg: '#0c131c',
        plot: '#101a25',
        text: '#f1f5f9',
        muted: '#94a3b8',
        grid: '#1e293b',
        axis: '#475569',
        crosshair: '#cbd5e1',
        accent: '#2dd4bf',
        blue: '#60a5fa',
        purple: '#a78bfa',
        orange: '#fb7185',
        amber: '#fbbf24',
      };
    }
    return {
      bg: '#ffffff',
      plot: '#f8fafc',
      text: '#0f172a',
      muted: '#475569',
      grid: '#e2e8f0',
      axis: '#94a3b8',
      crosshair: '#334155',
      accent: '#0d9488',
      blue: '#2563eb',
      purple: '#7c3aed',
      orange: '#e11d48',
      amber: '#d97706',
    };
  }

  function el(name, attrs = {}, text) {
    const node = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs)) {
      if (v !== undefined && v !== null) node.setAttribute(k, String(v));
    }
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function buildPath(pts, sx, sy) {
    let d = '';
    let open = false;
    for (const [x, y] of pts) {
      if (y === null || y === undefined || !Number.isFinite(y) || !Number.isFinite(x)) {
        open = false;
        continue;
      }
      d += `${open ? 'L' : 'M'}${sx(x).toFixed(2)},${sy(y).toFixed(2)} `;
      open = true;
    }
    return d.trim();
  }

  function renderPublicationSVG(host, legendHost, spec) {
    if (!host) return;
    const T = getPalette();
    const widthPx = host.getBoundingClientRect().width || 960;
    const compact = widthPx < 560;
    const W = 1000, H = spec.height || (compact ? 500 : 430);
    const padL = compact ? 98 : 108;
    const padR = spec.padR ?? 126;
    const padT = 26;
    const padB = compact ? 80 : 68;
    const pw = W - padL - padR;
    const ph = H - padT - padB;

    const tx = x => (spec.logX ? Math.log10(Math.max(x, 1e-12)) : x);
    const ty = y => (spec.logY ? Math.log10(Math.max(y, 1e-16)) : y);
    const x0 = tx(spec.xRange[0]), x1 = tx(spec.xRange[1]);
    const y0 = ty(spec.yRange[0]), y1 = ty(spec.yRange[1]);

    const sx = x => padL + ((tx(x) - x0) / Math.max(x1 - x0, 1e-12)) * pw;
    const sy = y => padT + ph - ((ty(y) - y0) / Math.max(y1 - y0, 1e-12)) * ph;

    const fontTick = compact ? 19 : 14.5;
    const fontAxis = compact ? 20 : 15;

    const svg = el('svg', {
      viewBox: `0 0 ${W} ${H}`,
      role: 'img',
      'aria-label': spec.title || 'WignerFlow benchmark chart',
      preserveAspectRatio: 'xMidYMid meet',
    });
    svg.style.cssText = 'display:block;width:100%;height:auto;font-family:inherit;user-select:none;';

    svg.append(el('rect', { width: W, height: H, fill: T.bg, rx: 8 }));
    const defs = el('defs');
    const clipId = 'clip-' + Math.random().toString(36).slice(2, 9);
    const clip = el('clipPath', { id: clipId });
    clip.append(el('rect', { x: padL, y: padT, width: pw, height: ph }));
    defs.append(clip);
    svg.append(defs);

    svg.append(el('rect', { x: padL, y: padT, width: pw, height: ph, fill: T.plot, rx: 6 }));

    for (const yv of spec.yTicks) {
      const py = sy(yv);
      if (py < padT - 2 || py > padT + ph + 2) continue;
      svg.append(el('line', {
        x1: padL, y1: py, x2: padL + pw, y2: py,
        stroke: T.grid, 'stroke-width': 1, 'vector-effect': 'non-scaling-stroke',
      }));
      const lbl = spec.formatY ? spec.formatY(yv) : String(yv);
      svg.append(el('text', {
        x: padL - 12, y: py, fill: T.muted, 'font-size': fontTick,
        'text-anchor': 'end', 'dominant-baseline': 'middle',
      }, lbl));
    }

    for (const xv of spec.xTicks) {
      const px = sx(xv);
      if (px < padL - 2 || px > padL + pw + 2) continue;
      svg.append(el('line', {
        x1: px, y1: padT + ph, x2: px, y2: padT + ph + 6,
        stroke: T.axis, 'stroke-width': 1, 'vector-effect': 'non-scaling-stroke',
      }));
      const lbl = spec.formatX ? spec.formatX(xv) : String(xv);
      svg.append(el('text', {
        x: px, y: padT + ph + fontTick + 10, fill: T.muted, 'font-size': fontTick,
        'text-anchor': 'middle',
      }, lbl));
    }

    const plotG = el('g', { 'clip-path': `url(#${clipId})` });

    if (spec.band) {
      const yTop = sy(spec.band.high);
      const yBot = sy(spec.band.low);
      plotG.append(el('rect', {
        x: padL, y: Math.min(yTop, yBot), width: pw, height: Math.abs(yBot - yTop),
        fill: spec.band.color || T.accent, 'fill-opacity': 0.12,
      }));
    }

    if (spec.hline !== undefined && spec.hline !== null) {
      const py = sy(spec.hline.y);
      plotG.append(el('line', {
        x1: padL, y1: py, x2: padL + pw, y2: py,
        stroke: spec.hline.color || T.blue,
        'stroke-width': 1.3,
        'stroke-dasharray': spec.hline.dash || '5 5',
        opacity: 0.78,
        'vector-effect': 'non-scaling-stroke',
      }));
    }

    const endCallouts = [];
    for (const s of spec.series) {
      const d = buildPath(s.points, sx, sy);
      if (d) {
        plotG.append(el('path', {
          d,
          fill: 'none',
          stroke: s.color,
          'stroke-width': s.width || (s.highlight ? 3.0 : 2.1),
          'stroke-linecap': 'round',
          'stroke-linejoin': 'round',
          'stroke-dasharray': s.dash && s.dash.length ? s.dash.join(' ') : 'none',
          'vector-effect': 'non-scaling-stroke',
        }));
      }
      if (s.markers && s.markers.length) {
        for (const [mx, my, kind] of s.markers) {
          const px = sx(mx), py = sy(my);
          if (kind === 'cross') {
            const r = 6;
            plotG.append(el('path', {
              d: `M${px - r},${py - r}L${px + r},${py + r}M${px - r},${py + r}L${px + r},${py - r}`,
              stroke: s.color, 'stroke-width': 2.5, 'stroke-linecap': 'round',
            }));
          } else {
            plotG.append(el('circle', {
              cx: px, cy: py, r: s.highlight ? 5.4 : 4.4,
              fill: s.color, stroke: T.bg, 'stroke-width': 1.8,
            }));
          }
        }
      }
      if (s.endLabel) {
        const valid = s.points.filter(p => p[1] !== null && Number.isFinite(p[1]));
        if (valid.length) {
          const last = valid[valid.length - 1];
          endCallouts.push({ y: sy(last[1]), color: s.color, text: s.endLabel, bold: !!s.highlight });
        }
      }
    }
    svg.append(plotG);

    svg.append(el('path', {
      d: `M${padL},${padT}V${padT + ph}H${padL + pw}`,
      fill: 'none', stroke: T.axis, 'stroke-width': 1.2, 'vector-effect': 'non-scaling-stroke',
    }));

    endCallouts.sort((a, b) => a.y - b.y);
    for (let i = 1; i < endCallouts.length; i++) {
      if (endCallouts[i].y - endCallouts[i - 1].y < 18) {
        endCallouts[i].y = endCallouts[i - 1].y + 18;
      }
    }
    for (const c of endCallouts) {
      svg.append(el('text', {
        x: padL + pw + 10,
        y: Math.min(padT + ph - 4, Math.max(padT + 10, c.y)),
        fill: c.color,
        'font-size': compact ? 15.5 : 13,
        'font-weight': c.bold ? '650' : '550',
        'dominant-baseline': 'middle',
      }, c.text));
    }

    svg.append(el('text', {
      x: padL + pw * 0.5, y: H - 12, fill: T.muted, 'font-size': fontAxis, 'text-anchor': 'middle',
    }, spec.xLabel || ''));
    svg.append(el('text', {
      transform: `translate(22 ${padT + ph * 0.5}) rotate(-90)`,
      fill: T.muted, 'font-size': fontAxis, 'text-anchor': 'middle',
    }, spec.yLabel || ''));

    const hoverLine = el('line', {
      x1: padL, y1: padT, x2: padL, y2: padT + ph,
      stroke: T.crosshair, 'stroke-width': 1, 'stroke-dasharray': '3 3',
      opacity: 0, 'pointer-events': 'none', 'vector-effect': 'non-scaling-stroke',
    });
    const hoverBadge = el('text', {
      x: padL + 10, y: padT + 18, fill: T.text, 'font-size': 13, 'font-weight': '600',
      opacity: 0, 'pointer-events': 'none',
    }, '');
    svg.append(hoverLine, hoverBadge);

    const hitRect = el('rect', {
      x: padL, y: padT, width: pw, height: ph, fill: 'transparent', style: 'cursor:crosshair',
    });
    hitRect.addEventListener('pointermove', ev => {
      const r = svg.getBoundingClientRect();
      const ux = ((ev.clientX - r.left) / Math.max(r.width, 1)) * W;
      const clampedX = Math.max(padL, Math.min(padL + pw, ux));
      const frac = (clampedX - padL) / pw;
      const dataX = spec.logX ? Math.pow(10, x0 + frac * (x1 - x0)) : (x0 + frac * (x1 - x0));
      hoverLine.setAttribute('x1', clampedX.toFixed(1));
      hoverLine.setAttribute('x2', clampedX.toFixed(1));
      hoverLine.setAttribute('opacity', '0.55');

      const snippets = [];
      for (const s of spec.series.slice(0, 3)) {
        let best = null, bestDist = Infinity;
        for (const p of s.points) {
          if (p[1] === null || !Number.isFinite(p[1])) continue;
          const d = Math.abs(tx(p[0]) - tx(dataX));
          if (d < bestDist) { bestDist = d; best = p; }
        }
        if (best) {
          const valStr = spec.formatTooltipY ? spec.formatTooltipY(best[1]) : Number(best[1].toPrecision(4));
          snippets.push(`${s.shortName || s.name.split(' ')[0]}: ${valStr}`);
        }
      }
      const xStr = spec.formatX ? spec.formatX(dataX) : Number(dataX.toPrecision(3));
      hoverBadge.textContent = `${xStr}  ·  ${snippets.join('   ')}`;
      hoverBadge.setAttribute('x', (clampedX > padL + pw * 0.58 ? padL + 12 : clampedX + 10).toFixed(1));
      hoverBadge.setAttribute('opacity', '0.96');
    });
    hitRect.addEventListener('pointerleave', () => {
      hoverLine.setAttribute('opacity', '0');
      hoverBadge.setAttribute('opacity', '0');
    });
    svg.append(hitRect);

    host.replaceChildren(svg);

    if (legendHost) {
      legendHost.replaceChildren();
      for (const s of spec.series) {
        const item = document.createElement('span');
        item.className = 'legend-item';
        const sw = el('svg', { viewBox: '0 0 40 12', width: 40, height: 12, 'aria-hidden': 'true' });
        sw.append(el('line', {
          x1: 2, y1: 6, x2: 38, y2: 6, stroke: s.color,
          'stroke-width': s.highlight ? 3.0 : 2.2,
          'stroke-linecap': 'round',
          'stroke-dasharray': s.dash && s.dash.length ? s.dash.join(' ') : 'none',
        }));
        const label = document.createElement('span');
        label.textContent = s.name;
        if (s.highlight) label.style.fontWeight = '600';
        item.append(sw, label);
        legendHost.append(item);
      }
    }
  }

  /**
   * Render CPU Inference API p50 / p95 latency grouped bar chart with 1,000 ms budget line.
   */
  function renderLatencyBarSVG(host, rows, activeSteps, budgetMs = 1000) {
    if (!host || !rows || !rows.length) return;
    const T = getPalette();
    const W = 620, H = 268;
    const padL = 62, padR = 24, padT = 22, padB = 46;
    const pw = W - padL - padR, ph = H - padT - padB;
    const maxY = Math.max(1500, ...rows.map(r => Math.min(4200, r.wall_p95_ms * 1.1)));
    const sy = v => padT + ph - (Math.min(v, maxY) / maxY) * ph;

    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'CPU latency per 50 fs step across flow steps S' });
    svg.style.cssText = 'display:block;width:100%;height:auto;font-family:inherit;';
    svg.append(el('rect', { width: W, height: H, fill: T.bg, rx: 8 }));
    svg.append(el('rect', { x: padL, y: padT, width: pw, height: ph, fill: T.plot, rx: 6 }));

    const ticks = maxY > 2400 ? [0, 1000, 2000, 3000, 4000] : [0, 500, 1000, 1500];
    for (const tv of ticks) {
      if (tv > maxY) continue;
      const py = sy(tv);
      svg.append(el('line', { x1: padL, y1: py, x2: padL + pw, y2: py, stroke: T.grid, 'stroke-width': 1 }));
      svg.append(el('text', {
        x: padL - 8, y: py, fill: T.muted, 'font-size': 11.5, 'text-anchor': 'end', 'dominant-baseline': 'middle',
      }, `${tv} ms`));
    }

    // 1,000 ms target line
    const yBud = sy(budgetMs);
    svg.append(el('line', {
      x1: padL, y1: yBud, x2: padL + pw, y2: yBud,
      stroke: T.amber, 'stroke-width': 1.6, 'stroke-dasharray': '5 4',
    }));
    svg.append(el('text', {
      x: padL + pw - 6, y: yBud - 6, fill: T.amber, 'font-size': 11, 'font-weight': '650', 'text-anchor': 'end',
    }, '1,000 ms target'));

    const colW = pw / rows.length;
    rows.forEach((r, idx) => {
      const cx = padL + colW * (idx + 0.5);
      const isAct = r.sampling_steps === activeSteps;
      const bw = 24;

      if (isAct) {
        svg.append(el('rect', {
          x: cx - 42, y: padT + 2, width: 84, height: ph - 4, rx: 6,
          fill: T.accent, 'fill-opacity': 0.09,
        }));
      }

      // p50 CPU bar & p95 Wall bar
      const yCpu = sy(r.cpu_p50_ms);
      const yWall95 = sy(r.wall_p95_ms);
      svg.append(el('rect', {
        x: cx - bw - 2, y: yCpu, width: bw, height: Math.max(2, padT + ph - yCpu), rx: 3,
        fill: isAct ? T.accent : T.blue, opacity: isAct ? 1 : 0.72,
      }));
      svg.append(el('rect', {
        x: cx + 2, y: yWall95, width: bw, height: Math.max(2, padT + ph - yWall95), rx: 3,
        fill: r.wall_p95_ms <= budgetMs ? T.purple : T.orange, opacity: isAct ? 0.92 : 0.58,
      }));

      svg.append(el('text', {
        x: cx - bw * 0.5 - 2, y: yCpu - 5, fill: T.text, 'font-size': 10.5, 'font-weight': '600', 'text-anchor': 'middle',
      }, `${Math.round(r.cpu_p50_ms)}`));
      const wallTxt = r.wall_p95_ms > maxY ? `${Math.round(r.wall_p95_ms)}↑` : `${Math.round(r.wall_p95_ms)}`;
      svg.append(el('text', {
        x: cx + bw * 0.5 + 2, y: yWall95 - 5, fill: T.muted, 'font-size': 10.5, 'text-anchor': 'middle',
      }, wallTxt));

      svg.append(el('text', {
        x: cx, y: padT + ph + 19, fill: isAct ? T.accent : T.text,
        'font-size': 12.5, 'font-weight': isAct ? '700' : '550', 'text-anchor': 'middle',
      }, `S = ${r.sampling_steps}${r.sampling_steps === 2 ? ' ★' : ''}`));
    });

    svg.append(el('text', {
      x: padL + pw * 0.5, y: H - 6, fill: T.muted, 'font-size': 11.5, 'text-anchor': 'middle',
    }, 'Left bar: CPU p50 (ms)   ·   Right bar: Wall p95 (ms)'));

    host.replaceChildren(svg);
  }

  window.WignerCharts = { renderPublicationSVG, renderLatencyBarSVG, getPalette };
})();
