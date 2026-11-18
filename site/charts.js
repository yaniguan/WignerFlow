/* Bespoke dependency-free SVG publication charts for WignerFlow.
 * Inspired by MolRSI's comparison-chart.js: obsidian dark surface (#0a0f15 / #0e161e),
 * subtle horizontal hairlines (#293541), vector-effect="non-scaling-stroke",
 * endpoint callouts, custom SVG legend swatches, and interactive hover crosshair.
 */
(() => {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';
  const THEME = {
    bg: '#0a0f15',
    plot: '#0e161e',
    text: '#edf1f7',
    muted: '#aab6c4',
    grid: '#24303c',
    axis: '#5f7285',
    lime: '#b5d88e',
    blue: '#91b6df',
    teal: '#79c2bd',
    purple: '#b9a1de',
    orange: '#d97757',
  };

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

  /**
   * Render a multi-series publication SVG chart with interactive crosshair readout.
   * spec: {
   *   title, xLabel, yLabel, xRange:[min,max], yRange:[min,max],
   *   xTicks:[], yTicks:[], logX?:bool, logY?:bool,
   *   band?: {low, high, color, label},
   *   hline?: {y, color, dash},
   *   series: [{name, color, width, dash, points:[[x,y],...], markers?:[[x,y,kind]], endLabel?:string}]
   * }
   */
  function renderPublicationSVG(host, legendHost, spec) {
    if (!host) return;
    const widthPx = host.getBoundingClientRect().width || 960;
    const compact = widthPx < 560;
    const W = 1000, H = compact ? 540 : 480;
    const padL = compact ? 104 : 114;
    const padR = spec.padR ?? 128;
    const padT = 32;
    const padB = compact ? 86 : 74;
    const pw = W - padL - padR;
    const ph = H - padT - padB;

    const tx = x => (spec.logX ? Math.log10(Math.max(x, 1e-12)) : x);
    const ty = y => (spec.logY ? Math.log10(Math.max(y, 1e-16)) : y);
    const x0 = tx(spec.xRange[0]), x1 = tx(spec.xRange[1]);
    const y0 = ty(spec.yRange[0]), y1 = ty(spec.yRange[1]);

    const sx = x => padL + ((tx(x) - x0) / Math.max(x1 - x0, 1e-12)) * pw;
    const sy = y => padT + ph - ((ty(y) - y0) / Math.max(y1 - y0, 1e-12)) * ph;

    const fontTick = compact ? 20 : 15;
    const fontAxis = compact ? 21 : 16;

    const svg = el('svg', {
      viewBox: `0 0 ${W} ${H}`,
      role: 'img',
      'aria-label': spec.title || 'WignerFlow benchmark chart',
      preserveAspectRatio: 'xMidYMid meet',
    });
    svg.style.cssText = 'display:block;width:100%;height:auto;font-family:inherit;user-select:none;';

    svg.append(el('rect', { width: W, height: H, fill: THEME.bg }));
    const defs = el('defs');
    const clipId = 'clip-' + Math.random().toString(36).slice(2, 9);
    const clip = el('clipPath', { id: clipId });
    clip.append(el('rect', { x: padL, y: padT, width: pw, height: ph }));
    defs.append(clip);
    svg.append(defs);

    svg.append(el('rect', { x: padL, y: padT, width: pw, height: ph, fill: THEME.plot, rx: 4 }));

    // Y gridlines + labels
    for (const yv of spec.yTicks) {
      const py = sy(yv);
      if (py < padT - 2 || py > padT + ph + 2) continue;
      svg.append(el('line', {
        x1: padL, y1: py, x2: padL + pw, y2: py,
        stroke: THEME.grid, 'stroke-width': 1, 'vector-effect': 'non-scaling-stroke',
      }));
      const lbl = spec.formatY ? spec.formatY(yv) : String(yv);
      svg.append(el('text', {
        x: padL - 14, y: py, fill: THEME.muted, 'font-size': fontTick,
        'text-anchor': 'end', 'dominant-baseline': 'middle',
      }, lbl));
    }

    // X ticks + labels
    for (const xv of spec.xTicks) {
      const px = sx(xv);
      if (px < padL - 2 || px > padL + pw + 2) continue;
      svg.append(el('line', {
        x1: px, y1: padT + ph, x2: px, y2: padT + ph + 7,
        stroke: THEME.axis, 'stroke-width': 1, 'vector-effect': 'non-scaling-stroke',
      }));
      const lbl = spec.formatX ? spec.formatX(xv) : String(xv);
      svg.append(el('text', {
        x: px, y: padT + ph + fontTick + 12, fill: THEME.muted, 'font-size': fontTick,
        'text-anchor': 'middle',
      }, lbl));
    }

    const plotG = el('g', { 'clip-path': `url(#${clipId})` });

    // Optional target band (e.g. +/-5% temperature window)
    if (spec.band) {
      const yTop = sy(spec.band.high);
      const yBot = sy(spec.band.low);
      plotG.append(el('rect', {
        x: padL, y: Math.min(yTop, yBot), width: pw, height: Math.abs(yBot - yTop),
        fill: spec.band.color || THEME.lime, 'fill-opacity': 0.11,
      }));
    }

    // Optional reference horizontal line
    if (spec.hline !== undefined && spec.hline !== null) {
      const py = sy(spec.hline.y);
      plotG.append(el('line', {
        x1: padL, y1: py, x2: padL + pw, y2: py,
        stroke: spec.hline.color || THEME.muted,
        'stroke-width': 1.2,
        'stroke-dasharray': spec.hline.dash || '5 5',
        opacity: 0.7,
        'vector-effect': 'non-scaling-stroke',
      }));
    }

    // Series lines & markers
    const endCallouts = [];
    for (const s of spec.series) {
      const d = buildPath(s.points, sx, sy);
      if (d) {
        plotG.append(el('path', {
          d,
          fill: 'none',
          stroke: s.color,
          'stroke-width': s.width || (s.highlight ? 3.2 : 2.0),
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
              cx: px, cy: py, r: s.highlight ? 5.6 : 4.6,
              fill: s.color, stroke: THEME.bg, 'stroke-width': 1.8,
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

    // Axis L-spine
    svg.append(el('path', {
      d: `M${padL},${padT}V${padT + ph}H${padL + pw}`,
      fill: 'none', stroke: THEME.axis, 'stroke-width': 1.2, 'vector-effect': 'non-scaling-stroke',
    }));

    // De-overlap right-hand endpoint callouts
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
        'font-size': compact ? 16 : 13.5,
        'font-weight': c.bold ? '650' : '500',
        'dominant-baseline': 'middle',
      }, c.text));
    }

    // Axis titles
    svg.append(el('text', {
      x: padL + pw * 0.5, y: H - 14, fill: THEME.muted, 'font-size': fontAxis, 'text-anchor': 'middle',
    }, spec.xLabel || ''));
    svg.append(el('text', {
      transform: `translate(24 ${padT + ph * 0.5}) rotate(-90)`,
      fill: THEME.muted, 'font-size': fontAxis, 'text-anchor': 'middle',
    }, spec.yLabel || ''));

    // Interactive hover vertical crosshair + readout pill
    const hoverLine = el('line', {
      x1: padL, y1: padT, x2: padL, y2: padT + ph,
      stroke: '#dce7f3', 'stroke-width': 1, 'stroke-dasharray': '3 3',
      opacity: 0, 'pointer-events': 'none', 'vector-effect': 'non-scaling-stroke',
    });
    const hoverBadge = el('text', {
      x: padL + 10, y: padT + 18, fill: THEME.text, 'font-size': 13,
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
      hoverBadge.setAttribute('opacity', '0.95');
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
        const sw = el('svg', { viewBox: '0 0 44 12', width: 44, height: 12, 'aria-hidden': 'true' });
        sw.append(el('line', {
          x1: 2, y1: 6, x2: 42, y2: 6, stroke: s.color,
          'stroke-width': s.highlight ? 3.2 : 2.2,
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

  window.WignerCharts = { renderPublicationSVG, THEME };
})();
