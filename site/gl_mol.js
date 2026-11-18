/* Hardware-accelerated WebGL2 ray-cast sphere & cylinder 3D molecular viewer.
 * Zero external dependencies. Renders analytical per-pixel ray-cast spheres
 * and capsules with gl_FragDepth, 3-point studio lighting, Fresnel rim light,
 * and smooth trackball orbit. Gracefully falls back to shaded 2D if WebGL2 is absent.
 */
(() => {
  'use strict';

  const CPK = {
    H:  [0.86, 0.90, 0.95, 0.28],
    C:  [0.27, 0.33, 0.40, 0.44],
    N:  [0.44, 0.63, 0.87, 0.42],
    O:  [0.86, 0.43, 0.36, 0.41],
    F:  [0.64, 0.85, 0.48, 0.39],
    P:  [0.89, 0.65, 0.34, 0.50],
    S:  [0.88, 0.78, 0.36, 0.50],
    Cl: [0.45, 0.80, 0.55, 0.48],
    Li: [0.73, 0.63, 0.87, 0.46],
    Na: [0.57, 0.71, 0.88, 0.52],
    Ar: [0.49, 0.76, 0.74, 0.56],
  };

  function mat4Perspective(fovRad, aspect, near, far) {
    const f = 1.0 / Math.tan(fovRad * 0.5);
    const nf = 1.0 / (near - far);
    return new Float32Array([
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (far + near) * nf, -1,
      0, 0, 2 * far * near * nf, 0,
    ]);
  }

  function mat4FromOrbit(rotX, rotY, dist, panX = 0, panY = 0) {
    const cx = Math.cos(rotX), sx = Math.sin(rotX);
    const cy = Math.cos(rotY), sy = Math.sin(rotY);
    // Ry * Rx followed by camera translation (panX, panY, -dist)
    return new Float32Array([
      cy, sx * sy, -cx * sy, 0,
      0, cx, sx, 0,
      sy, -sx * cy, cx * cy, 0,
      panX, panY, -dist, 1,
    ]);
  }

  function compileShader(gl, type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(sh);
      gl.deleteShader(sh);
      throw new Error(info || 'Shader compile error');
    }
    return sh;
  }

  function linkProgram(gl, vsSrc, fsSrc) {
    const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(prog) || 'Program link error');
    }
    return prog;
  }

  const SPHERE_VS = `#version 300 es
  precision highp float;
  layout(location = 0) in vec2 aCorner;
  layout(location = 1) in vec3 iCenter;
  layout(location = 2) in float iRadius;
  layout(location = 3) in vec4 iColor;

  uniform mat4 uView;
  uniform mat4 uProj;

  out vec3 vViewCenter;
  out vec3 vRayTarget;
  out float vRadius;
  out vec4 vColor;

  void main() {
    vec4 vc = uView * vec4(iCenter, 1.0);
    vViewCenter = vc.xyz;
    vRadius = iRadius;
    vColor = iColor;
    // Expand billboard slightly to prevent perspective grazing clipping
    vec2 offset = aCorner * iRadius * 1.32;
    vec3 quadPos = vc.xyz + vec3(offset, 0.0);
    vRayTarget = quadPos;
    gl_Position = uProj * vec4(quadPos, 1.0);
  }`;

  const SPHERE_FS = `#version 300 es
  precision highp float;
  in vec3 vViewCenter;
  in vec3 vRayTarget;
  in float vRadius;
  in vec4 vColor;

  uniform mat4 uProj;
  uniform vec3 uFogColor;
  uniform float uFogNear;
  uniform float uFogFar;

  out vec4 outColor;

  void main() {
    vec3 rayDir = normalize(vRayTarget);
    // Ray from origin (0,0,0) in view space to sphere at vViewCenter
    vec3 oc = -vViewCenter;
    float b = dot(oc, rayDir);
    float c = dot(oc, oc) - vRadius * vRadius;
    float disc = b * b - c;
    if (disc < 0.0) discard;

    float sqrtDisc = sqrt(disc);
    float t = -b - sqrtDisc;
    if (t < 0.02) t = -b + sqrtDisc;
    if (t < 0.02) discard;

    vec3 hitPos = rayDir * t;
    vec3 N = normalize(hitPos - vViewCenter);
    vec3 V = -rayDir;

    vec4 clip = uProj * vec4(hitPos, 1.0);
    float ndcDepth = clip.z / clip.w;
    gl_FragDepth = ndcDepth * 0.5 + 0.5;

    // 3-point studio illumination (ChimeraX / Blender Satin style)
    vec3 Lkey = normalize(vec3(0.52, 0.68, 0.78));
    vec3 Lfill = normalize(vec3(-0.68, 0.32, 0.42));
    vec3 Lrim = normalize(vec3(0.0, -0.62, -0.78));

    float diffKey = max(dot(N, Lkey), 0.0);
    float diffFill = max(dot(N, Lfill), 0.0);
    float hemi = 0.5 + 0.5 * N.y;

    vec3 H = normalize(Lkey + V);
    float spec = pow(max(dot(N, H), 0.0), 42.0);
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.2);
    float backRim = max(dot(N, Lrim), 0.0) * 0.22;

    vec3 base = vColor.rgb;
    vec3 ambient = base * mix(vec3(0.16, 0.19, 0.24), vec3(0.30, 0.34, 0.40), hemi);
    vec3 lit = ambient
             + base * (diffKey * 0.76 + diffFill * 0.24 + backRim)
             + vec3(0.96, 0.98, 1.0) * (spec * 0.42)
             + mix(base, vec3(0.75, 0.88, 0.96), 0.5) * (fresnel * 0.24);

    float dist = -hitPos.z;
    float fog = clamp((dist - uFogNear) / max(uFogFar - uFogNear, 0.001), 0.0, 0.72);
    vec3 finalRgb = mix(lit, uFogColor, fog);

    outColor = vec4(finalRgb, vColor.a);
  }`;

  const CYL_VS = `#version 300 es
  precision highp float;
  layout(location = 0) in vec3 aUnitCyl;     // x=cos, y=sin, z in [0,1]
  layout(location = 1) in vec3 aUnitNorm;
  layout(location = 2) in vec3 iP0;
  layout(location = 3) in vec3 iP1;
  layout(location = 4) in vec2 iRadTip;      // x=r0, y=r1 (tapered for cones!)
  layout(location = 5) in vec4 iColor;

  uniform mat4 uView;
  uniform mat4 uProj;

  out vec3 vViewPos;
  out vec3 vViewNorm;
  out vec4 vColor;

  void main() {
    vec3 axis = iP1 - iP0;
    float len = length(axis);
    if (len < 1e-5) {
      gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
      return;
    }
    vec3 w = axis / len;
    vec3 up = abs(w.y) < 0.92 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 u = normalize(cross(up, w));
    vec3 v = cross(w, u);

    float r = mix(iRadTip.x, iRadTip.y, aUnitCyl.z);
    vec3 worldPos = mix(iP0, iP1, aUnitCyl.z) + r * (aUnitCyl.x * u + aUnitCyl.y * v);
    vec3 worldNorm = normalize(aUnitNorm.x * u + aUnitNorm.y * v + ((iRadTip.x - iRadTip.y) / max(len, 1e-4)) * w);

    vec4 vp = uView * vec4(worldPos, 1.0);
    vViewPos = vp.xyz;
    vViewNorm = mat3(uView) * worldNorm;
    vColor = iColor;
    gl_Position = uProj * vp;
  }`;

  const CYL_FS = `#version 300 es
  precision highp float;
  in vec3 vViewPos;
  in vec3 vViewNorm;
  in vec4 vColor;

  uniform vec3 uFogColor;
  uniform float uFogNear;
  uniform float uFogFar;

  out vec4 outColor;

  void main() {
    vec3 N = normalize(vViewNorm);
    vec3 V = normalize(-vViewPos);
    vec3 Lkey = normalize(vec3(0.52, 0.68, 0.78));
    vec3 Lfill = normalize(vec3(-0.68, 0.32, 0.42));
    float diff = max(dot(N, Lkey), 0.0) * 0.72 + max(dot(N, Lfill), 0.0) * 0.25 + 0.24;
    vec3 H = normalize(Lkey + V);
    float spec = pow(max(dot(N, H), 0.0), 28.0) * 0.26;
    vec3 lit = vColor.rgb * diff + vec3(spec);

    float dist = -vViewPos.z;
    float fog = clamp((dist - uFogNear) / max(uFogFar - uFogNear, 0.001), 0.0, 0.72);
    outColor = vec4(mix(lit, uFogColor, fog), vColor.a);
  }`;

  function makeCylinderMesh(radialSegs = 14) {
    const verts = [];
    const norms = [];
    const idx = [];
    for (let i = 0; i <= radialSegs; i++) {
      const th = (i / radialSegs) * Math.PI * 2;
      const c = Math.cos(th), s = Math.sin(th);
      verts.push(c, s, 0, c, s, 1);
      norms.push(c, s, 0, c, s, 0);
    }
    for (let i = 0; i < radialSegs; i++) {
      const b = i * 2;
      idx.push(b, b + 1, b + 2, b + 1, b + 3, b + 2);
    }
    return {
      verts: new Float32Array(verts),
      norms: new Float32Array(norms),
      idx: new Uint16Array(idx),
    };
  }

  class WignerGLViewer {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.bg = opts.bg || [10 / 255, 15 / 255, 21 / 255];
      this.rotX = opts.rotX ?? 0.36;
      this.rotY = opts.rotY ?? -0.58;
      this.zoom = opts.zoom ?? 1.0;
      this.autoSpin = opts.autoSpin ?? 0.0;
      this.dragging = false;
      this.lastX = 0;
      this.lastY = 0;
      this.scene = { spheres: [], rods: [], center: [0, 0, 0], radius: 6.0 };

      this.gl = null;
      try {
        this.gl = canvas.getContext('webgl2', { antialias: true, alpha: false, premultipliedAlpha: false });
      } catch {
        this.gl = null;
      }
      if (this.gl) {
        try {
          this._initGL();
        } catch (err) {
          console.warn('WebGL2 shader init fallback:', err);
          this.gl = null;
        }
      }
      if (!this.gl) {
        this.ctx2d = canvas.getContext('2d');
      }
      this._bindEvents();
    }

    _initGL() {
      const gl = this.gl;
      this.sphereProg = linkProgram(gl, SPHERE_VS, SPHERE_FS);
      this.cylProg = linkProgram(gl, CYL_VS, CYL_FS);

      // Sphere quad VAO
      this.sphereVao = gl.createVertexArray();
      gl.bindVertexArray(this.sphereVao);
      const quadBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

      this.sphereInstBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.sphereInstBuf);
      // stride = 8 floats: center(3), radius(1), color(4)
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 32, 0);
      gl.vertexAttribDivisor(1, 1);
      gl.enableVertexAttribArray(2);
      gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 32, 12);
      gl.vertexAttribDivisor(2, 1);
      gl.enableVertexAttribArray(3);
      gl.vertexAttribPointer(3, 4, gl.FLOAT, false, 32, 16);
      gl.vertexAttribDivisor(3, 1);

      // Cylinder / Cone VAO
      const mesh = makeCylinderMesh(14);
      this.cylIndexCount = mesh.idx.length;
      this.cylVao = gl.createVertexArray();
      gl.bindVertexArray(this.cylVao);

      const vBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vBuf);
      gl.bufferData(gl.ARRAY_BUFFER, mesh.verts, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

      const nBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, nBuf);
      gl.bufferData(gl.ARRAY_BUFFER, mesh.norms, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

      const iBuf = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, iBuf);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.idx, gl.STATIC_DRAW);

      this.cylInstBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.cylInstBuf);
      // stride = 12 floats (48 bytes): p0(3), p1(3), radTip(2), color(4)
      gl.enableVertexAttribArray(2);
      gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 48, 0);
      gl.vertexAttribDivisor(2, 1);
      gl.enableVertexAttribArray(3);
      gl.vertexAttribPointer(3, 3, gl.FLOAT, false, 48, 12);
      gl.vertexAttribDivisor(3, 1);
      gl.enableVertexAttribArray(4);
      gl.vertexAttribPointer(4, 2, gl.FLOAT, false, 48, 24);
      gl.vertexAttribDivisor(4, 1);
      gl.enableVertexAttribArray(5);
      gl.vertexAttribPointer(5, 4, gl.FLOAT, false, 48, 32);
      gl.vertexAttribDivisor(5, 1);

      gl.bindVertexArray(null);
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);
    }

    _bindEvents() {
      const c = this.canvas;
      c.addEventListener('pointerdown', e => {
        this.dragging = true;
        this.lastX = e.clientX;
        this.lastY = e.clientY;
        c.setPointerCapture(e.pointerId);
      });
      c.addEventListener('pointermove', e => {
        if (!this.dragging) return;
        const dx = e.clientX - this.lastX;
        const dy = e.clientY - this.lastY;
        this.lastX = e.clientX;
        this.lastY = e.clientY;
        this.rotY += dx * 0.0085;
        this.rotX = Math.max(-1.48, Math.min(1.48, this.rotX + dy * 0.0085));
        this.render();
      });
      const stop = () => { this.dragging = false; };
      c.addEventListener('pointerup', stop);
      c.addEventListener('pointercancel', stop);
      c.addEventListener('wheel', e => {
        if (!e.ctrlKey && Math.abs(e.deltaY) < 4) return;
        e.preventDefault();
        this.zoom = Math.max(0.55, Math.min(2.4, this.zoom * (e.deltaY > 0 ? 0.92 : 1.08)));
        this.render();
      }, { passive: false });
    }

    resetView(rotX = 0.36, rotY = -0.58) {
      this.rotX = rotX;
      this.rotY = rotY;
      this.zoom = 1.0;
      this.render();
    }

    /**
     * Build scene geometry from molecular data:
     *   positions: [[x,y,z], ...]
     *   elements: ['C','O',...]
     *   bonds: [[i,j], ...]
     *   boxA: number | null (cubic box side in Angstrom)
     *   arrows: [{from:[x,y,z], vec:[dx,dy,dz], color:[r,g,b,a], scale, radius}]
     *   ghostFrames: [[[x,y,z],...], ...] (optional context trails)
     *   highlightAtom: index | null, cutoffA: number | null
     */
    setMolecule(spec) {
      const pos = spec.positions || [];
      const elems = spec.elements || [];
      const bonds = spec.bonds || [];
      const boxA = spec.boxA || null;
      const n = pos.length;
      if (!n) {
        this.scene = { spheres: [], rods: [], center: [0, 0, 0], radius: 6.0 };
        this.render();
        return;
      }

      let cx = 0, cy = 0, cz = 0;
      if (boxA) {
        cx = cy = cz = boxA * 0.5;
      } else {
        for (let i = 0; i < n; i++) {
          cx += pos[i][0]; cy += pos[i][1]; cz += pos[i][2];
        }
        cx /= n; cy /= n; cz /= n;
      }

      let maxR = boxA ? boxA * 0.78 : 2.5;
      if (!boxA) {
        for (let i = 0; i < n; i++) {
          const dx = pos[i][0] - cx, dy = pos[i][1] - cy, dz = pos[i][2] - cz;
          const d = Math.hypot(dx, dy, dz) + 0.8;
          if (d > maxR) maxR = d;
        }
      }

      const spheres = [];
      const rods = [];
      const atomScale = spec.atomScale || (n > 50 ? 1.05 : 0.96);

      // Optional periodic crystal cell box edges
      if (boxA) {
        const h = boxA * 0.5;
        const corners = [
          [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
          [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
        ];
        const edges = [
          [0, 1], [1, 2], [2, 3], [3, 0],
          [4, 5], [5, 6], [6, 7], [7, 4],
          [0, 4], [1, 5], [2, 6], [3, 7],
        ];
        const boxCol = [0.33, 0.45, 0.56, 0.55];
        for (const [a, b] of edges) {
          rods.push({ p0: corners[a], p1: corners[b], r0: 0.045, r1: 0.045, color: boxCol });
        }
      }

      // Optional ghost context frames (3-frame history trails)
      if (spec.ghostFrames && spec.ghostFrames.length) {
        const gf = spec.ghostFrames;
        for (let g = 0; g < gf.length; g++) {
          const frame = gf[g];
          const frac = (g + 1) / (gf.length + 1);
          const col = [0.48, 0.72, 0.86, 0.22 + 0.18 * frac];
          for (let i = 0; i < Math.min(n, frame.length); i++) {
            const p = [frame[i][0] - cx, frame[i][1] - cy, frame[i][2] - cz];
            spheres.push({ c: p, r: 0.24 * frac + 0.10, color: col });
            const nextF = g + 1 < gf.length ? gf[g + 1][i] : pos[i];
            const pn = [nextF[0] - cx, nextF[1] - cy, nextF[2] - cz];
            if (Math.hypot(pn[0] - p[0], pn[1] - p[1], pn[2] - p[2]) < (boxA ? boxA * 0.4 : 6.0)) {
              rods.push({ p0: p, p1: pn, r0: 0.035, r1: 0.05, color: [0.57, 0.78, 0.72, 0.45] });
            }
          }
        }
      }

      // Highlight coordination shell edges around a central atom (Wigner-D chapter)
      const hiIdx = spec.highlightAtom ?? null;
      const cutA = spec.cutoffA ?? 6.0;
      const inShell = new Set();
      if (hiIdx !== null && hiIdx >= 0 && hiIdx < n) {
        const pCenter = [pos[hiIdx][0] - cx, pos[hiIdx][1] - cy, pos[hiIdx][2] - cz];
        for (let j = 0; j < n; j++) {
          if (j === hiIdx) continue;
          const pj = [pos[j][0] - cx, pos[j][1] - cy, pos[j][2] - cz];
          const d = Math.hypot(pj[0] - pCenter[0], pj[1] - pCenter[1], pj[2] - pCenter[2]);
          if (d <= cutA) {
            inShell.add(j);
            const env = Math.max(0.15, 1.0 - (d / cutA) ** 3);
            rods.push({
              p0: pCenter,
              p1: pj,
              r0: 0.055 * env + 0.02,
              r1: 0.055 * env + 0.02,
              color: [0.71, 0.85, 0.56, 0.85],
            });
          }
        }
      }

      // Atoms
      for (let i = 0; i < n; i++) {
        const el = elems[i] || 'C';
        const cpk = CPK[el] || [0.58, 0.68, 0.78, 0.44];
        let col = [cpk[0], cpk[1], cpk[2], 1.0];
        let rad = cpk[3] * atomScale;
        if (hiIdx !== null) {
          if (i === hiIdx) {
            col = [0.71, 0.85, 0.56, 1.0];
            rad *= 1.24;
          } else if (inShell.has(i)) {
            col = [0.57, 0.71, 0.88, 1.0];
            rad *= 1.08;
          } else {
            col = [0.22, 0.28, 0.35, 1.0];
            rad *= 0.82;
          }
        }
        spheres.push({
          c: [pos[i][0] - cx, pos[i][1] - cy, pos[i][2] - cz],
          r: rad,
          color: col,
        });
      }

      // Covalent bonds (split half-and-half by element color like PyMOL/ChimeraX!)
      for (const [i, j] of bonds) {
        if (i >= n || j >= n) continue;
        const p0 = [pos[i][0] - cx, pos[i][1] - cy, pos[i][2] - cz];
        const p1 = [pos[j][0] - cx, pos[j][1] - cy, pos[j][2] - cz];
        if (boxA && Math.hypot(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]) > boxA * 0.45) continue;
        const pm = [(p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5, (p0[2] + p1[2]) * 0.5];
        const c0 = CPK[elems[i]] || [0.5, 0.56, 0.64];
        const c1 = CPK[elems[j]] || [0.5, 0.56, 0.64];
        rods.push({ p0, p1: pm, r0: 0.115, r1: 0.115, color: [c0[0], c0[1], c0[2], 1.0] });
        rods.push({ p0: pm, p1, r0: 0.115, r1: 0.115, color: [c1[0], c1[1], c1[2], 1.0] });
      }

      // 3D Vector Arrows (shaft cylinder + conical 3D arrowhead)
      if (spec.arrows && spec.arrows.length) {
        for (const arr of spec.arrows) {
          const p0 = [arr.from[0] - cx, arr.from[1] - cy, arr.from[2] - cz];
          const s = arr.scale ?? 1.0;
          const dx = arr.vec[0] * s, dy = arr.vec[1] * s, dz = arr.vec[2] * s;
          const len = Math.hypot(dx, dy, dz);
          if (len < 0.04) continue;
          const headLen = Math.min(0.42, len * 0.36);
          const shaftFrac = (len - headLen) / len;
          const pNeck = [p0[0] + dx * shaftFrac, p0[1] + dy * shaftFrac, p0[2] + dz * shaftFrac];
          const pTip = [p0[0] + dx, p0[1] + dy, p0[2] + dz];
          const col = arr.color || [0.71, 0.85, 0.56, 1.0];
          const rShaft = arr.radius ?? 0.055;
          rods.push({ p0, p1: pNeck, r0: rShaft, r1: rShaft, color: col });
          rods.push({ p0: pNeck, p1: pTip, r0: rShaft * 2.5, r1: 0.004, color: col });
        }
      }

      this.scene = { spheres, rods, center: [cx, cy, cz], radius: maxR };
      this._uploadScene();
      this.render();
    }

    _uploadScene() {
      if (!this.gl) return;
      const gl = this.gl;
      const s = this.scene.spheres;
      const sData = new Float32Array(s.length * 8);
      for (let i = 0; i < s.length; i++) {
        const o = i * 8, it = s[i];
        sData[o] = it.c[0]; sData[o + 1] = it.c[1]; sData[o + 2] = it.c[2];
        sData[o + 3] = it.r;
        sData[o + 4] = it.color[0]; sData[o + 5] = it.color[1];
        sData[o + 6] = it.color[2]; sData[o + 7] = it.color[3];
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, this.sphereInstBuf);
      gl.bufferData(gl.ARRAY_BUFFER, sData, gl.DYNAMIC_DRAW);
      this.sphereCount = s.length;

      const r = this.scene.rods;
      const rData = new Float32Array(r.length * 12);
      for (let i = 0; i < r.length; i++) {
        const o = i * 12, it = r[i];
        rData[o] = it.p0[0]; rData[o + 1] = it.p0[1]; rData[o + 2] = it.p0[2];
        rData[o + 3] = it.p1[0]; rData[o + 4] = it.p1[1]; rData[o + 5] = it.p1[2];
        rData[o + 6] = it.r0; rData[o + 7] = it.r1;
        rData[o + 8] = it.color[0]; rData[o + 9] = it.color[1];
        rData[o + 10] = it.color[2]; rData[o + 11] = it.color[3];
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, this.cylInstBuf);
      gl.bufferData(gl.ARRAY_BUFFER, rData, gl.DYNAMIC_DRAW);
      this.rodCount = r.length;
    }

    _syncSize() {
      const rect = this.canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(2, Math.round((rect.width || this.canvas.width || 640) * dpr));
      const h = Math.max(2, Math.round((rect.height || this.canvas.height || 400) * dpr));
      if (this.canvas.width !== w || this.canvas.height !== h) {
        this.canvas.width = w;
        this.canvas.height = h;
      }
      return { w, h, dpr };
    }

    render() {
      const { w, h } = this._syncSize();
      if (!this.gl) {
        this._renderFallback2D(w, h);
        return;
      }
      const gl = this.gl;
      gl.viewport(0, 0, w, h);
      gl.clearColor(this.bg[0], this.bg[1], this.bg[2], 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      if (!this.sphereCount && !this.rodCount) return;

      const rad = Math.max(2.5, this.scene.radius);
      const dist = (rad * 2.55) / this.zoom;
      const near = Math.max(0.2, dist - rad * 2.4);
      const far = dist + rad * 2.8;
      const proj = mat4Perspective(0.68, w / h, near, far);
      const view = mat4FromOrbit(this.rotX, this.rotY, dist);
      const fogNear = dist - rad * 0.35;
      const fogFar = dist + rad * 1.65;

      if (this.sphereCount > 0) {
        gl.useProgram(this.sphereProg);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.sphereProg, 'uView'), false, view);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.sphereProg, 'uProj'), false, proj);
        gl.uniform3fv(gl.getUniformLocation(this.sphereProg, 'uFogColor'), this.bg);
        gl.uniform1f(gl.getUniformLocation(this.sphereProg, 'uFogNear'), fogNear);
        gl.uniform1f(gl.getUniformLocation(this.sphereProg, 'uFogFar'), fogFar);
        gl.bindVertexArray(this.sphereVao);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.sphereCount);
      }

      if (this.rodCount > 0) {
        gl.useProgram(this.cylProg);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.cylProg, 'uView'), false, view);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.cylProg, 'uProj'), false, proj);
        gl.uniform3fv(gl.getUniformLocation(this.cylProg, 'uFogColor'), this.bg);
        gl.uniform1f(gl.getUniformLocation(this.cylProg, 'uFogNear'), fogNear);
        gl.uniform1f(gl.getUniformLocation(this.cylProg, 'uFogFar'), fogFar);
        gl.bindVertexArray(this.cylVao);
        gl.drawElementsInstanced(gl.TRIANGLES, this.cylIndexCount, gl.UNSIGNED_SHORT, 0, this.rodCount);
      }
      gl.bindVertexArray(null);
    }

    _renderFallback2D(w, h) {
      const ctx = this.ctx2d;
      if (!ctx) return;
      ctx.fillStyle = `rgb(${Math.round(this.bg[0] * 255)},${Math.round(this.bg[1] * 255)},${Math.round(this.bg[2] * 255)})`;
      ctx.fillRect(0, 0, w, h);
      const cx = Math.cos(this.rotX), sx = Math.sin(this.rotX);
      const cy = Math.cos(this.rotY), sy = Math.sin(this.rotY);
      const scale = (Math.min(w, h) * 0.42 * this.zoom) / Math.max(2.5, this.scene.radius);
      const proj = p => {
        const x1 = p[0] * cy + p[2] * sy;
        const z1 = -p[0] * sy + p[2] * cy;
        const y2 = p[1] * cx - z1 * sx;
        const z2 = p[1] * sx + z1 * cx;
        return [w * 0.5 + x1 * scale, h * 0.5 - y2 * scale, z2];
      };
      for (const rod of this.scene.rods) {
        const a = proj(rod.p0), b = proj(rod.p1);
        ctx.strokeStyle = `rgba(${Math.round(rod.color[0] * 255)},${Math.round(rod.color[1] * 255)},${Math.round(rod.color[2] * 255)},${rod.color[3]})`;
        ctx.lineWidth = Math.max(1.2, rod.r0 * scale * 1.5);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      }
      const sorted = this.scene.spheres.map(s => ({ ...s, p: proj(s.c) })).sort((a, b) => a.p[2] - b.p[2]);
      for (const s of sorted) {
        const rPx = Math.max(2.5, s.r * scale);
        const g = ctx.createRadialGradient(s.p[0] - rPx * 0.32, s.p[1] - rPx * 0.34, rPx * 0.08, s.p[0], s.p[1], rPx);
        const R = Math.round(s.color[0] * 255), G = Math.round(s.color[1] * 255), B = Math.round(s.color[2] * 255);
        g.addColorStop(0, `rgb(${Math.min(255, R + 85)},${Math.min(255, G + 85)},${Math.min(255, B + 85)})`);
        g.addColorStop(0.6, `rgb(${R},${G},${B})`);
        g.addColorStop(1, `rgb(${Math.round(R * 0.35)},${Math.round(G * 0.35)},${Math.round(B * 0.38)})`);
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(s.p[0], s.p[1], rPx, 0, Math.PI * 2); ctx.fill();
      }
    }
  }

  window.WignerGLViewer = WignerGLViewer;
})();
