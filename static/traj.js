// Interactive trajectory viewer: 16 curated ImageNet samples arranged on a sphere
// (overview), and for a selected sample its 250-step trajectory in the space of
// the first three principal directions, with level sets of the predicted
// distance u_theta.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { MarchingCubes } from 'three/addons/objects/MarchingCubes.js';

const ROOT = 'static/traj';
const SAMPLER_NAME = { st: 'Sphere Tracing', gd: 'Gradient Descent' };
const SAMPLER_COLOR = { st: '#eb6834', gd: '#2a78d6' };
// one-hue ramp for the level sets: light = far from the data, dark = close
const LEVEL_COLORS = ['#d9d9de', '#c4c4cb', '#a9a9b3', '#8b8b97', '#6b6b78'];
const LEVEL_FRACTIONS = [0.85, 0.6, 0.38, 0.2, 0.06];

let manifestPromise = null;
function loadManifest() {
  manifestPromise ??= fetch(`${ROOT}/manifest.json`).then((r) => r.json());
  return manifestPromise;
}

function textSprite(text, { size = 34, color = '#52514e', weight = 500 } = {}) {
  const pad = 8;
  const c = document.createElement('canvas');
  const ctx = c.getContext('2d');
  const font = `${weight} ${size}px Inter, -apple-system, sans-serif`;
  ctx.font = font;
  const w = Math.ceil(ctx.measureText(text).width) + pad * 2;
  const h = size + pad * 2;
  c.width = w;
  c.height = h;
  ctx.font = font;
  ctx.fillStyle = color;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, pad, h / 2);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  s.scale.set((w / h) * 0.09, 0.09, 1);
  s.renderOrder = 10;
  return s;
}

function disposeTree(obj) {
  obj.traverse((o) => {
    o.geometry?.dispose?.();
    const mats = Array.isArray(o.material) ? o.material : o.material ? [o.material] : [];
    mats.forEach((m) => { m.map?.dispose?.(); m.dispose(); });
  });
}

class Explorer {
  constructor(root, sampler) {
    this.root = root;
    this.sampler = sampler;
    this.color = new THREE.Color(SAMPLER_COLOR[sampler]);
    this.loader = new THREE.TextureLoader();
    this.mode = 'overview';
    this.visible = false;
    this.hovered = null;
    this.build();
  }

  build() {
    const r = this.root;
    r.innerHTML = `
      <div class="tv-stage">
        <canvas></canvas>
        <div class="tv-note tv-note-overview">
          <b>Not the actual data manifold.</b> The 16 samples are only arranged on a sphere for browsing.
          Select an image to view its trajectory.
        </div>
        <button class="tv-back" type="button" hidden>← All samples</button>
        <div class="tv-note tv-note-detail" hidden>Drag to rotate · scroll to zoom</div>
        <div class="tv-loading" hidden>Loading…</div>
        <div class="tv-hover" hidden></div>
        <div class="tv-legend" hidden></div>
      </div>
      <aside class="tv-panel" hidden></aside>`;
    this.stage = r.querySelector('.tv-stage');
    this.canvas = r.querySelector('canvas');
    this.panel = r.querySelector('.tv-panel');
    this.backBtn = r.querySelector('.tv-back');
    this.noteOverview = r.querySelector('.tv-note-overview');
    this.noteDetail = r.querySelector('.tv-note-detail');
    this.loading = r.querySelector('.tv-loading');
    this.hoverLabel = r.querySelector('.tv-hover');
    this.legend = r.querySelector('.tv-legend');

    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0xffffff, 0);

    // overview
    this.ovScene = new THREE.Scene();
    this.ovCamera = new THREE.PerspectiveCamera(35, 1, 0.1, 50);
    this.ovCamera.position.set(0, 0, 5.2);
    this.ovGroup = new THREE.Group();
    this.ovGroup.rotation.x = 0.25;
    this.ovScene.add(this.ovGroup);
    const sphere = new THREE.LineSegments(
      new THREE.WireframeGeometry(new THREE.SphereGeometry(1.32, 24, 12)),
      new THREE.LineBasicMaterial({ color: 0xe3e3e8, transparent: true, opacity: 0.7 }),
    );
    this.ovGroup.add(sphere);
    this.cards = [];

    // detail
    this.dtScene = new THREE.Scene();
    this.dtScene.add(new THREE.AmbientLight(0xffffff, 1.6));
    const light = new THREE.DirectionalLight(0xffffff, 1.4);
    light.position.set(2, 3, 4);
    this.dtScene.add(light);
    this.dtCamera = new THREE.PerspectiveCamera(35, 1, 0.05, 50);
    this.dtCamera.position.set(2.6, 1.7, 3.4);
    this.controls = new OrbitControls(this.dtCamera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.enablePan = false;
    this.controls.minDistance = 1.8;
    this.controls.maxDistance = 9;
    this.controls.enabled = false;
    this.dtGroup = null;

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2(-10, -10);
    this.canvas.addEventListener('pointermove', (e) => this.onPointerMove(e));
    this.canvas.addEventListener('pointerleave', () => this.pointer.set(-10, -10));
    this.canvas.addEventListener('click', (e) => this.onClick(e));
    this.backBtn.addEventListener('click', () => this.showOverview());

    new ResizeObserver(() => this.resize()).observe(this.stage);
    new IntersectionObserver((es) => {
      this.visible = es[0].isIntersecting;
      if (this.visible) this.loop();
    }).observe(this.stage);
    this.resize();
    this.initOverview();
  }

  async initOverview() {
    const m = await loadManifest();
    this.manifest = m;
    const n = m.indices.length;
    const golden = Math.PI * (3 - Math.sqrt(5));
    m.indices.forEach((idx, i) => {
      const y = 1 - (2 * (i + 0.5)) / n;
      const rad = Math.sqrt(1 - y * y);
      const th = golden * i;
      const pos = new THREE.Vector3(Math.cos(th) * rad, y, Math.sin(th) * rad).multiplyScalar(1.32);
      const tex = this.loader.load(`${ROOT}/${this.sampler}/${String(idx).padStart(6, '0')}.jpg`);
      tex.colorSpace = THREE.SRGBColorSpace;
      const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true });
      const card = new THREE.Mesh(new THREE.PlaneGeometry(0.46, 0.46), mat);
      card.position.copy(pos);
      card.userData = { index: idx, label: m.labels[i], k: i };
      this.ovGroup.add(card);
      this.cards.push(card);
    });
  }

  resize() {
    const w = this.stage.clientWidth;
    const h = this.stage.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    for (const cam of [this.ovCamera, this.dtCamera]) {
      cam.aspect = w / h;
      cam.updateProjectionMatrix();
    }
  }

  onPointerMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    this.pointer.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    this.hoverLabel.style.left = `${e.clientX - rect.left + 14}px`;
    this.hoverLabel.style.top = `${e.clientY - rect.top + 10}px`;
  }

  pickCard() {
    if (this.mode !== 'overview' || !this.cards.length) return null;
    this.raycaster.setFromCamera(this.pointer, this.ovCamera);
    const hits = this.raycaster.intersectObjects(this.cards, false);
    return hits.length ? hits[0].object : null;
  }

  onClick(e) {
    this.onPointerMove(e);
    const card = this.pickCard();
    if (card) this.showDetail(card.userData);
  }

  showOverview() {
    this.stopPlay();
    this.mode = 'overview';
    this.controls.enabled = false;
    this.panel.hidden = true;
    this.backBtn.hidden = true;
    this.noteDetail.hidden = true;
    this.legend.hidden = true;
    this.noteOverview.hidden = false;
    this.root.classList.remove('tv-detail');
    this.resize();
    this.loop();
  }

  async showDetail({ index, label }) {
    this.mode = 'loading';
    this.hoverLabel.hidden = true;
    this.canvas.style.cursor = 'default';
    this.loading.hidden = false;
    this.noteOverview.hidden = true;
    const base = `${ROOT}/${this.sampler}/${String(index).padStart(6, '0')}`;
    const [meta, buf] = await Promise.all([
      fetch(`${base}/meta.json`).then((r) => r.json()),
      fetch(`${base}/grid.bin`).then((r) => r.arrayBuffer()),
    ]);
    if (this.dtGroup) {
      this.dtScene.remove(this.dtGroup);
      disposeTree(this.dtGroup);
    }
    this.meta = meta;
    this.dtGroup = this.buildDetail(meta, new Uint16Array(buf), base);
    this.dtScene.add(this.dtGroup);
    this.buildPanel(meta, base);
    this.legend.innerHTML = `<div class="tv-legend-title">Level sets of u<sub>θ</sub></div>` +
      this.levels.map((lv, i) => `<div><span style="background:${LEVEL_COLORS[i]}"></span>u = ${lv.toFixed(1)}</div>`).join('') +
      `<div class="tv-legend-note">darkest: closest to the data manifold</div>`;
    this.legend.hidden = false;
    this.setStep(meta.coords.length - 1);
    this.dtCamera.position.set(3.1, 1.5, 4.6);
    this.controls.target.set(0, 0.3, 0);
    this.controls.enabled = true;
    this.mode = 'detail';
    this.loading.hidden = true;
    this.backBtn.hidden = false;
    this.noteDetail.hidden = false;
    this.panel.hidden = false;
    this.root.classList.add('tv-detail');
    this.resize();
    this.loop();
  }

  // grid index space -> MarchingCubes local space ([-1, 1) per axis)
  toLocal(c, a) {
    const N = this.meta.grid_res;
    const [lo, hi] = this.meta.axes[a];
    const i = ((c - lo) / (hi - lo)) * (N - 1);
    return (i - N / 2) / (N / 2);
  }

  buildDetail(meta, q, base) {
    const g = new THREE.Group();
    const N = meta.grid_res;
    const [gmin, gmax] = meta.grid_q;
    // reorder (PC1, PC2, PC3) C-order grid into MarchingCubes order (x fastest)
    const field = new Float32Array(N * N * N);
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++)
        for (let k = 0; k < N; k++)
          field[i + j * N + k * N * N] = gmin + (q[i * N * N + j * N + k] / 65535) * (gmax - gmin);

    // level sets between the endpoint value and the starting value of u
    const uEnd = meta.u[meta.u.length - 1];
    const uStart = meta.u[0];
    this.levels = LEVEL_FRACTIONS.map((f) => uEnd + f * (uStart - uEnd));
    this.levels.forEach((lv, li) => {
      const mat = new THREE.MeshLambertMaterial({
        color: LEVEL_COLORS[li], transparent: true, opacity: li === LEVEL_COLORS.length - 1 ? 0.3 : 0.13,
        side: THREE.DoubleSide, depthWrite: false,
      });
      const mc = new MarchingCubes(N, mat, false, false, 120000);
      mc.field.set(field);
      mc.isolation = lv;
      mc.update();
      mc.renderOrder = 1;
      g.add(mc);
    });

    // bounding box and axis labels
    const box = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(2, 2, 2)),
      new THREE.LineBasicMaterial({ color: 0xc9c9cf }),
    );
    box.position.set(-0.025, -0.025, -0.025);
    g.add(box);
    const ev = meta.explained_variance_ratio.map((r) => (r * 100).toFixed(r < 0.001 ? 3 : r < 0.1 ? 2 : 1));
    [['PC1', ev[0], [0, -1.12, 1.08]], ['PC2', ev[1], [-1.12, 0, 1.08]], ['PC3', ev[2], [1.08, -1.12, 0]]].forEach(([n, e, p]) => {
      const s = textSprite(`${n} · ${e}%`);
      s.position.set(...p);
      g.add(s);
    });

    // trajectory
    const pts = meta.coords.map((c) => new THREE.Vector3(this.toLocal(c[0], 0), this.toLocal(c[1], 1), this.toLocal(c[2], 2)));
    this.points = pts;
    const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal');
    const tube = new THREE.Mesh(
      new THREE.TubeGeometry(curve, 600, 0.011, 8, false),
      new THREE.MeshLambertMaterial({ color: this.color }),
    );
    tube.renderOrder = 2;
    g.add(tube);

    // thumbnails every 50 steps, on a rail above the box with leader lines to their states
    const steps = this.manifest.thumb_steps;
    steps.forEach((s, n) => {
      const p = pts[s];
      const dot = new THREE.Mesh(new THREE.SphereGeometry(0.032, 16, 12), new THREE.MeshLambertMaterial({ color: this.color }));
      dot.position.copy(p);
      g.add(dot);
      const tex = this.loader.load(`${base}/step_${String(s).padStart(3, '0')}.jpg`);
      tex.colorSpace = THREE.SRGBColorSpace;
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false }));
      sp.scale.set(0.36, 0.36, 1);
      sp.position.set(-1.25 + (2.5 * n) / (steps.length - 1), 1.5, 0);
      sp.renderOrder = 5;
      g.add(sp);
      const anchor = sp.position.clone().add(new THREE.Vector3(0, -0.19, 0));
      const lineGeo = new THREE.BufferGeometry().setFromPoints([p, anchor]);
      g.add(new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ color: 0xbdbdc4 })));
      const lab = textSprite(`step ${s}`, { size: 30 });
      lab.position.copy(sp.position).add(new THREE.Vector3(0, 0.25, 0));
      g.add(lab);
    });

    // current-step marker
    this.marker = new THREE.Mesh(
      new THREE.SphereGeometry(0.05, 20, 16),
      new THREE.MeshBasicMaterial({ color: 0x0b0b0b }),
    );
    this.marker.renderOrder = 6;
    g.add(this.marker);
    return g;
  }

  buildPanel(meta, base) {
    const name = this.manifest.class_names?.[String(meta.label)] ?? `class ${meta.label}`;
    const steps = this.manifest.thumb_steps;
    const ev = meta.explained_variance_ratio.map((r) => `${(r * 100).toFixed(2)}%`).join(' / ');
    this.panel.innerHTML = `
      <p class="tv-eyebrow">${SAMPLER_NAME[this.sampler]}, <span class="tv-nocase">η = ${meta.stepsize}</span></p>
      <h3>${name}</h3>
      <p class="tv-meta">ImageNet class ${meta.label} · XL/2 · CFG 4.0 · 250 steps</p>
      <div class="tv-strip">${steps.map((s) => `
        <figure data-step="${s}"><img src="${base}/step_${String(s).padStart(3, '0')}.jpg" alt="step ${s}"><figcaption>${s}</figcaption></figure>`).join('')}
      </div>
      <div class="tv-chart">
        <div class="tv-chart-title">Predicted distance  u<sub>θ</sub> along the trajectory</div>
        <svg viewBox="0 0 300 120" preserveAspectRatio="none" role="img" aria-label="Predicted distance over sampling steps"></svg>
        <div class="tv-tip" hidden></div>
      </div>
      <div class="tv-scrub">
        <button type="button" class="tv-play" aria-label="Play">▶</button>
        <input type="range" min="0" max="${meta.coords.length - 1}" value="${meta.coords.length - 1}" aria-label="Step">
        <span class="tv-readout"></span>
      </div>
      <p class="tv-explain">
        Coordinates: projection onto the first three principal directions of this trajectory's 250 latent states,
        centered at the generated sample (explained variance ${ev}). Shells: level sets of
        u<sub>θ</sub>(x<sub>end</sub> + Pz) for this class, darker = smaller distance. Axes are rescaled
        independently so that all three directions are visible.
      </p>`;
    this.slider = this.panel.querySelector('input[type=range]');
    this.readout = this.panel.querySelector('.tv-readout');
    this.slider.addEventListener('input', () => { this.stopPlay(); this.setStep(+this.slider.value); });
    this.playBtn = this.panel.querySelector('.tv-play');
    this.playBtn.addEventListener('click', () => (this.playing ? this.stopPlay() : this.startPlay()));
    this.panel.querySelectorAll('.tv-strip figure').forEach((f) =>
      f.addEventListener('click', () => { this.stopPlay(); this.setStep(+f.dataset.step); }));
    this.drawChart(meta);
  }

  drawChart(meta) {
    const svg = this.panel.querySelector('svg');
    const tip = this.panel.querySelector('.tv-tip');
    const W = 300, H = 120, L = 30, R = 6, T = 8, B = 18;
    const u = meta.u;
    const n = u.length;
    const umax = Math.ceil(Math.max(...u) / 10) * 10;
    const x = (i) => L + (i / (n - 1)) * (W - L - R);
    const y = (v) => T + (1 - v / umax) * (H - T - B);
    this.chartX = x;
    this.chartY = y;
    const ticks = [0, umax / 2, umax];
    const path = u.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('');
    svg.innerHTML = `
      ${ticks.map((t) => `<line x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}" class="grid"/>
        <text x="${L - 5}" y="${y(t) + 3}" class="tick" text-anchor="end">${t}</text>`).join('')}
      ${[0, 50, 100, 150, 200, 249].map((s) => `<text x="${x(s)}" y="${H - 4}" class="tick" text-anchor="middle">${s}</text>`).join('')}
      <path d="${path}" class="series" style="stroke:${SAMPLER_COLOR[this.sampler]}"/>
      <line class="cursor" y1="${T}" y2="${H - B}"/>
      <circle class="cursor-dot" r="4" style="fill:${SAMPLER_COLOR[this.sampler]}"/>
      <rect x="${L}" y="0" width="${W - L - R}" height="${H}" fill="transparent" class="hit"/>`;
    const hit = svg.querySelector('.hit');
    const toStep = (e) => {
      const r = svg.getBoundingClientRect();
      const px = ((e.clientX - r.left) / r.width) * W;
      return Math.max(0, Math.min(n - 1, Math.round(((px - L) / (W - L - R)) * (n - 1))));
    };
    hit.addEventListener('pointermove', (e) => {
      const s = toStep(e);
      tip.hidden = false;
      tip.textContent = `step ${s} · u = ${u[s].toFixed(2)}`;
      const r = svg.getBoundingClientRect();
      tip.style.left = `${(x(s) / W) * r.width}px`;
    });
    hit.addEventListener('pointerleave', () => { tip.hidden = true; });
    hit.addEventListener('click', (e) => { this.stopPlay(); this.setStep(toStep(e)); });
  }

  setStep(s) {
    this.step = s;
    if (this.marker) this.marker.position.copy(this.points[s]);
    if (this.slider) this.slider.value = s;
    const u = this.meta.u[s];
    if (this.readout) this.readout.textContent = `step ${s} · u = ${u.toFixed(2)}`;
    const svg = this.panel.querySelector('svg');
    if (svg) {
      const cx = this.chartX(s);
      svg.querySelector('.cursor').setAttribute('x1', cx);
      svg.querySelector('.cursor').setAttribute('x2', cx);
      const dot = svg.querySelector('.cursor-dot');
      dot.setAttribute('cx', cx);
      dot.setAttribute('cy', this.chartY(u));
    }
    const steps = this.manifest.thumb_steps;
    const near = steps.reduce((a, b) => (Math.abs(b - s) < Math.abs(a - s) ? b : a));
    this.panel.querySelectorAll('.tv-strip figure').forEach((f) => f.classList.toggle('active', +f.dataset.step === near));
    this.loop();
  }

  startPlay() {
    this.playing = true;
    this.playBtn.textContent = '❚❚';
    if (this.step >= this.meta.coords.length - 1) this.setStep(0);
    const tick = () => {
      if (!this.playing) return;
      const next = this.step + 1;
      if (next >= this.meta.coords.length) { this.stopPlay(); return; }
      this.setStep(next);
      this.playTimer = setTimeout(tick, 24);
    };
    tick();
  }

  stopPlay() {
    this.playing = false;
    clearTimeout(this.playTimer);
    if (this.playBtn) this.playBtn.textContent = '▶';
  }

  loop() {
    if (this.raf) return;
    const frame = () => {
      this.raf = null;
      if (!this.visible) return;
      if (this.mode === 'overview') {
        const card = this.pickCard();
        if (card !== this.hovered) {
          this.hovered = card;
          this.canvas.style.cursor = card ? 'pointer' : 'default';
          this.hoverLabel.hidden = !card;
          if (card) this.hoverLabel.textContent = this.manifest.class_names?.[String(card.userData.label)] ?? '';
        }
        if (!card) this.ovGroup.rotation.y += 0.0035;
        const tmp = new THREE.Vector3();
        for (const c of this.cards) {
          c.quaternion.copy(this.ovGroup.quaternion).invert().multiply(this.ovCamera.quaternion);
          c.getWorldPosition(tmp);
          const depth = THREE.MathUtils.clamp((tmp.z + 1.32) / 2.64, 0, 1);
          c.material.opacity = 0.25 + 0.75 * depth;
          c.renderOrder = Math.round(depth * 100);
          const target = c === card ? 1.35 : 1;
          c.scale.setScalar(c.scale.x + (target - c.scale.x) * 0.2);
        }
        this.renderer.render(this.ovScene, this.ovCamera);
        this.raf = requestAnimationFrame(frame);
      } else if (this.mode === 'detail') {
        const moving = this.controls.update();
        this.renderer.render(this.dtScene, this.dtCamera);
        this.raf = requestAnimationFrame(frame);
      }
    };
    this.raf = requestAnimationFrame(frame);
  }
}

document.querySelectorAll('[data-traj-viewer]').forEach((el) => new Explorer(el, el.dataset.trajViewer));
