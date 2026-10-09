import * as THREE from 'three';

// Display-only motion cues: soil clods, dust and cartoon puffs. None of it
// feeds back into the recorded Terra state.
const GRAVITY = 9.81;
const CLOD_COLORS = [0xa8713f, 0xb98049, 0x94602f, 0xc48f58];
const MAX_CLODS = 900, MAX_DUST = 240;
const up = new THREE.Vector3(0, 1, 0);

// Soft, camera-facing dust: additive-free alpha blending with a radial falloff.
const DUST_SHADER = {
  vertexShader: /* glsl */`
    attribute float size; attribute float alpha; varying float vAlpha;
    uniform float scale;
    void main() {
      vec4 view = modelViewMatrix * vec4(position, 1.);
      gl_PointSize = size * scale / -view.z; vAlpha = alpha;
      gl_Position = projectionMatrix * view;
    }`,
  fragmentShader: /* glsl */`
    uniform vec3 color; varying float vAlpha;
    void main() {
      float r = length(gl_PointCoord - .5) * 2.;
      float a = vAlpha * smoothstep(1., .15, r);
      if (a < .003) discard;
      gl_FragColor = vec4(color, a);
      #include <colorspace_fragment>
    }`,
};

export class Effects extends THREE.Group {
  constructor({ groundHeight = () => 0 } = {}) {
    super();
    this.name = 'Terra effects'; this.groundHeight = groundHeight;
    this.dummy = new THREE.Object3D(); this.color = new THREE.Color();
    const clodGeometry = new THREE.IcosahedronGeometry(1, 0);
    this.clodMesh = new THREE.InstancedMesh(clodGeometry, new THREE.MeshStandardMaterial({ roughness: 1, flatShading: true }), MAX_CLODS);
    this.clodMesh.castShadow = true; this.clodMesh.frustumCulled = false; this.clodMesh.count = 0; this.add(this.clodMesh);
    this.puffMesh = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 1), new THREE.MeshStandardMaterial({ roughness: 1, flatShading: true }), 260);
    this.puffMesh.frustumCulled = false; this.puffMesh.count = 0; this.add(this.puffMesh);
    for (const mesh of [this.clodMesh, this.puffMesh]) { mesh.setColorAt(0, this.color.setHex(0xffffff)); mesh.userData.skipAO = true; }
    const dustGeometry = new THREE.BufferGeometry();
    dustGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAX_DUST * 3), 3).setUsage(THREE.DynamicDrawUsage));
    dustGeometry.setAttribute('size', new THREE.BufferAttribute(new Float32Array(MAX_DUST), 1).setUsage(THREE.DynamicDrawUsage));
    dustGeometry.setAttribute('alpha', new THREE.BufferAttribute(new Float32Array(MAX_DUST), 1).setUsage(THREE.DynamicDrawUsage));
    this.dustMaterial = new THREE.ShaderMaterial({ ...DUST_SHADER, uniforms: { color: { value: new THREE.Color(0xc9b79c) }, scale: { value: 600 } }, transparent: true, depthWrite: false });
    this.dustPoints = new THREE.Points(dustGeometry, this.dustMaterial); this.dustPoints.frustumCulled = false; this.dustPoints.renderOrder = 40; this.dustPoints.userData.skipAO = true; this.add(this.dustPoints);
    this.clods = []; this.puffs = []; this.dusts = []; this.puffsEnabled = true; this.dustEnabled = false; this.palette = CLOD_COLORS;
  }

  /** Point-size scale for a viewport height in pixels and a vertical field of view. */
  setViewport(height, fov) { this.dustMaterial.uniforms.scale.value = height / (2 * Math.tan(THREE.MathUtils.degToRad(fov) / 2)); }

  /** Slow, translucent dust that billows, drifts and fades (studio style). */
  dust(position, { count = 6, size = .6, spread = .3, rise = .35, life = 1.6, opacity = .22, drift = null } = {}) {
    if (!this.dustEnabled) return;
    for (let i = 0; i < count && this.dusts.length < MAX_DUST; i++) {
      const offset = new THREE.Vector3((Math.random() - .5) * spread * 2, Math.random() * spread * .4, (Math.random() - .5) * spread * 2);
      const velocity = offset.clone().multiplyScalar(.8).add(up.clone().multiplyScalar(rise * (.5 + Math.random() * .7)));
      if (drift) velocity.add(drift);
      this.dusts.push({ position: position.clone().add(offset), velocity, size: size * (.6 + Math.random() * .7), age: -Math.random() * .15, life: life * (.75 + Math.random() * .5), opacity });
    }
  }

  /** Throw `count` clods from `from` so they land around `to` after `flight` seconds. */
  throwClods(from, to, { count = 10, flight = .45, spread = .3, size = .09, settle = true, jitter = .08 } = {}) {
    for (let i = 0; i < count && this.clods.length < MAX_CLODS; i++) {
      const start = from.clone().add(new THREE.Vector3((Math.random() - .5) * jitter * 2, (Math.random() - .5) * jitter, (Math.random() - .5) * jitter * 2));
      const end = to.clone().add(new THREE.Vector3((Math.random() - .5) * spread * 2, 0, (Math.random() - .5) * spread * 2));
      const time = flight * (.8 + Math.random() * .4), delay = Math.random() * flight * .6;
      const velocity = end.sub(start).multiplyScalar(1 / time); velocity.y += .5 * GRAVITY * time;
      this.clods.push({ position: start, velocity, spin: new THREE.Vector3(Math.random() * 9, Math.random() * 9, Math.random() * 9), rotation: new THREE.Euler(Math.random() * 6, Math.random() * 6, 0), size: size * (.6 + Math.random() * .8), age: -delay, life: time + (settle ? 1.4 : 0), arrive: time, settle, bounced: false, color: this.palette[Math.floor(Math.random() * this.palette.length)] });
    }
  }

  /** Spray clods upward/outward from a point, e.g. when the bucket bites. */
  burst(origin, { count = 12, speed = 2.2, size = .07 } = {}) {
    for (let i = 0; i < count && this.clods.length < MAX_CLODS; i++) {
      const angle = Math.random() * Math.PI * 2, lift = .55 + Math.random() * .5;
      const velocity = new THREE.Vector3(Math.cos(angle) * (1 - lift), lift * 1.4, Math.sin(angle) * (1 - lift)).multiplyScalar(speed * (.6 + Math.random() * .6));
      this.clods.push({ position: origin.clone(), velocity, spin: new THREE.Vector3(Math.random() * 12, Math.random() * 12, 0), rotation: new THREE.Euler(), size: size * (.6 + Math.random() * .8), age: -Math.random() * .08, life: 2.2, arrive: Infinity, settle: true, bounced: false, color: this.palette[Math.floor(Math.random() * this.palette.length)] });
    }
  }

  /** Solid cartoon puffs that swell, drift and shrink away. */
  puff(position, { count = 6, size = .25, spread = .35, rise = .6, life = .9, color = 0xe9dcc2, drift = null } = {}) {
    if (!this.puffsEnabled) return;
    for (let i = 0; i < count && this.puffs.length < 260; i++) {
      const offset = new THREE.Vector3((Math.random() - .5) * spread * 2, Math.random() * spread * .5, (Math.random() - .5) * spread * 2);
      const velocity = offset.clone().multiplyScalar(1.4).add(up.clone().multiplyScalar(rise * (.6 + Math.random() * .6)));
      if (drift) velocity.add(drift);
      this.puffs.push({ position: position.clone().add(offset), velocity, size: size * (.6 + Math.random() * .7), age: -Math.random() * .12, life: life * (.75 + Math.random() * .5), color });
    }
  }

  update(dt) {
    // Sub-step so accelerated playback keeps the same ballistic arcs.
    dt = Math.min(Math.max(dt, 0), .25);
    for (let left = dt; left > 1e-6; left -= .025) this.step(Math.min(.025, left));
    this.draw();
  }

  step(dt) {
    this.clods = this.clods.filter(clod => {
      clod.age += dt; if (clod.age < 0) return true;
      if (clod.age > clod.life) return false;
      if (clod.age < clod.arrive || clod.settle) {
        if (!clod.resting) {
          clod.velocity.y -= GRAVITY * dt; clod.position.addScaledVector(clod.velocity, dt);
          clod.rotation.x += clod.spin.x * dt; clod.rotation.y += clod.spin.y * dt; clod.rotation.z += clod.spin.z * dt;
          const ground = this.groundHeight(clod.position.x, clod.position.z) + clod.size * .5;
          if (clod.position.y < ground && clod.velocity.y < 0) {
            clod.position.y = ground;
            if (!clod.bounced && clod.velocity.y < -1.2) { clod.velocity.y *= -.28; clod.velocity.x *= .45; clod.velocity.z *= .45; clod.bounced = true; }
            else { clod.resting = true; clod.restAge = clod.age; }
          }
        }
      }
      return !(clod.age >= clod.arrive && !clod.settle);
    });
    this.puffs = this.puffs.filter(puff => { puff.age += dt; if (puff.age >= 0) { puff.position.addScaledVector(puff.velocity, dt); puff.velocity.multiplyScalar(Math.exp(-dt * 2.2)); } return puff.age < puff.life; });
    this.dusts = this.dusts.filter(dust => { dust.age += dt; if (dust.age >= 0) { dust.position.addScaledVector(dust.velocity, dt); dust.velocity.multiplyScalar(Math.exp(-dt * 1.6)); } return dust.age < dust.life; });
  }

  draw() {
    const dummy = this.dummy;
    const clodCount = this.clods.length;
    for (let i = 0; i < clodCount; i++) {
      const clod = this.clods[i], visible = clod.age >= 0;
      const fade = clod.resting ? Math.max(0, 1 - (clod.age - clod.restAge) / .9) : 1;
      dummy.position.copy(clod.position); dummy.rotation.copy(clod.rotation); dummy.scale.setScalar(visible ? clod.size * fade : 0); dummy.updateMatrix();
      this.clodMesh.setMatrixAt(i, dummy.matrix); this.clodMesh.setColorAt(i, this.color.setHex(clod.color));
    }
    this.clodMesh.count = clodCount; this.clodMesh.instanceMatrix.needsUpdate = true; if (this.clodMesh.instanceColor) this.clodMesh.instanceColor.needsUpdate = true;

    for (let i = 0; i < this.puffs.length; i++) {
      const puff = this.puffs[i];
      const t = Math.max(0, puff.age) / puff.life, grow = puff.age < 0 ? 0 : Math.sin(Math.min(1, t * 1.25) * Math.PI) * (1 + t * .6);
      dummy.position.copy(puff.position); dummy.rotation.set(puff.age * .7, puff.age, 0); dummy.scale.setScalar(puff.size * grow); dummy.updateMatrix();
      this.puffMesh.setMatrixAt(i, dummy.matrix); this.puffMesh.setColorAt(i, this.color.setHex(puff.color));
    }
    this.puffMesh.count = this.puffs.length; this.puffMesh.instanceMatrix.needsUpdate = true; if (this.puffMesh.instanceColor) this.puffMesh.instanceColor.needsUpdate = true;
    const position = this.dustPoints.geometry.attributes.position, size = this.dustPoints.geometry.attributes.size, alpha = this.dustPoints.geometry.attributes.alpha;
    for (let i = 0; i < this.dusts.length; i++) {
      const dust = this.dusts[i], t = Math.max(0, dust.age) / dust.life;
      position.setXYZ(i, dust.position.x, dust.position.y, dust.position.z);
      size.setX(i, dust.age < 0 ? 0 : dust.size * (.55 + t * 1.1));
      alpha.setX(i, dust.age < 0 ? 0 : dust.opacity * Math.min(1, t * 6) * (1 - t) ** 1.5);
    }
    this.dustPoints.geometry.setDrawRange(0, this.dusts.length);
    for (const attribute of [position, size, alpha]) attribute.needsUpdate = true;
  }

  clear() {
    this.clods = []; this.puffs = []; this.dusts = []; this.clodMesh.count = 0; this.puffMesh.count = 0; this.dustPoints.geometry.setDrawRange(0, 0);
  }

  dispose() {
    this.clear();
    for (const mesh of [this.clodMesh, this.puffMesh]) { mesh.geometry.dispose(); mesh.material.dispose(); mesh.dispose(); }
    this.dustPoints.geometry.dispose(); this.dustMaterial.dispose();
    this.removeFromParent();
  }
}
