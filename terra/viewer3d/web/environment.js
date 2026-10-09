import * as THREE from 'three';
import { PALETTES, earthMaterial, shared } from './materials.js';

// Decorative surroundings outside the Terra grid: a floating diorama island,
// site fence, trees, a road and a site office. Nothing here is inside a map
// cell, so it never reads as an obstacle, target or soil state.
function random(seed) {
  let state = seed >>> 0 || 1;
  return () => { state = Math.imul(state ^ (state >>> 15), 0x2c1b3c6d) + 0x6d2b79f5 >>> 0; state ^= state >>> 13; return (state >>> 0) / 0xffffffff; };
}

function roundedRect(shape, hx, hz, radius, clockwise = false) {
  const r = Math.min(radius, hx * .98, hz * .98), points = [];
  const corners = [[hx - r, hz - r, 0], [-hx + r, hz - r, Math.PI / 2], [-hx + r, -hz + r, Math.PI], [hx - r, -hz + r, Math.PI * 1.5]];
  for (const [cx, cz, start] of corners) for (let i = 0; i <= 6; i++) { const a = start + i / 6 * Math.PI / 2; points.push(new THREE.Vector2(cx + Math.cos(a) * r, cz + Math.sin(a) * r)); }
  if (clockwise) points.reverse();
  if (shape) { shape.setFromPoints(points); return shape; }
  return points;
}

function islandUnderside(hx, hz, radius, depth, seed) {
  const rand = random(seed), top = roundedRect(null, hx, hz, radius), rings = [top];
  const steps = [[.9, .34], [.66, .72], [.26, 1]];
  for (const [scale, drop] of steps) rings.push(top.map(p => { const jitter = .9 + rand() * .2; return new THREE.Vector3(p.x * scale * jitter, -depth * drop * (.85 + rand() * .3), p.y * scale * jitter); }));
  rings[0] = top.map(p => new THREE.Vector3(p.x, 0, p.y));
  const positions = [], colors = [], shade = new THREE.Color();
  const palette = [0x8c7a66, 0x7d6c5c, 0x96846e, 0x6f6152];
  const push = (a, b, c) => { shade.setHex(palette[Math.floor(rand() * palette.length)]); for (const v of [a, b, c]) { positions.push(v.x, v.y, v.z); colors.push(shade.r, shade.g, shade.b); } };
  for (let ring = 0; ring < rings.length - 1; ring++) for (let i = 0; i < top.length; i++) {
    const j = (i + 1) % top.length, a = rings[ring][i], b = rings[ring][j], c = rings[ring + 1][i], d = rings[ring + 1][j];
    push(a, c, b); push(b, c, d);
  }
  const tip = new THREE.Vector3(0, -depth * 1.25, 0), last = rings[rings.length - 1];
  for (let i = 0; i < last.length; i++) push(last[i], tip, last[(i + 1) % last.length]);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3)); geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3)); geometry.computeVertexNormals();
  return geometry;
}

function stripeTexture(colors, stripes = 8) {
  if (typeof document === 'undefined') return null;
  const canvas = document.createElement('canvas'); canvas.width = 64; canvas.height = 8;
  const ctx = canvas.getContext('2d');
  for (let i = 0; i < stripes; i++) { ctx.fillStyle = colors[i % colors.length]; ctx.beginPath(); const w = 64 / stripes; ctx.moveTo(i * w, 0); ctx.lineTo(i * w + w, 0); ctx.lineTo(i * w + w - 4, 8); ctx.lineTo(i * w - 4, 8); ctx.fill(); }
  const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace; texture.wrapS = THREE.RepeatWrapping; texture.anisotropy = 4;
  return texture;
}

function swayMaterial(parameters) {
  const material = new THREE.MeshStandardMaterial({ roughness: .9, flatShading: true, ...parameters });
  material.onBeforeCompile = shader => {
    shader.uniforms.uTime = shared.uTime;
    shader.vertexShader = `uniform float uTime;\n${shader.vertexShader}`.replace('#include <begin_vertex>', `#include <begin_vertex>
      #ifdef USE_INSTANCING
      float swayPhase = instanceMatrix[3].x * .37 + instanceMatrix[3].z * .23;
      float swayHeight = max(0., position.y + .5);
      transformed.x += sin(uTime * 1.3 + swayPhase) * .05 * swayHeight;
      transformed.z += cos(uTime * 1.1 + swayPhase) * .035 * swayHeight;
      #endif`);
  };
  material.customProgramCacheKey = () => 'terra-sway';
  return material;
}

class Instances {
  constructor(parent, geometry, material, capacity) {
    this.mesh = new THREE.InstancedMesh(geometry, material, capacity); this.mesh.count = 0;
    this.mesh.castShadow = true; this.mesh.receiveShadow = true; parent.add(this.mesh);
    this.dummy = new THREE.Object3D(); this.color = new THREE.Color();
  }
  add(x, y, z, sx, sy, sz, hex, rotation = 0, tilt = 0) {
    if (this.mesh.count >= this.mesh.instanceMatrix.count) return;
    const d = this.dummy; d.position.set(x, y, z); d.rotation.set(tilt, rotation, tilt * .6); d.scale.set(sx, sy, sz); d.updateMatrix();
    this.mesh.setMatrixAt(this.mesh.count, d.matrix); this.mesh.setColorAt(this.mesh.count, this.color.setHex(hex)); this.mesh.count++;
  }
  finish() { this.mesh.instanceMatrix.needsUpdate = true; if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true; this.mesh.computeBoundingSphere(); }
}

function box(parent, material, x, y, z, sx, sy, sz, geometry) {
  const mesh = new THREE.Mesh(geometry, material); mesh.position.set(x, y, z); mesh.scale.set(sx, sy, sz); mesh.castShadow = true; mesh.receiveShadow = true; parent.add(mesh); return mesh;
}

function siteOffice(parent, x, z, rotation, context) {
  const group = new THREE.Group(); group.position.set(x, 0, z); group.rotation.y = rotation; parent.add(group);
  const { cube } = context, m = context.materials;
  box(group, m.concrete, 0, .08, 0, 4.2, .16, 2.5, cube);
  box(group, m.office, 0, 1.4, 0, 4.0, 2.5, 2.3, cube);
  box(group, m.trim, 0, 2.7, 0, 4.15, .14, 2.45, cube);
  box(group, m.trim, 0, .22, 0, 4.1, .14, 2.4, cube);
  for (const wx of [-1.2, .15]) { box(group, m.window, wx, 1.6, 1.16, 1.0, .75, .04, cube); box(group, m.trim, wx, 1.18, 1.19, 1.1, .07, .08, cube); }
  box(group, m.door, 1.35, 1.15, 1.16, .8, 1.9, .05, cube);
  box(group, m.concrete, 1.35, .15, 1.55, 1.05, .3, .6, cube);
  box(group, m.window, -2.005, 1.6, 0, .04, .7, 1.0, cube);
  // Rooftop air conditioner and a small sign.
  box(group, m.metal, -1.2, 2.95, -.4, .8, .36, .6, cube);
  box(group, m.sign, .15, 3.08, 1.05, 1.7, .46, .06, cube);
  // A portable toilet beside the office.
  const loo = new THREE.Group(); loo.position.set(1.3, 0, -1.95); group.add(loo);
  box(loo, m.loo, 0, 1.12, 0, 1.05, 2.24, 1.05, cube); box(loo, m.looRoof, 0, 2.3, 0, 1.12, .12, 1.12, cube); box(loo, m.trim, 0, 1.12, .53, .7, 1.8, .03, cube);
  return group;
}

function pipeStack(parent, x, z, rotation, context) {
  const group = new THREE.Group(); group.position.set(x, 0, z); group.rotation.y = rotation; parent.add(group);
  const geometry = context.pipe;
  for (const [px, py] of [[-.55, .32], [0, .32], [.55, .32], [-.27, .8], [.27, .8], [0, 1.27]]) {
    const pipe = new THREE.Mesh(geometry, context.materials.pipe); pipe.rotation.x = Math.PI / 2; pipe.position.set(px, py, 0); pipe.scale.set(.3, 3.2, .3); pipe.castShadow = true; pipe.receiveShadow = true; group.add(pipe);
    const rim = new THREE.Mesh(context.ring, context.materials.pipeEnd); rim.position.set(px, py, 1.61); rim.scale.setScalar(.3); group.add(rim);
  }
  box(group, context.materials.wood, 0, .03, -1.1, 1.8, .06, .2, context.cube); box(group, context.materials.wood, 0, .03, 1.1, 1.8, .06, .2, context.cube);
  return group;
}

function pallet(parent, x, z, rotation, context, rand) {
  const group = new THREE.Group(); group.position.set(x, 0, z); group.rotation.y = rotation; parent.add(group);
  box(group, context.materials.wood, 0, .07, 0, 1.2, .14, 1.0, context.cube);
  const bags = 2 + Math.floor(rand() * 3);
  for (let i = 0; i < bags; i++) { const bag = box(group, context.materials.bag, (i % 2 - .5) * .52, .28 + Math.floor(i / 2) * .26, 0, .5, .24, .86, context.bagGeometry); bag.rotation.y = (rand() - .5) * .2; }
  return group;
}

// Figure styles: only a block of banded earth under the exact grid footprint,
// like a geological block diagram. Studio adds a floor that shows only the
// block's shadow, so it stands on the backdrop. No decoration at all.
function createPlinth(frame, palette = PALETTES.paper, studio = false) {
  const { rows, cols, tile_size_m: tile } = frame.grid, span = Math.max(rows, cols) * tile;
  const root = new THREE.Group(); root.name = 'Terra plinth';
  const geometry = new THREE.BoxGeometry(cols * tile, 1, rows * tile); geometry.translate(0, -.5, 0);
  const block = new THREE.Mesh(geometry, earthMaterial('soil', { polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 2 }, palette));
  block.name = 'plinth'; block.receiveShadow = true; block.castShadow = studio; root.add(block);
  let floor = null;
  if (studio) {
    floor = new THREE.Mesh(new THREE.PlaneGeometry(span * 12, span * 12), new THREE.ShadowMaterial({ color: 0x2c2620, opacity: .28 }));
    floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; floor.name = 'studio-floor'; root.add(floor);
  }
  root.userData.extent = { hx: cols * tile / 2, hz: rows * tile / 2 };
  root.setFloor = level => {
    // The block continues the terrain columns' walls below their shared floor.
    const bottom = Math.min(-Math.max(span * (studio ? .085 : .06), 1.8), level - tile * .8);
    block.position.y = level; block.scale.y = level - bottom; shared.uFloor.value = level;
    if (floor) floor.position.y = bottom - .002;
  };
  root.update = () => {};
  root.dispose = () => { geometry.dispose(); block.material.dispose(); if (floor) { floor.geometry.dispose(); floor.material.dispose(); } root.removeFromParent(); };
  return root;
}

/**
 * Build the surroundings of a rows × cols grid of `tile`-meter cells: the
 * stylized island ('diorama') or a plain block ('paper').
 * Returns a group with setFloor(floorY) and update(time) methods.
 */
export function createEnvironment(frame, { style = 'diorama' } = {}) {
  if (style === 'paper') return createPlinth(frame);
  if (style === 'studio') return createPlinth(frame, PALETTES.studio, true);
  const { rows, cols, tile_size_m: tile } = frame.grid, hx = cols * tile / 2, hz = rows * tile / 2, span = Math.max(rows, cols) * tile;
  const apron = THREE.MathUtils.clamp(span * .2, 6, 22), root = new THREE.Group(); root.name = 'Terra surroundings';
  const rand = random(rows * 7919 + cols * 104729 + Math.round(tile * 1000));
  const outerX = hx + apron, outerZ = hz + apron, radius = apron * 1.1;
  const context = {
    cube: new THREE.BoxGeometry(1, 1, 1), pipe: new THREE.CylinderGeometry(1, 1, 1, 10, 1, true), ring: new THREE.RingGeometry(.72, 1, 10),
    bagGeometry: new THREE.BoxGeometry(1, 1, 1),
    materials: {
      concrete: new THREE.MeshStandardMaterial({ color: 0xb9b6ad, roughness: 1 }), office: new THREE.MeshStandardMaterial({ color: 0xf3f0e6, roughness: .8 }),
      trim: new THREE.MeshStandardMaterial({ color: 0x3f6f9a, roughness: .7 }), window: new THREE.MeshStandardMaterial({ color: 0x9fd6e6, roughness: .15, metalness: .1, emissive: 0x1d3a44, emissiveIntensity: .3 }),
      door: new THREE.MeshStandardMaterial({ color: 0x2f5578, roughness: .7 }), metal: new THREE.MeshStandardMaterial({ color: 0xc9ced0, roughness: .5 }),
      sign: new THREE.MeshStandardMaterial({ color: 0xf2b231, roughness: .6 }), loo: new THREE.MeshStandardMaterial({ color: 0x3aa0d8, roughness: .6 }), looRoof: new THREE.MeshStandardMaterial({ color: 0xeaf2f5, roughness: .6 }),
      pipe: new THREE.MeshStandardMaterial({ color: 0xe57e3a, roughness: .7, side: THREE.DoubleSide }), pipeEnd: new THREE.MeshStandardMaterial({ color: 0xb8612b, roughness: .8, side: THREE.DoubleSide }),
      wood: new THREE.MeshStandardMaterial({ color: 0xb98a55, roughness: 1 }), bag: new THREE.MeshStandardMaterial({ color: 0xe9e0c9, roughness: 1 }),
    },
  };

  // Island: turf ring with banded earth walls around the exact grid rectangle.
  const shape = roundedRect(new THREE.Shape(), outerX, outerZ, radius);
  const hole = new THREE.Path(); hole.moveTo(-hx, -hz); hole.lineTo(-hx, hz); hole.lineTo(hx, hz); hole.lineTo(hx, -hz); hole.closePath(); shape.holes.push(hole);
  const ringGeometry = new THREE.ExtrudeGeometry(shape, { depth: 1, bevelEnabled: false, curveSegments: 6 }); ringGeometry.rotateX(Math.PI / 2);
  const island = new THREE.Mesh(ringGeometry, earthMaterial('island')); island.receiveShadow = true; island.castShadow = false; island.name = 'island-turf'; root.add(island);
  const underside = new THREE.Mesh(islandUnderside(outerX, outerZ, radius, Math.max(span * .16, 4), rows * 31 + cols), new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, roughness: 1, side: THREE.DoubleSide }));
  underside.name = 'island-underside'; root.add(underside);

  // Site entrance: a gravel road from the east fence to the island edge.
  const roadWidth = Math.min(4.2, hz * .5), road = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), earthMaterial('soil', { color: 0xcdb896 }));
  road.rotation.x = -Math.PI / 2; road.scale.set(apron, roadWidth, 1); road.position.set(hx + apron / 2 + .01, .012, 0); road.name = 'site-road'; road.receiveShadow = true; root.add(road);
  const gateHalf = roadWidth / 2 + .3;

  // Fence: short posts with striped barrier tape; a gap for the road.
  const posts = new Instances(root, context.cube, new THREE.MeshStandardMaterial({ color: 0xeeeae0, roughness: .7 }), 400);
  const tape = new THREE.InstancedMesh(context.cube, new THREE.MeshStandardMaterial({ map: stripeTexture(['#e8573a', '#f7f2e8']), color: typeof document === 'undefined' ? 0xe8573a : 0xffffff, roughness: .7 }), 800); tape.count = 0; tape.castShadow = true; root.add(tape);
  const dummy = new THREE.Object3D(), inset = .18;
  const runs = [
    [[-hx - inset, -hz - inset], [hx + inset, -hz - inset]], [[-hx - inset, hz + inset], [hx + inset, hz + inset]],
    [[-hx - inset, -hz - inset], [-hx - inset, hz + inset]], [[hx + inset, -hz - inset], [hx + inset, -gateHalf]], [[hx + inset, gateHalf], [hx + inset, hz + inset]],
  ];
  for (const [[x0, z0], [x1, z1]] of runs) {
    const length = Math.hypot(x1 - x0, z1 - z0), count = Math.max(1, Math.round(length / 2.4)), angle = Math.atan2(-(z1 - z0), x1 - x0);
    for (let i = 0; i <= count; i++) { const t = i / count; posts.add(x0 + (x1 - x0) * t, .45, z0 + (z1 - z0) * t, .1, .9, .1, i % 2 ? 0xeeeae0 : 0xe8573a); }
    for (let i = 0; i < count; i++) for (const y of [.42, .78]) {
      const t = (i + .5) / count; dummy.position.set(x0 + (x1 - x0) * t, y, z0 + (z1 - z0) * t); dummy.rotation.set(0, angle, 0); dummy.scale.set(length / count, .09, .02); dummy.updateMatrix(); tape.setMatrixAt(tape.count++, dummy.matrix);
    }
  }
  posts.finish(); tape.instanceMatrix.needsUpdate = true; tape.computeBoundingSphere();

  // Cones at the gate and along the road.
  const coneGeometry = new THREE.ConeGeometry(.2, .6, 8); coneGeometry.translate(0, .3, 0);
  const cones = new Instances(root, coneGeometry, new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: .6, flatShading: true }), 40);
  const stripes = new Instances(root, new THREE.CylinderGeometry(.115, .145, .1, 8), new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: .4 }), 40);
  for (let i = 0; i < Math.floor(apron / 2.2); i++) for (const side of [-1, 1]) {
    const x = hx + 1 + i * 2.2, z = side * (roadWidth / 2 + .35);
    cones.add(x, 0, z, 1, 1, 1, 0xf36f2a); stripes.add(x, .33, z, 1, 1, 1, 0xf7f4ea);
  }
  cones.finish(); stripes.finish();

  // Props on the apron, placed away from the road and fence.
  const occupied = [];
  const free = (x, z, r) => {
    if (Math.abs(x) < hx + 1.2 + r && Math.abs(z) < hz + 1.2 + r) return false;
    const ex = Math.max(0, Math.abs(x) - (outerX - radius)), ez = Math.max(0, Math.abs(z) - (outerZ - radius));
    if (Math.hypot(ex, ez) > radius - r - .5) return false;
    if (x > hx && Math.abs(z) < roadWidth / 2 + r + .6) return false;
    return occupied.every(([ox, oz, or]) => Math.hypot(x - ox, z - oz) > r + or);
  };
  const officeZ = -(roadWidth / 2 + 3.2), officeX = hx + Math.min(apron * .55, 6);
  if (apron >= 6 && free(officeX, officeZ, 2.3) && free(officeX + 1.3, officeZ - 1.95, .8)) { siteOffice(root, officeX, officeZ, 0, context); occupied.push([officeX, officeZ, 2.8], [officeX + 1.3, officeZ - 1.95, .9]); }
  const pipesX = hx + Math.min(apron * .6, 6.5), pipesZ = roadWidth / 2 + 3;
  if (apron >= 6 && free(pipesX, pipesZ, 1.7)) { pipeStack(root, pipesX, pipesZ, Math.PI / 2 + .1, context); occupied.push([pipesX, pipesZ, 2]); }
  for (let i = 0; i < 3; i++) {
    const x = -hx - apron * (.35 + rand() * .3), z = (rand() - .5) * hz * 1.4;
    if (free(x, z, .9)) { pallet(root, x, z, rand() * Math.PI, context, rand); occupied.push([x, z, .9]); }
  }

  // Trees and bushes: instanced, flat-shaded and gently swaying.
  const trunkGeometry = new THREE.CylinderGeometry(.5, .7, 1, 6); trunkGeometry.translate(0, .5, 0);
  const coneTier = new THREE.ConeGeometry(1, 1, 7); coneTier.translate(0, .5, 0);
  const blob = new THREE.IcosahedronGeometry(1, 0);
  const trunks = new Instances(root, trunkGeometry, new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, flatShading: true }), 400);
  const pines = new Instances(root, coneTier, swayMaterial({ color: 0xffffff }), 900);
  const crowns = new Instances(root, blob, swayMaterial({ color: 0xffffff }), 700);
  const rocks = new Instances(root, new THREE.DodecahedronGeometry(1, 0), new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, flatShading: true }), 200);
  const pineGreens = [0x4f8f5a, 0x5c9c5f, 0x467f52], leafGreens = [0x7db656, 0x8cc15d, 0x6aa84f, 0xa6c95a], blossoms = [0xf2b35c, 0xe98d6b];
  const area = 4 * (outerX * outerZ - hx * hz), attempts = Math.min(2600, Math.round(area / 2.2));
  for (let i = 0; i < attempts; i++) {
    const edge = rand(), side = Math.floor(rand() * 4), depth = Math.pow(rand(), .8);
    let x, z;
    if (side < 2) { x = (edge * 2 - 1) * outerX; z = (side ? 1 : -1) * (hz + 1.8 + depth * (apron - 1.8)); }
    else { z = (edge * 2 - 1) * outerZ; x = (side === 3 ? 1 : -1) * (hx + 1.8 + depth * (apron - 1.8)); }
    const kind = rand(), scale = .62 + rand() * .45;
    const r = kind < .45 ? 1.1 * scale : kind < .8 ? 1.3 * scale : .7 * scale;
    // Denser groves toward the island rim keep the site itself open.
    const rim = Math.min(Math.abs(Math.abs(x) - hx), Math.abs(Math.abs(z) - hz));
    if (rand() > .25 + Math.min(1, rim / apron) * .9 || !free(x, z, r)) continue;
    occupied.push([x, z, r]); const turn = rand() * Math.PI * 2, lean = (rand() - .5) * .08;
    if (kind < .45) {
      const height = (3.4 + rand() * 1.8) * scale;
      trunks.add(x, 0, z, .18 * scale, height * .3, .18 * scale, 0x8a5d3b, turn);
      for (let tier = 0; tier < 3; tier++) pines.add(x, height * (.22 + tier * .22), z, (1.25 - tier * .3) * scale, height * .42, (1.25 - tier * .3) * scale, pineGreens[(i + tier) % 3], turn + tier, lean);
    } else if (kind < .8) {
      const height = (2.2 + rand() * 1.4) * scale;
      trunks.add(x, 0, z, .16 * scale, height * .55, .16 * scale, 0x94653f, turn);
      crowns.add(x, height * .75, z, 1.25 * scale, 1.05 * scale, 1.2 * scale, rand() < .08 ? blossoms[i % 2] : leafGreens[i % 4], turn, lean);
      if (rand() < .6) crowns.add(x + .45 * scale, height * .98, z - .2 * scale, .8 * scale, .7 * scale, .8 * scale, leafGreens[(i + 1) % 4], turn + 1);
    } else if (kind < .93) crowns.add(x, .35 * scale, z, .7 * scale, .5 * scale, .7 * scale, leafGreens[(i + 2) % 4], turn);
    else rocks.add(x, .12 * scale, z, .55 * scale, .38 * scale, .5 * scale, [0x9a958c, 0x8a867f, 0xa7a197][i % 3], turn, lean * 3);
  }
  for (const group of [trunks, pines, crowns, rocks]) group.finish();

  // Clouds circle outside the island footprint so they never hide the grid.
  const clouds = new THREE.Group(), cloudMaterial = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, flatShading: true, emissive: 0xffffff, emissiveIntensity: .25 });
  const puff = new THREE.IcosahedronGeometry(1, 1), reach = Math.hypot(outerX, outerZ);
  for (let i = 0; i < 6; i++) {
    const cloud = new THREE.Group(), angle = i / 6 * Math.PI * 2 + rand() * .6, distance = reach * (1.25 + rand() * .45), size = span * (.035 + rand() * .025);
    cloud.position.set(Math.cos(angle) * distance, span * (.12 + rand() * .22), Math.sin(angle) * distance);
    for (let j = 0; j < 5; j++) { const part = new THREE.Mesh(puff, cloudMaterial); part.position.set((j - 2) * size * .9, (1 - Math.abs(j - 2) * .45) * size * .35, (rand() - .5) * size * .7); part.scale.setScalar(size * (1.1 - Math.abs(j - 2) * .22)); cloud.add(part); }
    cloud.userData = { angle, distance, height: cloud.position.y, speed: .006 + rand() * .006 };
    clouds.add(cloud);
  }
  root.add(clouds);

  root.userData.extent = { hx: outerX, hz: outerZ };
  root.setFloor = floor => {
    // Keep the banded walls below the deepest displayed cut.
    const bottom = Math.min(-Math.max(span * .07, 2.2), floor - tile * .8);
    island.scale.y = -bottom; underside.position.y = bottom; shared.uFloor.value = floor;
  };
  root.update = time => {
    for (const cloud of clouds.children) {
      const { angle, distance, height, speed } = cloud.userData, a = angle + time * speed;
      cloud.position.set(Math.cos(a) * distance, height + Math.sin(time * .3 + angle * 3) * span * .006, Math.sin(a) * distance);
    }
  };
  root.dispose = () => {
    const geometries = new Set(), materials = new Set();
    root.traverse(item => { if (item.geometry) geometries.add(item.geometry); if (item.material) materials.add(item.material); });
    for (const geometry of geometries) geometry.dispose();
    for (const material of materials) { material.map?.dispose(); material.dispose(); }
    root.removeFromParent();
  };
  return root;
}
