import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

// Original procedural toy-like assets. Dimensions follow each recorded machine
// footprint; motion is illustrative choreography of a discrete Terra action.
export const SLOT_COLORS = [0xffb01f, 0x31c6b0, 0xff6f61, 0x9b8cff];
const paintCache = new Map();
function paint(hex, options = {}) {
  const key = `${hex}:${JSON.stringify(options)}`;
  if (!paintCache.has(key)) paintCache.set(key, new THREE.MeshStandardMaterial({ color: hex, roughness: .52, metalness: 0, ...options }));
  return paintCache.get(key);
}
const materials = {
  dark: paint(0x3b4047, { roughness: .7 }), darker: paint(0x2b2f35, { roughness: .8 }), rubber: paint(0x2a2c30, { roughness: .95 }),
  metal: paint(0xa7b0b4, { roughness: .35, metalness: .55 }), chrome: paint(0xe3e8ea, { roughness: .18, metalness: .85 }),
  glass: paint(0x7dc4d6, { roughness: .08, metalness: .1, emissive: 0x16323a, emissiveIntensity: .35 }), seat: paint(0x2f3338, { roughness: .9 }),
  soil: paint(0xa8733f, { roughness: 1, flatShading: true }), lamp: paint(0xfff1c2, { emissive: 0xffe6a0, emissiveIntensity: .9 }),
  tail: paint(0xff4a3a, { emissive: 0xc01e10, emissiveIntensity: .6 }), white: paint(0xf4f1e8, { roughness: .6 }),
};
const LIVERIES = [
  { body: 0xf4b21b, accent: 0xffd35c, trim: 0x3b4047 },
  { body: 0xe4553a, accent: 0xf6a93b, trim: 0x3b4047, bed: 0xf2a93b },
  { body: 0xf47b20, accent: 0xffa04d, trim: 0x2f3338 },
];
// Studio style: one coated paint per machine. Excavators take their color from
// the state slot, so two excavators on one site stay distinguishable.
const STUDIO_EXCAVATORS = [0xe0a02b, 0x3f74a6, 0x5a8f5e, 0x8a6bb0];
const STUDIO_TRUCK = 0xc9563d, STUDIO_LOADER = 0xdf7a2c;
const coatCache = new Map();
function coat(hex) {
  if (!coatCache.has(hex)) coatCache.set(hex, new THREE.MeshPhysicalMaterial({ color: hex, roughness: .4, metalness: .05, clearcoat: .55, clearcoatRoughness: .28 }));
  return coatCache.get(hex);
}
const studio = {
  glass: paint(0x3a4d58, { roughness: .07, metalness: .25 }), steel: paint(0x4b5056, { roughness: .52, metalness: .3 }),
  worn: paint(0x9da2a6, { roughness: .32, metalness: .8 }), lamp: paint(0xf6efd8, { emissive: 0xffe9b8, emissiveIntensity: .25 }),
  soil: paint(0x6f5238, { roughness: 1, flatShading: true }),
};
function liveryFor(agent, style) {
  if (style !== 'studio') return { ...LIVERIES[agent.type], paint };
  const body = agent.type === 0 ? STUDIO_EXCAVATORS[agent.id % STUDIO_EXCAVATORS.length] : agent.type === 1 ? STUDIO_TRUCK : STUDIO_LOADER;
  return { body, accent: body, trim: 0x2e3237, bed: body, studio: true, paint: coat };
}
/** Body color of a machine in a presentation style, for legends. */
export function machineColor(agent, style = 'diorama') { return liveryFor(agent, style).body; }
const boxGeometry = new THREE.BoxGeometry(1, 1, 1);
const cylinderGeometry = new THREE.CylinderGeometry(1, 1, 1, 24);
const hexCylinder = new THREE.CylinderGeometry(1, 1, 1, 10);
function mesh(parent, geometry, material, x = 0, y = 0, z = 0) {
  const item = new THREE.Mesh(geometry, material);
  item.position.set(x, y, z); item.castShadow = true; item.receiveShadow = true; parent.add(item); return item;
}
function box(parent, material, x, y, z, dx, dy, dz) { const item = mesh(parent, boxGeometry, material, x, y, z); item.scale.set(dx, dy, dz); return item; }
function roundedBox(parent, material, x, y, z, dx, dy, dz, radius) {
  return mesh(parent, new RoundedBoxGeometry(dx, dy, dz, 3, Math.min(radius, dx / 2, dy / 2, dz / 2)), material, x, y, z);
}
function cylinder(parent, material, x, y, z, radius, height, rotation = 0, geometry = cylinderGeometry) { const item = mesh(parent, geometry, material, x, y, z); item.scale.set(radius, height, radius); item.rotation.x = rotation; return item; }
function rod(parent, material, from, to, radius) {
  const start = new THREE.Vector3(...from), end = new THREE.Vector3(...to), direction = end.clone().sub(start);
  const item = mesh(parent, cylinderGeometry, material); item.position.copy(start.add(end).multiplyScalar(.5));
  item.scale.set(radius, direction.length(), radius); item.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize()); return item;
}
function beam(parent, length, thickness, width, material, rounded = false) {
  const shape = new THREE.Shape();
  shape.moveTo(0, -thickness * .40);
  if (rounded) {
    shape.quadraticCurveTo(-thickness * .14, thickness * .15, length * .18, thickness * .50);
    shape.quadraticCurveTo(length * .24, thickness * .60, length * .33, thickness * .48);
    shape.lineTo(length * .89, thickness * .20);
    shape.quadraticCurveTo(length + thickness * .20, thickness * .20, length + thickness * .15, -thickness * .08);
    shape.quadraticCurveTo(length + thickness * .10, -thickness * .40, length * .90, -thickness * .32);
    shape.lineTo(length * .26, -thickness * .25);
  } else {
    shape.lineTo(length * .22, thickness * .55); shape.lineTo(length * .7, thickness * .3); shape.lineTo(length, thickness * .12); shape.lineTo(length, -thickness * .32); shape.lineTo(length * .24, -thickness * .25);
  }
  shape.closePath();
  const geometry = new THREE.ExtrudeGeometry(shape, { depth: width, bevelEnabled: true, bevelSegments: rounded ? 3 : 1, curveSegments: 10, steps: 1, bevelSize: thickness * .085, bevelThickness: thickness * .085 });
  return mesh(parent, geometry, material, 0, 0, -width / 2);
}
function anchor(parent, name, x, y, z) {
  const point = new THREE.Object3D(); point.name = name; point.position.set(x, y, z); parent.add(point); return point;
}
function pin(parent, material, x, y, z, radius, width) { return cylinder(parent, material, x, y, z, radius, width, Math.PI / 2); }
function placeRod(item, from, to, radius) {
  const direction = to.clone().sub(from);
  item.position.copy(from).add(to).multiplyScalar(.5);
  item.scale.set(radius, direction.length(), radius);
  item.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
}
function hydraulic(parent, start, end, radius, name) {
  const barrel = mesh(parent, cylinderGeometry, materials.dark), piston = mesh(parent, cylinderGeometry, materials.chrome);
  barrel.name = `${name}-barrel`; piston.name = `${name}-piston`;
  return {
    start, end, barrel, piston,
    update() {
      const from = parent.worldToLocal(start.getWorldPosition(new THREE.Vector3()));
      const to = parent.worldToLocal(end.getWorldPosition(new THREE.Vector3()));
      // The telescoping segments overlap; their outer ends remain on the pins.
      placeRod(barrel, from, from.clone().lerp(to, .58), radius);
      placeRod(piston, from.clone().lerp(to, .43), to, radius * .56);
    },
  };
}

/** Upper intersection of two pin-centered circles, in the stick's XY plane. */
function linkageJoint(origin, destination, firstLength, secondLength) {
  const dx = destination.x - origin.x, dy = destination.y - origin.y, distance = Math.hypot(dx, dy);
  if (distance <= Math.abs(firstLength - secondLength) || distance >= firstLength + secondLength) throw new Error('Bucket linkage pose is outside its mechanical range.');
  const along = (firstLength ** 2 - secondLength ** 2 + distance ** 2) / (2 * distance);
  const across = Math.sqrt(Math.max(0, firstLength ** 2 - along ** 2));
  return new THREE.Vector3(origin.x + (along * dx - across * dy) / distance, origin.y + (along * dy + across * dx) / distance, 0);
}

// Canvas-drawn decals; headless tests without a DOM fall back to flat colors.
function canvasTexture(width, height, draw) {
  if (typeof document === 'undefined') return null;
  const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height; draw(canvas.getContext('2d'));
  const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace; return texture;
}
function hazardTexture() {
  return canvasTexture(128, 32, ctx => {
    ctx.fillStyle = '#f4b21b'; ctx.fillRect(0, 0, 128, 32); ctx.fillStyle = '#2b2f35';
    for (let x = -32; x < 160; x += 24) { ctx.beginPath(); ctx.moveTo(x, 32); ctx.lineTo(x + 12, 32); ctx.lineTo(x + 44, 0); ctx.lineTo(x + 32, 0); ctx.fill(); }
  });
}
let hazardMaterial;

// Chunky tire with tread blocks; steer rotates the outer group, spin the inner one.
function wheel(parent, x, z, radius, width, rim) {
  const steer = new THREE.Group(); steer.position.set(x, radius, z); parent.add(steer);
  const spin = new THREE.Group(); steer.add(spin);
  cylinder(spin, materials.rubber, 0, 0, 0, radius * .9, width, Math.PI / 2);
  const side = Math.sign(z) || 1;
  cylinder(spin, rim, 0, 0, side * width * .47, radius * .56, width * .12, Math.PI / 2);
  cylinder(spin, materials.metal, 0, 0, side * width * .54, radius * .2, width * .1, Math.PI / 2, hexCylinder);
  for (let i = 0; i < 6; i++) { const angle = i * Math.PI / 3; box(spin, materials.darker, Math.cos(angle) * radius * .36, Math.sin(angle) * radius * .36, side * width * .53, radius * .09, radius * .09, width * .05); }
  for (let i = 0; i < 14; i++) {
    const angle = i * Math.PI * 2 / 14, block = box(spin, materials.rubber, Math.sin(angle) * radius * .93, Math.cos(angle) * radius * .93, (i % 2 ? .18 : -.18) * width, radius * .26, radius * .16, width * .6);
    block.rotation.z = -angle;
  }
  return { steer, spin, radius };
}

// A stadium track loop whose shoes travel around the loop as the machine drives.
function track(parent, L, W, S, side, material) {
  const group = new THREE.Group(); group.position.z = side * W * .35; parent.add(group);
  const radius = S * .13, half = L * .34, lift = S * .02, center = radius + lift, width = W * .2;
  const perimeter = 4 * half + 2 * Math.PI * radius;
  const frame = new THREE.Shape(); frame.moveTo(-half, lift + radius * .35); frame.lineTo(half, lift + radius * .35); frame.absarc(half, center, radius * .65, -Math.PI / 2, Math.PI / 2, false); frame.lineTo(-half, center + radius * .65); frame.absarc(-half, center, radius * .65, Math.PI / 2, Math.PI * 1.5, false);
  const frameGeometry = new THREE.ExtrudeGeometry(frame, { depth: width * .7, bevelEnabled: true, bevelSegments: 2, bevelSize: S * .012, bevelThickness: S * .012, steps: 1 });
  mesh(group, frameGeometry, material, 0, 0, -width * .35);
  const wheels = [];
  for (let i = 0; i < 5; i++) wheels.push(cylinder(group, materials.metal, L * (-.26 + i * .13), lift + radius * .42, side * width * .38, S * .045, width * .12, Math.PI / 2));
  for (const x of [-half, half]) {
    const hub = new THREE.Group(); hub.position.set(x, center, 0); group.add(hub); wheels.push(hub);
    cylinder(hub, materials.dark, 0, 0, 0, radius * .82, width * .86, Math.PI / 2);
    cylinder(hub, materials.metal, 0, 0, side * width * .44, radius * .38, width * .06, Math.PI / 2, hexCylinder);
    for (let i = 0; i < 8; i++) { const angle = i * Math.PI / 4; box(hub, materials.darker, Math.cos(angle) * radius * .6, Math.sin(angle) * radius * .6, side * width * .44, radius * .14, radius * .14, width * .04); }
  }
  const count = Math.max(24, Math.round(perimeter / (S * .055))), spacing = perimeter / count;
  const shoes = new THREE.InstancedMesh(boxGeometry, materials.rubber, count); shoes.castShadow = true; shoes.receiveShadow = true; group.add(shoes);
  const dummy = new THREE.Object3D(), thickness = S * .034;
  const place = (s, out) => {
    s = ((s % perimeter) + perimeter) % perimeter;
    if (s < 2 * half) { out.set(-half + s, lift, -Math.PI / 2); return; }
    s -= 2 * half;
    if (s < Math.PI * radius) { const a = -Math.PI / 2 + s / radius; out.set(half + Math.cos(a) * radius, center + Math.sin(a) * radius, a); return; }
    s -= Math.PI * radius;
    if (s < 2 * half) { out.set(half - s, center + radius, Math.PI / 2); return; }
    s -= 2 * half; const a = Math.PI / 2 + s / radius; out.set(-half + Math.cos(a) * radius, center + Math.sin(a) * radius, a);
  };
  const point = new THREE.Vector3();
  const update = phase => {
    for (let i = 0; i < count; i++) {
      place(i * spacing + phase, point);
      const angle = point.z;
      dummy.position.set(point.x + Math.cos(angle) * thickness * .4, point.y + Math.sin(angle) * thickness * .4, 0);
      dummy.rotation.set(0, 0, angle - Math.PI / 2); dummy.scale.set(spacing * .82, thickness, width); dummy.updateMatrix(); shoes.setMatrixAt(i, dummy.matrix);
    }
    shoes.instanceMatrix.needsUpdate = true;
    for (const item of wheels) if (item.isGroup) item.rotation.z = -phase / (radius * .82);
  };
  update(0);
  return { update, shoes };
}

function undercarriage(root, L, W, S, wheeled, livery, isTruck) {
  const steering = [], spinning = [], tracks = [];
  box(root, materials.dark, 0, S * .22, 0, L * .74, S * .17, W * .6);
  if (wheeled) {
    const radius = S * (isTruck ? .23 : .19), width = W * (isTruck ? .22 : .18);
    for (const x of [-.30, .30]) for (const z of [-.4, .4]) { const part = wheel(root, x * L, z * W, radius, width, paint(livery.accent)); spinning.push(part); if (x > 0) steering.push(part.steer); }
    for (const x of [-.30, .30]) rod(root, materials.darker, [x * L, radius, -.4 * W], [x * L, radius, .4 * W], S * .05);
  } else {
    for (const side of [-1, 1]) tracks.push({ side, ...track(root, L, W, S, side, paint(livery.trim, { roughness: .7 })) });
  }
  return { steering, spinning, tracks };
}

function glassHighlight(parent, x, y, z, dx, dy, dz, axis = 'x') {
  const material = paint(0xffffff, { transparent: true, opacity: .45, emissive: 0xffffff, emissiveIntensity: .4, depthWrite: false });
  for (const [offset, size] of [[-.18, .16], [.1, .07]]) {
    const strip = box(parent, material, x, y, z, dx, dy, dz);
    strip.castShadow = false;
    if (axis === 'x') { strip.scale.set(dx, dy * 1.2, dz * size); strip.position.z += dz * offset * 2.2; strip.rotation.x = .5; }
    else { strip.scale.set(dx * size, dy * 1.2, dz); strip.position.x += dx * offset * 2.2; strip.rotation.z = -.5; }
    strip.userData.skipAO = true; strip.name = 'glass-highlight';
  }
}

function cab(parent, L, W, S, x, z, livery) {
  const body = livery.paint(livery.body), group = new THREE.Group(); group.position.set(x, 0, z); parent.add(group);
  const cx = .30 * L, cz = .39 * W, height = .5 * S, glass = livery.studio ? studio.glass : materials.glass;
  roundedBox(group, body, 0, .12 * S, 0, cx, .2 * S, cz, S * .04);
  // Frame, then inset glass on the front, sides and rear.
  roundedBox(group, body, 0, .36 * S, 0, cx * .96, height * .72, cz * .96, S * .05).name = 'cab-shell';
  const glassY = .39 * S, glassH = height * .56;
  box(group, glass, cx * .485, glassY, 0, S * .012, glassH, cz * .84);
  glassHighlight(group, cx * .492, glassY, 0, S * .01, glassH * .8, cz * .84, 'x');
  for (const side of [-1, 1]) { box(group, glass, -cx * .04, glassY, side * cz * .485, cx * .76, glassH, S * .012); glassHighlight(group, -cx * .04, glassY, side * cz * .492, cx * .76, glassH * .8, S * .01, 'z'); }
  box(group, glass, -cx * .485, glassY + glassH * .1, 0, S * .012, glassH * .6, cz * .7);
  box(group, materials.seat, -cx * .12, .3 * S, 0, cx * .3, .16 * S, cz * .5);
  roundedBox(group, livery.paint(livery.studio ? livery.body : livery.trim), 0, .62 * S, 0, cx * (livery.studio ? .98 : 1.06), S * .05, cz * (livery.studio ? .98 : 1.06), S * .02);
  for (const side of [-1, 1]) box(group, livery.studio ? studio.lamp : materials.lamp, cx * .5, .6 * S, side * cz * .3, S * .02, S * .035, cz * .12);
  return group;
}

function extrudeShape(shape, depth, bevel, segments = 12) {
  const geometry = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: bevel > 0, bevelSegments: 2, bevelSize: bevel, bevelThickness: bevel, curveSegments: segments, steps: 1 });
  geometry.translate(0, 0, -depth / 2); return geometry;
}
function shapeOf(points) { const shape = new THREE.Shape(); shape.setFromPoints(points); return shape; }
/** Offset a sampled curve by `distance` toward `inside`, as a closed plate outline. */
function plateOutline(points, distance, inside) {
  const inner = points.map((point, i) => {
    const before = points[Math.max(0, i - 1)], after = points[Math.min(points.length - 1, i + 1)];
    const normal = new THREE.Vector2(before.y - after.y, after.x - before.x).normalize();
    if (normal.dot(inside.clone().sub(point)) < 0) normal.negate();
    return point.clone().addScaledVector(normal, distance);
  });
  return [...points, ...inner.reverse()];
}
function convexHull(points) {
  const sorted = [...points].sort((a, b) => a.x - b.x || a.y - b.y), cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower = [], upper = [];
  for (const point of sorted) { while (lower.length > 1 && cross(lower.at(-2), lower.at(-1), point) <= 0) lower.pop(); lower.push(point); }
  for (const point of sorted.reverse()) { while (upper.length > 1 && cross(upper.at(-2), upper.at(-1), point) <= 0) upper.pop(); upper.push(point); }
  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

// General-purpose loader bucket: flat floor, rolled back, side plates, spill
// guard, bolt-on steel cutting edge and a quick-attach plate. +x points ahead.
function loaderBucket(parent, S, width, livery) {
  const root = new THREE.Group(); parent.add(root);
  root.name = 'loader-bucket';
  const u = S, shell = livery.studio ? studio.steel : materials.dark, sides = livery.studio ? studio.steel : livery.paint(livery.body), edge = livery.studio ? studio.worn : materials.metal;
  const back = new THREE.Path();
  back.moveTo(.29 * u, -.155 * u); back.lineTo(-.05 * u, -.135 * u);
  back.quadraticCurveTo(-.16 * u, -.13 * u, -.165 * u, .0 * u); back.quadraticCurveTo(-.17 * u, .13 * u, -.12 * u, .185 * u);
  const outline = back.getPoints(10), inside = new THREE.Vector2(.05 * u, .02 * u);
  mesh(root, extrudeShape(shapeOf(plateOutline(outline, .022 * u, inside)), width * .97, .004 * u), shell).name = 'loader-bucket-shell';
  const side = shapeOf([...outline, new THREE.Vector2(-.04 * u, .2 * u), new THREE.Vector2(.07 * u, .2 * u)]);
  const sideGeometry = extrudeShape(side, width * .045, .006 * u);
  for (const z of [-1, 1]) mesh(root, sideGeometry, sides, 0, 0, z * width * .49);
  box(root, shell, .0 * u, .195 * u, 0, .17 * u, .02 * u, width * .99).rotation.z = .08;
  const lip = box(root, edge, .31 * u, -.157 * u, 0, .07 * u, .022 * u, width * 1.0); lip.rotation.z = -.06;
  box(root, shell, -.19 * u, .03 * u, 0, .03 * u, .26 * u, width * .62);
  for (const z of [-1, 1]) box(root, shell, -.21 * u, .03 * u, z * width * .2, .05 * u, .24 * u, .04 * u);
  anchor(root, 'loader-edge', .34 * u, -.15 * u, 0);
  anchor(root, 'loader-lip', .16 * u, .05 * u, 0);
  const soil = mesh(root, lumpGeometry(3), materials.soil, .06 * u, -.06 * u, 0); soil.name = 'bucket-soil'; soil.scale.set(u * .2, u * .14, width * .42); soil.visible = false;
  return { root, soil };
}

// A lumpy, faceted payload so carried soil reads as loose earth.
const lumpCache = new Map();
function lumpGeometry(seed) {
  if (lumpCache.has(seed)) return lumpCache.get(seed);
  const geometry = new THREE.IcosahedronGeometry(1, 1), position = geometry.attributes.position;
  let state = seed * 9301 + 49297;
  const rand = () => (state = (state * 16807) % 2147483647) / 2147483647;
  const offsets = new Map();
  for (let i = 0; i < position.count; i++) {
    const key = `${position.getX(i).toFixed(3)},${position.getY(i).toFixed(3)},${position.getZ(i).toFixed(3)}`;
    if (!offsets.has(key)) offsets.set(key, .82 + rand() * .3);
    const s = offsets.get(key), y = position.getY(i);
    position.setXYZ(i, position.getX(i) * s, (y < 0 ? y * .25 : y) * s, position.getZ(i) * s);
  }
  geometry.computeVertexNormals(); lumpCache.set(seed, geometry);
  return geometry;
}

// Backhoe bucket in its own side-profile frame: +x points back toward the
// machine (the opening faces the cab), +y up, main hinge pin at the origin.
// A curved back plate wraps from the mounting brackets around the heel to a
// steel cutting edge with teeth; flat side plates close the bowl.
function excavatorBucket(parent, S, width, hingeWidth, livery) {
  const curl = new THREE.Group(); curl.name = 'bucket-curl'; parent.add(curl);
  const orientation = new THREE.Group(); orientation.name = 'bucket-orientation';
  // Reverse the bucket around its vertical axis; curl remains a separate hinge.
  // The bowl and payload are never flipped around pitch or roll to reverse it.
  orientation.rotation.y = Math.PI; curl.add(orientation);
  const u = S, shell = livery.studio ? studio.steel : materials.dark, sides = livery.studio ? studio.steel : livery.paint(livery.body);
  const steel = livery.studio ? studio.worn : materials.metal, brackets = livery.studio ? studio.steel : livery.paint(livery.accent);
  const top = new THREE.Vector2(-.205 * u, -.07 * u), heel = new THREE.Vector2(.22 * u, -.53 * u), edge = new THREE.Vector2(.43 * u, -.41 * u), lip = new THREE.Vector2(.125 * u, -.05 * u);
  const back = new THREE.Path(); back.moveTo(top.x, top.y);
  back.bezierCurveTo(-.33 * u, -.15 * u, -.345 * u, -.36 * u, -.245 * u, -.47 * u);
  back.bezierCurveTo(-.14 * u, -.585 * u, .07 * u, -.61 * u, heel.x, heel.y);
  back.lineTo(edge.x, edge.y);
  const outline = back.getPoints(14), cavity = new THREE.Vector2(.07 * u, -.3 * u);
  mesh(orientation, extrudeShape(shapeOf(plateOutline(outline, .03 * u, cavity)), width * .96, .005 * u, 16), shell).name = 'bucket-shell';
  const cheekGeometry = extrudeShape(shapeOf([...outline, lip, top]), width * .05, .006 * u, 16);
  for (const side of [-1, 1]) mesh(orientation, cheekGeometry, sides, 0, 0, side * width * .475).name = `bucket-side-${side}`;
  const topPlate = box(orientation, shell, (top.x + lip.x) / 2, (top.y + lip.y) / 2 - .012 * u, 0, lip.distanceTo(top) + .02 * u, .028 * u, width * .97);
  topPlate.rotation.z = Math.atan2(lip.y - top.y, lip.x - top.x);
  // Lower wear straps follow the heel; a thicker steel lip carries the teeth.
  for (const z of [-.32, .32]) { const strap = box(orientation, steel, -.06 * u, -.585 * u, z * width, .26 * u, .018 * u, width * .07); strap.rotation.z = -.12; }
  const along = edge.clone().sub(heel).normalize(), angle = Math.atan2(along.y, along.x);
  const cuttingEdge = box(orientation, steel, edge.x - along.x * .02 * u, edge.y - along.y * .02 * u - .006 * u, 0, .11 * u, .032 * u, width * 1.0); cuttingEdge.rotation.z = angle; cuttingEdge.name = 'bucket-cutting-edge';
  const tooth = shapeOf([[0, .026], [.07, .021], [.13, .006], [.145, -.001], [.075, -.014], [0, -.02]].map(([x, y]) => new THREE.Vector2(x * u, y * u)));
  const toothWidth = Math.min(.055 * u, width * .13), toothGeometry = extrudeShape(tooth, toothWidth, .005 * u, 4), count = width > .3 * u ? 5 : 4;
  for (let i = 0; i < count; i++) {
    const z = (i / (count - 1) - .5) * width * .84, base = edge.clone().addScaledVector(along, .03 * u);
    const adapter = box(orientation, shell, edge.x, edge.y + .004 * u, z, .075 * u, .045 * u, toothWidth * 1.35); adapter.rotation.z = angle;
    const tip = mesh(orientation, toothGeometry, steel, base.x, base.y, z); tip.rotation.z = angle; tip.name = 'bucket-tooth';
  }
  const toothTip = edge.clone().addScaledVector(along, .175 * u);

  // Paired mounting brackets enclose the main hinge and the moving linkage pin.
  // The bowl shrinks independently of the unchanged stick-eye housing. Keep
  // clearance between the bracket bevels and that housing, with pins through both.
  const earThickness = width * .075, earBevel = u * .008, earGap = hingeWidth + u * .02;
  const earOffset = (earGap + earThickness) / 2 + earBevel;
  const pinWidth = Math.max(width * .72, earGap + 2 * earThickness + 4 * earBevel + u * .014);
  const linkCenter = new THREE.Vector2(-.08 * u, .12 * u), ring = (center, radius) => Array.from({ length: 20 }, (_, i) => center.clone().add(new THREE.Vector2(Math.cos(i / 20 * Math.PI * 2) * radius, Math.sin(i / 20 * Math.PI * 2) * radius)));
  const ear = convexHull([...ring(new THREE.Vector2(0, 0), .075 * u), ...ring(linkCenter, .06 * u), new THREE.Vector2(.09 * u, -.06 * u), new THREE.Vector2(-.175 * u, -.075 * u)]);
  const earGeometry = extrudeShape(shapeOf(ear), earThickness, earBevel, 10);
  for (const side of [-1, 1]) mesh(orientation, earGeometry, brackets, 0, 0, side * earOffset).name = `bucket-ear-${side}`;
  pin(orientation, steel, 0, 0, 0, u * .045, pinWidth).name = 'bucket-main-pin';
  const linkPin = anchor(orientation, 'bucket-link-pin', linkCenter.x, linkCenter.y, 0);
  pin(linkPin, steel, 0, 0, 0, u * .032, pinWidth);
  anchor(orientation, 'bucket-teeth', toothTip.x, toothTip.y, 0);
  const lipPoint = new THREE.Vector2((edge.x + lip.x) / 2 - .04 * u, (edge.y + lip.y) / 2);
  // Outward normal of the opening (cutting edge to top plate), for checks.
  const opening = new THREE.Vector2(lip.y - edge.y, edge.x - lip.x).normalize();
  orientation.userData.opening = new THREE.Vector3(opening.x, opening.y, 0);
  anchor(orientation, 'bucket-lip', lipPoint.x, lipPoint.y, 0);
  const soil = mesh(orientation, lumpGeometry(1), materials.soil, .09 * u, -.28 * u, 0); soil.name = 'bucket-soil'; soil.scale.set(u * .25, u * .2, width * .4); soil.visible = false;
  return { curl, orientation, soil, linkPin, toothTip, lipPoint };
}

function nameplate(agent, style) {
  const hex = `#${SLOT_COLORS[agent.id % 4].toString(16).padStart(6, '0')}`, text = String(agent.id + 1).padStart(2, '0');
  const texture = canvasTexture(160, 96, ctx => {
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    if (style === 'paper') {
      // Figure annotation: plain number with a halo and a slot-color bar.
      ctx.font = '600 44px Inter, "Helvetica Neue", Arial, sans-serif'; ctx.lineJoin = 'round';
      ctx.lineWidth = 10; ctx.strokeStyle = '#ffffffee'; ctx.strokeText(text, 80, 38); ctx.fillStyle = '#1f2426'; ctx.fillText(text, 80, 38);
      ctx.fillStyle = '#ffffffee'; ctx.beginPath(); ctx.roundRect(52, 66, 56, 14, 7); ctx.fill();
      ctx.fillStyle = hex; ctx.beginPath(); ctx.roundRect(56, 69, 48, 8, 4); ctx.fill();
      return;
    }
    ctx.fillStyle = '#00000033'; ctx.beginPath(); ctx.roundRect(22, 12, 116, 58, 29); ctx.fill();
    ctx.fillStyle = hex; ctx.beginPath(); ctx.roundRect(20, 8, 120, 58, 29); ctx.fill();
    ctx.beginPath(); ctx.moveTo(68, 62); ctx.lineTo(92, 62); ctx.lineTo(80, 80); ctx.closePath(); ctx.fill();
    ctx.lineWidth = 5; ctx.strokeStyle = '#ffffffcc'; ctx.beginPath(); ctx.roundRect(22.5, 10.5, 115, 53, 26.5); ctx.stroke();
    ctx.font = '800 38px ui-rounded, "SF Pro Rounded", system-ui, sans-serif'; ctx.fillStyle = '#1f2a2c';
    ctx.fillText(text, 80, 39);
  });
  const label = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false, transparent: true, sizeAttenuation: false }));
  label.center.set(.5, 0); label.renderOrder = 30; label.scale.set(.05, .03, 1);
  return label;
}

function selectionRing(color, glow = true) {
  const hex = `#${color.toString(16).padStart(6, '0')}`;
  const texture = canvasTexture(256, 256, ctx => {
    if (glow) {
      const gradient = ctx.createRadialGradient(128, 128, 60, 128, 128, 126);
      gradient.addColorStop(0, `${hex}00`); gradient.addColorStop(.82, `${hex}38`); gradient.addColorStop(1, `${hex}00`);
      ctx.fillStyle = gradient; ctx.fillRect(0, 0, 256, 256);
    }
    ctx.strokeStyle = hex; ctx.lineWidth = 9; ctx.lineCap = 'round';
    for (let i = 0; i < 16; i++) { ctx.beginPath(); ctx.arc(128, 128, 112, i * Math.PI / 8 + .06, (i + .62) * Math.PI / 8); ctx.stroke(); }
  });
  const ring = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: texture, transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -4 }));
  ring.rotation.x = -Math.PI / 2; ring.renderOrder = 16; ring.userData.skipAO = true;
  return ring;
}

const smooth = t => t * t * (3 - 2 * t);
const clamp01 = t => Math.min(1, Math.max(0, t));
const backOut = t => { const c = 1.9; return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2; };
const segment = (t, a, b) => clamp01((t - a) / (b - a));
function blend(poses, keys, t) {
  // keys: [[time, poseName, easing]], piecewise between consecutive keys.
  for (let i = 1; i < keys.length; i++) {
    const [t1, name1, ease = smooth] = keys[i], [t0, name0] = keys[i - 1];
    if (t <= t1 || i === keys.length - 1) {
      const u = ease(segment(t, t0, t1)), a = poses[name0], b = poses[name1];
      return Object.fromEntries(Object.keys(a).map(key => [key, a[key] + (b[key] - a[key]) * u]));
    }
  }
  return poses[keys[0][1]];
}


// Excavator arm poses: boom and stick angles (radians) plus the bucket's world pitch.
const ARM = {
  // Travel pose: boom raised, stick tucked, bucket close to the cab.
  carry: { boom: .86, stick: -1.92, pitch: .04 },
  reach: { boom: .30, stick: -1.05, pitch: -.42 },
  scoop: { boom: .20, stick: -1.30, pitch: .62 },
  raise: { boom: .80, stick: -1.02, pitch: .10 },
  pour: { boom: .74, stick: -.98, pitch: .94 },
};
const ARM_KEYS = {
  dig: [[0, 'carry'], [.3, 'reach'], [.56, 'scoop'], [1, 'carry', backOut]],
  dump: [[0, 'carry'], [.34, 'raise'], [.62, 'pour'], [1, 'carry', backOut]],
};
// Bucket world pitch while travelling: curled to hold a load, slightly open when empty.
const CARRY_PITCH = { empty: -.12, loaded: -.5 };
// Loader arm and bucket-curl angles for its work poses.
const LOADER = { ground: { arm: -.33, curl: -.06 }, raised: { arm: .45 }, tipped: { curl: -.95 } };
const easeInOut = t => t < .5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
const easeIn = t => t * t, easeOut = t => 1 - (1 - t) * (1 - t);
const steady = t => .75 * t + .25 * smooth(t);
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

/** Interpolate keyed work poses; numbers and Vector2 fields blend with the later key's easing. */
function keyframe(keys, t) {
  let i = 1;
  while (i < keys.length - 1 && t > keys[i].t) i++;
  const a = keys[i - 1], b = keys[i], u = (b.ease ?? smooth)(clamp01((t - a.t) / Math.max(1e-6, b.t - a.t)));
  const pose = {};
  for (const key of Object.keys(b)) {
    if (key === 't' || key === 'ease') continue;
    pose[key] = b[key]?.isVector2 ? a[key].clone().lerp(b[key], u) : a[key] + (b[key] - a[key]) * u;
  }
  return pose;
}

const rotate2 = (v, a) => new THREE.Vector2(v.x * Math.cos(a) - v.y * Math.sin(a), v.x * Math.sin(a) + v.y * Math.cos(a));

/** Planar boom/stick kinematics about the boom pivot, with a two-link IK for the bucket hinge. */
function excavatorArm(boom, first, second, bucket, size) {
  const pivot = new THREE.Vector2(boom.position.x, boom.position.y);
  // Bucket points live in the reversed orientation frame: flip x into the curl frame.
  const offset = point => new THREE.Vector2(-point.x, point.y);
  const teeth = offset(bucket.toothTip), lip = offset(bucket.lipPoint);
  const hinge = (boomAngle, stickAngle) => new THREE.Vector2(first * Math.cos(boomAngle) + second * Math.cos(boomAngle + stickAngle), first * Math.sin(boomAngle) + second * Math.sin(boomAngle + stickAngle));
  function solveHinge(target) {
    const point = target.clone(), longest = (first + second) * .995, shortest = Math.abs(first - second) * 1.05 + 1e-6;
    const length = point.length();
    if (length > longest) point.multiplyScalar(longest / length); else if (length < shortest) point.multiplyScalar(shortest / Math.max(length, 1e-6));
    const distance = point.length();
    // Elbow up: the stick folds down from the boom, as on a backhoe.
    const stickAngle = -Math.acos(clamp((distance * distance - first * first - second * second) / (2 * first * second), -1, 1));
    const boomAngle = Math.atan2(point.y, point.x) - Math.atan2(second * Math.sin(stickAngle), first + second * Math.cos(stickAngle));
    return { boom: boomAngle, stick: stickAngle };
  }
  return {
    pivot, size, hinge, solveHinge, teeth, lip,
    tip: (boomAngle, stickAngle, pitch) => hinge(boomAngle, stickAngle).add(rotate2(teeth, pitch)),
    solveTip: (tip, pitch) => solveHinge(tip.clone().sub(rotate2(teeth, pitch))),
  };
}

export function makeMachine(agent, tile, { labels = true, style = 'diorama' } = {}) {
  const root = new THREE.Group(), L = agent.height * tile, W = agent.width * tile, S = Math.min(L, W);
  const livery = liveryFor(agent, style), wheeled = agent.action_type === 1 || agent.type === 1, studioLook = style === 'studio';
  root.name = `machine-${agent.id}`;
  const chassis = undercarriage(root, L, W, S, wheeled, livery, agent.type === 1);
  const suspension = new THREE.Group(); suspension.name = 'suspension'; root.add(suspension);
  const upper = new THREE.Group(); upper.position.y = .32 * S; suspension.add(upper);
  const body = livery.paint(livery.body), accent = livery.paint(livery.accent), trim = livery.paint(livery.trim);
  const soilMaterial = studioLook ? studio.soil : materials.soil, pinPaint = studioLook ? studio.steel : body, pinAccent = studioLook ? studio.steel : accent;
  const beaconMaterial = new THREE.MeshStandardMaterial({ color: 0xffa21f, roughness: .3, emissive: 0xff7a00, emissiveIntensity: .2, transparent: true, opacity: .92 });
  let boom, stick, tool, load, bed, loaderArm, bucketRig, exhaust, beacon, arm = null, teeth = null, lip = null;
  const hydraulics = [];
  hazardMaterial ||= new THREE.MeshStandardMaterial({ map: hazardTexture(), color: typeof document === 'undefined' ? 0xf4b21b : 0xffffff, roughness: .6 });
  if (agent.type === 0) {
    cylinder(upper, materials.dark, 0, .045 * S, 0, S * .31, S * .10);
    roundedBox(upper, body, -.10 * L, .16 * S, 0, L * .65, S * .23, W * .66, S * .065).name = 'excavator-upper-body';
    roundedBox(upper, studioLook ? trim : materials.dark, -.33 * L, .255 * S, 0, L * .20, S * .20, W * .65, S * .068).name = 'excavator-counterweight';
    box(upper, studioLook ? trim : hazardMaterial, -.434 * L, .255 * S, 0, L * .012, S * .09, W * .56);
    for (const side of [-1, 1]) box(upper, materials.tail, -.434 * L, .3 * S, side * W * .29, L * .012, S * .03, W * .05);
    // Engine hood with vent slats.
    roundedBox(upper, accent, -.22 * L, .29 * S, .16 * W, L * .26, S * .05, W * .3, S * .02);
    for (let i = 0; i < 5; i++) box(upper, materials.darker, (-.3 + i * .04) * L, .318 * S, .16 * W, L * .018, S * .012, W * .22);
    cab(upper, L, W, S, -.02 * L, -.18 * W, livery);
    beacon = cylinder(upper, beaconMaterial, -.1 * L, .7 * S, -.18 * W, S * .032, S * .06);
    cylinder(upper, materials.dark, -.1 * L, .665 * S, -.18 * W, S * .04, S * .02);
    cylinder(upper, materials.dark, -.28 * L, .45 * S, .26 * W, S * .026, S * .32);
    cylinder(upper, materials.darker, -.28 * L, .62 * S, .26 * W, S * .034, S * .03);
    exhaust = anchor(upper, 'exhaust', -.28 * L, .66 * S, .26 * W);
    rod(upper, materials.metal, [-.33 * L, .4 * S, .32 * W], [-.12 * L, .4 * S, .32 * W], S * .012);
    for (const x of [-.33, -.12]) rod(upper, materials.metal, [x * L, .27 * S, .32 * W], [x * L, .4 * S, .32 * W], S * .012);
    const firstLength = Math.max(L * .63, agent.reach[1] * tile * .40), secondLength = Math.max(L * .48, agent.reach[1] * tile * .32);
    boom = new THREE.Group(); boom.name = 'boom-pivot'; boom.position.set(.16 * L, .23 * S, .09 * W); upper.add(boom);
    beam(boom, firstLength, S * .21, W * .12, accent, true);
    pin(boom, materials.dark, 0, 0, 0, S * .095, W * .19);
    pin(boom, materials.metal, 0, 0, 0, S * .050, W * .205);
    for (const side of [-1, 1]) box(boom, studioLook ? studio.lamp : materials.lamp, firstLength * .3, S * .1, side * W * .065, S * .04, S * .03, S * .012);
    for (const side of [-1, 1]) {
      const start = anchor(upper, `boom-cylinder-${side}-start`, .20 * L, .12 * S, (.09 + side * .12) * W);
      const end = anchor(boom, `boom-cylinder-${side}-end`, firstLength * .48, -.055 * S, side * W * .12);
      pin(start, pinPaint, 0, 0, 0, S * .047, W * .055); pin(end, pinPaint, 0, 0, 0, S * .047, W * .055);
      hydraulics.push(hydraulic(upper, start, end, S * .036, `boom-cylinder-${side}`));
    }
    stick = new THREE.Group(); stick.name = 'stick-pivot'; stick.position.x = firstLength; boom.add(stick);
    beam(stick, secondLength, S * .17, W * .09, body, true);
    roundedBox(stick, body, -.07 * secondLength, .045 * S, 0, .22 * secondLength, S * .105, W * .09, S * .035);
    pin(stick, materials.dark, 0, 0, 0, S * .078, W * .16); pin(stick, materials.metal, 0, 0, 0, S * .040, W * .175);
    const stickStart = anchor(boom, 'stick-cylinder-start', firstLength * .40, S * .145, 0);
    const stickEnd = anchor(stick, 'stick-cylinder-end', -.09 * secondLength, S * .080, 0);
    pin(stickStart, pinAccent, 0, 0, 0, S * .048, W * .11); pin(stickEnd, pinPaint, 0, 0, 0, S * .043, W * .115);
    hydraulics.push(hydraulic(upper, stickStart, stickEnd, S * .040, 'stick-cylinder'));

    // Keep the bucket compact relative to the cab. Scale its bowl, payload and
    // local linkage together around the unchanged stick-end hinge.
    const bucketScale = livery.studio ? .56 : .65, bucketSize = S * bucketScale, bucketWidth = W * .39 * bucketScale;
    const hingeWidth = W * .145, bucketPart = excavatorBucket(stick, bucketSize, bucketWidth, hingeWidth, livery);
    tool = bucketPart.curl; tool.position.x = secondLength; load = bucketPart.soil; load.material = soilMaterial;
    teeth = bucketPart.orientation.getObjectByName('bucket-teeth'); lip = bucketPart.orientation.getObjectByName('bucket-lip');
    anchor(stick, 'bucket-hinge', secondLength, 0, 0);
    pin(stick, body, secondLength, 0, 0, bucketSize * .068, hingeWidth).name = 'bucket-hinge-housing';
    const rockerOrigin = anchor(stick, 'bucket-rocker-pivot', secondLength - bucketSize * .24, bucketSize * .10, 0);
    roundedBox(stick, body, rockerOrigin.position.x, .04 * bucketSize, 0, bucketSize * .115, bucketSize * .17, W * .105, bucketSize * .025);
    pin(rockerOrigin, materials.metal, 0, 0, 0, bucketSize * .036, bucketWidth * .72);
    const rockerJoint = anchor(stick, 'bucket-rocker-joint', 0, 0, 0);
    pin(rockerJoint, materials.metal, 0, 0, 0, bucketSize * .036, bucketWidth * .72);
    const firstLinkLength = bucketSize * .22, secondLinkLength = bucketSize * .25, links = [];
    for (const side of [-1, 1]) {
      const first = mesh(stick, cylinderGeometry, body), second = mesh(stick, cylinderGeometry, materials.dark);
      first.name = `bucket-rocker-${side}`; second.name = `bucket-link-${side}`;
      links.push({ first, second, z: side * bucketWidth * .30 });
    }
    const bucketCylinderStart = anchor(stick, 'bucket-cylinder-start', secondLength * .24, S * .12, 0);
    pin(bucketCylinderStart, pinPaint, 0, 0, 0, bucketSize * .038, W * .115);
    hydraulics.push(hydraulic(stick, bucketCylinderStart, rockerJoint, bucketSize * .030, 'bucket-cylinder'));
    bucketRig = {
      origin: rockerOrigin, joint: rockerJoint, destination: bucketPart.linkPin,
      firstLength: firstLinkLength, secondLength: secondLinkLength, links,
      update() {
        const destination = stick.worldToLocal(bucketPart.linkPin.getWorldPosition(new THREE.Vector3()));
        const joint = linkageJoint(rockerOrigin.position, destination, firstLinkLength, secondLinkLength);
        rockerJoint.position.copy(joint);
        for (const link of links) {
          const start = rockerOrigin.position.clone(), end = destination.clone(), middle = joint.clone();
          start.z = link.z; middle.z = link.z; end.z = link.z;
          placeRod(link.first, start, middle, bucketSize * .029); placeRod(link.second, middle, end, bucketSize * .025);
        }
      },
    };
    arm = excavatorArm(boom, firstLength, secondLength, bucketPart, bucketSize);
  } else if (agent.type === 1) {
    box(upper, materials.dark, 0, .02 * S, 0, L * .92, .09 * S, W * .62);
    for (const side of [-1, 1]) box(upper, trim, .05 * L, .08 * S, side * W * .44, L * .7, .05 * S, W * .1);
    // Cab-forward hauler: chunky nose, grille and a canopy from the bed above.
    roundedBox(upper, body, .36 * L, .16 * S, 0, .22 * L, .26 * S, W * .82, S * .05);
    box(upper, materials.darker, .475 * L, .15 * S, 0, L * .02, .15 * S, W * .5);
    for (let i = 0; i < 4; i++) box(upper, materials.metal, .486 * L, (.1 + i * .035) * S, 0, L * .01, S * .012, W * .44);
    for (const side of [-1, 1]) { box(upper, materials.lamp, .478 * L, .24 * S, side * W * .32, L * .02, .05 * S, W * .1); box(upper, materials.chrome, .44 * L, .06 * S, side * W * .37, L * .08, .04 * S, W * .12); }
    cab(upper, L * .92, W * 1.62, S * .95, .30 * L, -.14 * W, livery);
    exhaust = anchor(upper, 'exhaust', .2 * L, .78 * S, .3 * W);
    cylinder(upper, materials.chrome, .2 * L, .5 * S, .3 * W, S * .03, S * .52);
    const pivot = new THREE.Group(); pivot.name = 'truck-bed'; pivot.position.set(-.43 * L, .12 * S, 0); upper.add(pivot); bed = pivot;
    const bedPaint = livery.paint(livery.bed);
    box(pivot, bedPaint, .30 * L, 0, 0, L * .64, S * .08, W * .86);
    for (const side of [-1, 1]) {
      const wall = box(pivot, bedPaint, .30 * L, .2 * S, side * W * .41, .66 * L, S * .38, W * .05); wall.rotation.x = side * .08;
      for (let i = 0; i < 4; i++) box(pivot, accent, (.06 + i * .16) * L, .22 * S, side * W * .44, L * .025, S * .34, W * .02);
      box(pivot, accent, .30 * L, .4 * S, side * W * .43, .66 * L, S * .04, W * .07);
    }
    box(pivot, bedPaint, .62 * L, .28 * S, 0, .04 * L, S * .52, W * .86);
    const canopy = box(pivot, bedPaint, .72 * L, .52 * S, 0, .22 * L, S * .04, W * .86); canopy.rotation.z = -.06;
    box(pivot, materials.dark, -.02 * L, .22 * S, 0, .03 * L, S * .3, W * .78);
    load = mesh(pivot, lumpGeometry(2), soilMaterial, L * .3, S * .2, 0); load.scale.set(L * .27, S * .2, W * .33);
    for (const side of [-1, 1]) box(upper, materials.tail, -.47 * L, .05 * S, side * .32 * W, .02 * L, .05 * S, .1 * W);
  } else {
    // Compact loader: engine at the rear, a cage cab, and side lift arms.
    roundedBox(upper, body, -.06 * L, .12 * S, 0, .72 * L, .26 * S, .64 * W, S * .05);
    roundedBox(upper, materials.dark, -.34 * L, .2 * S, 0, .14 * L, .22 * S, .6 * W, S * .04);
    for (let i = 0; i < 4; i++) box(upper, materials.darker, -.412 * L, (.12 + i * .045) * S, 0, L * .01, S * .018, W * .46);
    const cage = new THREE.Group(); cage.position.set(-.06 * L, .25 * S, 0); upper.add(cage);
    const cx = .34 * L, cz = .4 * W, ch = .46 * S;
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) box(cage, materials.darker, sx * cx * .47, ch / 2, sz * cz * .47, S * .035, ch, S * .035);
    roundedBox(cage, body, 0, ch, 0, cx * 1.06, S * .05, cz * 1.08, S * .02);
    box(cage, studioLook ? studio.glass : materials.glass, cx * .47, ch * .52, 0, S * .01, ch * .78, cz * .86);
    glassHighlight(cage, cx * .478, ch * .52, 0, S * .01, ch * .6, cz * .86, 'x');
    for (const side of [-1, 1]) for (let i = 0; i < 4; i++) box(cage, materials.darker, (-.3 + i * .2) * cx, ch * .55, side * cz * .47, S * .012, ch * .8, S * .012);
    box(cage, materials.seat, -cx * .1, ch * .25, 0, cx * .35, ch * .3, cz * .5);
    beacon = cylinder(cage, beaconMaterial, -cx * .3, ch + S * .05, 0, S * .03, S * .05);
    exhaust = anchor(upper, 'exhaust', -.36 * L, .42 * S, .2 * W);
    cylinder(upper, materials.dark, -.36 * L, .36 * S, .2 * W, S * .025, S * .14);
    loaderArm = new THREE.Group(); loaderArm.name = 'loader-arm'; loaderArm.position.set(-.18 * L, .22 * S, 0); upper.add(loaderArm);
    for (const side of [-1, 1]) {
      const beamGroup = new THREE.Group(); beamGroup.position.z = side * W * .36; loaderArm.add(beamGroup);
      beam(beamGroup, L * .81, S * .13, W * .075, accent);
      rod(beamGroup, materials.chrome, [L * .1, -.08 * S, 0], [L * .5, -.06 * S, 0], S * .022);
      pin(beamGroup, materials.metal, 0, 0, 0, S * .05, W * .09);
    }
    rod(loaderArm, trim, [L * .66, -.04 * S, -W * .36], [L * .66, -.04 * S, W * .36], S * .04);
    const bucketPart = loaderBucket(loaderArm, S * 1.22, W * .92, livery); tool = bucketPart.root; tool.position.set(.80 * L, -.10 * S, 0); load = bucketPart.soil; load.material = soilMaterial;
    teeth = tool.getObjectByName('loader-edge'); lip = tool.getObjectByName('loader-lip');
  }
  if (beacon) beacon.name = 'beacon';
  const ringColor = SLOT_COLORS[agent.id % 4], ring = selectionRing(ringColor, style === 'diorama'), paper = style === 'paper';
  if (style !== 'diorama') root.traverse(item => { if (item.name === 'glass-highlight') item.visible = false; });
  ring.scale.set(L * 1.34, W * 1.34 + (L - W) * .35, 1); ring.position.y = tile * .03; root.add(ring);
  let label = null;
  if (labels) { label = nameplate(agent, style === 'diorama' ? 'diorama' : 'paper'); label.position.set(-.05 * L, S * 1.12, 0); root.add(label); }
  const bucketTip = new THREE.Vector3();
  const motion = { last: null, treads: [0, 0], spin: 0, active: false, kind: '', phase: 1, lift: 0, tags: true };
  const wheelRadius = chassis.spinning[0]?.radius ?? S * .2;
  const load0 = load ? load.scale.clone() : null;

  /** Changed cells in a machine frame; heights are the displayed surface before and after. */
  function workCells(work, frame) {
    root.updateWorldMatrix(true, true);
    return work.cells.map(cell => {
      const before = frame.worldToLocal(new THREE.Vector3(cell.x, cell.before, cell.z)), after = frame.worldToLocal(new THREE.Vector3(cell.x, cell.after, cell.z));
      return { key: cell.key, x: before.x, z: before.z, before: before.y, after: after.y, weight: Math.max(1, Math.abs(cell.delta ?? 1)) };
    });
  }
  const weighted = (cells, read) => cells.reduce((sum, cell) => sum + read(cell) * cell.weight, 0) / cells.reduce((sum, cell) => sum + cell.weight, 0);
  const progressive = (cells, timing) => t => weighted(cells, cell => smooth(segment(t, ...timing.get(cell.key))));

  // Dig: open the bucket above the far edge of the changed cells, drag the
  // teeth along the new cut floor toward the cab, curl and lift. Dump: hold
  // the hinge above the deposit and open the bucket. A small extra slew
  // centers the boom on the cells and returns before the action ends.
  function planExcavator(work) {
    const cells = workCells(work, upper);
    if (!cells.length) return null;
    const cx = weighted(cells, cell => cell.x), cz = weighted(cells, cell => cell.z), radius = Math.hypot(cx, cz), lateral = boom.position.z;
    const yaw = radius > Math.abs(lateral) * 1.5 ? clamp(Math.asin(clamp(lateral / radius, -1, 1)) - Math.atan2(cz, cx), -.6, .6) : 0;
    const cos = Math.cos(yaw), sin = Math.sin(yaw), u = arm.size, timing = new Map();
    for (const cell of cells) { cell.s = cos * cell.x - sin * cell.z - arm.pivot.x; cell.before -= arm.pivot.y; cell.after -= arm.pivot.y; }
    const carryTip = loaded => arm.tip(ARM.carry.boom, ARM.carry.stick, loaded ? CARRY_PITCH.loaded : CARRY_PITCH.empty);
    if (work.kind === 'dig') {
      const far = Math.max(...cells.map(cell => cell.s)) + tile * .3, near = Math.min(...cells.map(cell => cell.s)) - tile * .35;
      const floor = Math.min(...cells.map(cell => cell.after)) - tile * .05, top = Math.max(...cells.map(cell => cell.before), floor + tile * .2);
      const keys = [
        { t: 0, point: carryTip(false), pitch: CARRY_PITCH.empty },
        { t: .22, point: new THREE.Vector2(far + u * .12, top + u * .45), pitch: 1.25, ease: easeInOut },
        { t: .34, point: new THREE.Vector2(far, floor + tile * .03), pitch: 1.05, ease: easeIn },
        { t: .68, point: new THREE.Vector2(near, floor), pitch: .4, ease: steady },
        { t: .8, point: new THREE.Vector2(near - u * .08, top + u * .3), pitch: -.55, ease: easeOut },
        { t: 1, point: carryTip(true), pitch: CARRY_PITCH.loaded, ease: easeInOut },
      ];
      const span = Math.max(far - near, 1e-6);
      for (const cell of cells) { const at = .34 + .34 * clamp01((far - cell.s) / span); timing.set(cell.key, [at - .05, at + .07]); }
      return { kind: 'dig', space: 'tip', keys, yaw, yawWindow: [.24, .84], timing, fill: progressive(cells, timing), events: { bite: .34, drag: [.34, .68], breakout: .76 } };
    }
    const pour = 1.8, centre = weighted(cells, cell => cell.s), peak = Math.max(...cells.map(cell => Math.max(cell.before, cell.after)));
    // Place the hinge so the open bucket's lip sits above the deposit centre.
    const above = new THREE.Vector2(centre - rotate2(arm.lip, pour).x, peak + u * .78), carryHinge = arm.hinge(ARM.carry.boom, ARM.carry.stick);
    const keys = [
      { t: 0, point: carryHinge, pitch: CARRY_PITCH.loaded },
      { t: .3, point: above, pitch: -.4, ease: easeInOut },
      { t: .6, point: above.clone().add(new THREE.Vector2(0, u * .04)), pitch: pour, ease: easeInOut },
      { t: .72, point: above.clone().add(new THREE.Vector2(0, u * .07)), pitch: pour + .12, ease: steady },
      { t: 1, point: carryHinge, pitch: CARRY_PITCH.empty, ease: easeInOut },
    ];
    for (const cell of cells) timing.set(cell.key, [.46, .8]);
    return { kind: 'dump', space: 'hinge', keys, yaw, yawWindow: [.3, .78], timing, fill: t => 1 - smooth(segment(t, .38, .66)), events: { pour: [.38, .7] } };
  }

  // Loader pickup: lower the bucket flat, drive into the soil, roll it back
  // and reverse. Dump: drive up with the arms raised and tip the bucket.
  function planLoader(work) {
    const cells = workCells(work, root);
    if (!cells.length) return null;
    const saved = [loaderArm.rotation.z, tool.rotation.z];
    const reach = (armAngle, curl, point) => { loaderArm.rotation.z = armAngle; tool.rotation.z = curl; root.updateWorldMatrix(true, true); return root.worldToLocal(point.getWorldPosition(new THREE.Vector3())).x; };
    const groundEdge = reach(LOADER.ground.arm, LOADER.ground.curl, teeth), raisedLip = reach(LOADER.raised.arm, LOADER.tipped.curl, lip);
    loaderArm.rotation.z = saved[0]; tool.rotation.z = saved[1];
    const near = Math.min(...cells.map(cell => cell.x)), far = Math.max(...cells.map(cell => cell.x)), timing = new Map();
    const idle = state => ({ arm: state.shovel_lifted ? .35 : -.2, curl: state.loaded > 0 ? .22 : 0, lunge: 0 });
    const from = idle(work.from), to = idle(work.to);
    if (work.kind === 'dig') {
      const lunge = clamp(near - groundEdge + tile * .35, 0, 3.5);
      const keys = [
        { t: 0, ...from },
        { t: .2, arm: LOADER.ground.arm, curl: LOADER.ground.curl, lunge: lunge * .2, ease: easeInOut },
        { t: .5, arm: LOADER.ground.arm, curl: LOADER.ground.curl, lunge, ease: steady },
        { t: .62, arm: LOADER.ground.arm + .03, curl: .5, lunge, ease: easeOut },
        { t: .8, arm: -.1, curl: .38, lunge: lunge * .5, ease: easeInOut },
        { t: 1, ...to, ease: easeInOut },
      ];
      const span = Math.max(far - near, 1e-6);
      for (const cell of cells) { const at = .3 + .2 * clamp01((cell.x - near) / span); timing.set(cell.key, [at - .05, at + .07]); }
      return { kind: 'dig', space: 'loader', keys, timing, fill: progressive(cells, timing), events: { bite: .3, drag: [.3, .55] } };
    }
    const lunge = clamp(weighted(cells, cell => cell.x) - raisedLip, 0, 3.5);
    const keys = [
      { t: 0, ...from },
      { t: .32, arm: LOADER.raised.arm, curl: .3, lunge, ease: easeInOut },
      { t: .55, arm: LOADER.raised.arm, curl: LOADER.tipped.curl, lunge, ease: easeInOut },
      { t: .68, arm: LOADER.raised.arm - .03, curl: LOADER.tipped.curl, lunge, ease: steady },
      { t: 1, ...to, ease: easeInOut },
    ];
    for (const cell of cells) timing.set(cell.key, [.46, .8]);
    return { kind: 'dump', space: 'loader', keys, timing, fill: t => 1 - smooth(segment(t, .36, .6)), events: { pour: [.36, .66] } };
  }

  function setPose(state, active, phase = 0, kind = '', plan = null) {
    upper.rotation.y = state.cabin_yaw;
    for (const wheelPart of chassis.steering) wheelPart.rotation.y = Math.max(-.6, Math.min(.6, state.wheel_angle * Math.PI / 9));
    motion.active = active; motion.kind = kind; motion.phase = phase;
    ring.visible = active && motion.tags && !studioLook;
    const carrying = state.loaded > 0;
    if (plan && load0) {
      const fill = plan.fill(phase), heap = .35 + .65 * fill;
      load.visible = fill > .03;
      load.scale.set(load0.x * (.65 + .35 * fill), load0.y * heap, load0.z * (.75 + .25 * fill));
    } else {
      load.visible = carrying || (kind === 'dump' && phase < .52) || (kind === 'transfer' && phase < .52) || (kind === 'dig' && phase > .5);
      if (kind === 'receive') load.visible = phase > .62;
      if (load0 && agent.type !== 0) {
        // Loose payload height saturates with the carried amount; display only.
        const before = kind === 'dump' ? Math.max(1, state.previous_loaded ?? state.loaded) : state.loaded;
        let fill = .55 + .45 * (1 - Math.exp(-Math.max(before, 1) / 18));
        if (kind === 'dump') { fill *= 1 - smooth(segment(phase, .22, .5)); load.visible = phase < .5; }
        load.scale.set(load0.x, load0.y * Math.max(fill, .02), load0.z * (kind === 'dump' ? .7 + .3 * fill : 1));
      }
    }
    if (boom) {
      if (plan) {
        const pose = keyframe(plan.keys, phase), [inside, outside] = plan.yawWindow;
        upper.rotation.y = state.cabin_yaw + plan.yaw * smooth(segment(phase, 0, inside)) * (1 - smooth(segment(phase, outside, 1)));
        const angles = plan.space === 'hinge' ? arm.solveHinge(pose.point) : arm.solveTip(pose.point, pose.pitch);
        boom.rotation.z = angles.boom; stick.rotation.z = angles.stick; tool.rotation.z = pose.pitch - angles.boom - angles.stick;
      } else {
        const keys = kind === 'dig' ? ARM_KEYS.dig : kind === 'dump' || kind === 'transfer' ? ARM_KEYS.dump : null;
        const poses = { ...ARM, carry: { ...ARM.carry, pitch: carrying ? CARRY_PITCH.loaded : CARRY_PITCH.empty } };
        const pose = keys ? blend(poses, keys, phase) : poses.carry;
        boom.rotation.z = pose.boom; stick.rotation.z = pose.stick;
        // An inward-facing bucket empties by lowering its inward cutting edge.
        // Compensate the arm pose so carry stays upright and yaw never becomes curl.
        tool.rotation.z = pose.pitch - boom.rotation.z - stick.rotation.z;
      }
      root.updateWorldMatrix(true, true); bucketRig.update();
      for (const actuator of hydraulics) actuator.update();
    }
    if (bed) {
      const tilt = kind === 'dump' ? (phase < .45 ? smooth(segment(phase, 0, .45)) : phase < .7 ? 1 : 1 - smooth(segment(phase, .7, 1))) : 0;
      bed.rotation.z = .62 * tilt;
    }
    if (loaderArm) {
      if (plan) {
        const pose = keyframe(plan.keys, phase);
        loaderArm.rotation.z = pose.arm; tool.rotation.z = pose.curl;
        if (pose.lunge) root.translateX(pose.lunge);
        return;
      }
      const lifted = state.shovel_lifted ? .35 : -.2, loadedCurl = carrying ? .22 : 0;
      let armAngle = lifted, curl = loadedCurl;
      if (kind === 'dig') {
        armAngle = phase < .35 ? THREE.MathUtils.lerp(-.2, -.3, smooth(segment(phase, 0, .35))) : phase < .6 ? -.3 : THREE.MathUtils.lerp(-.3, lifted, backOut(segment(phase, .6, 1)));
        curl = phase < .35 ? -.15 * smooth(segment(phase, 0, .35)) : phase < .6 ? THREE.MathUtils.lerp(-.15, .3, smooth(segment(phase, .35, .6))) : THREE.MathUtils.lerp(.3, loadedCurl, smooth(segment(phase, .6, 1)));
      } else if (kind === 'dump') {
        armAngle = phase < .6 ? THREE.MathUtils.lerp(.35, .45, smooth(segment(phase, 0, .35))) : THREE.MathUtils.lerp(.45, lifted, smooth(segment(phase, .6, 1)));
        curl = phase < .3 ? .22 * (1 - segment(phase, 0, .3)) : phase < .65 ? -.8 * smooth(segment(phase, .3, .5)) : THREE.MathUtils.lerp(-.8, loadedCurl, smooth(segment(phase, .65, 1)));
      }
      loaderArm.rotation.z = armAngle; tool.rotation.z = curl;
    }
  }

  return {
    root, agent, bucketRig, hydraulics, suspension, ringColor, arm,
    setPose,
    /** Plan the display motion of one dig/dump on its changed cells; null when unsupported. */
    plan(work) { return arm ? planExcavator(work) : loaderArm ? planLoader(work) : null; },
    /** Show or hide the number tag and the active-machine ring. */
    setTags(visible) { motion.tags = visible; if (label) label.visible = visible; ring.visible = visible && motion.active && !studioLook; },
    /** Accumulate tread and wheel travel from successive display poses. */
    drive(position, yaw) {
      const last = motion.last;
      motion.last = { x: position.x, z: position.z, yaw };
      if (!last) return;
      const dx = position.x - last.x, dz = position.z - last.z, turn = Math.atan2(Math.sin(yaw - last.yaw), Math.cos(yaw - last.yaw));
      if (Math.hypot(dx, dz) > L * 1.5 || Math.abs(turn) > 1.2) return;
      const ds = dx * Math.cos(yaw) - dz * Math.sin(yaw);
      motion.speed = ds;
      for (const [i, item] of chassis.tracks.entries()) { motion.treads[i] -= ds + item.side * W * .35 * turn; item.update(motion.treads[i]); }
      motion.spin -= ds / wheelRadius;
      for (const item of chassis.spinning) item.spin.rotation.z = motion.spin;
    },
    /** Idle life and move reactions; `move` is a signed 0..1 travel progress. */
    tick(time, { move = null, direction = 1, reducedMotion = false } = {}) {
      const active = motion.active;
      if (paper || studioLook) {
        // Figure styles: no flashing, pulsing or idle shake. Studio keeps a
        // slight pitch as the machine pulls away and stops.
        beaconMaterial.emissiveIntensity = studioLook ? .08 : .15; ring.material.opacity = active && paper ? .9 : 0; ring.rotation.z = 0;
        if (label) { label.material.opacity = 1; label.position.y = S * 1.12; }
        suspension.position.y = 0;
        suspension.rotation.z = studioLook && move !== null && !reducedMotion ? -Math.sin(move * Math.PI * 2) * .014 * direction : 0;
        return;
      }
      beaconMaterial.emissiveIntensity = active ? .5 + .9 * Math.max(0, Math.sin(time * 7)) ** 3 : .15;
      ring.material.opacity = active ? .75 + .25 * Math.sin(time * 3.2) : 0;
      ring.rotation.z = time * .25;
      if (label) { label.material.opacity = active ? 1 : .72; label.position.y = S * 1.12 + (active && !reducedMotion ? Math.sin(time * 3) * S * .03 : 0); }
      if (reducedMotion) { suspension.position.y = 0; suspension.rotation.z = 0; return; }
      // Rock back as the machine pulls away and nod forward as it stops.
      const pitch = move === null ? 0 : -Math.sin(move * Math.PI * 2) * .035 * direction;
      suspension.rotation.z = pitch;
      suspension.position.y = active ? Math.sin(time * 41) * S * .0015 : 0;
      if (move !== null) suspension.position.y += Math.abs(Math.sin(move * Math.PI * 3)) * S * .008;
    },
    tip() { load.getWorldPosition(bucketTip); return bucketTip.clone(); },
    /** World position of the excavator teeth or loader cutting edge. */
    teeth() { return (teeth ?? load).getWorldPosition(new THREE.Vector3()); },
    /** World position where soil leaves the open bucket. */
    lip() { return (lip ?? load).getWorldPosition(new THREE.Vector3()); },
    exhaust() { return exhaust ? exhaust.getWorldPosition(new THREE.Vector3()) : root.position.clone(); },
    bedLip() { return bed ? bed.localToWorld(new THREE.Vector3(-.02 * L, .05 * S, 0)) : this.tip(); },
    dispose() {
      const geometries = new Set();
      root.traverse(item => {
        if (item.geometry && item.geometry !== boxGeometry && item.geometry !== cylinderGeometry && item.geometry !== hexCylinder && ![...lumpCache.values()].includes(item.geometry)) geometries.add(item.geometry);
        if (item.isSprite) { item.material.map.dispose(); item.material.dispose(); }
      });
      for (const geometry of geometries) geometry.dispose();
      ring.material.map?.dispose(); ring.material.dispose(); beaconMaterial.dispose();
      for (const item of chassis.tracks) item.shoes.dispose();
    },
  };
}
