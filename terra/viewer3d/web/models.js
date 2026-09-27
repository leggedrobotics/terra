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
  const body = paint(livery.body), group = new THREE.Group(); group.position.set(x, 0, z); parent.add(group);
  const cx = .30 * L, cz = .39 * W, height = .5 * S;
  roundedBox(group, body, 0, .12 * S, 0, cx, .2 * S, cz, S * .04);
  // Frame, then inset glass on the front, sides and rear.
  roundedBox(group, body, 0, .36 * S, 0, cx * .96, height * .72, cz * .96, S * .05).name = 'cab-shell';
  const glassY = .39 * S, glassH = height * .56;
  box(group, materials.glass, cx * .485, glassY, 0, S * .012, glassH, cz * .84);
  glassHighlight(group, cx * .492, glassY, 0, S * .01, glassH * .8, cz * .84, 'x');
  for (const side of [-1, 1]) { box(group, materials.glass, -cx * .04, glassY, side * cz * .485, cx * .76, glassH, S * .012); glassHighlight(group, -cx * .04, glassY, side * cz * .492, cx * .76, glassH * .8, S * .01, 'z'); }
  box(group, materials.glass, -cx * .485, glassY + glassH * .1, 0, S * .012, glassH * .6, cz * .7);
  box(group, materials.seat, -cx * .12, .3 * S, 0, cx * .3, .16 * S, cz * .5);
  roundedBox(group, paint(livery.trim), 0, .62 * S, 0, cx * 1.06, S * .05, cz * 1.06, S * .02);
  for (const side of [-1, 1]) box(group, materials.lamp, cx * .5, .6 * S, side * cz * .3, S * .02, S * .035, cz * .12);
  return group;
}

function loaderBucket(parent, S, width, livery) {
  const root = new THREE.Group(); parent.add(root);
  root.name = 'loader-bucket';
  box(root, materials.dark, S * .07, -S * .1, 0, S * .35, S * .06, width);
  const back = box(root, materials.dark, -S * .12, .01 * S, 0, S * .06, S * .26, width); back.rotation.z = .25;
  box(root, materials.dark, -S * .04, .14 * S, 0, S * .14, S * .04, width * .98).rotation.z = -.5;
  for (const side of [-1, 1]) box(root, paint(livery.body), 0, .025 * S, side * width * .48, S * .31, S * .22, width * .055);
  box(root, materials.metal, S * .25, -S * .12, 0, S * .06, S * .03, width * 1.01);
  for (let i = 0; i < 6; i++) box(root, materials.metal, S * .29, -S * .115, (i / 5 - .5) * width * .86, S * .1, S * .035, width * .07);
  const soil = mesh(root, lumpGeometry(3), materials.soil, .04 * S, .03 * S, 0); soil.scale.set(S * .2, S * .12, width * .42); soil.visible = false;
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

function excavatorBucket(parent, S, width, hingeWidth, livery) {
  const curl = new THREE.Group(); curl.name = 'bucket-curl'; parent.add(curl);
  const orientation = new THREE.Group(); orientation.name = 'bucket-orientation';
  // Reverse the bucket around its vertical axis; curl remains a separate hinge.
  // The bowl and payload are never flipped around pitch or roll to reverse it.
  orientation.rotation.y = Math.PI; curl.add(orientation);
  const shell = new THREE.Shape();
  shell.moveTo(-.14 * S, -.05 * S);
  shell.bezierCurveTo(-.34 * S, -.12 * S, -.34 * S, -.34 * S, -.18 * S, -.43 * S);
  shell.quadraticCurveTo(.04 * S, -.50 * S, .30 * S, -.40 * S);
  shell.lineTo(.32 * S, -.34 * S);
  shell.quadraticCurveTo(.06 * S, -.43 * S, -.15 * S, -.37 * S);
  shell.bezierCurveTo(-.27 * S, -.30 * S, -.26 * S, -.16 * S, -.09 * S, -.105 * S);
  shell.closePath();
  const shellGeometry = new THREE.ExtrudeGeometry(shell, { depth: width, bevelEnabled: true, bevelSegments: 3, bevelSize: S * .008, bevelThickness: S * .008, curveSegments: 12, steps: 1 });
  mesh(orientation, shellGeometry, materials.dark, 0, 0, -width / 2).name = 'bucket-shell';

  const cheek = new THREE.Shape();
  cheek.moveTo(-.14 * S, -.06 * S);
  cheek.bezierCurveTo(-.34 * S, -.14 * S, -.34 * S, -.35 * S, -.17 * S, -.43 * S);
  cheek.quadraticCurveTo(.05 * S, -.48 * S, .31 * S, -.39 * S);
  cheek.lineTo(.19 * S, -.18 * S); cheek.quadraticCurveTo(.04 * S, -.10 * S, -.14 * S, -.06 * S);
  const cheekThickness = width * .05;
  const cheekGeometry = new THREE.ExtrudeGeometry(cheek, { depth: cheekThickness, bevelEnabled: true, bevelSegments: 3, bevelSize: S * .009, bevelThickness: S * .007, curveSegments: 12, steps: 1 });
  for (const side of [-1, 1]) mesh(orientation, cheekGeometry, paint(livery.body), 0, 0, side * width * .475 - cheekThickness / 2);
  roundedBox(orientation, materials.metal, S * .29, -S * .38, 0, S * .07, S * .06, width * 1.02, S * .012).rotation.z = .1;
  const teeth = new THREE.Shape(); teeth.moveTo(0, -.035 * S); teeth.lineTo(.17 * S, -.016 * S); teeth.lineTo(.17 * S, .009 * S); teeth.lineTo(0, .035 * S); teeth.closePath();
  const toothGeometry = new THREE.ExtrudeGeometry(teeth, { depth: width * .085, bevelEnabled: true, bevelSegments: 2, bevelSize: S * .006, bevelThickness: S * .006, steps: 1 });
  for (let i = 0; i < 5; i++) mesh(orientation, toothGeometry, materials.chrome, S * .31, -S * .385, (i / 4 - .5) * width * .82 - width * .0425);

  // Paired mounting ears enclose the main hinge and the moving linkage pin.
  const ear = new THREE.Shape(); ear.moveTo(-.14 * S, -.11 * S); ear.lineTo(-.14 * S, .115 * S); ear.quadraticCurveTo(-.10 * S, .18 * S, -.045 * S, .15 * S); ear.lineTo(.075 * S, .025 * S); ear.quadraticCurveTo(.10 * S, -.025 * S, .055 * S, -.10 * S); ear.closePath();
  // The bowl shrinks independently of the unchanged stick-eye housing. Keep
  // clearance between the ear bevels and that housing, with pins through both.
  const earThickness = width * .075, earBevel = S * .008, earGap = hingeWidth + S * .02;
  const earOffset = (earGap + earThickness) / 2 + earBevel;
  const pinWidth = Math.max(width * .72, earGap + 2 * earThickness + 4 * earBevel + S * .014);
  const earGeometry = new THREE.ExtrudeGeometry(ear, { depth: earThickness, bevelEnabled: true, bevelSegments: 3, bevelSize: S * .008, bevelThickness: earBevel, curveSegments: 10, steps: 1 });
  for (const side of [-1, 1]) mesh(orientation, earGeometry, paint(livery.accent), 0, 0, side * earOffset - earThickness / 2).name = `bucket-ear-${side}`;
  pin(orientation, materials.metal, 0, 0, 0, S * .045, pinWidth).name = 'bucket-main-pin';
  const linkPin = anchor(orientation, 'bucket-link-pin', -.08 * S, .12 * S, 0);
  pin(linkPin, materials.metal, 0, 0, 0, S * .032, pinWidth);
  const soil = mesh(orientation, lumpGeometry(1), materials.soil, .005 * S, -.245 * S, 0); soil.name = 'bucket-soil'; soil.scale.set(S * .225, S * .125, width * .405); soil.visible = false;
  return { curl, orientation, soil, linkPin };
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
  carry: { boom: .63, stick: -1.35, pitch: .04 },
  reach: { boom: .30, stick: -1.05, pitch: -.42 },
  scoop: { boom: .20, stick: -1.30, pitch: .62 },
  raise: { boom: .80, stick: -1.02, pitch: .10 },
  pour: { boom: .74, stick: -.98, pitch: .94 },
};
const ARM_KEYS = {
  dig: [[0, 'carry'], [.3, 'reach'], [.56, 'scoop'], [1, 'carry', backOut]],
  dump: [[0, 'carry'], [.34, 'raise'], [.62, 'pour'], [1, 'carry', backOut]],
};

export function makeMachine(agent, tile, { labels = true, style = 'diorama' } = {}) {
  const root = new THREE.Group(), L = agent.height * tile, W = agent.width * tile, S = Math.min(L, W);
  const livery = LIVERIES[agent.type], wheeled = agent.action_type === 1 || agent.type === 1;
  root.name = `machine-${agent.id}`;
  const chassis = undercarriage(root, L, W, S, wheeled, livery, agent.type === 1);
  const suspension = new THREE.Group(); suspension.name = 'suspension'; root.add(suspension);
  const upper = new THREE.Group(); upper.position.y = .32 * S; suspension.add(upper);
  const body = paint(livery.body), accent = paint(livery.accent), trim = paint(livery.trim);
  const beaconMaterial = new THREE.MeshStandardMaterial({ color: 0xffa21f, roughness: .3, emissive: 0xff7a00, emissiveIntensity: .2, transparent: true, opacity: .92 });
  let boom, stick, tool, load, bed, loaderArm, bucketRig, exhaust, beacon;
  const hydraulics = [];
  hazardMaterial ||= new THREE.MeshStandardMaterial({ map: hazardTexture(), color: typeof document === 'undefined' ? 0xf4b21b : 0xffffff, roughness: .6 });
  if (agent.type === 0) {
    cylinder(upper, materials.dark, 0, .045 * S, 0, S * .31, S * .10);
    roundedBox(upper, body, -.10 * L, .16 * S, 0, L * .65, S * .23, W * .66, S * .065).name = 'excavator-upper-body';
    roundedBox(upper, materials.dark, -.33 * L, .255 * S, 0, L * .20, S * .20, W * .65, S * .068).name = 'excavator-counterweight';
    box(upper, hazardMaterial, -.434 * L, .255 * S, 0, L * .012, S * .09, W * .56);
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
    for (const side of [-1, 1]) box(boom, materials.lamp, firstLength * .3, S * .1, side * W * .065, S * .04, S * .03, S * .012);
    for (const side of [-1, 1]) {
      const start = anchor(upper, `boom-cylinder-${side}-start`, .20 * L, .12 * S, (.09 + side * .12) * W);
      const end = anchor(boom, `boom-cylinder-${side}-end`, firstLength * .48, -.055 * S, side * W * .12);
      pin(start, body, 0, 0, 0, S * .047, W * .055); pin(end, body, 0, 0, 0, S * .047, W * .055);
      hydraulics.push(hydraulic(upper, start, end, S * .036, `boom-cylinder-${side}`));
    }
    stick = new THREE.Group(); stick.name = 'stick-pivot'; stick.position.x = firstLength; boom.add(stick);
    beam(stick, secondLength, S * .17, W * .09, body, true);
    roundedBox(stick, body, -.07 * secondLength, .045 * S, 0, .22 * secondLength, S * .105, W * .09, S * .035);
    pin(stick, materials.dark, 0, 0, 0, S * .078, W * .16); pin(stick, materials.metal, 0, 0, 0, S * .040, W * .175);
    const stickStart = anchor(boom, 'stick-cylinder-start', firstLength * .40, S * .145, 0);
    const stickEnd = anchor(stick, 'stick-cylinder-end', -.09 * secondLength, S * .080, 0);
    pin(stickStart, accent, 0, 0, 0, S * .048, W * .11); pin(stickEnd, body, 0, 0, 0, S * .043, W * .115);
    hydraulics.push(hydraulic(upper, stickStart, stickEnd, S * .040, 'stick-cylinder'));

    // Keep the bucket compact relative to the cab. Scale its bowl, payload and
    // local linkage together around the unchanged stick-end hinge.
    const bucketScale = .65, bucketSize = S * bucketScale, bucketWidth = W * .39 * bucketScale;
    const hingeWidth = W * .145, bucketPart = excavatorBucket(stick, bucketSize, bucketWidth, hingeWidth, livery);
    tool = bucketPart.curl; tool.position.x = secondLength; load = bucketPart.soil;
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
    pin(bucketCylinderStart, body, 0, 0, 0, bucketSize * .038, W * .115);
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
    const bedPaint = paint(livery.bed);
    box(pivot, bedPaint, .30 * L, 0, 0, L * .64, S * .08, W * .86);
    for (const side of [-1, 1]) {
      const wall = box(pivot, bedPaint, .30 * L, .2 * S, side * W * .41, .66 * L, S * .38, W * .05); wall.rotation.x = side * .08;
      for (let i = 0; i < 4; i++) box(pivot, accent, (.06 + i * .16) * L, .22 * S, side * W * .44, L * .025, S * .34, W * .02);
      box(pivot, accent, .30 * L, .4 * S, side * W * .43, .66 * L, S * .04, W * .07);
    }
    box(pivot, bedPaint, .62 * L, .28 * S, 0, .04 * L, S * .52, W * .86);
    const canopy = box(pivot, bedPaint, .72 * L, .52 * S, 0, .22 * L, S * .04, W * .86); canopy.rotation.z = -.06;
    box(pivot, materials.dark, -.02 * L, .22 * S, 0, .03 * L, S * .3, W * .78);
    load = mesh(pivot, lumpGeometry(2), materials.soil, L * .3, S * .2, 0); load.scale.set(L * .27, S * .2, W * .33);
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
    box(cage, materials.glass, cx * .47, ch * .52, 0, S * .01, ch * .78, cz * .86);
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
    const bucketPart = loaderBucket(loaderArm, S * 1.22, W * .92, livery); tool = bucketPart.root; tool.position.set(.80 * L, -.10 * S, 0); load = bucketPart.soil;
  }
  if (beacon) beacon.name = 'beacon';
  const ringColor = SLOT_COLORS[agent.id % 4], ring = selectionRing(ringColor, style !== 'paper'), paper = style === 'paper';
  if (paper) root.traverse(item => { if (item.name === 'glass-highlight') item.visible = false; });
  ring.scale.set(L * 1.34, W * 1.34 + (L - W) * .35, 1); ring.position.y = tile * .03; root.add(ring);
  let label = null;
  if (labels) { label = nameplate(agent, style); label.position.set(-.05 * L, S * 1.12, 0); root.add(label); }
  const bucketTip = new THREE.Vector3();
  const motion = { last: null, treads: [0, 0], spin: 0, active: false, kind: '', phase: 1, lift: 0, tags: true };
  const wheelRadius = chassis.spinning[0]?.radius ?? S * .2;
  const load0 = load ? load.scale.clone() : null;

  function setPose(state, active, phase = 0, kind = '') {
    upper.rotation.y = state.cabin_yaw;
    for (const wheelPart of chassis.steering) wheelPart.rotation.y = Math.max(-.6, Math.min(.6, state.wheel_angle * Math.PI / 9));
    motion.active = active; motion.kind = kind; motion.phase = phase;
    ring.visible = active && motion.tags;
    const carrying = state.loaded > 0;
    load.visible = carrying || (kind === 'dump' && phase < .52) || (kind === 'transfer' && phase < .52) || (kind === 'dig' && phase > .5);
    if (kind === 'receive') load.visible = phase > .62;
    if (load0 && agent.type !== 0) {
      // Loose payload height saturates with the carried amount; display only.
      const before = kind === 'dump' ? Math.max(1, state.previous_loaded ?? state.loaded) : state.loaded;
      let fill = .55 + .45 * (1 - Math.exp(-Math.max(before, 1) / 18));
      if (kind === 'dump') { fill *= 1 - smooth(segment(phase, .22, .5)); load.visible = phase < .5; }
      load.scale.set(load0.x, load0.y * Math.max(fill, .02), load0.z * (kind === 'dump' ? .7 + .3 * fill : 1));
    }
    if (boom) {
      const keys = kind === 'dig' ? ARM_KEYS.dig : kind === 'dump' || kind === 'transfer' ? ARM_KEYS.dump : null;
      const pose = keys ? blend(ARM, keys, phase) : ARM.carry;
      boom.rotation.z = pose.boom; stick.rotation.z = pose.stick;
      // An inward-facing bucket empties by lowering its inward cutting edge.
      // Compensate the arm pose so carry stays upright and yaw never becomes curl.
      tool.rotation.z = pose.pitch - boom.rotation.z - stick.rotation.z;
      root.updateWorldMatrix(true, true); bucketRig.update();
      for (const actuator of hydraulics) actuator.update();
    }
    if (bed) {
      const tilt = kind === 'dump' ? (phase < .45 ? smooth(segment(phase, 0, .45)) : phase < .7 ? 1 : 1 - smooth(segment(phase, .7, 1))) : 0;
      bed.rotation.z = .62 * tilt;
    }
    if (loaderArm) {
      const lifted = state.shovel_lifted ? .35 : -.2, loadedCurl = carrying ? .22 : 0;
      let arm = lifted, curl = loadedCurl;
      if (kind === 'dig') {
        arm = phase < .35 ? THREE.MathUtils.lerp(-.2, -.3, smooth(segment(phase, 0, .35))) : phase < .6 ? -.3 : THREE.MathUtils.lerp(-.3, lifted, backOut(segment(phase, .6, 1)));
        curl = phase < .35 ? -.15 * smooth(segment(phase, 0, .35)) : phase < .6 ? THREE.MathUtils.lerp(-.15, .3, smooth(segment(phase, .35, .6))) : THREE.MathUtils.lerp(.3, loadedCurl, smooth(segment(phase, .6, 1)));
      } else if (kind === 'dump') {
        arm = phase < .6 ? THREE.MathUtils.lerp(.35, .45, smooth(segment(phase, 0, .35))) : THREE.MathUtils.lerp(.45, lifted, smooth(segment(phase, .6, 1)));
        curl = phase < .3 ? .22 * (1 - segment(phase, 0, .3)) : phase < .65 ? -.8 * smooth(segment(phase, .3, .5)) : THREE.MathUtils.lerp(-.8, loadedCurl, smooth(segment(phase, .65, 1)));
      } else if (kind === 'turn') {
        arm = lifted;
      }
      loaderArm.rotation.z = arm; tool.rotation.z = curl;
    }
  }

  return {
    root, agent, bucketRig, hydraulics, suspension, ringColor,
    setPose,
    /** Show or hide the number tag and the active-machine ring. */
    setTags(visible) { motion.tags = visible; if (label) label.visible = visible; ring.visible = visible && motion.active; },
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
      if (paper) {
        // Figure style: no flashing, pulsing or body sway; machines move rigidly.
        beaconMaterial.emissiveIntensity = .15; ring.material.opacity = active ? .9 : 0; ring.rotation.z = 0;
        if (label) { label.material.opacity = 1; label.position.y = S * 1.12; }
        suspension.position.y = 0; suspension.rotation.z = 0; return;
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
