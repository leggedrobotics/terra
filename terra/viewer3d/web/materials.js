import * as THREE from 'three';

// Stylized, procedural surface shading shared by the terrain, island and
// overlays. Everything is computed in world space, so neighboring cells and
// separate meshes continue the same strata and mottling without seams.
export const PALETTE = {
  name: 'diorama',
  sand: 0xd6ae76, dug: [0xc28a58, 0xad7248, 0x96603d, 0x7f5134], loose: 0xa86c3a,
  strata: [0xc6905c, 0xae7549, 0xd0a26c, 0x9c6a45], rock: 0x8e867d,
  grass: [0x8cc152, 0x74ad45], grassEdge: 0x5f933c,
  sky: [0x8cc6ea, 0xcfe5ef, 0xf7e7cc],
};
// Muted earth tones for figures: neutral tone mapping keeps these close to print.
export const PALETTES = {
  diorama: PALETTE,
  paper: { ...PALETTE, name: 'paper', sand: 0xd8cdb9, dug: [0xc4ad8d, 0xae9575, 0x977e60, 0x80684e], loose: 0xab8a64, strata: [0xc8b494, 0xb29c7c, 0xd3c3a6, 0xa38d6e], rock: 0x9b968e },
};

export const shared = {
  uTime: { value: 0 }, uUnit: { value: .27 }, uTile: { value: .57 }, uFloor: { value: -1 }, uMotion: { value: 1 },
};

const NOISE = /* glsl */`
varying vec3 vTWorld;
varying vec3 vTNormal;
float tHash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float tNoise(vec2 p) {
  vec2 i = floor(p), f = fract(p); f = f * f * (3. - 2. * f);
  return mix(mix(tHash(i), tHash(i + vec2(1, 0)), f.x), mix(tHash(i + vec2(0, 1)), tHash(i + vec2(1, 1)), f.x), f.y);
}
float tFbm(vec2 p) { return tNoise(p) * .55 + tNoise(p * 2.13 + 7.1) * .3 + tNoise(p * 4.7 + 3.3) * .15; }
`;
const WORLD = /* glsl */`
vec4 tWorld = vec4(transformed, 1.);
vec3 tNormal = objectNormal;
#ifdef USE_INSTANCING
tWorld = instanceMatrix * tWorld; tNormal = mat3(instanceMatrix) * tNormal;
#endif
tWorld = modelMatrix * tWorld; vTWorld = tWorld.xyz; vTNormal = normalize(mat3(modelMatrix) * tNormal);
`;

const color = hex => new THREE.Color(hex);
const vec3 = c => `vec3(${c.r.toFixed(4)}, ${c.g.toFixed(4)}, ${c.b.toFixed(4)})`;

function strataCode(lip, palette) {
  const [a, b, c, d] = palette.strata.map(hex => vec3(color(hex)));
  return /* glsl */`
  {
    float depth = -vTWorld.y / uUnit;
    float along = vTWorld.x * .83 + vTWorld.z * 1.17;
    float wobble = (tNoise(vec2(along * 1.4, depth * .35)) - .5) * .32;
    float band = depth + wobble, index = mod(floor(band), 4.), phase = fract(band);
    vec3 stratum = index < 1. ? ${a} : index < 2. ? ${b} : index < 3. ? ${c} : ${d};
    stratum *= .93 + tNoise(vec2(along * 5.1, vTWorld.y * 9.)) * .12;
    stratum *= mix(.8, 1., smoothstep(0., .1, phase));
    stratum *= 1. - clamp(depth * .025, 0., .28);
    if (vTWorld.y < uFloor) stratum = ${vec3(color(palette.rock))} * (.86 + tNoise(vec2(along * 2.3, vTWorld.y * 2.7)) * .2);
    ${lip ? `if (vTWorld.y > -uTile * .2) stratum = ${vec3(color(palette.grassEdge))} * (.92 + tNoise(vec2(along * 3., 1.)) * .14);` : ''}
    diffuseColor.rgb = stratum;
  }`;
}

/** Patch a lit material with world-space soil shading.
 * mode 'soil': mottled cell tops (instance color) and banded cut walls.
 * mode 'island': grass top and banded outer walls with a turf lip.
 * mode 'pile': loose-soil mounds, lighter toward their crests.
 */
export function earthMaterial(mode, parameters = {}, palette = PALETTE) {
  const material = new THREE.MeshStandardMaterial({ roughness: 1, metalness: 0, ...parameters });
  const [grassA, grassB] = palette.grass.map(hex => vec3(color(hex)));
  const top = mode === 'island'
    ? `float g = smoothstep(.3, .72, tFbm(vTWorld.xz * .28)); diffuseColor.rgb = mix(${grassA}, ${grassB}, g) * (.94 + tNoise(vTWorld.xz * 3.1) * .1);`
    : mode === 'pile'
      ? `diffuseColor.rgb *= (.84 + .2 * smoothstep(0., 5., vTWorld.y / uUnit)) * (.92 + tFbm(vTWorld.xz * 2.6) * .16);`
      : `diffuseColor.rgb *= .9 + tFbm(vTWorld.xz * 1.15) * .2;`;
  material.onBeforeCompile = shader => {
    Object.assign(shader.uniforms, shared);
    shader.vertexShader = `varying vec3 vTWorld;\nvarying vec3 vTNormal;\n${shader.vertexShader}`
      .replace('#include <begin_vertex>', `#include <begin_vertex>\n${WORLD}`);
    shader.fragmentShader = `uniform float uUnit;\nuniform float uTile;\nuniform float uFloor;\n${NOISE}\n${shader.fragmentShader}`
      .replace('#include <color_fragment>', `#include <color_fragment>
      {
        vec3 tn = normalize(vTNormal);
        if (tn.y > .5) { ${top} }
        else if (tn.y > -.5) { ${mode === 'pile' ? top : strataCode(mode === 'island', palette)} }
      }`);
  };
  material.customProgramCacheKey = () => `terra-earth-${mode}-${palette.name}`;
  return material;
}

// Hatch/dots/solid patterns keep zones distinguishable without relying only on hue.
export const PATTERNS = { hatch: 0, dots: 1, solid: 2, cross: 3, stripes: 4 };
export function zoneMaterial({ color: hex, opacity, pattern = 'solid', ...rest }) {
  const material = new THREE.MeshBasicMaterial({ color: hex, transparent: true, opacity, depthWrite: false, ...rest });
  const kind = PATTERNS[pattern];
  material.onBeforeCompile = shader => {
    Object.assign(shader.uniforms, shared);
    shader.vertexShader = `varying vec3 vTWorld;\nvarying vec3 vTNormal;\n${shader.vertexShader}`
      .replace('#include <begin_vertex>', `#include <begin_vertex>\nvec3 objectNormal = vec3(0., 1., 0.);\n${WORLD}`);
    shader.fragmentShader = `uniform float uTile;\nuniform float uTime;\nuniform float uMotion;\n${NOISE}\n${shader.fragmentShader}`
      .replace('#include <color_fragment>', `#include <color_fragment>
      {
        vec2 p = vTWorld.xz / uTile;
        float a = 1.;
        ${kind === 0 ? 'a = mix(.42, 1., step(.5, fract((p.x + p.y) * .7 - uTime * .12 * uMotion)));' : ''}
        ${kind === 1 ? 'vec2 q = fract(p * 1.5) - .5; a = mix(.5, 1., 1. - smoothstep(.2, .26, length(q)));' : ''}
        ${kind === 3 ? 'a = mix(.35, 1., max(step(.72, fract((p.x + p.y) * .7)), step(.72, fract((p.x - p.y) * .7))));' : ''}
        ${kind === 4 ? 'a = mix(.3, 1., step(.62, fract((p.x - p.y) * .55)));' : ''}
        diffuseColor.a *= a;
      }`);
  };
  material.customProgramCacheKey = () => `terra-zone-${kind}`;
  return material;
}

/** Vertical sky gradient as a screen-space background texture. */
export function skyTexture() {
  const canvas = document.createElement('canvas'); canvas.width = 4; canvas.height = 256;
  const ctx = canvas.getContext('2d'), gradient = ctx.createLinearGradient(0, 0, 0, 256);
  const [top, middle, bottom] = PALETTE.sky.map(hex => `#${hex.toString(16).padStart(6, '0')}`);
  gradient.addColorStop(0, top); gradient.addColorStop(.58, middle); gradient.addColorStop(1, bottom);
  ctx.fillStyle = gradient; ctx.fillRect(0, 0, 4, 256);
  const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}
