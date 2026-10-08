import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { FXAAPass } from 'three/addons/postprocessing/FXAAPass.js';

// Ambient occlusion that ignores sprites, labels, particles and overlays; the
// normal pass also skips the sky so silhouettes stay clean.
class SceneAOPass extends GTAOPass {
  _overrideVisibility() {
    super._overrideVisibility();
    const cache = this._visibilityCache;
    this.scene.traverse(object => { if ((object.isSprite || object.isLineSegments2 || object.userData.skipAO) && object.visible) { object.visible = false; cache.push(object); } });
  }
  _renderOverride(renderer, ...rest) {
    const background = this.scene.background; this.scene.background = null;
    super._renderOverride(renderer, ...rest);
    this.scene.background = background;
  }
}

// Soft "ink" outlines from depth creases and normal changes. The line color is
// a darker tint of the surface beneath it, never pure black.
const InkShader = {
  uniforms: {
    tDiffuse: { value: null }, tDepth: { value: null }, tNormal: { value: null },
    resolution: { value: new THREE.Vector2(1, 1) }, cameraNear: { value: .1 }, cameraFar: { value: 1000 },
    thickness: { value: 1 }, strength: { value: .92 }, vignette: { value: .16 },
  },
  vertexShader: /* glsl */`varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.); }`,
  fragmentShader: /* glsl */`
    #include <packing>
    uniform sampler2D tDiffuse, tDepth, tNormal;
    uniform vec2 resolution; uniform float cameraNear, cameraFar, thickness, strength, vignette;
    varying vec2 vUv;
    float depthAt(vec2 uv) { float z = texture2D(tDepth, uv).x; return z >= 1. ? 1e6 : -perspectiveDepthToViewZ(z, cameraNear, cameraFar); }
    vec3 normalAt(vec2 uv) { return texture2D(tNormal, uv).xyz * 2. - 1.; }
    void main() {
      // Optional soft vignette (diorama style only).
      vec2 centered = vUv - .5;
      vec4 base = texture2D(tDiffuse, vUv) * vec4(vec3(1. - smoothstep(.35, .95, length(centered * vec2(1.1, 1.))) * vignette), 1.);
      float c = depthAt(vUv);
      if (c > 1e5) { gl_FragColor = base; return; }
      vec2 px = thickness / resolution;
      vec2 ox = vec2(px.x, 0.), oy = vec2(0., px.y);
      float l = depthAt(vUv - ox), r = depthAt(vUv + ox), d = depthAt(vUv - oy), u = depthAt(vUv + oy);
      // Silhouettes: draw on the near side where a neighbor is clearly farther.
      float silhouette = smoothstep(.012, .03, (max(max(l, r), max(d, u)) - c) / c);
      // 1/z is affine across a plane: its Laplacian separates real creases
      // from sub-pixel cracks between coplanar cells, which stay unlined.
      float ic = 1. / c;
      float curvature = (abs(1. / l + 1. / r - 2. * ic) + abs(1. / d + 1. / u - 2. * ic)) / ic;
      vec3 n = normalAt(vUv);
      float bend = max(max(1. - dot(n, normalAt(vUv - ox)), 1. - dot(n, normalAt(vUv + ox))), max(1. - dot(n, normalAt(vUv - oy)), 1. - dot(n, normalAt(vUv + oy))));
      float crease = smoothstep(.3, .7, bend) * smoothstep(.0015, .006, curvature);
      float edge = max(silhouette, crease) * strength;
      gl_FragColor = vec4(mix(base.rgb, base.rgb * vec3(.3, .26, .28), edge), base.a);
    }`,
};

export class PostPipeline {
  constructor(renderer, scene, camera) {
    this.renderer = renderer; this.scene = scene; this.camera = camera;
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    const target = new THREE.WebGLRenderTarget(size.x, size.y, { type: THREE.HalfFloatType, samples: 4 });
    this.composer = new EffectComposer(renderer, target);
    this.composer.addPass(new RenderPass(scene, camera));
    this.ao = new SceneAOPass(scene, camera, size.x, size.y);
    this.ao.blendIntensity = .9;
    this.composer.addPass(this.ao);
    this.ink = new ShaderPass(InkShader);
    this.ink.uniforms.tDepth.value = this.ao.depthTexture; this.ink.uniforms.tNormal.value = this.ao.normalTexture;
    this.composer.addPass(this.ink);
    this.composer.addPass(new OutputPass());
    this.composer.addPass(new FXAAPass());
  }
  configure({ span, tile }) {
    this.ao.updateGtaoMaterial({ radius: Math.max(tile * 1.6, span * .018), distanceExponent: 1.4, thickness: 1.2, scale: 1.05, samples: 16 });
    this.ao.updatePdMaterial({ lumaPhi: 10, depthPhi: 2, normalPhi: 3, radius: 6, rings: 2, samples: 12 });
  }
  setSize(width, height, pixelRatio) {
    this.composer.setPixelRatio(pixelRatio); this.composer.setSize(width, height);
    const size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    this.ink.uniforms.resolution.value.copy(size); this.ink.uniforms.thickness.value = 1.35 * pixelRatio;
  }
  setLook({ vignette = 0 } = {}) { this.ink.uniforms.vignette.value = vignette; }
  setSamples(samples) {
    for (const target of [this.composer.renderTarget1, this.composer.renderTarget2]) if (target.samples !== samples) { target.samples = samples; target.dispose(); }
  }
  render() {
    this.ink.uniforms.cameraNear.value = this.camera.near; this.ink.uniforms.cameraFar.value = this.camera.far;
    this.composer.render();
  }
  dispose() { this.ao.dispose(); this.composer.dispose(); }
}
