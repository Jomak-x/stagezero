import React, { forwardRef, useEffect, useMemo } from "react";
import * as THREE from "three";
import { PointCloudMessage } from "./WebsocketMessages";

/** The point-cloud envelope preserves compatibility with ordinary Viser clients.
 * Four float32 triplets: (timeline seconds, intensity, seed), size, RGB, reserved.
 * Only packet time drives animation: pause, seek, and recording are reproducible.
 */
export function cinematicEffectKind(name: string) {
  return /^\/effects\/[^/]+\/cinematic_(explosion|energy_burst)$/.exec(name)?.[1];
}

const vertexShader = /* glsl */ `
  varying vec3 vPosition;
  void main() {
    vPosition = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const noiseGLSL = /* glsl */ `
  // Smooth, continuous 3D value noise. Fixed octave and march counts bound cost.
  float hash(vec3 p) {
    p = fract(p * .3183099 + vec3(.13, .37, .71));
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
  }
  float noise3(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash(i), hash(i + vec3(1,0,0)), f.x),
                   mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
               mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
                   mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z);
  }
  float fbm(vec3 p) {
    float n = .57 * noise3(p);
    p = p * 2.07 + vec3(7.1, 3.8, 1.3);
    n += .28 * noise3(p);
    return n + .15 * noise3(p * 2.03 + 11.7);
  }
`;

const volumeFragment = /* glsl */ `
  uniform vec3 uCamera;
  uniform mat4 uLocalToClip;
  uniform vec3 uColor;
  uniform float uTime;
  uniform float uIntensity;
  uniform float uSeed;
  uniform float uEnergy;
  varying vec3 vPosition;
  ${noiseGLSL}
  void main() {
    if (uIntensity <= 0.0) discard;
    vec3 rd = normalize(vPosition - uCamera);
    // Ray / local box intersection also works when the camera is inside.
    vec3 inv = 1.0 / (rd + vec3(0.000001));
    vec3 ta = (-vec3(1.0) - uCamera) * inv;
    vec3 tb = ( vec3(1.0) - uCamera) * inv;
    vec3 lo = min(ta, tb), hi = max(ta, tb);
    float nearT = max(0.0, max(lo.x, max(lo.y, lo.z)));
    float farT = min(hi.x, min(hi.y, hi.z));
    if (nearT >= farT) discard;
    float t = mod(max(uTime, 0.0), 6.0);
    float birth = smoothstep(0.0, .11, t);
    float death = 1.0 - smoothstep(3.7, 5.8, t);
    float radius = .10 + .55 * (1.0 - exp(-t * 3.0));
    float heatLife = exp(-max(0.0, t - .42) * .48);
    float stepSize = (farT - nearT) / 64.0;
    // Stable per-pixel dithering eliminates banding without temporal shimmer.
    float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898,78.233))) * 43758.5453);
    vec3 sum = vec3(0.0);
    float alpha = 0.0;
    float depthT = farT;
    bool foundDepth = false;
    vec3 seedOffset = vec3(uSeed * .073, uSeed * .113, uSeed * .037);
    for (int i = 0; i < 64; i++) {
      vec3 p = uCamera + rd * (nearT + (float(i) + jitter) * stepSize);
      float edge = 1.0 - smoothstep(.83, 1.0, max(abs(p.x), max(abs(p.y), abs(p.z))));
      vec3 q = p;
      q.y -= -.50 + .17 * min(t, 4.0);
      vec3 flow = q * 6.5 + seedOffset - vec3(0.0, t * 1.6, 0.0);
      float swirl = atan(q.z, q.x) + t * .42;
      flow.xz += .48 * vec2(sin(swirl * 3.0 + q.y * 5.0), cos(swirl * 3.0 - q.y * 4.0));
      float n = fbm(flow);
      float fine = noise3(flow * 3.7 - vec3(0.0, t * 1.4, 0.0));
      float dist = length(q * vec3(1.0, 1.12, 1.0));
      // Overlapping asymmetric lobes create a rolling combustion silhouette,
      // rather than a noisy sphere. Erosion reveals crisp cauliflower folds.
      float grow = 1.0 - exp(-t * 3.0);
      float shape = .40 * grow - length(q * vec3(1.0, 1.25, 1.0));
      shape = max(shape, .33 * grow - length(q - vec3(-.31, .14, .02) * grow));
      shape = max(shape, .35 * grow - length(q - vec3(.24, .26, -.10) * grow));
      shape = max(shape, .27 * grow - length(q - vec3(-.06, .40, .14) * grow));
      shape = max(shape, .32 * grow - length(q - vec3(.08, -.09, .32) * grow));
      shape = max(shape, .29 * grow - length(q - vec3(-.16, .08, -.34) * grow));
      shape = max(shape, .24 * grow - length(q - vec3(.38, -.02, .19) * grow));
      float body = (shape + (n - .48) * .34 + (fine - .5) * .065) * 18.0;
      vec3 stem = p - vec3(0.0, -.61 + .10 * t, 0.0);
      float column = (.15 + .08 * n - length(stem.xz)) * 10.0;
      column *= 1.0 - smoothstep(.16, .60, abs(stem.y));
      float density = max(0.0, max(body, column * .45)) * edge * birth * death;
      vec3 radiance;
      float emission;
      if (uEnergy > .5) {
        float r = length(p.xz);
        float angle = atan(p.z, p.x);
        float waveR = .12 + .68 * (1.0 - exp(-t * 1.5));
        float twist = angle * 4.0 + p.y * 8.0 - t * 3.0;
        float filament = pow(max(0.0, .5 + .5 * sin(twist + n * 5.0)), 12.0);
        float torus = length(vec2(r - waveR, p.y * 1.8));
        float ring = exp(-torus * torus * 140.0) * (0.35 + 1.8 * filament);
        float funnel = exp(-pow(r - (.16 + .36 * abs(p.y)), 2.0) * 160.0);
        funnel *= (1.0 - smoothstep(.35, .87, abs(p.y))) * filament;
        float core = exp(-dot(p,p) * (35.0 + t * 8.0));
        density = (ring * 1.8 + funnel * 1.8 + core * 3.0) * edge * birth * death;
        radiance = mix(uColor * .8, vec3(.85,.96,1.0), clamp(core * .85 + filament * .035, 0.0, 1.0));
        emission = 1.8;
      } else {
        // Independent turbulent temperature pockets leave opaque soot folds
        // between red-hot fissures. Only tiny fresh pockets approach white.
        float thermal = clamp((n - .30) * 2.6 + (fine - .5) * .65 - dist * .48, 0.0, 1.0);
        float hot = thermal * heatLife;
        float litFold = clamp(n * 1.4 + q.y * .22, 0.0, 1.0);
        vec3 soot = mix(vec3(.008,.009,.012), vec3(.085,.092,.105), litFold);
        soot += vec3(.13,.025,.002) * heatLife * thermal;
        vec3 fire = mix(vec3(.24,.003,.0003), uColor * vec3(1.0,.22,.08), smoothstep(.12,.35,hot));
        fire = mix(fire, vec3(1.0,.16,.003), smoothstep(.30,.54,hot));
        fire = mix(fire, vec3(1.0,.53,.035), smoothstep(.56,.80,hot));
        fire = mix(fire, vec3(1.0,.91,.57), smoothstep(.85,1.0,hot));
        float burning = smoothstep(.19,.37,hot);
        radiance = mix(soot, fire, burning);
        emission = 1.0 + smoothstep(.55,.98,hot) * 1.0;
      }
      float a = 1.0 - exp(-density * stepSize * 8.0 * min(uIntensity, 3.0));
      sum += (1.0 - alpha) * a * radiance * emission;
      alpha += (1.0 - alpha) * a;
      if (!foundDepth && alpha > .12) {
        depthT = nearT + (float(i) + jitter) * stepSize;
        foundDepth = true;
      }
      if (alpha > .985) break;
    }
    if (alpha < .003) discard;
    vec4 clip = uLocalToClip * vec4(uCamera + rd * depthT, 1.0);
    gl_FragDepth = clamp(clip.z / clip.w * .5 + .5, 0.0, 1.0);
    gl_FragColor = vec4(sum / max(alpha, .001), alpha);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }
`;

const sparkVertex = /* glsl */ `
  attribute vec3 aRandom;
  uniform float uTime;
  uniform float uSeed;
  uniform float uIntensity;
  uniform float uEnergy;
  uniform float uPixelRatio;
  varying float vAlpha;
  varying float vHeat;
  void main() {
    float t = mod(max(uTime, 0.0), 6.0);
    float delay = aRandom.z * .28;
    float age = max(0.0, t - delay);
    float angle = aRandom.x * 6.2831853 + uSeed;
    float speed = .35 + aRandom.y * .65;
    vec3 p;
    if (uEnergy > .5) {
      float r = (.12 + age * speed * .35);
      angle += age * (1.4 + aRandom.z);
      p = vec3(cos(angle) * r, (aRandom.z - .5) * sin(age * 1.2) * 1.5, sin(angle) * r);
    } else {
      float travel = (1.0 - exp(-age * .9)) * speed * 1.5;
      p = vec3(cos(angle) * travel, -.60 + age * (.60 + aRandom.z * .7) - age * age * .33, sin(angle) * travel);
    }
    float life = 1.0 - smoothstep(.6 + aRandom.y * .5, 1.8 + aRandom.y * 1.6, age);
    vAlpha = life * smoothstep(0.0, .035, t - delay) * min(uIntensity, 1.5);
    vHeat = 1.0 - age * .3;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = clamp((3.0 + aRandom.z * 5.0) * uPixelRatio / max(.6, -mv.z * .18), 1.0, 24.0);
  }
`;
const sparkFragment = /* glsl */ `
  uniform vec3 uColor;
  varying float vAlpha;
  varying float vHeat;
  void main() {
    vec2 p = gl_PointCoord - .5;
    float r = length(p);
    float a = exp(-r * r * 28.0) * (1.0 - smoothstep(.35,.5,r)) * vAlpha;
    if(a < .01) discard;
    gl_FragColor = vec4(mix(uColor, vec3(1.0,.94,.74), clamp(vHeat,0.0,1.0)) * 2.5, a);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }
`;

const waveFragment = /* glsl */ `
  uniform float uTime;
  uniform float uIntensity;
  uniform float uEnergy;
  uniform vec3 uColor;
  varying vec3 vPosition;
  void main() {
    float t = mod(max(uTime, 0.0), 6.0);
    float r = length(vPosition.xy);
    float radius = .03 + min(t * 1.2, 1.8);
    float ring = exp(-pow((r - radius) / (.014 + t * .027), 2.0));
    float trail = exp(-abs(r - radius) * 17.0) * .15;
    float fade = smoothstep(0.0,.05,t) * (1.0 - smoothstep(.25,1.4,t));
    float a = (ring + trail) * fade * min(uIntensity, 2.0);
    if (a < .005) discard;
    gl_FragColor = vec4(mix(uColor,vec3(1.0),.45) * 2.0, a * .65);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }
`;

export const CinematicEffects = forwardRef<THREE.Group, PointCloudMessage & {
  children?: React.ReactNode;
}>(({ name, props, children }, ref) => {
  const packet = useMemo(() => {
    if (props.precision !== "float32" || props.points.byteLength !== 48) return null;
    const data = new DataView(props.points.buffer, props.points.byteOffset, props.points.byteLength);
    const values = Array.from({ length: 12 }, (_, i) => data.getFloat32(i * 4, true));
    if (!values.every(Number.isFinite)) return null;
    return values;
  }, [props.points, props.precision]);
  const energy = cinematicEffectKind(name) === "energy_burst";
  const resources = useMemo(() => {
    const uniforms = {
      uCamera: { value: new THREE.Vector3() },
      uLocalToClip: { value: new THREE.Matrix4() },
      uColor: { value: new THREE.Color() },
      uTime: { value: 0 },
      uIntensity: { value: 0 },
      uSeed: { value: 0 },
      uEnergy: { value: energy ? 1 : 0 },
      uPixelRatio: { value: 1 },
    };
    const volume = new THREE.ShaderMaterial({
      uniforms, vertexShader, fragmentShader: volumeFragment,
      transparent: true, side: THREE.BackSide, depthWrite: !energy,
    });
    const sparks = new THREE.ShaderMaterial({
      uniforms, vertexShader: sparkVertex, fragmentShader: sparkFragment,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    const wave = new THREE.ShaderMaterial({
      uniforms, vertexShader, fragmentShader: waveFragment,
      transparent: true, depthWrite: false, side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
    });
    const geometry = new THREE.BufferGeometry();
    const count = 220;
    const random = new Float32Array(count * 3);
    // A deterministic low-discrepancy distribution, independent of mount order.
    for (let i = 0; i < count; i++) {
      random[i * 3] = (i * .61803398875) % 1;
      random[i * 3 + 1] = (i * .754877666 + .17) % 1;
      random[i * 3 + 2] = (i * .569840291 + .43) % 1;
    }
    geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(count * 3), 3));
    geometry.setAttribute("aRandom", new THREE.BufferAttribute(random, 3));
    return { uniforms, volume, sparks, wave, geometry };
  }, [energy]);
  useEffect(() => () => {
    resources.volume.dispose();
    resources.sparks.dispose();
    resources.wave.dispose();
    resources.geometry.dispose();
  }, [resources]);
  const { uniforms } = resources;
  const values = packet ?? [0, 0, 0, 1, 1, 1, 1, .3, .04];
  uniforms.uTime.value = values[0];
  uniforms.uIntensity.value = Math.max(0, Math.min(values[1], 5));
  uniforms.uSeed.value = values[2] % 10000;
  uniforms.uColor.value.setRGB(
    THREE.MathUtils.clamp(values[6], 0, 1),
    THREE.MathUtils.clamp(values[7], 0, 1),
    THREE.MathUtils.clamp(values[8], 0, 1),
    THREE.SRGBColorSpace,
  );
  const scale = values.slice(3, 6).map((v) => Math.max(.001, Math.abs(v)) * .5) as [number, number, number];
  const phase = Math.max(0, values[0]) % 6;
  const flash = (1 - Math.exp(-phase * 30)) * Math.exp(-phase * (energy ? .9 : 2.2));
  return <group ref={ref}>
    <pointLight color={uniforms.uColor.value} intensity={values[1] * flash * 32} distance={Math.max(...scale) * 7} decay={2} />
    <group scale={scale} visible={packet !== null && values[1] > 0}>
      <mesh
        material={resources.volume}
        frustumCulled={false}
        ref={(mesh) => {
          if (mesh) mesh.onBeforeRender = (renderer, _scene, camera) => {
            mesh.worldToLocal(camera.getWorldPosition(uniforms.uCamera.value));
            uniforms.uPixelRatio.value = renderer.getPixelRatio();
            uniforms.uLocalToClip.value.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).multiply(mesh.matrixWorld);
          };
        }}
      >
        <boxGeometry args={[2, 2, 2]} />
      </mesh>
      <mesh position={[0, energy ? 0 : -.83, 0]} rotation={[-Math.PI / 2, 0, 0]} material={resources.wave}>
        <planeGeometry args={[4, 4]} />
      </mesh>
      <points geometry={resources.geometry} material={resources.sparks} frustumCulled={false} />
    </group>
    {children}
  </group>;
});
CinematicEffects.displayName = "CinematicEffects";
