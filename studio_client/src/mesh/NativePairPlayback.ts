/** Preloaded exact segment-affine native animation. Socket commands control time. */
import type { NativePairClipMessage, NativePairTransportMessage } from "../WebsocketMessages";
import * as THREE from "three";
const BONES = 22;
export type NativePairSample = { frame: number; first: number; second: number; alpha: number; revision: number; playing: boolean; capturing: boolean };
export type NativePairPart = { name: string; vertexCount: number; localBind: Float32Array; bones: Uint16Array; weights: Float32Array; linear: Float32Array; targets: Float32Array; frames: number };
type PreparedClip = { revision: number; fps: number; frames: number; parts: Map<string, NativePairPart>; bytes: number };
function integer(value: number, min: number, max: number, label: string) {
  if (!Number.isSafeInteger(value) || value < min || value > max) throw new Error(`Invalid native pair ${label}`);
}
function prepareClip(message: NativePairClipMessage): PreparedClip {
  integer(message.revision, 0, Number.MAX_SAFE_INTEGER, "revision");
  integer(message.frames, 1, 1000, "frame count");
  if (!Number.isFinite(message.fps) || message.fps <= 0 || message.fps > 120) throw new Error("Invalid native pair fps");
  if (!Array.isArray(message.actors) || message.actors.length < 1 || message.actors.length > 8) throw new Error("Invalid native pair cast");
  let bytes = 0, totalVertices = 0;
  const decode = (value: Uint8Array, count: number, uint16 = false) => {
    if (!(value instanceof Uint8Array) || value.byteLength !== count * (uint16 ? 2 : 4)) throw new Error("Invalid native pair array length");
    bytes += value.byteLength;
    if (bytes > 64 * 1024 * 1024) throw new Error("Native pair exceeds memory limit");
    const copy = new Uint8Array(value).buffer;
    const array = uint16 ? new Uint16Array(copy) : new Float32Array(copy);
    if (!uint16 && !array.every(Number.isFinite)) throw new Error("Nonfinite native pair transform");
    return array;
  };
  const parts = new Map<string, NativePairPart>();
  for (const actor of message.actors) {
    const rest = decode(actor.rest, BONES * 3) as Float32Array;
    const linear = decode(actor.linear, message.frames * BONES * 9) as Float32Array;
    const targets = decode(actor.targets, message.frames * BONES * 3) as Float32Array;
    if (!Array.isArray(actor.parts) || actor.parts.length < 1 || actor.parts.length > 64) throw new Error("Invalid native pair mesh count");
    for (const part of actor.parts) {
      if (typeof part.name !== "string" || !part.name.startsWith("/") || part.name.length > 512 || parts.has(part.name)) throw new Error("Invalid native pair mesh name");
      if (!(part.weights instanceof Uint8Array)) throw new Error("Missing native pair weights");
      const vertexCount = part.weights.byteLength / 16;
      integer(vertexCount, 1, 500000, "vertex count");
      totalVertices += vertexCount;
      if (totalVertices > 500000) throw new Error("Native pair exceeds vertex limit");
      const weights = decode(part.weights, vertexCount * 4) as Float32Array;
      const bones = decode(part.bones, vertexCount * 4, true) as Uint16Array;
      const localBind = decode(part.bind_world, vertexCount * 12) as Float32Array;
      for (let vertex = 0; vertex < vertexCount; vertex++) {
        let total = 0;
        for (let influence = 0; influence < 4; influence++) {
          const slot = vertex * 4 + influence, bone = bones[slot];
          if (bone >= BONES || weights[slot] < 0 || weights[slot] > 1) throw new Error("Invalid native pair skin influence");
          total += weights[slot];
          for (let axis = 0; axis < 3; axis++) localBind[slot * 3 + axis] -= rest[bone * 3 + axis];
        }
        if (Math.abs(total - 1) > 0.0001) throw new Error("Native pair weights must sum to one");
      }
      parts.set(part.name, { name: part.name, vertexCount, weights, bones, localBind, linear, targets, frames: message.frames });
    }
  }
  return { revision: message.revision, fps: message.fps, frames: message.frames, parts, bytes };
}
/** Matches NativeRigAsset.skin at source frames, including affine limb scaling. */
export function skinNativePairFrame(part: NativePairPart, frame: number, output: Float32Array) {
  integer(frame, 0, part.frames - 1, "sample frame");
  if (output.length !== part.vertexCount * 3) throw new Error("Native pair output size mismatch");
  const ls = frame * BONES * 9, ts = frame * BONES * 3;
  for (let vertex = 0; vertex < part.vertexCount; vertex++) {
    let x = 0, y = 0, z = 0;
    for (let influence = 0; influence < 4; influence++) {
      const slot = vertex * 4 + influence, weight = part.weights[slot];
      if (weight === 0) continue;
      const bone = part.bones[slot], mi = ls + bone * 9, ti = ts + bone * 3, pi = slot * 3;
      const px = part.localBind[pi], py = part.localBind[pi + 1], pz = part.localBind[pi + 2];
      const m = part.linear, t = part.targets;
      x += weight * (m[mi] * px + m[mi + 1] * py + m[mi + 2] * pz + t[ti]);
      y += weight * (m[mi + 3] * px + m[mi + 4] * py + m[mi + 5] * pz + t[ti + 1]);
      z += weight * (m[mi + 6] * px + m[mi + 7] * py + m[mi + 8] * pz + t[ti + 2]);
    }
    output[vertex * 3] = x; output[vertex * 3 + 1] = y; output[vertex * 3 + 2] = z;
  }
}
export class NativePairPlayback {
  clip: PreparedClip | undefined;
  sample: NativePairSample | undefined;
  private transport: NativePairTransportMessage | undefined;
  private pending: NativePairTransportMessage | undefined;
  private anchorMs = 0;
  private latestRevision = -1;
  private latestSequence = -1;
  commandDifferenceFrames = 0;
  load(message: NativePairClipMessage, now = performance.now()) {
    if (message.revision <= this.latestRevision) return false;
    const clip = prepareClip(message);
    this.clip = clip; this.latestRevision = clip.revision;
    this.transport = undefined; this.latestSequence = -1; this.sample = undefined;
    if (this.pending?.revision === clip.revision) {
      const pending = this.pending; this.pending = undefined; this.command(pending, now);
    } else if (this.pending && this.pending.revision < clip.revision) this.pending = undefined;
    return true;
  }
  command(message: NativePairTransportMessage, now = performance.now()) {
    integer(message.revision, 0, Number.MAX_SAFE_INTEGER, "transport revision");
    integer(message.sequence, 0, Number.MAX_SAFE_INTEGER, "sequence");
    if (!Number.isFinite(message.frame) || message.frame < 0 || typeof message.playing !== "boolean" || typeof message.enabled !== "boolean" || (message.capturing !== undefined && typeof message.capturing !== "boolean")) throw new Error("Invalid native pair transport");
    if (message.revision < this.latestRevision) return false;
    if (!this.clip || message.revision > this.clip.revision) {
      if (!this.pending || message.revision > this.pending.revision || (message.revision === this.pending.revision && message.sequence > this.pending.sequence)) this.pending = { ...message };
      return false;
    }
    if (message.sequence <= this.latestSequence) return false;
    if (message.frame > this.clip.frames - 1 || (message.capturing && !Number.isInteger(message.frame))) throw new Error("Native pair transport frame outside clip");
    const prior = this.advance(now);
    this.commandDifferenceFrames = prior ? message.frame - prior.frame : 0;
    this.transport = { ...message }; this.latestSequence = message.sequence; this.anchorMs = now;
    this.advance(now); return true;
  }
  advance(now = performance.now()) {
    const clip = this.clip, transport = this.transport;
    if (!clip || !transport || !transport.enabled) return this.sample = undefined;
    const playing = transport.playing && !transport.capturing;
    const elapsed = playing ? Math.max(0, now - this.anchorMs) * clip.fps / 1000 : 0;
    const frame = Math.min(clip.frames - 1, transport.frame + elapsed), first = Math.floor(frame);
    return this.sample = { revision: clip.revision, frame, first, second: transport.capturing ? first : Math.min(first + 1, clip.frames - 1), alpha: transport.capturing ? 0 : frame - first, playing: playing && frame < clip.frames - 1, capturing: !!transport.capturing };
  }
  remove(name: string) {
    if (!this.clip) return;
    for (const key of this.clip.parts.keys()) if (key === name || key.startsWith(name + "/")) this.clip.parts.delete(key);
    if (!this.clip.parts.size) { this.clip = undefined; this.sample = undefined; this.transport = undefined; }
  }
  reset() {
    this.clip = undefined; this.sample = undefined; this.transport = undefined; this.pending = undefined;
    this.latestRevision = this.latestSequence = -1; this.commandDifferenceFrames = 0;
  }
}
export const nativePairPlayback = new NativePairPlayback();
const renderers = new Set<() => void>();
export function registerNativePairRenderer(render: () => void) { renderers.add(render); return () => { renderers.delete(render); }; }
/** Capture calls this immediately before its offscreen render. */
export function renderNativePairNow() { renderers.forEach((render) => render()); }
let lastTick = 0, lastReport = 0, reportFrames = 0, droppedFrames = 0, runtimeMs = 0;
export function recordNativePairRenderCost(ms: number) { runtimeMs += ms; }
export function advanceNativePairPlayback(now = performance.now()) {
  const sample = nativePairPlayback.advance(now);
  if (sample?.playing && lastTick && now - lastTick > 50) droppedFrames += Math.max(0, Math.round((now - lastTick) / (1000 / 60)) - 1);
  lastTick = now; reportFrames++;
  if (typeof document !== "undefined" && now - lastReport >= 500) {
    document.documentElement.dataset.nativePairPlayback = JSON.stringify({ mode: sample ? "local-affine" : "inactive", revision: nativePairPlayback.clip?.revision, frame: sample?.frame, playing: sample?.playing ?? false, capturing: sample?.capturing ?? false, renderFps: lastReport ? Math.round(reportFrames * 1000 / (now - lastReport)) : null, droppedFrames, commandDifferenceFrames: nativePairPlayback.commandDifferenceFrames, runtimeMsPerFrame: runtimeMs / reportFrames, payloadBytes: nativePairPlayback.clip?.bytes ?? 0 });
    reportFrames = 0; runtimeMs = 0; lastReport = now;
  }
}
export function resetNativePairPlayback() {
  nativePairPlayback.reset(); lastTick = lastReport = reportFrames = droppedFrames = runtimeMs = 0;
  if (typeof document !== "undefined") delete document.documentElement.dataset.nativePairPlayback;
}

/** Skin each source endpoint once; blend positions/normals without reallocating geometry. */
export class NativePairGeometry {
  private part: NativePairPart | undefined;
  private previousFrame = -1;
  private endpoints = new Map<number, { positions: Float32Array; normals: Float32Array }>();
  private scratch = new THREE.BufferGeometry();
  private geometry: THREE.BufferGeometry;
  constructor(geometry: THREE.BufferGeometry) { this.geometry = geometry; }
  update(part: NativePairPart, sample: NativePairSample) {
    if (this.part !== part) {
      this.part = part; this.previousFrame = -1; this.endpoints.clear();
      this.scratch.setIndex(this.geometry.index);
    }
    if (sample.frame === this.previousFrame) return;
    const positions = this.geometry.getAttribute("position") as THREE.BufferAttribute;
    if (positions.count !== part.vertexCount) throw new Error("Native pair geometry vertex count mismatch");
    const endpoint = (frame: number) => {
      let value = this.endpoints.get(frame);
      if (value) return value;
      const points = new Float32Array(part.vertexCount * 3);
      skinNativePairFrame(part, frame, points);
      this.scratch.setAttribute("position", new THREE.BufferAttribute(points, 3));
      this.scratch.computeVertexNormals();
      value = { positions: points, normals: new Float32Array(this.scratch.getAttribute("normal").array) };
      this.endpoints.set(frame, value);
      return value;
    };
    const a = endpoint(sample.first), b = endpoint(sample.second);
    for (const frame of this.endpoints.keys()) if (frame !== sample.first && frame !== sample.second) this.endpoints.delete(frame);
    const normals = this.geometry.getAttribute("normal") as THREE.BufferAttribute;
    const positionArray = positions.array, normalArray = normals.array, alpha = sample.alpha;
    for (let i = 0; i < positionArray.length; i++) {
      positionArray[i] = a.positions[i] + (b.positions[i] - a.positions[i]) * alpha;
      normalArray[i] = a.normals[i] + (b.normals[i] - a.normals[i]) * alpha;
    }
    positions.needsUpdate = normals.needsUpdate = true;
    this.geometry.computeBoundingSphere();
    this.previousFrame = sample.frame;
  }
  dispose() { this.endpoints.clear(); this.scratch.dispose(); }
}
