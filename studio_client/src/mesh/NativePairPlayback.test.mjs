import test from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";
import { NativePairPlayback, NativePairGeometry, skinNativePairFrame } from "./NativePairPlayback.ts";

function bytes(values, Type = Float32Array) {
  const typed = new Type(values);
  // Deliberately unaligned msgpack-like view.
  const data = new Uint8Array(typed.byteLength + 3);
  data.set(new Uint8Array(typed.buffer), 1);
  return data.subarray(1, typed.byteLength + 1);
}
function fixture(revision = 1) {
  const frames = 4, rest = new Float32Array(66), linear = new Float32Array(frames * 198), targets = new Float32Array(frames * 66);
  rest[0] = 1; rest[3] = 2;
  for (let frame = 0; frame < frames; frame++) {
    for (let bone = 0; bone < 22; bone++) {
      const m = frame * 198 + bone * 9;
      linear[m] = 1 + frame * 0.25;
      linear[m + 1] = frame * 0.1; // Affine shear cannot be reduced to a rigid rotation.
      linear[m + 4] = linear[m + 8] = 1;
      const t = frame * 66 + bone * 3;
      targets[t] = bone * 2 + frame; targets[t + 1] = frame * 0.5;
    }
  }
  return { type: "NativePairClipMessage", revision, fps: 30, frames,
    actors: [{ rest: bytes(rest), linear: bytes(linear), targets: bytes(targets), parts: [{
      name: "/native-pair/a/part-0",
      // Authored inverse bind points differ per influence, intentionally.
      bind_world: bytes([1,0,0, 3,0,0, 0,0,0, 0,0,0, 2,0,0, 4,0,0, 0,0,0, 0,0,0, 1,1,0, 3,1,0, 0,0,0, 0,0,0]),
      bones: bytes([0,1,0,0, 0,1,0,0, 0,1,0,0], Uint16Array),
      weights: bytes([0.25,0.75,0,0, 0.25,0.75,0,0, 0.25,0.75,0,0]),
    }] }] };
}
const command = (overrides = {}) => ({ type: "NativePairTransportMessage", revision: 1, sequence: 1, frame: 0, playing: true, enabled: true, ...overrides });
const close = (actual, expected, tolerance = 1e-6) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, i) => assert.ok(Math.abs(value - expected[i]) <= tolerance, `${i}: ${value} != ${expected[i]}`));
};

test("integer skin matches authored influence affine sum with source limb scaling", () => {
  const player = new NativePairPlayback(); player.load(fixture(), 0);
  const part = player.clip.parts.values().next().value;
  const result = new Float32Array(9);
  skinNativePairFrame(part, 0, result);
  close(result, [2.25,0,0, 3.25,0,0, 2.25,1,0]);
  skinNativePairFrame(part, 2, result);
  close(result, [4.625,1,0, 6.125,1,0, 4.825,2,0]);
});

test("local clock continues through absent, delayed, and duplicated socket updates", () => {
  const player = new NativePairPlayback(); player.load(fixture(), 0);
  player.command(command(), 1000);
  assert.equal(player.advance(1030).frame, 0.9);
  assert.equal(player.command(command(), 1060), false);
  assert.equal(player.advance(1060).frame, 1.8);
  assert.equal(player.command(command({ sequence: 0, frame: 0 }), 1070), false);
  assert.equal(player.advance(1100).frame, 3);
  assert.equal(player.sample.playing, false);
});

test("pause seek resume and exact capture remain authoritative", () => {
  const player = new NativePairPlayback(); player.load(fixture(), 0);
  player.command(command(), 0);
  player.command(command({ sequence: 2, frame: 2, playing: false }), 20);
  assert.equal(player.advance(2000).frame, 2);
  player.command(command({ sequence: 3, frame: 1 }), 2000);
  assert.equal(player.advance(2030).frame, 1.9);
  player.command(command({ sequence: 4, frame: 2, capturing: true }), 2040);
  assert.deepEqual(player.advance(9999), { revision: 1, frame: 2, first: 2, second: 2, alpha: 0, playing: false, capturing: true });
  player.command(command({ sequence: 5, enabled: false }), 10000);
  assert.equal(player.advance(10050), undefined);
});

test("queued controls, revisions, removal, and reconnect cannot revive stale takes", () => {
  const player = new NativePairPlayback();
  player.command(command({ revision: 2, sequence: 9, playing: false, frame: 2 }), 0);
  player.command(command({ revision: 1, sequence: 99, frame: 0 }), 10);
  player.load(fixture(2), 30);
  assert.equal(player.advance(300).frame, 2);
  assert.equal(player.load(fixture(1), 400), false);
  assert.equal(player.command(command(), 400), false);
  player.remove("/native-pair/a");
  assert.equal(player.clip, undefined);
  assert.equal(player.advance(1000), undefined);
  assert.equal(player.load(fixture(2), 1100), false);
  player.reset();
  assert.equal(player.load(fixture(1), 2000), true);
});

test("malformed binary, weights, indices, and memory shapes retain the valid take", () => {
  const player = new NativePairPlayback(); player.load(fixture(), 0);
  const valid = player.clip;
  for (const mutation of [
    (m) => { m.actors[0].linear = bytes([0]); },
    (m) => { m.actors[0].parts[0].bones = bytes([22,1,0,0, 0,1,0,0, 0,1,0,0], Uint16Array); },
    (m) => { m.actors[0].parts[0].weights = bytes([0,0,0,0, 0.25,0.75,0,0, 0.25,0.75,0,0]); },
    (m) => { m.frames = 1001; },
    (m) => { m.fps = NaN; },
    (m) => { m.actors[0].rest = bytes(new Array(66).fill(Infinity)); },
  ]) {
    const bad = fixture(2); mutation(bad);
    assert.throws(() => player.load(bad, 100)); assert.equal(player.clip, valid);
  }
});

test("mesh interpolation reuses geometry and buffers and capture lands exactly", () => {
  const player = new NativePairPlayback(); player.load(fixture(), 0);
  const part = player.clip.parts.values().next().value;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(9), 3));
  geometry.setIndex([0,1,2]); geometry.computeVertexNormals();
  const renderer = new NativePairGeometry(geometry);
  const originalPositions = geometry.getAttribute("position");
  const originalNormals = geometry.getAttribute("normal");
  player.command(command(), 0);
  renderer.update(part, player.advance(50));
  const a = new Float32Array(9), b = new Float32Array(9);
  skinNativePairFrame(part, 1, a); skinNativePairFrame(part, 2, b);
  close(geometry.getAttribute("position").array, a.map((value, i) => (value + b[i]) / 2));
  assert.equal(geometry.getAttribute("position"), originalPositions);
  assert.equal(geometry.getAttribute("normal"), originalNormals);
  player.command(command({ sequence: 2, frame: 2, playing: false, capturing: true }), 60);
  renderer.update(part, player.advance(1000));
  close(geometry.getAttribute("position").array, b);
  assert.ok(geometry.boundingSphere.radius > 0);
  renderer.dispose(); geometry.dispose();
});
