import test from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";
import {
  MIN_PITCH_MARGIN,
  isDragGesture,
  lookTargetAfterDrag,
  panTranslation,
  walkTranslation,
  wheelDollyDistance,
} from "./cameraNavigation.ts";

const up = new THREE.Vector3(0, 1, 0);

test("a click and tiny pointer jitter never begin a camera drag", () => {
  assert.equal(isDragGesture(100, 100, 100, 100), false);
  assert.equal(isDragGesture(100, 100, 102, 101), false);
  assert.equal(isDragGesture(100, 100, 103, 104), true);
});

test("pan moves the camera and target together in the screen plane", () => {
  const position = new THREE.Vector3(0, 2, 5);
  const target = new THREE.Vector3(0, 0, 0);
  const camera = new THREE.PerspectiveCamera(60);
  camera.position.copy(position);
  camera.lookAt(target);
  const delta = panTranslation(position, target, camera.quaternion, 60, 600, 40, -20);
  assert.ok(delta.x < 0);
  assert.ok(delta.length() > 0);
  assert.ok(Math.abs(position.clone().add(delta).distanceTo(target.clone().add(delta)) - position.distanceTo(target)) < 1e-10);
});

test("look keeps the camera fixed, target distance constant, and pitch upright", () => {
  const position = new THREE.Vector3(0, 1, 5);
  const target = new THREE.Vector3(0, 1, 0);
  const newTarget = lookTargetAfterDrag(position, target, up, 90, -10000);
  assert.ok(Math.abs(newTarget.distanceTo(position) - 5) < 1e-8);
  const direction = newTarget.clone().sub(position).normalize();
  assert.ok(Math.asin(direction.dot(up)) <= Math.PI / 2 - MIN_PITCH_MARGIN + 1e-9);
  assert.ok(direction.dot(up) > 0);
});

test("walk forward stays level even when looking down", () => {
  const delta = walkTranslation(
    new THREE.Vector3(0, 4, 5), new THREE.Vector3(0, 0, 0), up,
    1, 0, 0,
  );
  assert.ok(Math.abs(delta.y) < 1e-10);
  assert.ok(delta.z < 0);
});

test("dolly has bounded, monotonic sensitivity", () => {
  assert.ok(wheelDollyDistance(5, -40) < 5);
  assert.ok(wheelDollyDistance(5, 40) > 5);
  assert.equal(wheelDollyDistance(0.1, -10000), 0.1);
  assert.equal(wheelDollyDistance(1000, 10000), 1000);
});
