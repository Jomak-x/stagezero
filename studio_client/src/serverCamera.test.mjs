import test from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";
import CameraControlsPackage from "camera-controls";
import { setServerCameraTarget } from "./serverCamera.ts";

const CameraControls = CameraControlsPackage.default;
globalThis.DOMRect ??= class DOMRect {
  constructor(x = 0, y = 0, width = 0, height = 0) {
    Object.assign(this, { x, y, width, height });
  }
};
CameraControls.install({ THREE });

function controlsFor(camera) {
  const controls = new CameraControls(camera);
  controls.minPolarAngle = 0.06;
  controls.maxPolarAngle = Math.PI - 0.06;
  return controls;
}

function assertVector(actual, expected, label) {
  assert.ok(actual.distanceTo(expected) < 1e-9,
    `${label}: expected ${expected.toArray()}, received ${actual.toArray()}`);
}

function applyServerPose(controls, camera, position, target, up) {
  // Match the server's position, target, up message order. The up handler
  // preserves both endpoints when it changes camera-controls' orbit basis.
  controls.setPosition(position.x, position.y, position.z, false);
  setServerCameraTarget(controls, target);
  assertVector(controls.getPosition(new THREE.Vector3()), position, "position before up");
  assertVector(controls.getTarget(new THREE.Vector3()), target, "target before up");

  camera.up.copy(up);
  const savedPosition = controls.getPosition(new THREE.Vector3());
  const savedTarget = controls.getTarget(new THREE.Vector3());
  controls.updateCameraUp();
  controls.setLookAt(
    savedPosition.x, savedPosition.y, savedPosition.z,
    savedTarget.x, savedTarget.y, savedTarget.z, false,
  );
  controls.update(0);

  assertVector(controls.getPosition(new THREE.Vector3()), position, "position after up");
  assertVector(controls.getTarget(new THREE.Vector3()), target, "target after up");
  assertVector(camera.position, position, "rendered position");
  assertVector(camera.up, up, "rendered up");
  const expectedCamera = new THREE.PerspectiveCamera();
  expectedCamera.position.copy(position);
  expectedCamera.up.copy(up);
  expectedCamera.lookAt(target);
  assert.ok(camera.quaternion.angleTo(expectedCamera.quaternion) < 1e-7,
    "rendered orientation must match the server pose");
}

test("a cut from Y-up to a straight-down shot preserves the exact target", () => {
  const camera = new THREE.PerspectiveCamera();
  camera.position.set(4, 2, 6);
  const controls = controlsFor(camera);
  controls.setLookAt(4, 2, 6, 0, 1, 0, false);
  controls.update(0);

  const oldCamera = new THREE.PerspectiveCamera();
  oldCamera.position.set(4, 2, 6);
  const oldControls = controlsFor(oldCamera);
  oldControls.setLookAt(4, 2, 6, 0, 1, 0, false);
  oldControls.setPosition(0, 5, 0, false);
  oldControls.setTarget(0, 0, 0, false);
  assert.ok(oldControls.getPosition(new THREE.Vector3()).distanceTo(
    new THREE.Vector3(0, 5, 0)) > 0.2,
  "the old setTarget path moves the camera before the up-vector update");

  applyServerPose(controls, camera,
    new THREE.Vector3(0, 5, 0),
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0, 0, -1));
});

test("successive rolled and up-vector cuts preserve both endpoints", () => {
  const camera = new THREE.PerspectiveCamera();
  camera.position.set(0, 2, 5);
  const controls = controlsFor(camera);
  controls.setLookAt(0, 2, 5, 0, 0, 0, false);
  controls.update(0);

  const shots = [
    [[3, 4, 5], [0, 1, 0], [0, 0, -1]],
    [[-2, 3, 7], [1, 0, -1], [1, 0, 0]],
    [[0, 5, 0], [0, 0, 0], [0, 0, -1]],
    [[2, 1, 4], [0, 1, 0], [0, 1, 0]],
  ];
  for (const [position, target, up] of shots) {
    applyServerPose(controls, camera,
      new THREE.Vector3(...position),
      new THREE.Vector3(...target),
      new THREE.Vector3(...up));
  }
});
