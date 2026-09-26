import test from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";
import { applyRootPoseImmediately, computeT_threeworld_world } from "./WorldTransformUtils.ts";
import { walkTranslation } from "./cameraNavigation.ts";

test("root +Y pose is available to camera commands in the same socket batch", () => {
  // The client starts with Viser's +Z root. A fresh StageZero connection sends
  // a +Y root rotation followed by camera up, position, and look-at commands.
  const initial = new THREE.Quaternion().setFromEuler(
    new THREE.Euler(Math.PI / 2, Math.PI, -Math.PI / 2),
  );
  let root = { wxyz: [initial.w, initial.x, initial.y, initial.z], position: [0, 0, 0] };
  const viewer = {
    useSceneTree: { getState: () => ({ "": root }) },
    sceneTreeActions: {
      updateNodeAttributes: (_name, updates) => { root = { ...root, ...updates }; },
    },
  };

  const serverUp = new THREE.Vector3(0, 1, 0);
  const staleUp = serverUp.clone().transformDirection(computeT_threeworld_world(viewer));
  assert.ok(Math.abs(staleUp.y) < 1e-9);

  const yUpRoot = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI);
  applyRootPoseImmediately(viewer, {
    wxyz: [yUpRoot.w, yUpRoot.x, yUpRoot.y, yUpRoot.z],
    position: [0, 0, 0],
  });
  const worldToThree = computeT_threeworld_world(viewer);
  const up = serverUp.clone().transformDirection(worldToThree);
  const position = new THREE.Vector3(4, 2.25, 4.2).applyMatrix4(worldToThree);
  const target = new THREE.Vector3(0, 0.9, 0).applyMatrix4(worldToThree);
  const camera = new THREE.PerspectiveCamera(42);
  camera.position.copy(position);
  camera.up.copy(up);
  camera.lookAt(target);

  assert.ok(up.distanceTo(new THREE.Vector3(0, 1, 0)) < 1e-9);
  assert.ok(new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion).dot(up) > 0.95);
  const walk = walkTranslation(position, target, up, 1, 0, 0);
  assert.ok(Math.abs(walk.dot(up)) < 1e-9);
  const rise = walkTranslation(position, target, up, 0, 0, 1);
  assert.ok(rise.distanceTo(up) < 1e-9);
});
