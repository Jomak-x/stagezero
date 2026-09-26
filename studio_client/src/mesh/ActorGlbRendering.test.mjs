import test from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";

import { configureActorMeshes, createActorPlacement } from "./ActorGlbRendering.ts";

test("ground offset uses world units while model scale remains local", () => {
  for (const scale of [0.5, 1, 2]) {
    const importedScene = new THREE.Group();
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshBasicMaterial(),
    );
    importedScene.add(body);
    const offset = scale * 0.5;
    const placement = createActorPlacement(importedScene, scale, offset);
    const initialBounds = new THREE.Box3().setFromObject(placement);
    assert.ok(Math.abs(initialBounds.min.y) < 1e-8);
    assert.equal(placement.position.y, offset);
    assert.equal(importedScene.scale.y, scale);

    // Animated node motion must remain relative to the single load-time
    // placement. A jump is not pulled back onto the ground every frame.
    body.position.y = 1;
    const jumpedBounds = new THREE.Box3().setFromObject(placement);
    assert.ok(Math.abs(jumpedBounds.min.y - scale) < 1e-8);
  }
});

test("placement defaults legacy loads to zero and stays isolated per actor", () => {
  const active = createActorPlacement(new THREE.Group(), 1, 0.25);
  const candidate = createActorPlacement(new THREE.Group(), 1, -0.75);
  const legacy = createActorPlacement(new THREE.Group(), 1, undefined);
  assert.equal(active.position.y, 0.25);
  assert.equal(candidate.position.y, -0.75);
  assert.equal(legacy.position.y, 0);
  candidate.position.y = 10;
  assert.equal(active.position.y, 0.25);
  assert.throws(() => createActorPlacement(new THREE.Group(), 1, NaN),
    /ground_offset must be finite/);
});

test("controlled skinned GLB is not culled against stale bind bounds", () => {
  const scene = new THREE.Group();
  const skinnedMaterial = new THREE.MeshStandardMaterial({ color: 0xb06a52 });
  const skinned = new THREE.SkinnedMesh(
    new THREE.BufferGeometry(), skinnedMaterial,
  );
  skinned.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 0.25);
  const staticMesh = new THREE.Mesh(
    new THREE.BufferGeometry(), new THREE.MeshStandardMaterial(),
  );
  scene.add(skinned, staticMesh);

  // A moving bone can put skinned vertices far from the cached sphere. The
  // browser renderer must still draw it while the camera follows that pose.
  configureActorMeshes(scene, true, true);

  assert.equal(skinned.frustumCulled, false);
  assert.equal(staticMesh.frustumCulled, true);
  assert.equal(skinned.boundingSphere.radius, 0.25);
  assert.equal(skinned.material, skinnedMaterial);
  assert.equal(skinned.castShadow, true);
  assert.equal(skinned.receiveShadow, true);
  assert.equal(staticMesh.castShadow, true);
  assert.equal(staticMesh.receiveShadow, true);
});
