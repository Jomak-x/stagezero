import test from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";

import { configureActorMeshes } from "./ActorGlbRendering.ts";

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
