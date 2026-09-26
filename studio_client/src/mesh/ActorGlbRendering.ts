import * as THREE from "three";

/** Keep the load-time floor placement outside the GLTF nodes driven by poses.
 * The offset is already measured in scene units, so scaling the imported
 * scene must not scale it a second time. Older load messages omit the field.
 */
export function createActorPlacement(
  importedScene: THREE.Group,
  scale: number,
  groundOffset: number | undefined,
): THREE.Group {
  const offset = groundOffset ?? 0;
  if (!Number.isFinite(offset)) throw new Error("ground_offset must be finite");
  const placement = new THREE.Group();
  placement.position.y = offset;
  importedScene.scale.setScalar(scale);
  placement.add(importedScene);
  placement.updateMatrixWorld(true);
  return placement;
}

/** Configure a controlled GLB without changing its imported materials.
 *
 * Three.js caches a SkinnedMesh bounding sphere. Bone poses can move vertices
 * far outside that bind-pose sphere, so the renderer must not use it to cull
 * the actor. Ordinary static meshes retain their normal frustum culling.
 */
export function configureActorMeshes(
  scene: THREE.Object3D,
  castShadow: boolean,
  receiveShadow: boolean,
) {
  scene.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return;
    object.castShadow = castShadow;
    object.receiveShadow = receiveShadow;
    if (object instanceof THREE.SkinnedMesh) object.frustumCulled = false;
  });
}
