import * as THREE from "three";

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
