import * as THREE from "three";
import React from "react";
import { disposeMaterial } from "./MeshUtils";
import { GLTF, GLTFLoader, DRACOLoader } from "three-stdlib";
import { createLoadGuard } from "./GlbLoadGuard";

// We use a CDN for Draco. We could move this locally if we want to use Viser offline.
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath("https://www.gstatic.com/draco/v1/decoders/");

/**
 * Dispose a 3D object and its resources
 */
export function disposeNode(node: any) {
  if (node instanceof THREE.Mesh) {
    if (node.geometry) {
      node.geometry.dispose();
    }
    if (node.material) {
      if (Array.isArray(node.material)) {
        node.material.forEach((material) => {
          disposeMaterial(material);
        });
      } else {
        disposeMaterial(node.material);
      }
    }
  }
}

/**
 * Custom hook for loading a GLB model
 */
export function useGlbLoader(glb_data: Uint8Array) {
  const [model, setModel] = React.useState<{
    gltf: GLTF;
    meshes: THREE.Mesh[];
    mixer: THREE.AnimationMixer | null;
  }>();

  // Animation mixer reference
  const mixerRef = React.useRef<THREE.AnimationMixer | null>(null);

  // Load the GLB model
  React.useEffect(() => {
    const loader = new GLTFLoader();
    loader.setDRACOLoader(dracoLoader);
    const request = createLoadGuard(
      (loaded: GLTF) => {
        // Process all meshes in the scene.
        const meshes: THREE.Mesh[] = [];
        loaded.scene.traverse((obj) => {
          if (obj instanceof THREE.Mesh) {
            obj.geometry.computeVertexNormals();
            obj.geometry.computeBoundingSphere();
            meshes.push(obj);
          }
        });

        let mixer: THREE.AnimationMixer | null = null;
        if (loaded.animations.length) {
          mixer = new THREE.AnimationMixer(loaded.scene);
          loaded.animations.forEach((clip) => mixer!.clipAction(clip).play());
        }

        // Keep the previous scene visible until this model is ready to render.
        setModel({ gltf: loaded, meshes, mixer });
      },
      (stale: GLTF) => stale.scene.traverse(disposeNode),
    );
    loader.parse(
      new Uint8Array(glb_data).buffer,
      "",
      request.complete,
      (error) => {
        if (!request.active) return;
        console.log("Error loading GLB!");
        console.log(error);
      },
    );

    return request.cancel;
  }, [glb_data]);

  // Dispose only after React has replaced the rendered scene, or on unmount.
  React.useEffect(() => {
    if (!model) return;
    mixerRef.current = model.mixer;
    return () => {
      if (mixerRef.current === model.mixer) mixerRef.current = null;
      model.mixer?.stopAllAction();
      model.gltf.scene.traverse(disposeNode);
    };
  }, [model]);

  // Return the loaded model, meshes, and mixer for animation updates
  return { gltf: model?.gltf, meshes: model?.meshes ?? [], mixerRef };
}
