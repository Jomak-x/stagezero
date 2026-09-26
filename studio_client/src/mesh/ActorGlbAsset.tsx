import React from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { notifications } from "@mantine/notifications";
import * as THREE from "three";
import { GLTF, GLTFLoader } from "three-stdlib";

import { ViewerContext } from "../ViewerContext";
import {
  ActorGlbLoadMessage,
  ActorGlbPoseMessage,
} from "../WebsocketMessages";
import {
  ActorGlbControl,
  controls,
  g1Revision,
  poses,
  subscribeActorGlbControl,
} from "./ActorGlbProtocol";
import { configureActorMeshes, createActorPlacement } from "./ActorGlbRendering";
import { committedLoadToRetry, rejectionCancelsInFlightLoad } from "./ActorGlbLifecycle";

type LoadedActor = {
  assetId: string;
  revision: number;
  scene: THREE.Group;
  placement: THREE.Group;
  nodes: Map<number, THREE.Object3D[]>;
};

function disposeActor(actor: LoadedActor) {
  const geometries = new Set<THREE.BufferGeometry>();
  const materials = new Set<THREE.Material>();
  const textures = new Set<THREE.Texture>();
  actor.scene.traverse((object) => {
    if (object instanceof THREE.Mesh) {
      geometries.add(object.geometry);
      const entries = Array.isArray(object.material)
        ? object.material
        : [object.material];
      entries.forEach((material) => materials.add(material));
      if (object instanceof THREE.SkinnedMesh && object.skeleton.boneTexture) {
        textures.add(object.skeleton.boneTexture);
      }
    }
  });
  materials.forEach((material) => {
    Object.values(material).forEach((value) => {
      if (value instanceof THREE.Texture) textures.add(value);
    });
    material.dispose();
  });
  textures.forEach((texture) => {
    texture.dispose();
    const image = texture.image;
    if (typeof ImageBitmap !== "undefined" && image instanceof ImageBitmap) {
      image.close();
    }
  });
  geometries.forEach((geometry) => geometry.dispose());
}

function nodeMap(gltf: GLTF): Map<number, THREE.Object3D[]> {
  const map = new Map<number, THREE.Object3D[]>();
  const associations = gltf.parser.associations;
  const addObject = (object: THREE.Object3D) => {
    const index = associations.get(object)?.nodes;
    if (typeof index !== "number") return;
    const matches = map.get(index) ?? [];
    if (!matches.includes(object)) matches.push(object);
    map.set(index, matches);
  };
  gltf.scene.traverse((object) => {
    addObject(object);
    if (object instanceof THREE.SkinnedMesh) {
      object.skeleton.bones.forEach(addObject);
    }
  });
  return map;
}

function applyPose(actor: LoadedActor, message: ActorGlbPoseMessage) {
  const indices = new Uint32Array(
    message.node_indices.buffer.slice(
      message.node_indices.byteOffset,
      message.node_indices.byteOffset + message.node_indices.byteLength,
    ),
  );
  const values = new Float32Array(
    message.local_matrices.buffer.slice(
      message.local_matrices.byteOffset,
      message.local_matrices.byteOffset + message.local_matrices.byteLength,
    ),
  );
  if (values.length !== indices.length * 16) {
    console.error("Actor GLB pose has mismatched node/matrix counts");
    return;
  }
  indices.forEach((index, slot) => {
    const nodes = actor.nodes.get(index);
    if (!nodes) return;
    nodes.forEach((node) => {
      node.matrixAutoUpdate = false;
      node.matrix.fromArray(values, slot * 16);
      node.matrixWorldNeedsUpdate = true;
    });
  });
  actor.placement.updateMatrixWorld(true);
}

/** The controlled GLB actor. Normal GlbMessage assets keep their usual loader. */
export const ActorGlbAsset = React.forwardRef<
  THREE.Group,
  ActorGlbLoadMessage & { children?: React.ReactNode }
>(function ActorGlbAsset({ children, ...message }, ref) {
  const viewer = React.useContext(ViewerContext)!;
  const gl = useThree((state) => state.gl);
  const camera = useThree((state) => state.camera);
  const scene = useThree((state) => state.scene);
  const activeRef = React.useRef<LoadedActor | null>(null);
  const candidateRef = React.useRef<LoadedActor | null>(null);
  const [visible, setVisible] = React.useState<LoadedActor | null>(null);
  const retired = React.useRef<LoadedActor[]>([]);
  const loadToken = React.useRef(0);
  const inFlightRevision = React.useRef<number | null>(null);
  const latestRevision = React.useRef(-1);
  const wantedRevision = React.useRef<number | null>(null);
  const requestedLoad = React.useRef<ActorGlbLoadMessage>(message);
  const committedLoad = React.useRef<ActorGlbLoadMessage | null>(null);
  const [retryLoad, setRetryLoad] = React.useState<{
    failedRevision: number;
    load: ActorGlbLoadMessage;
  } | null>(null);
  if (message.revision >= requestedLoad.current.revision) requestedLoad.current = message;

  const loadMessage = retryLoad?.failedRevision === message.revision
    ? retryLoad.load
    : message;

  function setFallbackVisible(visible: boolean) {
    const fallback = viewer.mutable.current.nodeRefFromName[message.fallback_name];
    if (fallback) fallback.visible = visible;
  }

  function retryCommittedAfter(failedRevision: number) {
    const fallback = committedLoadToRetry(
      committedLoad.current,
      activeRef.current?.revision ?? null,
      failedRevision,
    );
    if (fallback === null) return;
    setRetryLoad({ failedRevision, load: fallback });
  }

  function reportError(load: ActorGlbLoadMessage, error: string) {
    notifications.show({
      id: `actor-glb-${load.revision}`,
      title: "Character load failed in this tab",
      message: `${error}. ${activeRef.current ? "The previous character remains visible." : "The G1 robot remains visible."}`,
      color: "red",
      autoClose: 7000,
    });
    viewer.mutable.current.sendMessage({
      type: "ActorGlbStatusMessage",
      asset_id: load.asset_id,
      revision: load.revision,
      status: "error",
      error,
    });
    retryCommittedAfter(load.revision);
  }

  function showCandidate(revision: number) {
    const candidate = candidateRef.current;
    if (!candidate || candidate.revision !== revision) return;
    candidateRef.current = null;
    const old = activeRef.current;
    activeRef.current = candidate;
    if (old) retired.current.push(old);
    if (requestedLoad.current.revision === revision) {
      committedLoad.current = requestedLoad.current;
    }
    setFallbackVisible(false);
    setVisible(candidate);
  }

  function handleControl(control: ActorGlbControl) {
    if (control.type === "ActorGlbPoseMessage") {
      if (activeRef.current?.revision === control.revision) {
        applyPose(activeRef.current, control);
      }
      if (candidateRef.current?.revision === control.revision) {
        applyPose(candidateRef.current, control);
      }
      return;
    }
    if (control.action === "g1") {
      latestRevision.current = Math.max(latestRevision.current, control.revision);
      loadToken.current++;
      inFlightRevision.current = null;
      wantedRevision.current = null;
      committedLoad.current = null;
      setRetryLoad(null);
      if (candidateRef.current) disposeActor(candidateRef.current);
      candidateRef.current = null;
      if (activeRef.current) retired.current.push(activeRef.current);
      activeRef.current = null;
      setFallbackVisible(true);
      setVisible(null);
      return;
    }
    if (control.action === "reject") {
      if (candidateRef.current?.revision === control.revision) {
        disposeActor(candidateRef.current);
        candidateRef.current = null;
      }
      if (rejectionCancelsInFlightLoad(control.revision, inFlightRevision.current)) {
        loadToken.current++;
        inFlightRevision.current = null;
      }
      if (wantedRevision.current === control.revision) wantedRevision.current = null;
      retryCommittedAfter(control.revision);
      return;
    }
    wantedRevision.current = control.revision;
    if (committedLoad.current === null && requestedLoad.current.revision === control.revision) {
      committedLoad.current = requestedLoad.current;
    }
    showCandidate(control.revision);
  }

  React.useEffect(() => {
    const unsubscribe = subscribeActorGlbControl(handleControl);
    return () => {
      unsubscribe();
      loadToken.current++;
      inFlightRevision.current = null;
      if (candidateRef.current) disposeActor(candidateRef.current);
      candidateRef.current = null;
      if (activeRef.current) disposeActor(activeRef.current);
      activeRef.current = null;
      retired.current.forEach(disposeActor);
      retired.current = [];
      setFallbackVisible(true);
    };
  }, []);

  React.useEffect(() => {
    retired.current.forEach(disposeActor);
    retired.current = [];
  }, [visible]);

  React.useEffect(() => {
    if (retryLoad !== null && retryLoad.failedRevision !== message.revision) {
      setRetryLoad(null);
    }
  }, [message.revision, retryLoad]);

  React.useEffect(() => {
    const isRetry = retryLoad?.failedRevision === message.revision;
    if (loadMessage.revision <= g1Revision ||
        (!isRetry && loadMessage.revision < latestRevision.current) ||
        (!isRetry && controls.get(loadMessage.revision)?.action === "reject")) return;
    latestRevision.current = Math.max(latestRevision.current, loadMessage.revision);
    if (activeRef.current?.revision === loadMessage.revision) return;
    if (committedLoad.current === null && controls.get(loadMessage.revision)?.action === "commit") {
      committedLoad.current = loadMessage;
    }
    if (candidateRef.current) disposeActor(candidateRef.current);
    candidateRef.current = null;
    const token = ++loadToken.current;
    inFlightRevision.current = loadMessage.revision;
    const loader = new GLTFLoader();
    const buffer = new Uint8Array(loadMessage.glb_data).buffer;
    try {
      loader.parse(
        buffer,
        "",
        (gltf) => {
          const actor: LoadedActor = {
            assetId: loadMessage.asset_id,
            revision: loadMessage.revision,
            scene: gltf.scene,
            placement: new THREE.Group(),
            nodes: new Map(),
          };
          if (token !== loadToken.current || loadMessage.revision <= g1Revision) {
            disposeActor(actor);
            return;
          }
          try {
            actor.nodes = nodeMap(gltf);
            const missing = loadMessage.required_nodes.filter((index) => !actor.nodes.has(index));
            if (missing.length) {
              throw new Error(`GLB is missing required node indices: ${missing.join(", ")}`);
            }
            configureActorMeshes(actor.scene, loadMessage.cast_shadow, loadMessage.receive_shadow);
            actor.placement = createActorPlacement(
              actor.scene, loadMessage.scale, loadMessage.ground_offset,
            );
          } catch (error) {
            disposeActor(actor);
            if (token === loadToken.current) inFlightRevision.current = null;
            reportError(loadMessage, error instanceof Error ? error.message : String(error));
            return;
          }
          void Promise.resolve().then(() => gl.compileAsync(actor.scene, camera, scene)).then(() => {
            if (token !== loadToken.current || loadMessage.revision <= g1Revision) {
              disposeActor(actor);
              return;
            }
            const pose = poses.get(loadMessage.revision);
            if (pose) applyPose(actor, pose);
            inFlightRevision.current = null;
            candidateRef.current = actor;
            viewer.mutable.current.sendMessage({
              type: "ActorGlbStatusMessage",
              asset_id: loadMessage.asset_id,
              revision: loadMessage.revision,
              status: "loaded",
              error: null,
            });
            if (isRetry || wantedRevision.current === loadMessage.revision ||
                controls.get(loadMessage.revision)?.action === "commit") {
              showCandidate(loadMessage.revision);
            }
          }).catch((error) => {
            disposeActor(actor);
            if (token === loadToken.current) {
              inFlightRevision.current = null;
              reportError(loadMessage, error instanceof Error ? error.message : String(error));
            }
          });
        },
        (error) => {
          if (token !== loadToken.current) return;
          inFlightRevision.current = null;
          reportError(loadMessage, error instanceof Error ? error.message : String(error));
        },
      );
    } catch (error) {
      if (token === loadToken.current) {
        inFlightRevision.current = null;
        reportError(loadMessage, error instanceof Error ? error.message : String(error));
      }
    }
    return () => {
      if (token === loadToken.current) {
        loadToken.current++;
        inFlightRevision.current = null;
      }
    };
  }, [loadMessage.asset_id, loadMessage.revision, loadMessage.glb_data,
      loadMessage.scale, loadMessage.ground_offset, gl, camera, scene]);

  // SceneTree applies server visibility at priority -1000. Override only the
  // local G1 subtree after that pass; no shared server node state is changed.
  useFrame(() => {
    setFallbackVisible(activeRef.current === null);
  });

  return (
    <group ref={ref}>
      {visible && <primitive object={visible.placement} />}
      {children}
    </group>
  );
});
