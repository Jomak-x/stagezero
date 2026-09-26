import React from "react";
import * as THREE from "three";
import { Html, Line, PivotControls } from "@react-three/drei";
import { useThree } from "@react-three/fiber";
import { ViewerContext } from "./ViewerContext";
import { getCameraStore, sendCameraCommand } from "./cameraStore";
import type { CameraRecord } from "./cameraProtocol";
import { computeT_threeworld_world } from "./WorldTransformUtils";

function cameraMatrix(worldToThree: THREE.Matrix4, camera: CameraRecord) {
  const worldPose = new THREE.Matrix4().makeRotationFromQuaternion(
    new THREE.Quaternion(camera.wxyz[1], camera.wxyz[2], camera.wxyz[3], camera.wxyz[0]),
  ).setPosition(...camera.position);
  return worldToThree.clone().multiply(worldPose);
}

function CameraGlyph({ camera, aspect, selected, onSelect }: {
  camera: CameraRecord;
  aspect: number;
  selected: boolean;
  onSelect: () => void;
}) {
  const corners = React.useMemo(() => {
    const z = 0.8;
    const halfHeight = z * Math.tan(camera.fov / 2);
    const halfWidth = halfHeight * aspect;
    return [
      [-halfWidth, -halfHeight, z], [halfWidth, -halfHeight, z],
      [halfWidth, halfHeight, z], [-halfWidth, halfHeight, z],
    ] as [number, number, number][];
  }, [camera.fov, aspect]);
  const points = React.useMemo(() => {
    const segments: [number, number, number][] = [];
    corners.forEach((corner, index) => {
      segments.push([0, 0, 0], corner);
      segments.push(corner, corners[(index + 1) % 4]);
    });
    // A small roof line distinguishes image up from down in OpenCV axes.
    segments.push(corners[0], [0, corners[0][1] - 0.12, corners[0][2]]);
    segments.push([0, corners[0][1] - 0.12, corners[0][2]], corners[1]);
    return segments;
  }, [corners]);
  const color = selected ? "#ffc56e" : "#6ec7ec";
  return (
    <group onClick={(event) => { event.stopPropagation(); onSelect(); }} onDoubleClick={(event) => { event.stopPropagation(); onSelect(); }}>
      <Line points={points} segments color={color} lineWidth={selected ? 2.5 : 1.6} depthTest={false} />
      <mesh>
        <boxGeometry args={[0.14, 0.1, 0.2]} />
        <meshBasicMaterial color={color} depthTest={false} />
      </mesh>
      <Html position={[0, -0.23, 0]} center distanceFactor={8} style={{ pointerEvents: "auto" }}>
        <button
          type="button"
          onClick={(event) => { event.stopPropagation(); onSelect(); }}
          title={`Select camera ${camera.name}`}
          style={{
            border: `1px solid ${color}`, borderRadius: 5, padding: "2px 5px",
            background: "rgba(17,24,37,.9)", color: "white", cursor: "pointer",
            font: "11px system-ui, sans-serif", whiteSpace: "nowrap",
          }}
        >{camera.name}</button>
      </Html>
    </group>
  );
}

function CameraMarker({ camera, worldToThree, aspect, selected }: {
  camera: CameraRecord;
  worldToThree: THREE.Matrix4;
  aspect: number;
  selected: boolean;
}) {
  const viewer = React.useContext(ViewerContext)!;
  const store = React.useMemo(() => getCameraStore(viewer), [viewer.mutable]);
  const matrixRef = React.useRef(cameraMatrix(worldToThree, camera));
  const dragPoseRef = React.useRef<THREE.Matrix4 | null>(null);
  const draggingRef = React.useRef(false);
  const dragContextRef = React.useRef<{
    project_id: string;
    revision: number;
    take_id: string | null;
    root: string;
  } | null>(null);
  const rootSignature = worldToThree.elements.join("|");
  const poseSignature = [...camera.position, ...camera.wxyz, camera.fov, ...worldToThree.elements].join("|");
  React.useLayoutEffect(() => {
    if (!draggingRef.current) {
      matrixRef.current.copy(cameraMatrix(worldToThree, camera));
    }
  }, [poseSignature]);
  React.useLayoutEffect(() => {
    if (!selected && draggingRef.current) {
      draggingRef.current = false;
      dragPoseRef.current = null;
      dragContextRef.current = null;
      viewer.mutable.current.transformControlsDraggingNames.delete(`camera:${camera.id}`);
      matrixRef.current.copy(cameraMatrix(worldToThree, camera));
    }
  });
  React.useEffect(() => () => {
    viewer.mutable.current.transformControlsDraggingNames.delete(`camera:${camera.id}`);
  }, [viewer.mutable, camera.id]);

  const select = () => sendCameraCommand(viewer, "select", { camera_id: camera.id });
  const glyph = <CameraGlyph camera={camera} aspect={aspect} selected={selected} onSelect={select} />;
  if (!selected) {
    const position = new THREE.Vector3();
    const rotation = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    cameraMatrix(worldToThree, camera).decompose(position, rotation, scale);
    return <group position={position} quaternion={rotation}>{glyph}</group>;
  }
  return (
    <PivotControls
      matrix={matrixRef.current}
      fixed
      scale={75}
      lineWidth={2}
      depthTest={false}
      disableScaling
      disableSliders
      onDragStart={() => {
        const current = store.getState();
        dragContextRef.current = {
          project_id: current.project_id,
          revision: current.revision,
          take_id: current.take_id,
          root: rootSignature,
        };
        draggingRef.current = true;
        dragPoseRef.current = null;
        viewer.mutable.current.transformControlsDraggingNames.add(`camera:${camera.id}`);
      }}
      onDrag={(local) => { dragPoseRef.current = local.clone(); }}
      onDragEnd={() => {
        draggingRef.current = false;
        viewer.mutable.current.transformControlsDraggingNames.delete(`camera:${camera.id}`);
        const context = dragContextRef.current;
        dragContextRef.current = null;
        const current = store.getState();
        const valid = context && current.received && current.mode === "free" &&
          context.project_id === current.project_id &&
          context.revision === current.revision &&
          context.take_id === current.take_id &&
          context.root === rootSignature &&
          current.cameras.some((item) => item.id === camera.id);
        if (!valid || !dragPoseRef.current) {
          dragPoseRef.current = null;
          matrixRef.current.copy(cameraMatrix(worldToThree, camera));
          return;
        }
        const worldPose = worldToThree.clone().invert().multiply(dragPoseRef.current);
        const position = new THREE.Vector3();
        const rotation = new THREE.Quaternion();
        const scale = new THREE.Vector3();
        worldPose.decompose(position, rotation, scale);
        rotation.normalize();
        // Keep the saved pose visible until the server confirms the edit.
        // A rejected command may return identical state/error, so no local
        // dragged pose may remain waiting for a response to clear it.
        dragPoseRef.current = null;
        matrixRef.current.copy(cameraMatrix(worldToThree, camera));
        sendCameraCommand(viewer, "update", {
          camera_id: camera.id,
          camera: {
            position: position.toArray() as [number, number, number],
            wxyz: [rotation.w, rotation.x, rotation.y, rotation.z],
          },
        });
      }}
    >{glyph}</PivotControls>
  );
}

/** Client-side editable frustums in the same Three-world basis as scene nodes. */
export function CameraScene() {
  const viewer = React.useContext(ViewerContext)!;
  const store = React.useMemo(() => getCameraStore(viewer), [viewer.mutable]);
  const { received, cameras, mode, selected_camera_id } = store((state) => ({
    received: state.received,
    cameras: state.cameras,
    mode: state.mode,
    selected_camera_id: state.selected_camera_id,
  }));
  const rootWxyz = viewer.useSceneTree((state) => state[""]?.wxyz);
  const rootPosition = viewer.useSceneTree((state) => state[""]?.position);
  const worldToThree = React.useMemo(
    () => computeT_threeworld_world(viewer),
    [rootWxyz, rootPosition],
  );
  const size = useThree((state) => state.size);
  if (!received || mode !== "free") return null;
  const aspect = Math.max(0.01, size.width / Math.max(1, size.height));
  return (
    <group name="Camera helpers">
      {cameras.map((camera) => (
        <CameraMarker key={camera.id} camera={camera} worldToThree={worldToThree} aspect={aspect} selected={camera.id === selected_camera_id} />
      ))}
    </group>
  );
}
