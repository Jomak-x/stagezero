// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ViewerContext } from "./ViewerContext";
import { CameraControls, CameraControlsImpl, Instance, Instances } from "@react-three/drei";
import { useThree } from "@react-three/fiber";
import React, { useContext, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import { PerspectiveCamera } from "three";
import * as THREE from "three";
import { computeT_threeworld_world } from "./WorldTransformUtils";
import { useThrottledMessageSender } from "./WebsocketUtils";
import { Grid, PivotControls } from "@react-three/drei";
import {
  isDragGesture,
  lookTargetAfterDrag,
  panTranslation,
  walkTranslation,
  wheelDollyDistance,
} from "./cameraNavigation";
import { getCameraStore } from "./cameraStore";

type NavigationMode = "Pan" | "Orbit" | "Look";

function NavigationOverlay({
  canvas,
  mode,
  setMode,
}: {
  canvas: HTMLCanvasElement;
  mode: NavigationMode;
  setMode: (mode: NavigationMode) => void;
}) {
  const buttonsRef = useRef<Partial<Record<NavigationMode, HTMLButtonElement>>>({});
  const hintRef = useRef<HTMLDivElement | null>(null);

  React.useEffect(() => {
    const parent = canvas.parentElement;
    if (!parent) return;
    const container = document.createElement("div");
    Object.assign(container.style, {
      // Below the Welcome return button and clear of the right-hand inspector.
      position: "absolute", left: "20px", top: "108px",
      maxWidth: "calc(100% - 24px)", zIndex: "20",
      display: "flex", flexDirection: "column", alignItems: "flex-start", gap: "5px",
      pointerEvents: "none", fontFamily: "system-ui, sans-serif",
      fontSize: "11px", color: "white", textShadow: "0 1px 3px #000",
    });
    const row = document.createElement("div");
    row.setAttribute("role", "group");
    row.setAttribute("aria-label", "Camera drag mode");
    Object.assign(row.style, {
      display: "flex", gap: "3px", padding: "4px", borderRadius: "8px",
      background: "rgba(20, 24, 32, 0.76)", pointerEvents: "auto",
    });
    for (const option of ["Pan", "Orbit", "Look"] as const) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = option;
      button.title = `${option} with a plain drag`;
      Object.assign(button.style, {
        border: "0", borderRadius: "5px", padding: "5px 7px",
        cursor: "pointer", color: "white", fontSize: "11px",
      });
      button.onclick = () => setMode(option);
      buttonsRef.current[option] = button;
      row.appendChild(button);
    }
    const hint = document.createElement("div");
    Object.assign(hint.style, {
      padding: "8px 10px", borderRadius: "5px",
      background: "rgba(20, 24, 32, 0.94)", textAlign: "left",
      display: "none", lineHeight: "1.6", maxWidth: "min(184px, 100%)",
    });
    const help = document.createElement("button");
    help.type = "button";
    help.textContent = "?";
    help.title = "Camera controls";
    help.setAttribute("aria-label", "Camera controls help");
    help.setAttribute("aria-expanded", "false");
    Object.assign(help.style, { border: "0", borderRadius: "5px", padding: "5px 9px", color: "#bad1dd", background: "transparent", cursor: "pointer" });
    help.onclick = () => {
      const expanded = help.getAttribute("aria-expanded") !== "true";
      help.setAttribute("aria-expanded", String(expanded));
      hint.style.display = expanded ? "block" : "none";
    };
    row.appendChild(help);
    hintRef.current = hint;
    container.append(row, hint);
    parent.appendChild(container);
    return () => {
      container.remove();
      buttonsRef.current = {};
      hintRef.current = null;
    };
  }, [canvas, setMode]);

  React.useEffect(() => {
    for (const option of ["Pan", "Orbit", "Look"] as const) {
      const button = buttonsRef.current[option];
      if (!button) continue;
      button.setAttribute("aria-pressed", String(option === mode));
      button.style.background = option === mode ? "#6888bd" : "transparent";
    }
    if (hintRef.current) {
      hintRef.current.textContent = `Drag ${mode.toLowerCase()} · Alt/middle drag orbit · Scroll pan · Pinch/Alt scroll zoom · WASD/QE move`;
    }
  }, [mode]);
  return null;
}

function CrosshairVisual({
  visible,
  children,
}: {
  visible: boolean;
  children?: React.ReactNode;
}) {
  const { camera } = useThree();
  const groupRef = useRef<THREE.Group>(null);

  const worldPos = new THREE.Vector3();
  useFrame(() => {
    if (groupRef.current && visible) {
      // Get world position of the crosshair.
      groupRef.current.getWorldPosition(worldPos);
      // Scale based on distance and FOV to maintain consistent visual size.
      const distance = camera.position.distanceTo(worldPos);
      const fovScale = Math.tan(
        ((camera as THREE.PerspectiveCamera).fov * Math.PI) / 360,
      );
      groupRef.current.scale.setScalar((distance / 20) * fovScale);
    }
  });

  return (
    <group ref={groupRef} visible={visible}>
      <Instances limit={6}>
        <boxGeometry args={[0.4, 0.02, 0.02]} />
        <meshBasicMaterial opacity={0.625} transparent />
        {/* Horizontal line segments */}
        <Instance position={[0.5, 0.0, 0.0]} color="#777777" />
        <Instance position={[-0.5, 0.0, 0.0]} color="#777777" />
        <Instance
          position={[0.0, 0.0, 0.5]}
          rotation={new THREE.Euler(0.0, Math.PI / 2.0, 0.0)}
          color="#777777"
        />
        <Instance
          position={[0.0, 0.0, -0.5]}
          rotation={new THREE.Euler(0.0, Math.PI / 2.0, 0.0)}
          color="#777777"
        />
        {/* Vertical line segments */}
        <Instance
          position={[0.0, 0.5, 0.0]}
          rotation={new THREE.Euler(0.0, 0.0, Math.PI / 2.0)}
          color="#999999"
        />
        <Instance
          position={[0.0, -0.5, 0.0]}
          rotation={new THREE.Euler(0.0, 0.0, Math.PI / 2.0)}
          color="#999999"
        />
      </Instances>
      <mesh>
        <sphereGeometry args={[0.04, 8, 8]} />
        <meshBasicMaterial color="#999999" opacity={0.625} transparent />
      </mesh>
      {children}
    </group>
  );
}

function OrbitOriginTool({
  forceShow,
  locked,
  pivotRef,
  onPivotChange,
  update,
  crosshairVisible,
}: {
  forceShow: boolean;
  locked: boolean;
  pivotRef: React.RefObject<THREE.Group>;
  onPivotChange: (matrix: THREE.Matrix4) => void;
  update: () => void;
  crosshairVisible: boolean;
}) {
  const viewer = useContext(ViewerContext)!;
  const showOrbitOriginTool = viewer.useGui(
    (state) => state.showOrbitOriginTool,
  );
  const enableOrbitCrosshair = viewer.useDevSettings(
    (state) => state.enableOrbitCrosshair,
  );
  React.useEffect(update, [showOrbitOriginTool]);

  const show = !locked && (showOrbitOriginTool || forceShow);
  return (
    <PivotControls
      ref={pivotRef}
      scale={200}
      lineWidth={3}
      fixed={true}
      axisColors={["#ffaaff", "#ff33ff", "#ffaaff"]}
      disableScaling={true}
      disableAxes={!show}
      disableRotations={!show}
      disableSliders={!show}
      onDragEnd={() => {
        onPivotChange(pivotRef.current!.matrix);
      }}
    >
      <Grid
        args={[10, 10, 10, 10]}
        infiniteGrid
        fadeStrength={0}
        fadeFrom={0}
        fadeDistance={1000}
        sectionColor={"#ffaaff"}
        cellColor={"#ffccff"}
        side={THREE.DoubleSide}
        visible={show}
      />
      {/* Crosshair visualization at look-at point */}
      <CrosshairVisual visible={!locked && enableOrbitCrosshair && crosshairVisible} />
    </PivotControls>
  );
}

export function SynchronizedCameraControls() {
  const viewer = useContext(ViewerContext)!;
  const camera = useThree((state) => state.camera as PerspectiveCamera);
  const cameraStore = React.useMemo(() => getCameraStore(viewer), [viewer.mutable]);
  const cameraLocked = cameraStore((state) => state.mode !== "free");

  const sendCameraThrottled = useThrottledMessageSender(20).send;

  // Helper for resetting camera poses.
  const initialCameraRef = useRef<{
    camera: PerspectiveCamera;
    lookAt: THREE.Vector3;
  } | null>(null);
  const receivedInitialServerViewRef = useRef(false);

  const pivotRef = useRef<THREE.Group>(null);
  const [navigationMode, setNavigationMode] = useState<NavigationMode>("Pan");
  const worldUpRef = useRef(new THREE.Vector3(0, 1, 0));
  const lastCameraUpRef = useRef(new THREE.Vector3(0, 1, 0));
  const heldKeysRef = useRef(new Set<string>());
  const pointerRef = useRef<{
    id: number;
    startX: number;
    startY: number;
    lastX: number;
    lastY: number;
    moving: boolean;
    mode: NavigationMode;
  } | null>(null);

  const viewerMutable = viewer.mutable.current;

  // Crosshair visibility state: separate counter for keyboard and flag for pointer interactions.
  const [keyboardCrosshairCounter, setKeyboardCrosshairCounter] = useState(0);
  const [pointerInteractionActive, setPointerInteractionActive] =
    useState(false);

  // Crosshair is visible if either keyboard keys are held or pointer interaction is active.
  const crosshairVisible =
    keyboardCrosshairCounter > 0 || pointerInteractionActive;

  // Animation state interface.
  interface CameraAnimation {
    startUp: THREE.Vector3;
    targetUp: THREE.Vector3;
    startLookAt: THREE.Vector3;
    targetLookAt: THREE.Vector3;
    startTime: number;
    duration: number;
  }

  const [cameraAnimation, setCameraAnimation] =
    useState<CameraAnimation | null>(null);

  React.useEffect(() => {
    if (!cameraLocked) return;
    heldKeysRef.current.clear();
    pointerRef.current = null;
    setKeyboardCrosshairCounter(0);
    setPointerInteractionActive(false);
    setCameraAnimation(null);
  }, [cameraLocked]);

  // Animation parameters.
  const ANIMATION_DURATION = 0.5; // seconds

  useFrame((state) => {
    if (!cameraLocked && cameraAnimation && viewerMutable.cameraControl) {
      const cameraControls = viewerMutable.cameraControl;
      const camera = cameraControls.camera;

      const elapsed = state.clock.getElapsedTime() - cameraAnimation.startTime;
      const progress = Math.min(elapsed / cameraAnimation.duration, 1);

      // Smooth step easing.
      const t = progress * progress * (3 - 2 * progress);

      // Interpolate up vector.
      const newUp = new THREE.Vector3()
        .copy(cameraAnimation.startUp)
        .lerp(cameraAnimation.targetUp, t)
        .normalize();

      // Interpolate look-at position.
      const newLookAt = new THREE.Vector3()
        .copy(cameraAnimation.startLookAt)
        .lerp(cameraAnimation.targetLookAt, t);

      camera.up.copy(newUp);

      // Back up position.
      const prevPosition = new THREE.Vector3();
      cameraControls.getPosition(prevPosition);

      cameraControls.updateCameraUp();

      // Restore position and set new look-at.
      cameraControls.setPosition(
        prevPosition.x,
        prevPosition.y,
        prevPosition.z,
        false,
      );

      cameraControls.setLookAt(
        prevPosition.x,
        prevPosition.y,
        prevPosition.z,
        newLookAt.x,
        newLookAt.y,
        newLookAt.z,
        false,
      );

      // Clear animation when complete.
      if (progress >= 1) {
        setCameraAnimation(null);
      }
    }
  });

  const { clock } = useThree();

  const updateCameraLookAtAndUpFromPivotControl = (matrix: THREE.Matrix4) => {
    if (cameraLocked) return;
    if (!viewerMutable.cameraControl) return;

    const targetPosition = new THREE.Vector3();
    targetPosition.setFromMatrixPosition(matrix);

    const cameraControls = viewerMutable.cameraControl;
    const camera = viewerMutable.cameraControl.camera;

    // Get target up vector from matrix.
    const targetUp = new THREE.Vector3().setFromMatrixColumn(matrix, 1);

    // Get current look-at position.
    const currentLookAt = cameraControls.getTarget(new THREE.Vector3());

    // Start new animation.
    setCameraAnimation({
      startUp: camera.up.clone(),
      targetUp: targetUp,
      startLookAt: currentLookAt,
      targetLookAt: targetPosition,
      startTime: clock.getElapsedTime(),
      duration: ANIMATION_DURATION,
    });
  };

  const updatePivotControlFromCameraLookAtAndup = () => {
    if (cameraAnimation !== null) return;
    if (!viewerMutable.cameraControl) return;
    if (!pivotRef.current) return;

    const cameraControls = viewerMutable.cameraControl;
    const lookAt = cameraControls.getTarget(new THREE.Vector3());

    // Rotate matrix s.t. it's y-axis aligns with the camera's up vector.
    // We'll do this with math.
    const origRotation = new THREE.Matrix4().extractRotation(
      pivotRef.current.matrix,
    );

    const cameraUp = camera.up.clone().normalize();
    const pivotUp = new THREE.Vector3(0, 1, 0)
      .applyMatrix4(origRotation)
      .normalize();
    const axis = new THREE.Vector3()
      .crossVectors(pivotUp, cameraUp)
      .normalize();
    const angle = Math.acos(Math.min(1, Math.max(-1, cameraUp.dot(pivotUp))));

    // Create rotation matrix.
    const rotationMatrix = new THREE.Matrix4();
    if (axis.lengthSq() > 0.0001) {
      // Check if cross product is valid.
      rotationMatrix.makeRotationAxis(axis, angle);
    }
    // rotationMatrix.premultiply(origRotation);

    // Combine rotation with position.
    const matrix = new THREE.Matrix4();
    matrix.multiply(rotationMatrix);
    matrix.multiply(origRotation);
    matrix.setPosition(lookAt);

    pivotRef.current.matrix.copy(matrix);
    pivotRef.current.updateMatrixWorld(true);
  };

  viewerMutable.resetCameraView = () => {
    if (getCameraStore(viewer).getState().mode !== "free") return;
    if (!initialCameraRef.current || !viewerMutable.cameraControl) return;
    camera.up.set(
      initialCameraRef.current.camera.up.x,
      initialCameraRef.current.camera.up.y,
      initialCameraRef.current.camera.up.z,
    );
    worldUpRef.current.copy(camera.up).normalize();
    lastCameraUpRef.current.copy(worldUpRef.current);
    viewerMutable.cameraControl.updateCameraUp();
    viewerMutable.cameraControl.setLookAt(
      initialCameraRef.current.camera.position.x,
      initialCameraRef.current.camera.position.y,
      initialCameraRef.current.camera.position.z,
      initialCameraRef.current.lookAt.x,
      initialCameraRef.current.lookAt.y,
      initialCameraRef.current.lookAt.z,
      true,
    );
  };

  viewerMutable.captureInitialCameraView = () => {
    if (receivedInitialServerViewRef.current || !viewerMutable.cameraControl) return;
    initialCameraRef.current = {
      camera: camera.clone(),
      lookAt: viewerMutable.cameraControl.getTarget(new THREE.Vector3()),
    };
    worldUpRef.current.copy(camera.up).normalize();
    lastCameraUpRef.current.copy(worldUpRef.current);
    receivedInitialServerViewRef.current = true;
  };

  // Callback for sending cameras.
  // It makes the code more chaotic, but we preallocate a bunch of things to
  // minimize garbage collection!
  const R_threecam_cam = new THREE.Quaternion().setFromEuler(
    new THREE.Euler(Math.PI, 0.0, 0.0),
  );
  const R_world_threeworld = new THREE.Quaternion();
  const tmpMatrix4 = new THREE.Matrix4();
  const lookAt = new THREE.Vector3();
  const R_world_camera = new THREE.Quaternion();
  const t_world_camera = new THREE.Vector3();
  const scale = new THREE.Vector3();
  const sendCamera = React.useCallback(() => {
    if (viewerMutable.applyingServerCameraBatch) return;
    updatePivotControlFromCameraLookAtAndup();

    const three_camera = camera;
    const camera_control = viewerMutable.cameraControl;
    const canvas = viewerMutable.canvas!;

    if (camera_control === null) {
      // Camera controls not yet ready, let's re-try later.
      setTimeout(sendCamera, 10);
      return;
    }

    // We put Z up to match the scene tree, and convert threejs camera convention
    // to the OpenCV one.
    const T_world_threeworld = computeT_threeworld_world(viewer).invert();
    const T_world_camera = T_world_threeworld.clone()
      .multiply(
        tmpMatrix4
          .makeRotationFromQuaternion(three_camera.quaternion)
          .setPosition(three_camera.position),
      )
      .multiply(tmpMatrix4.makeRotationFromQuaternion(R_threecam_cam));
    R_world_threeworld.setFromRotationMatrix(T_world_threeworld);

    camera_control.getTarget(lookAt).applyMatrix4(T_world_threeworld);
    const up = three_camera.up.clone().applyQuaternion(R_world_threeworld);

    // Store initial camera values.
    if (initialCameraRef.current === null) {
      initialCameraRef.current = {
        camera: three_camera.clone(),
        lookAt: camera_control.getTarget(new THREE.Vector3()),
      };
    }

    T_world_camera.decompose(t_world_camera, R_world_camera, scale);

    sendCameraThrottled({
      type: "ViewerCameraMessage",
      wxyz: [
        R_world_camera.w,
        R_world_camera.x,
        R_world_camera.y,
        R_world_camera.z,
      ],
      position: t_world_camera.toArray(),
      image_height: canvas.height,
      image_width: canvas.width,
      fov: (three_camera.fov * Math.PI) / 180.0,
      near: three_camera.near,
      far: three_camera.far,
      look_at: [lookAt.x, lookAt.y, lookAt.z],
      up_direction: [up.x, up.y, up.z],
    });

    // Log camera.
    if (logCamera) {
      console.log(
        `&initialCameraPosition=${t_world_camera.x.toFixed(
          3,
        )},${t_world_camera.y.toFixed(3)},${t_world_camera.z.toFixed(3)}` +
          `&initialCameraLookAt=${lookAt.x.toFixed(3)},${lookAt.y.toFixed(
            3,
          )},${lookAt.z.toFixed(3)}` +
          `&initialCameraUp=${up.x.toFixed(3)},${up.y.toFixed(
            3,
          )},${up.z.toFixed(3)}`,
      );
    }
  }, [camera, sendCameraThrottled]);

  // Camera control search parameters.
  // EXPERIMENTAL: these may be removed or renamed in the future. Please pin to
  // a commit/version if you're relying on this (undocumented) feature.
  const searchParams = new URLSearchParams(window.location.search);
  const initialCameraPosString = searchParams.get("initialCameraPosition");
  const initialCameraLookAtString = searchParams.get("initialCameraLookAt");
  const initialCameraUpString = searchParams.get("initialCameraUp");
  const forceOrbitOriginTool = searchParams.get("forceOrbitOriginTool") === "1";
  const logCamera = viewer.useDevSettings((state) => state.logCamera);

  // Send camera for new connections.
  // We add a small delay to give the server time to add a callback.
  const connected = viewer.useGui((state) => state.websocketConnected);
  const initialCameraPositionSet = React.useRef(false);
  React.useEffect(() => {
    if (!initialCameraPositionSet.current) {
      const initialCameraPos = new THREE.Vector3(
        ...((initialCameraPosString
          ? (initialCameraPosString.split(",").map(Number) as [
              number,
              number,
              number,
            ])
          : [3.0, 3.0, 3.0]) as [number, number, number]),
      );
      initialCameraPos.applyMatrix4(computeT_threeworld_world(viewer));
      const initialCameraLookAt = new THREE.Vector3(
        ...((initialCameraLookAtString
          ? (initialCameraLookAtString.split(",").map(Number) as [
              number,
              number,
              number,
            ])
          : [0, 0, 0]) as [number, number, number]),
      );
      initialCameraLookAt.applyMatrix4(computeT_threeworld_world(viewer));
      const initialCameraUp = new THREE.Vector3(
        ...((initialCameraUpString
          ? (initialCameraUpString.split(",").map(Number) as [
              number,
              number,
              number,
            ])
          : [0, 0, 1]) as [number, number, number]),
      );
      initialCameraUp.transformDirection(computeT_threeworld_world(viewer));
      worldUpRef.current.copy(initialCameraUp);
      lastCameraUpRef.current.copy(initialCameraUp);

      camera.up.set(initialCameraUp.x, initialCameraUp.y, initialCameraUp.z);
      viewerMutable.cameraControl!.updateCameraUp();

      viewerMutable.cameraControl!.setLookAt(
        initialCameraPos.x,
        initialCameraPos.y,
        initialCameraPos.z,
        initialCameraLookAt.x,
        initialCameraLookAt.y,
        initialCameraLookAt.z,
        false,
      );
      initialCameraPositionSet.current = true;
    }

    viewerMutable.sendCamera = sendCamera;
    if (!connected) return;
    setTimeout(() => sendCamera(), 50);
  }, [connected, sendCamera]);

  // Send camera for 3D viewport changes.
  const canvas = viewerMutable.canvas!; // R3F canvas.
  React.useEffect(() => {
    // Create a resize observer to resize the CSS canvas when the window is resized.
    const resizeObserver = new ResizeObserver(() => {
      sendCamera();
    });
    resizeObserver.observe(canvas);

    // Cleanup.
    return () => resizeObserver.disconnect();
  }, [canvas]);

  const canvasElement = useThree((state) => state.gl.domElement);
  const enableArrowKeys = viewer.cameraKeyboardControlsEnabled.value;

  const alignToWorldUp = React.useCallback(() => {
    const controls = viewerMutable.cameraControl;
    if (!controls) return;
    // A server camera-up command can arrive after initial setup. Treat that
    // direction as authoritative for this scene (StageZero uses +Y).
    if (camera.up.distanceToSquared(lastCameraUpRef.current) > 1e-6 && camera.up.lengthSq() > 1e-8) {
      worldUpRef.current.copy(camera.up).normalize();
      lastCameraUpRef.current.copy(worldUpRef.current);
    }
    if (camera.up.dot(worldUpRef.current) > 0.9999) return;
    const position = controls.getPosition(new THREE.Vector3());
    const target = controls.getTarget(new THREE.Vector3());
    camera.up.copy(worldUpRef.current);
    lastCameraUpRef.current.copy(worldUpRef.current);
    controls.updateCameraUp();
    controls.setLookAt(position.x, position.y, position.z, target.x, target.y, target.z, false);
  }, [camera, viewerMutable]);

  const translateCamera = React.useCallback((translation: THREE.Vector3) => {
    const controls = viewerMutable.cameraControl;
    if (!controls || translation.lengthSq() === 0) return;
    const position = controls.getPosition(new THREE.Vector3()).add(translation);
    const target = controls.getTarget(new THREE.Vector3()).add(translation);
    controls.setLookAt(position.x, position.y, position.z, target.x, target.y, target.z, false);
  }, [viewerMutable]);

  React.useEffect(() => {
    const canvas = canvasElement;
    canvas.tabIndex = 0;
    const controls = viewerMutable.cameraControl;
    if (!controls) return;
    // The camera-controls instance still handles pose updates and server commands.
    // Gesture input is owned here so a click cannot accidentally start a rotate.
    controls.mouseButtons.left = CameraControlsImpl.ACTION.NONE;
    controls.mouseButtons.middle = CameraControlsImpl.ACTION.NONE;
    controls.mouseButtons.right = CameraControlsImpl.ACTION.NONE;
    controls.mouseButtons.wheel = CameraControlsImpl.ACTION.NONE;
    controls.touches.one = CameraControlsImpl.ACTION.NONE;
    controls.touches.two = CameraControlsImpl.ACTION.NONE;
    controls.touches.three = CameraControlsImpl.ACTION.NONE;
    // Drei connects its default document listeners in the child's effect.
    // Their pointermove handler can suppress scene clicks even with ACTION.NONE.
    controls.disconnect();

    const onPointerDown = (event: PointerEvent) => {
      if (cameraLocked) return;
      if (event.pointerType === "touch" || ![0, 1, 2].includes(event.button)) return;
      canvas.focus({ preventScroll: true });
      if (viewerMutable.scenePointerInfo.enabled !== false) return;
      // Let clickable scene objects retain their plain left-drag behavior.
      if (event.button === 0 && !event.altKey && viewerMutable.hoveredElementsCount > 0) return;
      if (!controls.enabled) return;
      alignToWorldUp();
      const mode: NavigationMode = event.button === 1 || event.altKey
        ? "Orbit" : event.button === 2 ? "Pan" : navigationMode;
      pointerRef.current = {
        id: event.pointerId, startX: event.clientX, startY: event.clientY,
        lastX: event.clientX, lastY: event.clientY, moving: false, mode,
      };
      if (event.button !== 0) event.preventDefault();
    };
    const onPointerMove = (event: PointerEvent) => {
      if (cameraLocked) return;
      const gesture = pointerRef.current;
      if (!gesture || event.pointerId !== gesture.id || !controls.enabled) return;
      if (viewerMutable.transformControlsDraggingNames.size > 0 || viewerMutable.scenePointerInfo.enabled !== false) {
        pointerRef.current = null;
        setPointerInteractionActive(false);
        return;
      }
      if (!gesture.moving) {
        if (!isDragGesture(gesture.startX, gesture.startY, event.clientX, event.clientY)) return;
        gesture.moving = true;
        setPointerInteractionActive(true);
      }
      const dx = THREE.MathUtils.clamp(event.clientX - gesture.lastX, -80, 80);
      const dy = THREE.MathUtils.clamp(event.clientY - gesture.lastY, -80, 80);
      gesture.lastX = event.clientX;
      gesture.lastY = event.clientY;
      if (gesture.mode === "Pan") {
        translateCamera(panTranslation(
          controls.getPosition(new THREE.Vector3()),
          controls.getTarget(new THREE.Vector3()), camera.quaternion,
          camera.fov, canvas.clientHeight, dx, dy,
        ));
      } else if (gesture.mode === "Orbit") {
        controls.rotate(-dx * 0.005, -dy * 0.005, false);
      } else {
        const position = controls.getPosition(new THREE.Vector3());
        const target = lookTargetAfterDrag(
          position, controls.getTarget(new THREE.Vector3()),
          worldUpRef.current, dx, dy,
        );
        controls.setLookAt(position.x, position.y, position.z, target.x, target.y, target.z, false);
      }
      if (event.cancelable) event.preventDefault();
    };
    const onPointerEnd = (event: PointerEvent) => {
      if (pointerRef.current?.id !== event.pointerId) return;
      pointerRef.current = null;
      setPointerInteractionActive(false);
    };
    const onWindowBlur = () => {
      pointerRef.current = null;
      setPointerInteractionActive(false);
    };
    const onWheel = (event: WheelEvent) => {
      if (cameraLocked) { event.preventDefault(); return; }
      if (!controls.enabled) return;
      event.preventDefault();
      alignToWorldUp();
      const factor = event.deltaMode === WheelEvent.DOM_DELTA_LINE ? 16
        : event.deltaMode === WheelEvent.DOM_DELTA_PAGE ? canvas.clientHeight : 1;
      if (event.ctrlKey || event.metaKey || event.altKey) {
        const position = controls.getPosition(new THREE.Vector3());
        const target = controls.getTarget(new THREE.Vector3());
        controls.dollyTo(wheelDollyDistance(position.distanceTo(target), event.deltaY * factor), false);
      } else {
        translateCamera(panTranslation(
          controls.getPosition(new THREE.Vector3()),
          controls.getTarget(new THREE.Vector3()), camera.quaternion,
          camera.fov, canvas.clientHeight,
          THREE.MathUtils.clamp(event.deltaX * factor, -150, 150),
          THREE.MathUtils.clamp(event.deltaY * factor, -150, 150),
        ));
      }
    };
    const onContextMenu = (event: MouseEvent) => event.preventDefault();
    canvas.addEventListener("pointerdown", onPointerDown, { capture: true });
    document.addEventListener("pointermove", onPointerMove, { capture: true });
    document.addEventListener("pointerup", onPointerEnd, { capture: true });
    document.addEventListener("pointercancel", onPointerEnd, { capture: true });
    window.addEventListener("blur", onWindowBlur);
    canvas.addEventListener("wheel", onWheel, { passive: false, capture: true });
    canvas.addEventListener("contextmenu", onContextMenu);
    return () => {
      canvas.removeEventListener("pointerdown", onPointerDown, true);
      document.removeEventListener("pointermove", onPointerMove, true);
      document.removeEventListener("pointerup", onPointerEnd, true);
      document.removeEventListener("pointercancel", onPointerEnd, true);
      window.removeEventListener("blur", onWindowBlur);
      canvas.removeEventListener("wheel", onWheel, true);
      canvas.removeEventListener("contextmenu", onContextMenu);
      pointerRef.current = null;
      setPointerInteractionActive(false);
    };
  }, [alignToWorldUp, camera, cameraLocked, canvasElement, navigationMode, translateCamera, viewerMutable]);

  React.useEffect(() => {
    const canvas = canvasElement;
    const onKeyDown = (event: KeyboardEvent) => {
      if (cameraLocked) return;
      if (event.ctrlKey || event.metaKey || event.altKey) {
        heldKeysRef.current.clear();
        setKeyboardCrosshairCounter(0);
        return;
      }
      const editable = event.target instanceof HTMLElement &&
        (event.target.isContentEditable || !!event.target.closest("input,textarea,select,[contenteditable]"));
      if (editable) return;
      const allowed = ["KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE"];
      if (enableArrowKeys) allowed.push("ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight");
      if (!allowed.includes(event.code)) return;
      event.preventDefault();
      heldKeysRef.current.add(event.code);
      setKeyboardCrosshairCounter(heldKeysRef.current.size);
    };
    const onKeyUp = (event: KeyboardEvent) => {
      heldKeysRef.current.delete(event.code);
      setKeyboardCrosshairCounter(heldKeysRef.current.size);
    };
    const onBlur = () => {
      heldKeysRef.current.clear();
      setKeyboardCrosshairCounter(0);
    };
    const onVisibilityChange = () => {
      if (document.hidden) onBlur();
    };
    canvas.addEventListener("keydown", onKeyDown);
    // Keyup may arrive on another element after focus changes mid-press.
    window.addEventListener("keyup", onKeyUp);
    canvas.addEventListener("blur", onBlur);
    window.addEventListener("blur", onBlur);
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      canvas.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      canvas.removeEventListener("blur", onBlur);
      window.removeEventListener("blur", onBlur);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      onBlur();
    };
  }, [cameraLocked, canvasElement, enableArrowKeys]);

  useFrame((_, delta) => {
    if (cameraLocked) return;
    const keys = heldKeysRef.current;
    const controls = viewerMutable.cameraControl;
    if (keys.size === 0 || !controls || !controls.enabled) return;
    alignToWorldUp();
    const position = controls.getPosition(new THREE.Vector3());
    const target = controls.getTarget(new THREE.Vector3());
    const speed = Math.max(0.5, position.distanceTo(target) * 0.7) * Math.min(delta, 0.05);
    const translation = walkTranslation(position, target, worldUpRef.current,
      Number(keys.has("KeyW")) - Number(keys.has("KeyS")),
      Number(keys.has("KeyD")) - Number(keys.has("KeyA")),
      Number(keys.has("KeyE")) - Number(keys.has("KeyQ")),
    ).multiplyScalar(speed);
    translateCamera(translation);
    if (enableArrowKeys) {
      const yaw = Number(keys.has("ArrowRight")) - Number(keys.has("ArrowLeft"));
      const pitch = Number(keys.has("ArrowDown")) - Number(keys.has("ArrowUp"));
      if (yaw || pitch) controls.rotate(yaw * delta, pitch * delta, false);
    }
  });

  return (
    <>
      <CameraControls
        ref={(controls) => (viewerMutable.cameraControl = controls)}
        minDistance={0.01}
        enabled={!cameraLocked}
        minPolarAngle={0.06}
        maxPolarAngle={Math.PI - 0.06}
        dollySpeed={0.3}
        smoothTime={0.05}
        draggingSmoothTime={0.0}
        onChange={sendCamera}
        onStart={() => {
          setPointerInteractionActive(true);
        }}
        onEnd={() => {
          setPointerInteractionActive(false);
        }}
        makeDefault
      />
      {!cameraLocked && <NavigationOverlay canvas={canvasElement} mode={navigationMode} setMode={setNavigationMode} />}
      <OrbitOriginTool
        forceShow={forceOrbitOriginTool}
        locked={cameraLocked}
        pivotRef={pivotRef}
        onPivotChange={(matrix) => {
          updateCameraLookAtAndUpFromPivotControl(matrix);
        }}
        update={updatePivotControlFromCameraLookAtAndup}
        crosshairVisible={crosshairVisible}
      />
    </>
  );
}
