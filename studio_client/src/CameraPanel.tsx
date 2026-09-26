import React from "react";
import * as THREE from "three";
import { ViewerContext } from "./ViewerContext";
import { getCameraStore, sendCameraCommand } from "./cameraStore";
import type { CameraRecord } from "./cameraProtocol";
import "./CameraPanel.css";

type CameraDraft = {
  name: string;
  position: [string, string, string];
  rotation: [string, string, string];
  fov: string;
};

const degrees = (radians: number) => (radians * 180) / Math.PI;
const radians = (degreesValue: number) => (degreesValue * Math.PI) / 180;
const displayNumber = (value: number) => String(Number(value.toFixed(3)));

function draftFromCamera(camera: CameraRecord): CameraDraft {
  const rotation = new THREE.Euler().setFromQuaternion(
    new THREE.Quaternion(camera.wxyz[1], camera.wxyz[2], camera.wxyz[3], camera.wxyz[0]),
    "XYZ",
  );
  return {
    name: camera.name,
    position: camera.position.map(displayNumber) as CameraDraft["position"],
    rotation: [rotation.x, rotation.y, rotation.z].map((v) => displayNumber(degrees(v))) as CameraDraft["rotation"],
    fov: displayNumber(degrees(camera.fov)),
  };
}

function CameraEditor({ camera }: { camera: CameraRecord }) {
  const viewer = React.useContext(ViewerContext)!;
  const [draft, setDraft] = React.useState(() => draftFromCamera(camera));
  const [localError, setLocalError] = React.useState<string | null>(null);
  const cameraSignature = [camera.id, camera.name, ...camera.position, ...camera.wxyz, camera.fov].join("|");
  React.useEffect(() => {
    setDraft(draftFromCamera(camera));
    setLocalError(null);
  }, [cameraSignature]);

  const setVectorValue = (key: "position" | "rotation", index: number, value: string) => {
    setDraft((current) => {
      const next = [...current[key]] as [string, string, string];
      next[index] = value;
      return { ...current, [key]: next };
    });
  };

  const apply = (event: React.FormEvent) => {
    event.preventDefault();
    const position = draft.position.map(Number);
    const angles = draft.rotation.map(Number);
    const fovDegrees = Number(draft.fov);
    if (
      !draft.name.trim() ||
      [...draft.position, ...draft.rotation, draft.fov].some((value) => value.trim() === "") ||
      [...position, ...angles, fovDegrees].some((value) => !Number.isFinite(value)) ||
      fovDegrees <= 1 || fovDegrees >= 179
    ) {
      setLocalError("Enter a name, finite coordinates and angles, and a FOV between 1° and 179°.");
      return;
    }
    const q = new THREE.Quaternion().setFromEuler(
      new THREE.Euler(radians(angles[0]), radians(angles[1]), radians(angles[2]), "XYZ"),
    );
    setLocalError(null);
    sendCameraCommand(viewer, "update", {
      camera_id: camera.id,
      camera: {
        name: draft.name.trim(),
        position: position as [number, number, number],
        wxyz: [q.w, q.x, q.y, q.z],
        fov: radians(fovDegrees),
      },
    });
  };

  return (
    <form className="sz-camera-editor" onSubmit={apply}>
      <label className="sz-camera-field">
        <span>Name</span>
        <input aria-label="Camera name" value={draft.name} onChange={(event) => setDraft({ ...draft, name: event.target.value })} />
      </label>
      <fieldset className="sz-camera-vector">
        <legend>Position</legend>
        {(["X", "Y", "Z"] as const).map((axis, index) => (
          <label key={axis}>
            <span>{axis}</span>
            <input type="number" step="any" aria-label={`Position ${axis}`} value={draft.position[index]} onChange={(event) => setVectorValue("position", index, event.target.value)} />
          </label>
        ))}
      </fieldset>
      <fieldset className="sz-camera-vector">
        <legend>Rotation · degrees</legend>
        {(["X", "Y", "Z"] as const).map((axis, index) => (
          <label key={axis}>
            <span>{axis}</span>
            <input type="number" step="any" aria-label={`Rotation ${axis} degrees`} value={draft.rotation[index]} onChange={(event) => setVectorValue("rotation", index, event.target.value)} />
          </label>
        ))}
      </fieldset>
      <label className="sz-camera-field sz-camera-fov">
        <span>Field of view · degrees</span>
        <input type="number" step="any" min="1" max="179" aria-label="Field of view degrees" value={draft.fov} onChange={(event) => setDraft({ ...draft, fov: event.target.value })} />
      </label>
      {localError && <p className="sz-camera-error" role="alert">{localError}</p>}
      <div className="sz-camera-actions">
        <button type="submit" className="sz-camera-primary">Apply changes</button>
        <button type="button" onClick={() => sendCameraCommand(viewer, "view", { camera_id: camera.id })}>View</button>
        <button type="button" onClick={() => sendCameraCommand(viewer, "duplicate", { camera_id: camera.id })}>Duplicate</button>
        <button type="button" className="sz-camera-danger" onClick={() => {
          if (window.confirm(`Delete camera “${camera.name}”? Clear any cuts using it first.`)) {
            sendCameraCommand(viewer, "delete", { camera_id: camera.id });
          }
        }}>Delete</button>
      </div>
    </form>
  );
}

/** Mounted at the exact server HTML marker in the generated inspector. */
export function CameraPanel() {
  const viewer = React.useContext(ViewerContext)!;
  const store = React.useMemo(() => getCameraStore(viewer), [viewer.mutable]);
  const state = store((current) => current);
  if (!state.received) return null;
  const selected = state.cameras.find((camera) => camera.id === state.selected_camera_id) ?? null;

  return (
    <section className="sz-camera-panel" aria-label="Cameras">
      <div className="sz-camera-header">
        <div>
          <h3>Cameras</h3>
          <span>{state.mode === "sequence" ? "Take sequence" : state.mode === "fixed" ? "Fixed view" : "Free view"}</span>
        </div>
        <button type="button" className="sz-camera-primary" onClick={() => sendCameraCommand(viewer, "add")}>+ Add camera</button>
      </div>
      <div className="sz-camera-mode" role="group" aria-label="Camera viewing mode">
        <button type="button" aria-pressed={state.mode === "free"} onClick={() => sendCameraCommand(viewer, "free")}>Free</button>
        <button type="button" aria-pressed={state.mode === "sequence"} disabled={!state.take_id || state.cuts.length === 0} onClick={() => sendCameraCommand(viewer, "sequence")}>Sequence</button>
      </div>
      {state.busy && <p className="sz-camera-note">Motion generation is running. Camera edits are still available.</p>}
      {state.error && <p className="sz-camera-error" role="alert">{state.error}</p>}
      {state.cameras.length === 0 ? (
        <p className="sz-camera-empty">Add a camera to save the current view. Then place its cuts on the timeline.</p>
      ) : (
        <div className="sz-camera-list" role="list" aria-label="Saved cameras">
          {state.cameras.map((camera) => (
            <div className="sz-camera-item" role="listitem" key={camera.id} data-selected={camera.id === state.selected_camera_id}>
              <button type="button" className="sz-camera-name" aria-pressed={camera.id === state.selected_camera_id} onClick={() => sendCameraCommand(viewer, "select", { camera_id: camera.id })} title={`Edit ${camera.name}`}>
                <span className="sz-camera-dot" aria-hidden="true" />
                <span>{camera.name}</span>
                {camera.id === state.active_camera_id && <small>Live</small>}
              </button>
              <button type="button" className="sz-camera-item-view" onClick={() => sendCameraCommand(viewer, "view", { camera_id: camera.id })}>View</button>
            </div>
          ))}
        </div>
      )}
      {selected && <CameraEditor key={selected.id} camera={selected} />}
    </section>
  );
}
