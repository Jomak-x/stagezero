import { create } from "zustand";
import type { StoreApi, UseBoundStore } from "zustand";
import type { ViewerContextContents } from "./ViewerContext";
import type {
  CameraAction,
  CameraCommandPayload,
  CameraStudioCommandMessage,
  CameraStudioStateMessage,
} from "./cameraProtocol";

export type CameraStoreState = Omit<CameraStudioStateMessage, "type"> & {
  /** False until this viewer receives the first authoritative camera state. */
  received: boolean;
};

const emptyState = (): CameraStoreState => ({
  received: false,
  project_id: "",
  revision: 0,
  take_id: null,
  frame: 0,
  length: 0,
  fps: 30,
  cameras: [],
  cuts: [],
  mode: "free",
  active_camera_id: null,
  selected_camera_id: null,
  busy: false,
  error: null,
});

const cameraStores = new WeakMap<object, UseBoundStore<StoreApi<CameraStoreState>>>();

/** One store per mounted viewer, including file-playback and multi-viewer pages. */
export function getCameraStore(viewer: ViewerContextContents) {
  const key = viewer.mutable.current;
  let store = cameraStores.get(key);
  if (!store) {
    store = create<CameraStoreState>(() => emptyState());
    cameraStores.set(key, store);
  }
  return store;
}

export function applyCameraState(
  viewer: ViewerContextContents,
  message: CameraStudioStateMessage,
) {
  const store = getCameraStore(viewer);
  const previous = store.getState();
  // A delayed message from the same project cannot undo a later edit. A
  // different project id is an authoritative reset even if its revision is 0.
  if (
    previous.received &&
    previous.project_id === message.project_id &&
    message.revision < previous.revision
  ) return;
  store.setState({
    received: true,
    project_id: message.project_id,
    revision: message.revision,
    take_id: message.take_id,
    frame: message.frame,
    length: message.length,
    fps: message.fps,
    cameras: message.cameras,
    cuts: message.cuts,
    mode: message.mode,
    active_camera_id: message.active_camera_id,
    selected_camera_id: message.selected_camera_id,
    busy: message.busy,
    error: message.error,
  });
}

export function resetCameraState(viewer: ViewerContextContents) {
  getCameraStore(viewer).setState(emptyState());
}

/** Send a command against the last authoritative project/take revision. */
export function sendCameraCommand(
  viewer: ViewerContextContents,
  action: CameraAction,
  payload: CameraCommandPayload = {},
): boolean {
  const state = getCameraStore(viewer).getState();
  if (!state.received) return false;
  const command: CameraStudioCommandMessage = {
    type: "CameraStudioCommandMessage",
    project_id: state.project_id,
    revision: state.revision,
    take_id: state.take_id,
    action,
    ...payload,
  };
  // A retry should show its own server response even when it repeats the
  // previous error text. The next state remains authoritative.
  if (state.error !== null) getCameraStore(viewer).setState({ error: null });
  viewer.mutable.current.sendMessage(command);
  return true;
}
