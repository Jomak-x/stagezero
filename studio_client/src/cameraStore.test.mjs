import test from "node:test";
import assert from "node:assert/strict";
import { applyCameraState, getCameraStore, resetCameraState, sendCameraCommand } from "./cameraStore.ts";

function viewer() {
  const messages = [];
  return { mutable: { current: { sendMessage: (message) => messages.push(message) } }, messages };
}

function state(project_id, revision, overrides = {}) {
  return {
    type: "CameraStudioStateMessage", project_id, revision, take_id: "take-1",
    frame: 12, length: 100, fps: 30, cameras: [], cuts: [], mode: "free",
    active_camera_id: null, selected_camera_id: null, busy: false, error: null,
    ...overrides,
  };
}

test("camera state and command envelopes are isolated per viewer", () => {
  const first = viewer();
  const second = viewer();
  assert.equal(sendCameraCommand(first, "add"), false);
  applyCameraState(first, state("project-a", 4));
  applyCameraState(second, state("project-b", 9, { take_id: "take-b" }));
  sendCameraCommand(first, "cut_add", { camera_id: "cam-a", frame: 12 });
  sendCameraCommand(second, "free");
  assert.deepEqual(first.messages[0], {
    type: "CameraStudioCommandMessage", project_id: "project-a", revision: 4,
    take_id: "take-1", action: "cut_add", camera_id: "cam-a", frame: 12,
  });
  assert.equal(second.messages[0].project_id, "project-b");
  assert.equal(second.messages[0].take_id, "take-b");
  resetCameraState(first);
  assert.equal(getCameraStore(first).getState().received, false);
  assert.equal(getCameraStore(second).getState().revision, 9);
});

test("older same-project states cannot roll back a camera edit", () => {
  const client = viewer();
  applyCameraState(client, state("project-a", 8, { mode: "fixed" }));
  applyCameraState(client, state("project-a", 7, { mode: "free" }));
  assert.equal(getCameraStore(client).getState().mode, "fixed");
  applyCameraState(client, state("project-new", 0, { mode: "free" }));
  assert.equal(getCameraStore(client).getState().project_id, "project-new");
});
