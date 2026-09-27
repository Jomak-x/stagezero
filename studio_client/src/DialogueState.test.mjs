import test from "node:test";
import assert from "node:assert/strict";
import {
  receiveDialogueState, receiveDialogueStatus, receiveDialogueVoices,
  resetDialogueState, retryDialogueLine, submitDialogueLine, useDialogueState,
} from "./ControlPanel/DialogueState.ts";

const line = (request_id, status = "queued") => ({
  line_id: `line-${request_id}`, request_id, take_id: "take-1", start_frame: 12,
  character_id: "main", text: "Hello there.", voice_id: "female-1",
  status, detail: status, retryable: false,
});
const snapshot = (lines) => ({
  take_id: "take-1", take_name: "Intro", frame: 12,
  characters: [{ id: "main", name: "Character" }], lines,
  available: true, detail: "",
});

test("an older snapshot keeps a newly submitted line until acknowledged", () => {
  resetDialogueState();
  submitDialogueLine(line("first"));
  receiveDialogueState(snapshot([]));
  assert.deepEqual(useDialogueState.getState().lines.map((item) => item.request_id), ["first"]);

  receiveDialogueState(snapshot([line("first", "generating")]));
  assert.deepEqual(useDialogueState.getState().lines.map((item) => item.request_id), ["first"]);
  assert.equal(useDialogueState.getState().lines[0].status, "generating");
});

test("a newer status wins over an older snapshot for the same attempt", () => {
  resetDialogueState();
  submitDialogueLine(line("second"));
  receiveDialogueStatus({ request_id: "second", line_id: "saved-second",
    status: "completed", detail: "Ready", retryable: false });
  receiveDialogueState(snapshot([line("second", "queued")]));
  assert.equal(useDialogueState.getState().lines[0].status, "completed");
  assert.equal(useDialogueState.getState().lines[0].line_id, "saved-second");

  receiveDialogueState(snapshot([{ ...line("second", "completed"), line_id: "saved-second" }]));
  assert.equal(useDialogueState.getState().lines[0].status, "completed");
});

test("catalog and take eligibility work in either arrival order", () => {
  resetDialogueState();
  receiveDialogueState(snapshot([]));
  assert.equal(useDialogueState.getState().available, false);
  receiveDialogueVoices({ voices: [{ id: "female-1", name: "Voice 1", gender: "female" }],
    available: true, detail: "" });
  assert.equal(useDialogueState.getState().available, true);
  receiveDialogueState({ ...snapshot([]), available: false, detail: "Select a take" });
  assert.equal(useDialogueState.getState().available, false);
});

test("retry stays queued through a stale saved-line snapshot", () => {
  resetDialogueState();
  receiveDialogueState(snapshot([line("old", "failed")]));
  retryDialogueLine("line-old", "new");
  receiveDialogueState(snapshot([line("old", "failed")]));
  assert.equal(useDialogueState.getState().lines.length, 1);
  assert.equal(useDialogueState.getState().lines[0].request_id, "new");
  assert.equal(useDialogueState.getState().lines[0].status, "queued");
  receiveDialogueState(snapshot([{ ...line("new", "generating"), line_id: "line-old" }]));
  assert.equal(useDialogueState.getState().lines[0].status, "generating");
});
