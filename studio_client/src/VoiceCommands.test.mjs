import test from "node:test";
import assert from "node:assert/strict";
import { receiveVoiceQueue, receiveVoiceStatus, resetVoiceCommands, submitVoiceRequest, useVoiceCommands } from "./VoiceCommands.ts";

const request = (request_id, transcript) => ({
  request_id, status: "submitting", detail: "Sending direction…", transcript,
  retryable: false, target: "auto",
});

test("queue snapshots preserve newer local requests until acknowledged", () => {
  resetVoiceCommands();
  submitVoiceRequest(request("first", "Generate scene one"));
  submitVoiceRequest(request("second", "Generate scene two"));
  receiveVoiceQueue({ type: "VoiceQueueMessage", requests: [
    { ...request("first", "Generate scene one"), status: "running", detail: "Generating" },
  ] });
  assert.deepEqual(useVoiceCommands.getState().requests.map((item) => item.request_id), ["first", "second"]);
  assert.equal(useVoiceCommands.getState().activeRequestId, "second");

  receiveVoiceStatus({ type: "VoiceStatusMessage", request_id: "first", status: "completed",
    detail: "Done", transcript: "Generate scene one", retryable: false });
  assert.equal(useVoiceCommands.getState().status.request_id, "second",
    "an older request must not replace the latest status");
  assert.equal(useVoiceCommands.getState().requests[0].status, "completed");

  receiveVoiceQueue({ type: "VoiceQueueMessage", requests: [
    { ...request("first", "Generate scene one"), status: "completed", detail: "Done" },
    { ...request("second", "Generate scene two"), status: "queued", detail: "Queued" },
  ] });
  assert.equal(useVoiceCommands.getState().status.status, "queued");
  assert.deepEqual(useVoiceCommands.getState().requests.map((item) => item.request_id), ["first", "second"]);
  resetVoiceCommands();
  assert.deepEqual(useVoiceCommands.getState().requests, []);
});
