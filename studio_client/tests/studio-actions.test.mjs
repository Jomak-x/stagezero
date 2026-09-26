import test from "node:test";
import assert from "node:assert/strict";
import { parseStudioActionUuid, studioActionMessage } from "../src/timeline/studioActions.ts";

test("studio action identifiers carry the take, index, and command address", () => {
  assert.deepEqual(parseStudioActionUuid("stagezero|take-7|2|gui-42"), {
    takeId: "take-7", index: 2, commandUuid: "gui-42",
  });
  assert.equal(parseStudioActionUuid("stagezero-segment-2"), null);
  assert.equal(parseStudioActionUuid("stagezero|take-7|-1|gui-42"), null);
  assert.equal(parseStudioActionUuid("stagezero|take-7|2|gui-42|extra"), null);
  assert.equal(parseStudioActionUuid("stagezero|take-7|2|")?.commandUuid, "");
});

test("each timeline command uses the button update protocol", () => {
  const action = parseStudioActionUuid("stagezero|take-7|2|gui-42");
  assert.ok(action);
  for (const operation of ["replace", "insert_before", "insert_after"]) {
    const message = studioActionMessage(action, operation, "nonce-1");
    assert.equal(message.type, "GuiUpdateMessage");
    assert.equal(message.uuid, "gui-42");
    assert.deepEqual(JSON.parse(message.updates.value), {
      take_id: "take-7", index: 2, operation, nonce: "nonce-1",
    });
  }
});

test("transport commands use action zero and the same update protocol", () => {
  const action = parseStudioActionUuid("stagezero|take-7|0|gui-42");
  assert.ok(action);
  for (const operation of ["start", "play", "pause"]) {
    const message = studioActionMessage(action, operation, "transport-1");
    assert.equal(message.type, "GuiUpdateMessage");
    assert.equal(message.uuid, "gui-42");
    assert.deepEqual(JSON.parse(message.updates.value), {
      take_id: "take-7", index: 0, operation, nonce: "transport-1",
    });
  }
});
