import test from "node:test";
import assert from "node:assert/strict";
import { boundedCutFrame, boundedPlayheadFrame, cameraCutPreviewSegments, cameraCutSegments, frameAfterCutDrag, visibleCameraBlock } from "./cameraTimelineMath.ts";

test("camera cuts form hard-cut durations ending at the take length", () => {
  const cuts = [
    { id: "b", frame: 42, camera_id: "side" },
    { id: "a", frame: 0, camera_id: "front" },
  ];
  const segments = cameraCutSegments(cuts, 90);
  assert.deepEqual(segments.map(({ cut, endFrame, firstCut }) => [cut.id, endFrame - cut.frame, firstCut]), [
    ["a", 42, true], ["b", 48, false],
  ]);
  assert.deepEqual(cuts.map((cut) => cut.id), ["b", "a"]);
});

test("a dragged cut replaces an occupied frame in either direction", () => {
  const cuts = [
    { id: "anchor", frame: 0, camera_id: "wide" },
    { id: "A", frame: 5, camera_id: "left" },
    { id: "B", frame: 10, camera_id: "right" },
  ];
  const forward = cameraCutPreviewSegments(cuts, 20, { id: "A", frame: 10 });
  assert.deepEqual(forward.map(({ cut, endFrame }) => [cut.id, cut.frame, endFrame]), [
    ["anchor", 0, 10], ["A", 10, 20],
  ]);
  const backward = cameraCutPreviewSegments(cuts, 20, { id: "B", frame: 5 });
  assert.deepEqual(backward.map(({ cut, endFrame }) => [cut.id, cut.frame, endFrame]), [
    ["anchor", 0, 5], ["B", 5, 20],
  ]);
  assert.deepEqual(cuts.map((cut) => cut.frame), [0, 5, 10]);
});

test("the final one-frame cut keeps a clickable block at the edge", () => {
  assert.deepEqual(visibleCameraBlock(299.4, 300, 92, 300), { left: 288, width: 12 });
  assert.deepEqual(visibleCameraBlock(300.5, 301, 92, 300, true), { left: 288, width: 12 });
  assert.equal(visibleCameraBlock(300.5, 301, 92, 300), null);
});

test("dragging snaps to whole frames and stays within the take", () => {
  assert.equal(frameAfterCutDrag(10, 25, 10, 50), 13);
  assert.equal(frameAfterCutDrag(10, -999, 10, 50), 1);
  assert.equal(frameAfterCutDrag(10, 999, 10, 50), 49);
  assert.equal(boundedCutFrame(23, 50, true), 0);
  assert.equal(boundedPlayheadFrame(0, 50), 0);
  assert.equal(boundedPlayheadFrame(17, 50), 17);
  assert.equal(boundedPlayheadFrame(999, 50), 49);
});
