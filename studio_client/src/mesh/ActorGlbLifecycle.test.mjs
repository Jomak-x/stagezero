import test from "node:test";
import assert from "node:assert/strict";

import {
  committedLoadToRetry,
  rejectionCancelsInFlightLoad,
} from "./ActorGlbLifecycle.ts";

test("a failed replacement retries an interrupted committed load", () => {
  const previous = { revision: 4, asset_id: "actor-a" };
  assert.equal(committedLoadToRetry(previous, null, 5), previous);
  assert.equal(committedLoadToRetry(previous, 4, 5), null);
  assert.equal(committedLoadToRetry(previous, null, 4), null);
  assert.equal(committedLoadToRetry(null, null, 5), null);
});

test("rejecting B leaves an in-flight retry of committed A intact", () => {
  const actorA = { revision: 4, asset_id: "actor-a" };
  const retry = committedLoadToRetry(actorA, null, 5);
  assert.equal(retry, actorA);

  // B failed locally, so the browser began parsing A again. The later server
  // reject(B) must not invalidate A's parse token.
  let parseToken = 12;
  const inFlightRevision = retry.revision;
  if (rejectionCancelsInFlightLoad(5, inFlightRevision)) parseToken++;
  assert.equal(parseToken, 12);
  assert.equal(rejectionCancelsInFlightLoad(5, 5), true);
  assert.equal(rejectionCancelsInFlightLoad(5, null), false);
});
