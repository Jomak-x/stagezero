import test from 'node:test';
import assert from 'node:assert/strict';
import { createLoadGuard } from '../src/mesh/GlbLoadGuard.ts';

test('an older GLB result is discarded after a replacement starts', () => {
  const accepted = [];
  const disposed = [];
  const first = createLoadGuard(value => accepted.push(value), value => disposed.push(value));
  first.cancel();
  const second = createLoadGuard(value => accepted.push(value), value => disposed.push(value));

  second.complete('new scene');
  first.complete('old scene');

  assert.deepEqual(accepted, ['new scene']);
  assert.deepEqual(disposed, ['old scene']);
});

test('unmounted GLB results are disposed and stale errors can be suppressed', () => {
  const accepted = [];
  const disposed = [];
  const request = createLoadGuard(value => accepted.push(value), value => disposed.push(value));
  assert.equal(request.active, true);
  request.cancel();
  assert.equal(request.active, false);
  request.complete('late scene');
  assert.deepEqual(accepted, []);
  assert.deepEqual(disposed, ['late scene']);
});
