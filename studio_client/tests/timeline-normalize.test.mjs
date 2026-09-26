import test from 'node:test';
import assert from 'node:assert/strict';

import { normalizeTimelineMessage } from '../src/timeline/normalize.ts';

const timeline = () => ({
  type: 'TimelineMessage',
  enabled: true,
  fps: 25,
  start_frame: 0,
  end_frame: 100,
  current_frame: 1,
  prompts: [
    { uuid: 'later', text: 'Walk', start_frame: 51, end_frame: 100, color: [72, 202, 183] },
    { uuid: 'first', text: 'Wave', start_frame: 0, end_frame: 50, color: [96, 165, 250] },
  ],
  tracks: [
    { uuid: 'actor', name: 'Actor', track_type: 'value', color: [1, 2, 3], height_scale: 1 },
    { uuid: 'camera', name: 'Camera', track_type: 'value', color: null, height_scale: 1 },
  ],
  keyframes: [
    { uuid: 'pose-1', track_id: 'actor', frame: 1, value: 0.5, opacity: 1, locked: false },
  ],
  intervals: [
    { uuid: 'clip-1', track_id: 'actor', start_frame: 0, end_frame: 50, value: null, opacity: 1, locked: false },
  ],
  mode: 'splittext',
  constraints_enabled: false,
  default_text: '',
  default_duration: 25,
  min_prompt_duration: 1,
  max_prompt_duration: null,
  default_num_frames_zoom: 101,
  max_frames_zoom: 101,
});

const clone = (value) => structuredClone(value);

test('playhead-only update reuses all static timeline arrays', () => {
  const first = normalizeTimelineMessage(timeline());
  const nextMessage = timeline();
  nextMessage.current_frame = 2;
  const next = normalizeTimelineMessage(nextMessage, first);

  assert.notStrictEqual(next, first);
  assert.equal(next.current_frame, 2);
  for (const key of ['prompts', 'tracks', 'keyframes', 'intervals']) {
    assert.strictEqual(next[key], first[key], `${key} should retain its identity`);
  }
  assert.deepEqual(next.prompts.map((prompt) => prompt.uuid), ['first', 'later']);
});

test('a changed entry invalidates only its array and keeps the new value', () => {
  const first = normalizeTimelineMessage(timeline());
  for (const [key, change, expected] of [
    ['prompts', (message) => { message.prompts[1].text = 'Dance'; }, 'Dance'],
    ['tracks', (message) => { message.tracks[0].color[0] = 9; }, 9],
    ['keyframes', (message) => { message.keyframes[0].opacity = 0.4; }, 0.4],
    ['intervals', (message) => { message.intervals[0].end_frame = 60; }, 60],
  ]) {
    const message = timeline();
    change(message);
    const next = normalizeTimelineMessage(message, first);
    assert.notStrictEqual(next[key], first[key], `${key} changed`);
    for (const other of ['prompts', 'tracks', 'keyframes', 'intervals']) {
      if (other !== key) assert.strictEqual(next[other], first[other], `${other} unchanged`);
    }
    const actual = {
      prompts: () => next.prompts[0].text,
      tracks: () => next.tracks[0].color[0],
      keyframes: () => next.keyframes[0].opacity,
      intervals: () => next.intervals[0].end_frame,
    }[key]();
    assert.equal(actual, expected);
  }
});

test('entry order, prompt constraints, and changed scalar fields survive reuse', () => {
  const first = normalizeTimelineMessage(timeline());
  const message = clone(timeline());
  message.tracks.reverse();
  message.prompts[1].end_frame = -4;
  message.min_prompt_duration = 3;
  message.fps = 30;
  message.enabled = false;
  message.current_frame = 500;
  const next = normalizeTimelineMessage(message, first);

  assert.deepEqual(next.tracks.map((track) => track.uuid), ['camera', 'actor']);
  assert.notStrictEqual(next.tracks, first.tracks);
  assert.equal(next.prompts[0].start_frame, -4);
  assert.equal(next.prompts[0].end_frame, 0);
  assert.notStrictEqual(next.prompts, first.prompts);
  assert.equal(next.fps, 30);
  assert.equal(next.enabled, false);
  assert.equal(next.current_frame, 100);
});

test('layout and configuration changes invalidate every static array even if input arrays are reused', () => {
  const first = normalizeTimelineMessage(timeline());
  const changes = [
    ['fps', (message) => { message.fps = 30; }],
    ['proportional zoom', (message) => {
      message.default_num_frames_zoom = 202;
      message.max_frames_zoom = 202;
    }],
    ['start frame', (message) => { message.start_frame = 1; }],
    ['end frame', (message) => { message.end_frame = 120; }],
    ['minimum prompt duration', (message) => { message.min_prompt_duration = 2; }],
    ['maximum prompt duration', (message) => { message.max_prompt_duration = 60; }],
    ['visibility', (message) => { message.enabled = false; }],
    ['mode', (message) => { message.mode = 'other'; }],
    ['constraints', (message) => { message.constraints_enabled = true; }],
    ['default text', (message) => { message.default_text = 'New'; }],
    ['default duration', (message) => { message.default_duration = 20; }],
  ];

  for (const [label, change] of changes) {
    const message = { ...first, current_frame: 2 };
    change(message);
    const next = normalizeTimelineMessage(message, first);
    for (const key of ['prompts', 'tracks', 'keyframes', 'intervals']) {
      assert.notStrictEqual(next[key], first[key], `${label} must invalidate ${key}`);
    }
  }
});
