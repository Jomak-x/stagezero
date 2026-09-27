import test from "node:test";
import assert from "node:assert/strict";
import { VoicePlayback, useVoiceView } from "./VoicePlayback.ts";

class FakeSource {
  buffer = null;
  playbackRate = { value: 1 };
  onended = null;
  started = null;
  stopped = false;
  connect() {}
  disconnect() {}
  start(when, offset, duration) { this.started = { when, offset, duration, speed: this.playbackRate.value }; }
  stop() { this.stopped = true; }
}

class FakeAudioContext {
  static instances = [];
  static decoder = async () => ({ duration: 5 });
  currentTime = 0;
  state = "running";
  destination = {};
  sources = [];
  constructor() { FakeAudioContext.instances.push(this); }
  resume() { this.state = "running"; return Promise.resolve(); }
  decodeAudioData(bytes) { return FakeAudioContext.decoder(bytes); }
  createBufferSource() {
    const source = new FakeSource();
    this.sources.push(source);
    return source;
  }
}
globalThis.AudioContext = FakeAudioContext;

const cue = { text: "Follow me.", voice_id: "voice", audio_id: "line", start_frame: 0, end_frame: 125 };
const assets = (take_id, cues = [cue], audio = { line: new Uint8Array([1, 2, 3]) }) =>
  ({ type: "DialogueAssetsMessage", take_id, assets: audio, cues });
const clock = (take_id, frame, playing = true, speed = 1, revision = 1) =>
  ({ type: "DialoguePlaybackMessage", take_id, frame, fps: 25, playing, speed, revision });

test("dialogue follows pause, resume, seek, speed, loop, and take switch", async () => {
  FakeAudioContext.instances = [];
  FakeAudioContext.decoder = async () => ({ duration: 5 });
  const player = new VoicePlayback();
  const ready = [];
  await player.receiveAssets(assets("take-one"), (id) => ready.push(id));
  assert.deepEqual(ready, ["take-one"]);
  const audio = FakeAudioContext.instances.at(-1);

  player.receivePlayback(clock("take-one", 0));
  assert.equal(audio.sources.length, 1);
  assert.equal(audio.sources[0].started.offset, 0);
  assert.equal(useVoiceView.getState().caption, "Follow me.");
  audio.currentTime = 1;
  player.receivePlayback(clock("take-one", 25));
  assert.equal(audio.sources.length, 1, "ordinary clock ticks do not restart audio");

  player.receivePlayback(clock("take-one", 25, false));
  assert.equal(audio.sources[0].stopped, true);
  player.receivePlayback(clock("take-one", 25));
  assert.equal(audio.sources.at(-1).started.offset, 1);

  player.receivePlayback(clock("take-one", 25, true, 2));
  assert.equal(audio.sources.at(-1).started.speed, 2);
  assert.equal(audio.sources.at(-2).stopped, true);

  player.receivePlayback(clock("take-one", 75, true, 2, 2));
  assert.equal(audio.sources.at(-1).started.offset, 3);
  player.receivePlayback(clock("take-one", 0, true, 2, 2));
  assert.equal(audio.sources.at(-1).started.offset, 0, "loop rewinds the dialogue");

  await player.receiveAssets(assets("take-two", [], {}), (id) => ready.push(id));
  assert.equal(audio.sources.at(-1).stopped, true);
  player.receivePlayback(clock("take-two", 0));
  assert.equal(useVoiceView.getState().caption, "");
  assert.deepEqual(ready, ["take-one", "take-two"]);
  player.reset();
});

test("superseded decode cannot install stale audio or acknowledge an old take", async () => {
  FakeAudioContext.instances = [];
  const pending = [];
  FakeAudioContext.decoder = () => new Promise((resolve) => pending.push(resolve));
  const player = new VoicePlayback();
  const ready = [];
  const first = player.receiveAssets(assets("old"), (id) => ready.push(id));
  const second = player.receiveAssets(assets("new"), (id) => ready.push(id));
  pending[1]({ duration: 5 });
  await second;
  pending[0]({ duration: 5 });
  await first;
  assert.deepEqual(ready, ["new"]);
  player.receivePlayback(clock("new", 0));
  assert.equal(FakeAudioContext.instances.at(-1).sources.length, 1);
  player.reset();
  assert.equal(FakeAudioContext.instances.at(-1).sources.at(-1).stopped, true);
  assert.equal(useVoiceView.getState().caption, "");
});

test("decode failure waits for user recovery; silent playback invalidates late decodes", async () => {
  FakeAudioContext.instances = [];
  let resolveLate;
  let calls = 0;
  FakeAudioContext.decoder = () => {
    calls++;
    return calls === 1 ? Promise.reject(new Error("bad audio")) : new Promise((resolve) => { resolveLate = resolve; });
  };
  const player = new VoicePlayback();
  const ready = [];
  await player.receiveAssets(assets("take", [cue], {
    line: new Uint8Array([1]), other: new Uint8Array([2]),
  }), (id) => ready.push(id));
  assert.deepEqual(ready, []);
  assert.equal(useVoiceView.getState().canRetryAssets, true);
  player.playWithoutAudio((id) => ready.push(id));
  resolveLate({ duration: 5 });
  await Promise.resolve();
  player.receivePlayback(clock("take", 0));
  assert.deepEqual(ready, ["take"]);
  assert.equal(FakeAudioContext.instances.at(-1).sources.length, 0);
  assert.equal(useVoiceView.getState().caption, "Follow me.");
  player.reset();
});
