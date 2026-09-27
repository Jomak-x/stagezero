import { create } from "zustand";

export type DialogueCue = {
  text: string;
  voice_id: string;
  audio_id: string;
  start_frame: number;
  end_frame: number;
  audio_offset_seconds?: number;
  character_id?: string;
  line_id?: string;
};
export type DialogueAssetsMessage = {
  type: "DialogueAssetsMessage";
  take_id: string;
  assets: Record<string, Uint8Array>;
  cues: DialogueCue[];
};
export type DialoguePlaybackMessage = {
  type: "DialoguePlaybackMessage";
  take_id: string;
  frame: number;
  fps: number;
  playing: boolean;
  speed: number;
  revision: number;
};

type VoiceViewState = {
  caption: string;
  audioError: string | null;
  takeId: string | null;
  canRetryAssets: boolean;
};

export const useVoiceView = create<VoiceViewState>(() => ({
  caption: "",
  audioError: null,
  takeId: null,
  canRetryAssets: false,
}));

type PlayingCue = { source: AudioBufferSourceNode; startedAt: number; offset: number; speed: number };

export class VoicePlayback {
  private context: AudioContext | null = null;
  private buffers = new Map<string, AudioBuffer>();
  private cues: DialogueAssetsMessage["cues"] = [];
  private takeId: string | null = null;
  private playing = new Map<number, PlayingCue>();
  private mutedForRecording = false;
  private loadVersion = 0;
  private lastRevision = -1;
  private pendingAssets: DialogueAssetsMessage | null = null;
  private readyCallback: ((takeId: string) => void) | null = null;

  private getContext(): AudioContext {
    if (!this.context) this.context = new AudioContext();
    return this.context;
  }

  /** Call from a user gesture so browsers permit the later automatic playback. */
  unlock() {
    try {
      void this.getContext().resume().then(() => {
        if (!this.pendingAssets && useVoiceView.getState().audioError === "Click Enable audio to hear dialogue.") {
          useVoiceView.setState({ audioError: null });
        }
      }).catch(() => useVoiceView.setState({ audioError: "Click Enable audio to hear dialogue." }));
    } catch {
      useVoiceView.setState({ audioError: "Audio playback is unavailable in this browser." });
    }
  }

  async receiveAssets(message: DialogueAssetsMessage, ready: (takeId: string) => void) {
    const version = ++this.loadVersion;
    this.readyCallback = ready;
    this.pendingAssets = message;
    this.stopAll();
    this.buffers.clear();
    this.cues = message.cues;
    this.takeId = message.take_id;
    this.lastRevision = -1;
    useVoiceView.setState({ takeId: message.take_id, caption: "", audioError: null, canRetryAssets: false });
    try {
      if (Object.keys(message.assets).length) {
        const context = this.getContext();
        await Promise.all(Object.entries(message.assets).map(async ([id, bytes]) => {
          const copy = new Uint8Array(bytes.byteLength);
          copy.set(bytes);
          const decoded = await context.decodeAudioData(copy.buffer);
          if (version === this.loadVersion) this.buffers.set(id, decoded);
        }));
      }
    } catch {
      if (version === this.loadVersion) {
        ++this.loadVersion;
        this.buffers.clear();
        useVoiceView.setState({ audioError: "Dialogue audio could not be decoded. Retry audio or play motion without sound.", canRetryAssets: true });
      }
      return;
    }
    if (version === this.loadVersion) {
      this.pendingAssets = null;
      useVoiceView.setState({ canRetryAssets: false });
      ready(message.take_id);
    }
  }

  retryAssets(ready = this.readyCallback) {
    if (this.pendingAssets && ready) void this.receiveAssets(this.pendingAssets, ready);
  }

  playWithoutAudio(ready = this.readyCallback) {
    if (!this.pendingAssets || !ready) return;
    ++this.loadVersion;
    const takeId = this.pendingAssets.take_id;
    this.pendingAssets = null;
    this.readyCallback = null;
    this.buffers.clear();
    useVoiceView.setState({ audioError: "Dialogue sound is unavailable; playing motion and captions.", canRetryAssets: false });
    ready(takeId);
  }

  receivePlayback(message: DialoguePlaybackMessage) {
    if (message.take_id !== this.takeId) {
      this.stopAll();
      useVoiceView.setState({ caption: "", takeId: message.take_id });
      return;
    }
    const frame = message.frame;
    const fps = message.fps;
    const speed = message.speed;
    if (!Number.isFinite(frame) || fps <= 0 || speed <= 0) return;
    const active = new Set<number>();
    const captions: string[] = [];
    this.cues.forEach((cue, index) => {
      if (frame < cue.start_frame || frame >= cue.end_frame) return;
      captions.push(cue.text);
      if (message.playing && !this.mutedForRecording && this.buffers.has(cue.audio_id)) active.add(index);
    });
    const caption = captions.join("\n");
    if (useVoiceView.getState().caption !== caption) useVoiceView.setState({ caption });
    for (const [index, playing] of this.playing) {
      const cue = this.cues[index];
      const expected = (cue.audio_offset_seconds ?? 0) + (frame - cue.start_frame) / fps;
      const current = playing.offset + (this.getContext().currentTime - playing.startedAt) * playing.speed;
      if (!active.has(index) || Math.abs(current - expected) > 0.22 || Math.abs(playing.speed - speed) > 0.001 || message.revision !== this.lastRevision) {
        this.stop(index);
      }
    }
    if (active.size) {
      const context = this.getContext();
      if (context.state !== "running") {
        useVoiceView.setState({ audioError: "Click Enable audio to hear dialogue." });
      } else {
        for (const index of active) {
          if (this.playing.has(index)) continue;
          const cue = this.cues[index];
          const buffer = this.buffers.get(cue.audio_id)!;
          const offset = Math.max(0, (cue.audio_offset_seconds ?? 0) + (frame - cue.start_frame) / fps);
          if (offset >= buffer.duration) continue;
          const source = context.createBufferSource();
          source.buffer = buffer;
          source.playbackRate.value = speed;
          source.connect(context.destination);
          const maxDuration = Math.min(buffer.duration - offset, (cue.end_frame - frame) / fps);
          if (maxDuration <= 0) continue;
          source.start(0, offset, maxDuration);
          const playing: PlayingCue = { source, startedAt: context.currentTime, offset, speed };
          this.playing.set(index, playing);
          source.onended = () => {
            if (this.playing.get(index) === playing) this.playing.delete(index);
          };
        }
      }
    }
    this.lastRevision = message.revision;
  }

  setRecording(recording: boolean) {
    this.mutedForRecording = recording;
    if (recording) this.stopAll();
  }

  private stop(index: number) {
    const item = this.playing.get(index);
    if (!item) return;
    item.source.onended = null;
    try { item.source.stop(); } catch { /* source already ended */ }
    item.source.disconnect();
    this.playing.delete(index);
  }

  private stopAll() {
    for (const index of this.playing.keys()) this.stop(index);
  }

  reset() {
    ++this.loadVersion;
    this.stopAll();
    this.buffers.clear();
    this.cues = [];
    this.takeId = null;
    this.pendingAssets = null;
    this.readyCallback = null;
    this.lastRevision = -1;
    useVoiceView.setState({ caption: "", audioError: null, takeId: null, canRetryAssets: false });
  }
}

export const voicePlayback = new VoicePlayback();
