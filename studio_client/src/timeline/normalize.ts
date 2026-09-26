// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { TimelineMessage } from "../WebsocketMessages";

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function sameEntry(left: object, right: object): boolean {
  const a = left as Record<string, unknown>;
  const b = right as Record<string, unknown>;
  const keys = Object.keys(a);
  if (keys.length !== Object.keys(b).length) return false;
  return keys.every((key) => {
    if (!Object.prototype.hasOwnProperty.call(b, key)) return false;
    const value = a[key];
    const other = b[key];
    return Array.isArray(value) && Array.isArray(other)
      ? value.length === other.length && value.every((item, index) => Object.is(item, other[index]))
      : Object.is(value, other);
  });
}

function reuseUnchangedEntries<T extends object>(next: T[], previous?: T[]): T[] {
  if (!previous || next.length !== previous.length) return next;
  return next.every((entry, index) => sameEntry(entry, previous[index]!))
    ? previous
    : next;
}

const dynamicTimelineFields = new Set([
  "current_frame", "prompts", "tracks", "keyframes", "intervals",
]);

function sameStaticConfiguration(next: TimelineMessage, previous: TimelineMessage): boolean {
  const a = next as unknown as Record<string, unknown>;
  const b = previous as unknown as Record<string, unknown>;
  const keys = Object.keys(a).filter((key) => !dynamicTimelineFields.has(key));
  if (keys.length !== Object.keys(b).filter((key) => !dynamicTimelineFields.has(key)).length) {
    return false;
  }
  return keys.every((key) =>
    Object.prototype.hasOwnProperty.call(b, key) && Object.is(a[key], b[key]),
  );
}

/** Keep static timeline data stable while only the playback frame changes. */
export function normalizeTimelineMessage(
  message: TimelineMessage,
  previous?: TimelineMessage | null,
): TimelineMessage {
  const minPromptDuration = Math.max(1, message.min_prompt_duration ?? 1);
  const maxPromptDuration =
    message.max_prompt_duration == null
      ? null
      : Math.max(minPromptDuration, message.max_prompt_duration);

  const normalizedPrompts = (message.prompts ?? []).map((prompt) => {
    const start = Math.min(prompt.start_frame, prompt.end_frame);
    let end = Math.max(prompt.start_frame, prompt.end_frame);
    const minEnd = start + minPromptDuration;
    end = Math.max(end, minEnd);
    if (maxPromptDuration != null) {
      end = Math.min(end, start + maxPromptDuration);
    }
    return { ...prompt, start_frame: start, end_frame: end };
  });

  normalizedPrompts.sort((a, b) => a.start_frame - b.start_frame);

  const defaultNumFramesZoom = Math.max(1, message.default_num_frames_zoom ?? 1);
  const maxFramesZoom = Math.max(defaultNumFramesZoom, message.max_frames_zoom ?? defaultNumFramesZoom);
  const startFrame = message.start_frame ?? 0;
  const endFrame = Math.max(message.end_frame ?? startFrame, startFrame);
  const fps = message.fps > 0 ? message.fps : 30.0;
  const currentFrame = clamp(message.current_frame ?? startFrame, startFrame, endFrame);

  const normalized: TimelineMessage = {
    ...message,
    prompts: normalizedPrompts,
    tracks: message.tracks ?? [],
    keyframes: message.keyframes ?? [],
    intervals: message.intervals ?? [],
    min_prompt_duration: minPromptDuration,
    max_prompt_duration: maxPromptDuration,
    default_num_frames_zoom: defaultNumFramesZoom,
    max_frames_zoom: maxFramesZoom,
    start_frame: startFrame,
    end_frame: endFrame,
    fps,
    current_frame: currentFrame,
  };

  if (!previous) return normalized;
  if (!sameStaticConfiguration(normalized, previous)) {
    // Timeline's visible canvas redraw depends on these identities. Invalidate
    // even if a caller reused an incoming array while changing layout scalars.
    return {
      ...normalized,
      tracks: [...normalized.tracks],
      keyframes: [...normalized.keyframes],
      intervals: [...normalized.intervals],
    };
  }

  return {
    ...normalized,
    prompts: reuseUnchangedEntries(normalized.prompts, previous.prompts),
    tracks: reuseUnchangedEntries(normalized.tracks, previous.tracks),
    keyframes: reuseUnchangedEntries(normalized.keyframes, previous.keyframes),
    intervals: reuseUnchangedEntries(normalized.intervals, previous.intervals),
  };
}
