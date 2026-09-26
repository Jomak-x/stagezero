// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export interface TimedCameraCut {
  id: string;
  frame: number;
  camera_id: string;
}

export function boundedPlayheadFrame(frame: number, length: number): number {
  return Math.max(0, Math.min(Math.max(0, Math.floor(length) - 1), Math.round(frame)));
}

export function boundedCutFrame(frame: number, length: number, firstCut: boolean): number {
  const end = Math.max(0, Math.floor(length) - 1);
  return firstCut ? 0 : Math.max(1, Math.min(end, Math.round(frame)));
}

export function frameAfterCutDrag(
  initialFrame: number,
  pixelDelta: number,
  pixelsPerFrame: number,
  length: number,
): number {
  if (!Number.isFinite(pixelsPerFrame) || pixelsPerFrame <= 0) return initialFrame;
  return boundedCutFrame(initialFrame + pixelDelta / pixelsPerFrame, length, false);
}

export function cameraCutSegments<T extends TimedCameraCut>(cuts: T[], length: number) {
  const ordered = [...cuts].sort((a, b) => a.frame - b.frame);
  return ordered.map((cut, index) => ({
    cut,
    endFrame: index + 1 < ordered.length ? ordered[index + 1]!.frame : Math.max(cut.frame + 1, length),
    firstCut: index === 0 && cut.frame === 0,
  }));
}

/** The server replaces a cut already at the dragged cut's destination. */
export function cameraCutPreviewSegments<T extends TimedCameraCut>(
  cuts: T[], length: number, drag: { id: string; frame: number } | null,
) {
  if (!drag || !cuts.some((cut) => cut.id === drag.id)) return cameraCutSegments(cuts, length);
  const previewCuts = cuts
    .filter((cut) => cut.id === drag.id || cut.frame !== drag.frame)
    .map((cut) => cut.id === drag.id ? { ...cut, frame: drag.frame } : cut);
  return cameraCutSegments(previewCuts, length);
}

/** Keep a very short or captured cut reachable at the viewport edge. */
export function visibleCameraBlock(
  rawLeft: number, rawRight: number, minX: number, maxX: number,
  keepMounted = false, minimumWidth = 12,
): { left: number; width: number } | null {
  const available = maxX - minX;
  if (available <= 0) return keepMounted ? { left: minX, width: 1 } : null;
  const narrowWidth = Math.min(minimumWidth, available);
  const left = Math.max(minX, rawLeft);
  const right = Math.min(maxX, rawRight);
  if (right <= left) {
    if (!keepMounted) return null;
    return { left: rawLeft >= maxX ? maxX - narrowWidth : minX, width: narrowWidth };
  }
  if (right - left >= narrowWidth) return { left, width: right - left };
  const fittedLeft = Math.max(minX, Math.min(left, maxX - narrowWidth));
  return { left: fittedLeft, width: narrowWidth };
}
