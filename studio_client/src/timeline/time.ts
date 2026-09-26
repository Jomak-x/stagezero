/** Frame positions are zero based; the last pose occurs at (length - 1) / fps. */
export function frameSeconds(frame: number, fps: number): number {
  return frame / Math.max(fps, 1e-9);
}

export function formatSeconds(seconds: number, decimals = 1): string {
  const value = Math.max(0, seconds);
  if (value >= 60) {
    const minutes = Math.floor(value / 60);
    return `${minutes}:${(value - minutes * 60).toFixed(decimals).padStart(3 + decimals, "0")}`;
  }
  return `${value.toFixed(decimals)}s`;
}

/** Tick spacing with labels at readable 1/2/5 multiples of seconds. */
export function secondsTickStep(fps: number, pixelsPerFrame: number, minPixels = 72): number {
  const requested = minPixels / Math.max(fps * pixelsPerFrame, 1e-9);
  const magnitude = Math.pow(10, Math.floor(Math.log10(requested)));
  for (const multiplier of [1, 2, 5, 10]) {
    const step = multiplier * magnitude;
    if (step >= requested) return step;
  }
  return 10 * magnitude;
}
