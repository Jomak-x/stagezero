/** A failed candidate can fall back to a committed load that was still in
 * flight when the candidate replaced its scene-node message. Never retry the
 * failed asset itself, or parse a previous asset that is already visible. */
export function committedLoadToRetry<T extends { revision: number }>(
  committed: T | null,
  activeRevision: number | null,
  failedRevision: number,
): T | null {
  if (committed === null || activeRevision !== null ||
      committed.revision === failedRevision) return null;
  return committed;
}

/** A server rejection must cancel only that asset's current parse/compile.
 * A previous committed actor may already be loading again as a fallback. */
export function rejectionCancelsInFlightLoad(
  rejectedRevision: number,
  inFlightRevision: number | null,
): boolean {
  return inFlightRevision === rejectedRevision;
}
