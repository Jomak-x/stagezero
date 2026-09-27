/** Ignore parse results from a request that has been replaced or unmounted. */
export function createLoadGuard<T>(
  accept: (value: T) => void,
  discard: (value: T) => void,
) {
  let active = true;
  return {
    get active() {
      return active;
    },
    complete(value: T) {
      if (active) accept(value);
      else discard(value);
    },
    cancel() {
      active = false;
    },
  };
}
