/** Coalesce high-frequency updates per control, never across unrelated inputs. */
export function createMessageThrottle<T extends { type: string }>(
  deliver: (message: T) => void,
  milliseconds: number,
) {
  const latest = new Map<string, T>();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const keyFor = (message: T) => {
    const fields = message as Record<string, unknown>;
    const id = ['uuid', 'prompt_id', 'interval_id', 'keyframe_id', 'handle_id', 'node_id', 'scene_node_id', 'camera_id', 'material_id']
      .find(key => typeof fields[key] === 'string');
    return id ? `${message.type}:${id}:${fields[id]}` : message.type;
  };
  const flush = () => {
    for (const message of latest.values()) deliver(message);
    latest.clear();
  };
  const drain = () => {
    timer = undefined;
    if (latest.size) {
      flush();
      timer = setTimeout(drain, milliseconds);
    }
  };
  return {
    send(message: T) {
      latest.set(keyFor(message), message);
      if (timer === undefined) {
        flush();
        timer = setTimeout(drain, milliseconds);
      }
    },
    flush,
    dispose() {
      if (timer !== undefined) clearTimeout(timer);
      timer = undefined;
      latest.clear();
    },
  };
}
