import {
  ActorGlbCommandMessage,
  ActorGlbPoseMessage,
} from "../WebsocketMessages";

export type ActorGlbControl = ActorGlbCommandMessage | ActorGlbPoseMessage;

// Control and pose can arrive before the GLB parser completes. Keep only the
// current revision's messages and replay them when that candidate is ready.
export const controls = new Map<number, ActorGlbCommandMessage>();
export const poses = new Map<number, ActorGlbPoseMessage>();
const subscribers = new Set<(message: ActorGlbControl) => void>();
export let g1Revision = -1;

export function resetActorGlbProtocol() {
  controls.clear();
  poses.clear();
  g1Revision = -1;
}

export function subscribeActorGlbControl(callback: (message: ActorGlbControl) => void) {
  subscribers.add(callback);
  return () => { subscribers.delete(callback); };
}

export function publishActorGlbControl(message: ActorGlbControl) {
  if (message.type === "ActorGlbPoseMessage") {
    poses.set(message.revision, message);
  } else {
    controls.set(message.revision, message);
    if (message.action === "g1") {
      g1Revision = Math.max(g1Revision, message.revision);
      poses.clear();
      controls.clear();
      controls.set(message.revision, message);
    } else if (message.action === "commit") {
      for (const revision of poses.keys()) {
        if (revision < message.revision) poses.delete(revision);
      }
      for (const revision of controls.keys()) {
        if (revision < message.revision) controls.delete(revision);
      }
    } else {
      poses.delete(message.revision);
    }
  }
  subscribers.forEach((callback) => callback(message));
}
