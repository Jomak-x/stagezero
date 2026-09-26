import type { GuiUpdateMessage } from "../WebsocketMessages";

export type StudioActionOperation = "replace" | "insert_before" | "insert_after" | "start" | "play" | "pause";

export interface StudioActionRef {
  takeId: string;
  index: number;
  commandUuid: string;
}

/** A prompt identifier is also the address of its Director command control. */
export function parseStudioActionUuid(uuid: string): StudioActionRef | null {
  const parts = uuid.split("|");
  if (parts.length !== 4 || parts[0] !== "stagezero" || !parts[1] || !/^\d+$/.test(parts[2]!)) {
    return null;
  }
  const index = Number(parts[2]);
  if (!Number.isSafeInteger(index)) return null;
  return { takeId: parts[1]!, index, commandUuid: parts[3]! };
}

export function studioActionMessage(
  action: StudioActionRef,
  operation: StudioActionOperation,
  nonce: string,
): GuiUpdateMessage {
  return {
    type: "GuiUpdateMessage",
    uuid: action.commandUuid,
    updates: {
      value: JSON.stringify({
        take_id: action.takeId,
        index: action.index,
        operation,
        nonce,
      }),
    },
  };
}
