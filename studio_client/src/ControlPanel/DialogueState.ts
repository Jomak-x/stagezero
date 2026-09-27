import { create } from "zustand";

export type DialogueVoice = { id: string; name: string; gender: string };
export type DialogueCharacter = { id: string; name: string };
export type DialogueLine = {
  line_id: string;
  request_id: string;
  take_id: string;
  start_frame: number;
  character_id: string;
  text: string;
  voice_id: string;
  status: string;
  detail: string;
  retryable: boolean;
};

type DialogueStore = {
  voices: DialogueVoice[];
  available: boolean;
  catalogDetail: string;
  catalogAvailable: boolean;
  contextAvailable: boolean;
  contextDetail: string;
  takeId: string;
  takeName: string;
  frame: number;
  characters: DialogueCharacter[];
  lines: DialogueLine[];
};

const initialState: DialogueStore = {
  voices: [], available: false, catalogDetail: "", catalogAvailable: false,
  contextAvailable: false, contextDetail: "",
  takeId: "", takeName: "", frame: 0, characters: [], lines: [],
};

export const useDialogueState = create<DialogueStore>(() => initialState);

// A snapshot can be in flight while a new line or a newer status arrives.
const pendingLines = new Map<string, DialogueLine>();
const latestStatuses = new Map<string, { line_id: string; status: string; detail: string; retryable: boolean }>();
const statusRank = (status: string) => status === "queued" ? 0 : status === "generating" ? 1 : 2;

export function receiveDialogueVoices(message: {
  voices: DialogueVoice[]; available: boolean; detail: string;
}) {
  useDialogueState.setState((state) => ({
    voices: message.voices, catalogDetail: message.detail,
    catalogAvailable: message.available, contextAvailable: state.contextAvailable,
    available: message.available && state.contextAvailable,
  }));
}

export function receiveDialogueState(message: {
  take_id: string; take_name: string; frame: number;
  characters: DialogueCharacter[]; lines: DialogueLine[]; available: boolean; detail: string;
}) {
  const seen = new Set(message.lines.map((line) => line.request_id));
  for (const requestId of seen) pendingLines.delete(requestId);
  const lines = message.lines.map((line) => {
    const retried = [...pendingLines.values()].find((pending) => pending.line_id === line.line_id && pending.request_id !== line.request_id);
    if (retried) return retried;
    const newer = latestStatuses.get(line.request_id);
    if (!newer || statusRank(newer.status) <= statusRank(line.status)) {
      if (newer) latestStatuses.delete(line.request_id);
      return line;
    }
    return { ...line, line_id: newer.line_id || line.line_id,
      status: newer.status, detail: newer.detail, retryable: newer.retryable };
  });
  useDialogueState.setState((state) => ({
    takeId: message.take_id, takeName: message.take_name, frame: message.frame,
    characters: message.characters,
    lines: [...lines, ...[...pendingLines.values()].filter((line) => line.take_id === message.take_id && !seen.has(line.request_id) && !lines.some((current) => current.line_id === line.line_id))],
    contextAvailable: message.available, contextDetail: message.detail,
    available: state.catalogAvailable && message.available,
  }));
}

export function receiveDialogueStatus(message: {
  request_id: string; line_id: string; status: string; detail: string; retryable: boolean;
}) {
  latestStatuses.set(message.request_id, message);
  const pending = pendingLines.get(message.request_id);
  if (pending) pendingLines.set(message.request_id, { ...pending, line_id: message.line_id || pending.line_id,
    status: message.status, detail: message.detail, retryable: message.retryable });
  useDialogueState.setState((state) => ({
    lines: state.lines.map((line) => line.line_id === message.line_id || line.request_id === message.request_id
      ? { ...line, line_id: message.line_id || line.line_id, request_id: message.request_id,
        status: message.status, detail: message.detail, retryable: message.retryable }
      : line),
  }));
}

export function submitDialogueLine(line: DialogueLine) {
  pendingLines.set(line.request_id, line);
  useDialogueState.setState((state) => ({ lines: [...state.lines.filter((item) => item.request_id !== line.request_id), line] }));
}

export function retryDialogueLine(lineId: string, requestId: string) {
  const existing = useDialogueState.getState().lines.find((line) => line.line_id === lineId);
  if (existing) pendingLines.set(requestId, { ...existing, request_id: requestId,
    status: "queued", detail: "Retrying speech generation…", retryable: false });
  useDialogueState.setState((state) => ({
    lines: state.lines.map((line) => line.line_id === lineId
      ? { ...line, request_id: requestId, status: "queued", detail: "Retrying speech generation…", retryable: false }
      : line),
  }));
}

export function resetDialogueState() {
  pendingLines.clear();
  latestStatuses.clear();
  useDialogueState.setState(initialState);
}
