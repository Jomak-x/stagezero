import { create } from "zustand";
import type { VoiceQueueMessage, VoiceQueueRequest, VoiceStatusMessage } from "./WebsocketMessages";

type VoiceCommandState = {
  requests: VoiceQueueRequest[];
  activeRequestId: string | null;
  status: VoiceStatusMessage | null;
};

export const useVoiceCommands = create<VoiceCommandState>(() => ({
  requests: [],
  activeRequestId: null,
  status: null,
}));

// Keep locally submitted rows visible until the server acknowledges them.
const pendingSubmissions = new Set<string>();

export function submitVoiceRequest(request: VoiceQueueRequest) {
  pendingSubmissions.add(request.request_id);
  useVoiceCommands.setState((state) => ({
    activeRequestId: request.request_id,
    status: {
      type: "VoiceStatusMessage", request_id: request.request_id,
      status: request.status, detail: request.detail,
      transcript: request.transcript, retryable: request.retryable,
    },
    requests: [...state.requests.filter((item) => item.request_id !== request.request_id), request],
  }));
}

export function receiveVoiceQueue(message: VoiceQueueMessage) {
  const ids = new Set(message.requests.map((item) => item.request_id));
  for (const id of ids) pendingSubmissions.delete(id);
  useVoiceCommands.setState((state) => {
    const active = message.requests.find((item) => item.request_id === state.activeRequestId);
    return {
      requests: [
        ...message.requests,
        ...state.requests.filter((item) => pendingSubmissions.has(item.request_id) && !ids.has(item.request_id)),
      ],
      status: active ? {
        type: "VoiceStatusMessage", request_id: active.request_id,
        status: active.status, detail: active.detail,
        transcript: active.transcript, retryable: active.retryable,
      } : state.status,
    };
  });
}

export function receiveVoiceStatus(message: VoiceStatusMessage) {
  pendingSubmissions.delete(message.request_id);
  useVoiceCommands.setState((state) => {
    const existing = state.requests.find((item) => item.request_id === message.request_id);
    const request: VoiceQueueRequest = {
      request_id: message.request_id, status: message.status, detail: message.detail,
      transcript: message.transcript, retryable: message.retryable,
      target: existing?.target ?? "auto",
    };
    return {
      status: state.activeRequestId === message.request_id ? message : state.status,
      requests: existing
        ? state.requests.map((item) => item.request_id === message.request_id ? request : item)
        : [...state.requests, request],
    };
  });
}

export function resetVoiceCommands() {
  pendingSubmissions.clear();
  useVoiceCommands.setState({ requests: [], activeRequestId: null, status: null });
}
