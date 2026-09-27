import React from "react";
import { Box, Button, Group, SegmentedControl, Switch, Text, Textarea } from "@mantine/core";
import { IconMicrophone, IconPlayerStop } from "@tabler/icons-react";
import { ViewerContext } from "../ViewerContext";
import { submitVoiceRequest, useVoiceCommands } from "../VoiceCommands";

const MAX_RECORDING_MS = 30_000;
const MAX_RECORDING_BYTES = 8 * 1024 * 1024;
const MIME_TYPES = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
type VoiceTarget = "auto" | "single_action" | "full_scene";
const RUNNING_STATUSES = new Set(["submitting", "queued", "transcribing", "planning", "running", "preparing"]);
const TARGET_LABELS: Record<VoiceTarget, string> = { auto: "Auto", single_action: "Single action", full_scene: "Full scene" };

function recordingMimeType() {
  return MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type));
}

export function VoiceDirector() {
  const viewer = React.useContext(ViewerContext)!;
  const connected = viewer.useGui((state) => state.websocketConnected);
  const requests = useVoiceCommands((state) => state.requests);
  const [enabled, setEnabled] = React.useState(false);
  const [text, setText] = React.useState("");
  const [target, setTarget] = React.useState<VoiceTarget>("auto");
  const [recordState, setRecordState] = React.useState<"idle" | "permission" | "recording" | "sending">("idle");
  const [localError, setLocalError] = React.useState("");
  const holdToken = React.useRef(0);
  const holdActive = React.useRef(false);
  const recordingTarget = React.useRef<VoiceTarget>("auto");
  const recorder = React.useRef<MediaRecorder | null>(null);
  const stream = React.useRef<MediaStream | null>(null);
  const chunks = React.useRef<Blob[]>([]);
  const recordingTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const cancelCurrentRecording = React.useRef<() => void>(() => {});
  const finishCurrentRecording = React.useRef<() => void>(() => {});

  const send = React.useCallback((command: "submit" | "cancel", id: string, content = "") => {
    viewer.mutable.current.sendMessage({ type: "VoiceCommandMessage", request_id: id, command, text: content, target });
  }, [viewer, target]);

  const closeStream = React.useCallback(() => {
    if (recordingTimer.current) clearTimeout(recordingTimer.current);
    recordingTimer.current = null;
    stream.current?.getTracks().forEach((track) => track.stop());
    stream.current = null;
    recorder.current = null;
  }, []);

  const cancelRecording = React.useCallback(() => {
    holdActive.current = false;
    holdToken.current++;
    if (recorder.current && recorder.current.state !== "inactive") {
      recorder.current.onstop = null;
      recorder.current.stop();
    }
    chunks.current = [];
    closeStream();
    setRecordState("idle");
  }, [closeStream]);
  cancelCurrentRecording.current = cancelRecording;

  React.useEffect(() => () => cancelCurrentRecording.current(), []);
  React.useEffect(() => {
    if (!connected) cancelCurrentRecording.current();
  }, [connected]);
  React.useEffect(() => {
    if (!enabled) cancelCurrentRecording.current();
  }, [enabled]);
  React.useEffect(() => {
    const cancelHeldRecording = () => {
      if (holdActive.current) cancelCurrentRecording.current();
    };
    const cancelWhenHidden = () => {
      if (document.hidden) cancelHeldRecording();
    };
    window.addEventListener("blur", cancelHeldRecording);
    document.addEventListener("visibilitychange", cancelWhenHidden);
    return () => {
      window.removeEventListener("blur", cancelHeldRecording);
      document.removeEventListener("visibilitychange", cancelWhenHidden);
    };
  }, []);

  const startRecording = React.useCallback(async () => {
    if (recordState !== "idle" || !connected || !enabled || !holdActive.current) return;
    setLocalError("");
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      setLocalError("Microphone access needs localhost or HTTPS. You can type a direction below.");
      return;
    }
    if (typeof MediaRecorder === "undefined") {
      setLocalError("This browser cannot record audio. You can type a direction below.");
      return;
    }
    const token = ++holdToken.current;
    recordingTarget.current = target;
    setRecordState("permission");
    try {
      const media = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (token !== holdToken.current || !holdActive.current) {
        media.getTracks().forEach((track) => track.stop());
        return;
      }
      stream.current = media;
      chunks.current = [];
      const mimeType = recordingMimeType();
      const mediaRecorder = new MediaRecorder(media, mimeType ? { mimeType } : undefined);
      recorder.current = mediaRecorder;
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size) chunks.current.push(event.data);
        if (chunks.current.reduce((sum, part) => sum + part.size, 0) > MAX_RECORDING_BYTES) {
          setLocalError("Recording exceeded 8 MB. Try a shorter direction.");
          cancelCurrentRecording.current();
        }
      };
      mediaRecorder.onerror = () => {
        setLocalError("The recording stopped unexpectedly. Try again or type your direction.");
        cancelCurrentRecording.current();
      };
      mediaRecorder.start(250);
      setRecordState("recording");
      recordingTimer.current = setTimeout(() => finishCurrentRecording.current(), MAX_RECORDING_MS);
    } catch (error) {
      if (token !== holdToken.current) return;
      setLocalError(error instanceof DOMException && error.name === "NotAllowedError"
        ? "Microphone permission was denied. You can enable it in your browser or type a direction."
        : "Could not open the microphone. You can type a direction below.");
      closeStream();
      setRecordState("idle");
    }
  }, [closeStream, connected, enabled, recordState, target]);

  // The recorder's stop event owns the final chunk, so build the socket payload there.
  const finishRecording = React.useCallback(() => {
    holdActive.current = false;
    const current = recorder.current;
    if (!current || current.state === "inactive") {
      cancelRecording();
      return;
    }
    if (recordingTimer.current) clearTimeout(recordingTimer.current);
    recordingTimer.current = null;
    const token = holdToken.current;
    setRecordState("sending");
    current.onstop = async () => {
      const blob = new Blob(chunks.current, { type: current.mimeType || "audio/webm" });
      chunks.current = [];
      closeStream();
      setRecordState("idle");
      if (!blob.size || blob.size > MAX_RECORDING_BYTES || !connected) {
        setLocalError("No usable recording was captured. Hold the microphone a little longer.");
        return;
      }
      const id = crypto.randomUUID();
      const audio = new Uint8Array(await blob.arrayBuffer());
      if (token !== holdToken.current || !viewer.useGui.getState().websocketConnected) return;
      submitVoiceRequest({ request_id: id, status: "submitting", detail: "Sending recording…",
        transcript: "", retryable: false, target: recordingTarget.current });
      viewer.mutable.current.sendMessage({
        type: "VoiceRecordingMessage",
        request_id: id,
        mime_type: blob.type,
        audio,
        target: recordingTarget.current,
      });
    };
    current.stop();
  }, [cancelRecording, closeStream, connected, viewer]);
  finishCurrentRecording.current = finishRecording;

  const onRelease = () => {
    const wasHeld = holdActive.current;
    holdActive.current = false;
    if (recorder.current?.state === "recording") finishRecording();
    else if (wasHeld && recordState !== "sending") {
      cancelRecording();
      if (recordState === "permission") setLocalError("Hold again after allowing microphone access to record.");
    }
  };

  const submitText = () => {
    const direction = text.trim();
    if (!direction || direction.length > (target === "single_action" ? 500 : 2000) || !connected || !enabled) return;
    setLocalError("");
    const id = crypto.randomUUID();
    submitVoiceRequest({ request_id: id, status: "submitting", detail: "Sending direction…",
      transcript: direction, retryable: false, target });
    send("submit", id, direction);
    setText("");
  };

  return <Box px="xs" pt="xs" pb="sm" style={{ borderBottom: "1px solid #ffffff24" }}>
    <Group justify="space-between" mb={6}>
      <Text fw={700} size="sm">Voice direction</Text>
      <Switch size="xs" label="Activate voice tools" checked={enabled} onChange={(event) => setEnabled(event.currentTarget.checked)} />
    </Group>
    {!enabled && <Text size="xs" c="dimmed">Turn on to enter or record directions. Queued requests keep running.</Text>}
    {enabled && <>
    <SegmentedControl
      fullWidth size="xs" mb="xs" aria-label="Generation destination"
      value={target} disabled={recordState !== "idle"}
      onChange={(value) => setTarget(value as VoiceTarget)}
      data={[{ label: "Auto", value: "auto" }, { label: "Single action", value: "single_action" }, { label: "Full scene", value: "full_scene" }]}
    />
    <Group gap="xs" wrap="nowrap" align="stretch">
      <Button
        color={recordState === "recording" ? "red" : "teal"}
        variant={recordState === "recording" ? "filled" : "light"}
        aria-label={recordState === "recording" ? "Recording: release to queue" : "Hold to speak a direction"}
        aria-pressed={recordState === "recording"}
        disabled={!connected}
        leftSection={recordState === "recording" ? <IconPlayerStop size={16} /> : <IconMicrophone size={16} />}
        onPointerDown={(event) => {
          if (event.button !== 0) return;
          event.currentTarget.setPointerCapture(event.pointerId);
          holdActive.current = true;
          void startRecording();
        }}
        onPointerUp={onRelease}
        onPointerCancel={cancelRecording}
        onKeyDown={(event) => {
          if ((event.key === " " || event.key === "Enter") && !event.repeat) {
            event.preventDefault();
            holdActive.current = true;
            void startRecording();
          }
        }}
        onKeyUp={(event) => {
          if (event.key === " " || event.key === "Enter") {
            event.preventDefault();
            onRelease();
          }
        }}
      >{recordState === "recording" ? "Recording…" : recordState === "permission" ? "Allow mic…" : "Hold to speak"}</Button>
      {(recordState === "permission" || recordState === "recording") && <Button color="gray" variant="subtle" onClick={cancelRecording}>Discard</Button>}
    </Group>
    <Text size="xs" c="dimmed" mt={4}>
      {target === "auto"
        ? "Try ‘generate scene…’, ‘generate a short…’, or ‘edit scene [name]…’. Hold to record; release to queue."
        : target === "single_action"
          ? "One motion instruction. Hold to record; release to queue."
          : "A sequence of actions. Hold to record; release to queue. Recording stops after 30 seconds."}
    </Text>
    <Textarea
      mt="xs" minRows={2} maxRows={4} autosize value={text}
      onChange={(event) => setText(event.currentTarget.value)}
      placeholder={target === "auto" ? 'Generate a short: wave. / Edit scene "Walk and wave" to walk slowly.' : target === "single_action" ? "Walk forward and wave." : "Walk to the door, then wave."}
      aria-label="Type a scene direction"
      maxLength={target === "single_action" ? 500 : 2000}
    />
    <Group gap="xs" mt="xs">
      <Button size="xs" onClick={submitText} disabled={!connected || !text.trim() || text.trim().length > (target === "single_action" ? 500 : 2000) || recordState !== "idle"}>Queue direction</Button>
    </Group>
    </>}
    {requests.length > 0 && <Box mt="sm" aria-label="Voice request queue">
      <Text size="xs" fw={700} mb={4}>Requests</Text>
      {[...requests].reverse().map((request) => <Box key={request.request_id} py={5} style={{ borderTop: "1px solid #ffffff24" }}>
        <Group justify="space-between" gap="xs" wrap="nowrap">
          <Text size="xs" fw={600}>{TARGET_LABELS[request.target] ?? "Voice"} · {request.status.replaceAll("_", " ")}</Text>
          <Group gap={4} wrap="nowrap">
            {connected && RUNNING_STATUSES.has(request.status) && <Button size="compact-xs" variant="subtle" color="gray" onClick={() => send("cancel", request.request_id)}>Cancel</Button>}
          </Group>
        </Group>
        <Text size="xs" c="dimmed" role="status">{request.detail || request.status}</Text>
        {request.transcript && <Text size="xs" style={{ overflowWrap: "anywhere" }}>“{request.transcript.slice(0, 240)}{request.transcript.length > 240 ? "…" : ""}”</Text>}
      </Box>)}
    </Box>}
    {localError && <Text size="xs" mt="xs" c="red" role="alert">{localError}</Text>}
  </Box>;
}
