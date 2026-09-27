"""Studio-only extensions to the pinned Viser socket protocol."""
from dataclasses import dataclass
from typing import Any
from viser._messages import Message


@dataclass
class VoiceRecordingMessage(Message):
    request_id: str
    mime_type: str
    audio: bytes
    target: str = 'full_scene'


@dataclass
class VoiceCommandMessage(Message):
    request_id: str
    command: str
    text: str = ''
    target: str = 'full_scene'


@dataclass
class VoiceStatusMessage(Message):
    request_id: str
    status: str
    detail: str
    transcript: str
    retryable: bool

    def redundancy_key(self) -> str:
        # A new queued request must not replace the previous request's
        # completion before Viser flushes its outgoing message buffer.
        return f'{type(self).__name__}:{self.request_id}'


@dataclass
class VoiceQueueMessage(Message):
    requests: list[dict[str, Any]]

