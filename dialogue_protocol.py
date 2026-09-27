"""Character dialogue commands and status over the existing Viser socket."""

from dataclasses import dataclass
from typing import Any

from viser._messages import Message


@dataclass
class DialogueCommandMessage(Message):
    request_id: str
    command: str
    take_id: str = ''
    start_frame: int = -1
    character_id: str = ''
    text: str = ''
    voice_id: str = ''
    line_id: str = ''


@dataclass
class DialogueVoicesMessage(Message):
    voices: list[dict[str, str]]
    available: bool
    detail: str


@dataclass
class DialogueStatusMessage(Message):
    request_id: str
    line_id: str
    status: str
    detail: str
    retryable: bool

    def redundancy_key(self) -> str:
        return f'{type(self).__name__}:{self.request_id}:{self.status}'


@dataclass
class DialogueStateMessage(Message):
    take_id: str
    take_name: str
    frame: int
    characters: list[dict[str, str]]
    lines: list[dict[str, Any]]
    available: bool
    detail: str


@dataclass
class DialogueAssetsMessage(Message):
    take_id: str
    assets: dict[str, bytes]
    cues: list[dict[str, Any]]


@dataclass
class DialoguePlaybackMessage(Message):
    take_id: str
    frame: int
    fps: float
    playing: bool
    speed: float
    revision: int
