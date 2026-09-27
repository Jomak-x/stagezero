"""Resolve the private native pair provider for every local launcher.

Configuration precedence is an explicit CLI path, the
``STAGEZERO_NATIVE_PAIR_CONFIG`` environment variable, then the repository's
``.runtime/prompt-native-provider.json``. The repository-local file is private
runtime configuration; callers should display ``status`` when it is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping

from native_pair_provider import NativePairProvider


ENV_NAME = "STAGEZERO_NATIVE_PAIR_CONFIG"
LOCAL_CONFIG = Path(".runtime/prompt-native-provider.json")


@dataclass(frozen=True)
class NativePairResolution:
    provider: NativePairProvider | None
    source: str
    path: Path | None
    status: str


def resolve_native_pair_provider(
    explicit_config: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> NativePairResolution:
    """Load the selected provider, or describe how to configure a missing one.

    A selected but invalid path or JSON raises an actionable ``ValueError``.
    Missing implicit local configuration is the only non-error unavailable
    state, allowing the rest of the viewer to open for reviewed playback.
    """
    environment = os.environ if environ is None else environ
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent
    if explicit_config is not None:
        source = "cli"
        path = Path(explicit_config)
    elif ENV_NAME in environment:
        source = "environment"
        value = environment[ENV_NAME]
        if not value.strip():
            raise ValueError(f"{ENV_NAME} is empty; provide a native pair config path or unset it.")
        path = Path(value)
    else:
        source = "local"
        path = root / LOCAL_CONFIG
        if not path.is_file():
            return NativePairResolution(
                provider=None, source="missing", path=path,
                status=(f"Native two-person generation is unavailable. Put the private provider "
                        f"configuration at {path}, set {ENV_NAME}, or pass --native-pair-config."),
            )
    try:
        provider = NativePairProvider.from_config(path)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid native pair config from {source} at {path}: {exc}") from exc
    return NativePairResolution(provider=provider, source=source, path=path,
                                status=f"Native pair provider configured from {path}.")
