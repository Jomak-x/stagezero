"""Private, reusable storage for validated procedural scene props.

Scenes carry the full asset definitions so they stay portable. This library only
keeps copies for future generation requests; it never acts as a scene dependency.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from asset_geometry import validate_assets


MAX_LIBRARY_ASSETS = 128
MAX_ASSET_BYTES = 100 * 1024
_ID_PATTERN = re.compile(r"a-[0-9a-f]{16}\Z")


class AssetLibrary:
    """Store each prop as a validated JSON file under ``.runtime/scene-assets``.

    ``save`` returns the canonical assets, including their stable library IDs.
    ``load_all`` returns recent assets for a generator prompt, and ``get`` reads
    one asset by ID. Missing or corrupt files are ignored on reads.
    """

    def __init__(self, directory=None):
        self.directory = Path(directory) if directory is not None else Path(__file__).resolve().parent / ".runtime" / "scene-assets"
        if self.directory.is_symlink():
            raise ValueError("Asset library directory must not be a symlink")
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not self.directory.is_dir():
            raise ValueError("Asset library path must be a directory")
        self.directory.chmod(0o700)

    @staticmethod
    def _asset_id(asset):
        # A name or model-supplied ID does not change the reusable geometry.
        geometry = json.dumps(asset["parts"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return "a-" + hashlib.sha256(geometry).hexdigest()[:16]

    def _path(self, identifier):
        if not isinstance(identifier, str) or not _ID_PATTERN.fullmatch(identifier):
            raise ValueError("Invalid scene asset identifier")
        return self.directory / (identifier + ".json")

    def get(self, identifier):
        """Return a detached canonical asset, or None when it is unavailable."""
        path = self._path(identifier)
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_ASSET_BYTES:
                return None
            with path.open("rb") as stream:
                payload = stream.read(MAX_ASSET_BYTES + 1)
            if len(payload) > MAX_ASSET_BYTES:
                return None
            raw = json.loads(payload)
            canonical = validate_assets([raw])[0]
            if canonical["id"] != identifier or self._asset_id(canonical) != identifier:
                return None
            return canonical
        except (OSError, ValueError, TypeError, IndexError, KeyError):
            return None

    def load_all(self, limit=16):
        """Return up to ``limit`` recent valid assets (default: prompt sized)."""
        if not isinstance(limit, int) or isinstance(limit, bool) or not 0 <= limit <= MAX_LIBRARY_ASSETS:
            raise ValueError(f"Asset limit must be between 0 and {MAX_LIBRARY_ASSETS}")
        if limit == 0:
            return []
        candidates = []
        try:
            for path in self.directory.glob("a-*.json"):
                if path.is_symlink() or not _ID_PATTERN.fullmatch(path.stem):
                    continue
                try:
                    candidates.append((path.stat().st_mtime_ns, path.stem))
                except OSError:
                    continue
        except OSError:
            return []
        assets = []
        for _, identifier in sorted(candidates, reverse=True):
            asset = self.get(identifier)
            if asset is not None:
                assets.append(asset)
                if len(assets) == limit:
                    break
        return assets

    def list_assets(self):
        """Return all stored valid assets, newest first, up to library capacity."""
        return self.load_all(limit=MAX_LIBRARY_ASSETS)

    def save(self, assets):
        """Validate and save one asset pack, returning the stored canonical assets.

        Validation and quota checks finish before any file changes. Identical
        geometry reuses an existing file and its original display name.
        """
        canonical = validate_assets(assets)
        prepared = []
        for asset in canonical:
            identifier = self._asset_id(asset)
            stored = {**asset, "id": identifier}
            payload = (json.dumps(stored, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
            if len(payload) > MAX_ASSET_BYTES:
                raise ValueError("Scene asset exceeds storage size limit")
            prepared.append((identifier, stored, payload))

        existing = {asset["id"]: asset for asset in self.list_assets()}
        additions = {identifier for identifier, _, _ in prepared if identifier not in existing}
        if len(existing) + len(additions) > MAX_LIBRARY_ASSETS:
            raise ValueError("Scene asset library is full")

        result = []
        committed = []
        try:
            for identifier, asset, payload in prepared:
                if identifier in existing:
                    result.append(existing[identifier])
                    continue
                path = self._path(identifier)
                # A malformed file with a matching name must never be replaced
                # silently. Removing it is a separate, deliberate user action.
                if path.exists() or path.is_symlink():
                    raise ValueError(f"Scene asset ID conflict: {identifier}")
                fd, temporary = tempfile.mkstemp(prefix=".asset-", suffix=".tmp", dir=self.directory)
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    # Hard-linking is atomic and fails if another writer has
                    # created this ID since the existence check above.
                    os.link(temporary, path)
                    committed.append(path)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
                existing[identifier] = asset
                result.append(asset)
        except (OSError, ValueError):
            for path in committed:
                path.unlink(missing_ok=True)
            raise
        return result
