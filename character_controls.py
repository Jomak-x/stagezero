"""Shared character selection with client-confirmed GLB activation."""
from __future__ import annotations

from dataclasses import dataclass, replace
from html import escape
from io import BytesIO
from pathlib import Path
from queue import SimpleQueue, Empty
from threading import Lock, RLock
import hashlib
import json
import tempfile
import numpy as np
from PIL import Image

from bounded_upload import (ScopedUploadLimits, acquire_scoped_upload_limits,
                            release_scoped_upload_limits)
from character_assets import DEFAULT_LIMITS, import_glb, load_character_asset
from character_compatibility import assess_character, inspect_character, unsupported_character
from character_diagnostics import standing_reference
from character_geometry import ground_offset
from character_guide import add_character_guide
from character_renderer import GlbCharacterRenderer


@dataclass
class CharacterEntry:
    asset: object
    retargeter: object | None
    reason: str
    compatibility: object
    ground_offset: float
    mapping: object = None
    preview: bool = False


class CharacterControls:
    """Keep character changes separate from motion history and inference IDs.

    Upload parsing happens in the GUI worker. Render acknowledgements are drained
    by tick(), so inference state is only changed after a successful browser load.
    """

    def __init__(self, server, session, skeleton, storage_root, *, renderer=None):
        self.server, self.session, self.skeleton = server, session, skeleton
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.entries = {}
        self.active_id = None
        self.active_entry = None
        self.revision = 0
        self.status = 'G1 robot · ready for motion'
        self._lock = RLock()
        self._import_lock = Lock()
        self._results = SimpleQueue()
        self._ticket = 0
        self._pending = None
        self._candidate_id = None
        self.compatibility = None
        self._requested_id = None
        self._initial_id = None
        self._last_pose_key = None
        self._root = None
        self._controls = None
        self._diagnostics = None
        self._mapping_binding = None
        self._mapping_folder = None
        self._mapping_gui = None
        self.renderer = renderer or GlbCharacterRenderer(server, on_result=self._on_result)
        self.upload_limits = None
        self.creation = None
        self._upload_gui = None
        self._upload_handle = None
        self._restore_catalog()

    def _on_result(self, client_id, asset_id, revision, status, error):
        self._results.put((client_id, asset_id, revision, status, error))

    def _set_error(self, message):
        with self._lock:
            self.status = 'Load failed · ' + str(message)[:240]

    def _entry(self, asset, mapping=None, *, report=None, preview=False):
        report = report or assess_character(asset, mapping=mapping, skeleton=self.skeleton)
        rig = None if preview else report.retargeter
        generated = not preview and mapping is None and report.motion_ready and self._generated_origin(asset.sha256)
        if generated:
            from generated_character_rig import build_generated_retargeter
            rig = build_generated_retargeter(asset, self.skeleton)
            report = replace(report, retargeter=rig,
                             title='Generated character',
                             reasons=('Automatic body fit. Fingers and face remain still.',))
        world = None
        if rig is not None:
            world = rig.retarget(*standing_reference(rig.skeleton)).world_matrices
        reason = report.title + ' · ' + ' '.join(report.reasons)
        if preview:
            reason = 'Static preview · motion generation disabled. ' + ' '.join(report.reasons)
        return CharacterEntry(asset, rig, reason, report, 0.0 if generated else ground_offset(asset, world), mapping, preview)

    def _generated_origin(self, asset_id):
        path = self.storage_root / asset_id / 'generated.json'
        try:
            if path.is_symlink() or path.stat().st_size > 4096:
                return False
            doc = json.loads(path.read_text())
            return (isinstance(doc, dict) and doc.get('version') == 1
                    and doc.get('source') in ('neon-trellis', 'gemini-neon-trellis', 'preset', 'legacy')
                    and doc.get('asset_id') == asset_id)
        except (OSError, ValueError):
            return False

    @property
    def selection_revision(self):
        with self._lock:
            return self._ticket

    def select_generated(self, asset_id, client_id, expected_revision):
        """Do not replace a character explicitly selected while generation ran."""
        with self._lock:
            if self._ticket != expected_revision:
                return False
            self.select(asset_id, client_id)
            return True

    def add_generated_file(self, data, prompt, *, source='gemini-neon-trellis', display_name=None,
                           preset_key=None, legacy_id=None):
        """Fit a generated body, then store it in the existing GLB catalog."""
        from generated_character_rig import export_generated_character, build_generated_retargeter
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 800:
            raise ValueError('Describe a person in 1–800 characters')
        with self._import_lock:
            with self._lock:
                if len(self.entries) >= 16:
                    raise ValueError('Character library is full (16 models)')
            rigged = export_generated_character(data, self.skeleton)
            name = display_name or 'Generated - ' + ' '.join(prompt.split())[:60]
            asset, report = inspect_character(rigged, display_name=name, skeleton=self.skeleton)
            if asset is None or not report.motion_ready:
                raise ValueError('Could not fit this character. Try one person with separated arms and legs.')
            # Validate our fitted adapter before writing anything to the catalog.
            build_generated_retargeter(asset, self.skeleton)
            import_glb(rigged, self.storage_root, display_name=name)
            folder = self.storage_root / asset.sha256
            staged = None
            try:
                origin_file = folder / 'generated.json'
                aliases = []
                try:
                    if not origin_file.is_symlink() and origin_file.stat().st_size <= 4096:
                        old = json.loads(origin_file.read_text())
                        if old.get('version') == 1 and old.get('asset_id') == asset.sha256:
                            aliases = [alias for alias in old.get('aliases', [])
                                       if isinstance(alias, dict) and set(alias) == {'source', 'field', 'value'}][:16]
                            for field in ('preset_key', 'legacy_id'):
                                if old.get(field):
                                    previous = dict(source=old.get('source'), field=field, value=old[field])
                                    if previous not in aliases:
                                        aliases.append(previous)
                except (OSError, ValueError, AttributeError, TypeError):
                    pass
                for field, value in (('preset_key', preset_key), ('legacy_id', legacy_id)):
                    alias = dict(source=source, field=field, value=value)
                    if value and alias not in aliases:
                        aliases.append(alias)
                with tempfile.NamedTemporaryFile('w', dir=folder, prefix='.generated-', suffix='.tmp', encoding='utf-8', delete=False) as f:
                    staged = Path(f.name)
                    json.dump({'version': 1, 'source': source, 'asset_id': asset.sha256,
                               'prompt': prompt.strip(), **({'preset_key': preset_key} if preset_key else {}),
                               **({'legacy_id': legacy_id} if legacy_id else {}),
                               'aliases': aliases[:16]},
                              f, ensure_ascii=False)
                staged.replace(folder / 'generated.json')
            finally:
                if staged is not None:
                    staged.unlink(missing_ok=True)
            entry = self._entry(asset, report=report)
            with self._lock:
                self.entries[asset.sha256] = entry
            return asset.sha256

    def _saved_origin(self, field, value, source):
        """Find a previously fitted trusted local source in this catalog."""
        for asset_id in self.entries:
            path = self.storage_root / asset_id / 'generated.json'
            try:
                if path.is_symlink() or path.stat().st_size > 4096:
                    continue
                doc = json.loads(path.read_text())
                alias = dict(source=source, field=field, value=value)
                if (doc.get('version') == 1 and doc.get('asset_id') == asset_id
                        and ((doc.get('source') == source and doc.get(field) == value)
                             or alias in doc.get('aliases', []))):
                    return asset_id
            except (OSError, ValueError, AttributeError):
                continue
        return None

    def _saved_preset(self, key):
        return self._saved_origin('preset_key', key, 'preset')

    def use_preset(self, key, client_id):
        """Fit a bundled static model once, then use the normal confirmed swap."""
        from character_presets import PRESETS, preset_data
        import io
        import trimesh
        preset = next((item for item in PRESETS if item['key'] == key), None)
        if preset is None:
            raise ValueError('Unknown character preset')
        with self._lock:
            existing = self._saved_preset(key)
        if existing is None:
            data, _ = preset_data(key)
            # Two bundled source meshes omit NORMAL accessors. Re-exporting
            # locally supplies them while retaining their embedded texture.
            scene = trimesh.load(io.BytesIO(data), file_type='glb', force='scene', process=False)
            data = trimesh.exchange.gltf.export_glb(scene, include_normals=True)
            existing = self.add_generated_file(data, preset['name'], source='preset',
                                               display_name=preset['name'], preset_key=key)
        self.select(existing, client_id)
        return existing

    def import_legacy_catalog(self, folder):
        """Copy a bounded flat cast into this catalog; leave source files intact.

        The caller should point this at a separate legacy directory during
        startup. A source active ID becomes the initial browser selection.
        """
        from character_library import CharacterLibrary
        import io
        import trimesh
        source = Path(folder)
        if source.resolve() == self.storage_root.resolve():
            raise ValueError('Legacy source and current character catalog must be separate')
        if not source.is_dir() or source.is_symlink():
            return {}
        library = CharacterLibrary(source)
        active = library.active()
        imported = {}
        for item in library.entries()[:16]:
            legacy_id = item['id']
            current = self._saved_origin('legacy_id', legacy_id, 'legacy')
            if current is None:
                try:
                    data = library.read(legacy_id)
                    scene = trimesh.load(io.BytesIO(data), file_type='glb', force='scene', process=False)
                    data = trimesh.exchange.gltf.export_glb(scene, include_normals=True)
                    current = self.add_generated_file(data, str(item.get('prompt') or item['name'])[:800],
                                                      source='legacy', display_name=item['name'][:70],
                                                      legacy_id=legacy_id)
                except (OSError, ValueError):
                    continue
            imported[legacy_id] = current
        if active in imported:
            self.set_initial_asset(imported[active])
        return imported

    def _restore_catalog(self):
        # A bounded startup catalog; imports are local to this installation.
        for manifest in sorted(self.storage_root.glob('*/manifest.json'))[:16]:
            try:
                asset = load_character_asset(self.storage_root, manifest.parent.name)
                mapping_file = manifest.parent / 'mapping.json'
                mapping = mapping_file.read_text() if mapping_file.is_file() and mapping_file.stat().st_size <= 1_048_576 else None
                self.entries[asset.sha256] = self._entry(asset, mapping)
            except (ValueError, OSError, KeyError) as exc:
                self.status = 'Skipped an invalid saved character · ' + str(exc)[:160]
                self.compatibility = unsupported_character(exc)

    def add_file(self, data, name, *, ticket=None):
        if not name.lower().endswith('.glb'):
            raise ValueError('Choose a .glb file')
        if not isinstance(data, bytes):
            raise ValueError('GLB upload must be bytes')
        if len(data) > DEFAULT_LIMITS.max_file_bytes:
            raise ValueError('GLB exceeds the file size limit')
        digest = hashlib.sha256(data).hexdigest()
        with self._import_lock:
            with self._lock:
                if digest in self.entries:
                    return digest
                if len(self.entries) >= 16:
                    raise ValueError('Character library is full (16 models); use a new character directory')
            asset, report = inspect_character(data, display_name=Path(name).name, skeleton=self.skeleton)
            if asset is None:
                with self._lock:
                    if ticket is None or ticket == self._ticket:
                        self.compatibility = report
                raise ValueError(report.title + ' · ' + ' '.join(report.reasons))
            entry = self._entry(asset, report=report)
            import_glb(data, self.storage_root, display_name=Path(name).name)
            with self._lock:
                self.entries[asset.sha256] = entry
        return asset.sha256

    def set_initial_asset(self, asset_id):
        self._initial_id = asset_id

    def on_client_connect(self, client):
        with self._lock:
            if self._initial_id is not None:
                asset_id, self._initial_id = self._initial_id, None
                self.select(asset_id, client.client_id)
            if self.active_entry is not None:
                self.frame_character(client)

    def select(self, asset_id, client_id):
        with self._lock:
            self._ticket += 1
            self._cancel_pending()
            self._candidate_id = None
            if asset_id is None:
                self.renderer.restore_g1()
                self._pending = None
                self._requested_id = None
                self.active_id = None
                self.active_entry = None
                self.revision += 1
                self._last_pose_key = None
                self._root = None
                self.compatibility = None
                self.session.set_character_motion_enabled(True)
                self.status = 'G1 robot · ready for motion'
                return
            saved = self.entries[asset_id]
            entry = self._entry(saved.asset, saved.mapping)
            self.entries[asset_id] = entry
            self.compatibility = entry.compatibility
            self._requested_id = asset_id
            if not entry.compatibility.motion_ready:
                self._candidate_id = asset_id
                self.status = entry.asset.display_name + ' · ' + entry.reason + ' Current character is unchanged.'
                return
            self._start_load(asset_id, entry, client_id)

    def _cancel_pending(self):
        if self._pending is not None:
            self.renderer.reject(self._pending[0])
            self._pending = None

    def _start_load(self, asset_id, entry, client_id):
        required_nodes = tuple(sorted(set(entry.retargeter.profile.bones.values()))) if entry.retargeter else ()
        revision = self.renderer.load(asset_id, entry.asset.glb_bytes, client_id,
                                      required_nodes=required_nodes, ground_offset=entry.ground_offset)
        self._pending = (revision, asset_id, client_id, entry)
        self._candidate_id = None
        self._requested_id = asset_id
        self.status = 'Loading · ' + entry.asset.display_name

    def open_static_preview(self, client_id):
        """Explicit consent to replace the actor with the candidate's rest pose."""
        with self._lock:
            if self._candidate_id is None:
                return
            asset_id = self._candidate_id
            saved = self.entries[asset_id]
            if not saved.compatibility.can_preview:
                return
            self._ticket += 1
            entry = self._entry(saved.asset, saved.mapping, preview=True)
            self._start_load(asset_id, entry, client_id)

    def choose_another_file(self):
        """Dismiss a proposed selection without affecting the committed actor."""
        with self._lock:
            self._ticket += 1
            self._cancel_pending()
            self._candidate_id = None
            self._requested_id = self.active_id
            self.compatibility = self.active_entry.compatibility if self.active_entry else None
            self.status = 'Choose another .glb file with Load GLB. Current character is unchanged.'

    def apply_mapping(self, asset_id, mapping, client_id):
        """Validate and persist a mapping, then request a browser-confirmed swap."""
        with self._lock:
            saved = self.entries[asset_id]
            if not saved.compatibility.can_map:
                raise ValueError('This model needs rig preparation in Blender; a mapping cannot create bones or weights')
            entry = self._entry(saved.asset, mapping)
            if not entry.compatibility.motion_ready:
                self.compatibility = entry.compatibility
                raise ValueError(entry.reason)
            text = mapping.decode('utf-8') if isinstance(mapping, bytes) else mapping
            parsed = json.loads(text) if isinstance(text, str) else text
            (self.storage_root / asset_id / 'mapping.json').write_text(json.dumps(parsed, indent=2))
            self.entries[asset_id] = entry
            self.select(asset_id, client_id)

    def _pose(self, entry):
        if entry.retargeter is None:
            return None
        with self.session.lock:
            positions = self.session.positions[self.session.frame].copy()
            rotations = self.session.rotations[self.session.frame].copy()
        return entry.retargeter.retarget(positions, rotations)

    def tick(self, pose_key=None):
        self.renderer.poll()
        if self.upload_limits is not None:
            self.upload_limits.poll()
        frame_request = None
        with self._lock:
            while True:
                try:
                    client, asset_id, revision, status, error = self._results.get_nowait()
                except Empty:
                    break
                pending = self._pending
                if pending is None or (revision, asset_id, client) != pending[:3]:
                    continue
                if status != 'loaded':
                    self.renderer.reject(revision)
                    self._pending = None
                    self._requested_id = self.active_id
                    self.compatibility = self.active_entry.compatibility if self.active_entry else None
                    self._set_error(error or 'Browser could not load this model')
                    continue
                entry = pending[3]
                try:
                    pose = self._pose(entry)
                    if pose is not None:
                        self.renderer.set_pose(pose.local_matrices, revision=revision)
                    if not self.renderer.commit(revision):
                        continue
                except (ValueError, RuntimeError) as exc:
                    self.renderer.reject(revision)
                    self._pending = None
                    self._requested_id = self.active_id
                    self.compatibility = self.active_entry.compatibility if self.active_entry else None
                    self._set_error(exc)
                    continue
                self.active_id = asset_id
                self.active_entry = entry
                self._pending = None
                self.revision += 1
                self._last_pose_key = None
                self.session.set_character_motion_enabled(entry.retargeter is not None)
                self.status = entry.asset.display_name + ' · ' + entry.reason
                frame_request = (client, self.revision)
            key = (self.revision, pose_key)
            if self.active_id is not None and key != self._last_pose_key:
                entry = self.active_entry
                try:
                    pose = self._pose(entry)
                    if pose is not None:
                        self.renderer.set_pose(pose.local_matrices)
                        self._root = pose.root_position.copy()
                    else:
                        self._root = entry.asset.bounds.mean(axis=0)
                        self._root[1] = 0.
                    self._root[1] += entry.ground_offset
                    self._last_pose_key = key
                except (ValueError, RuntimeError) as exc:
                    # A bad transform must never reach the GPU or lose the source take.
                    self.select(None, None)
                    self._set_error('Motion mapping failed; showing G1. ' + str(exc))
            if frame_request is not None and frame_request[1] == self.revision and self.active_entry is not None:
                self.frame_character(self.server.get_clients().get(frame_request[0]))
            self._update_gui()

    def actor_root(self):
        with self._lock:
            if self._root is not None:
                return self._root.copy()
        with self.session.lock:
            return self.session.positions[self.session.frame, 0].copy()

    @property
    def mapping_asset_id(self):
        """The same asset the selection control currently presents to the user."""
        with self._lock:
            if self._candidate_id is not None:
                return self._candidate_id
            return self._pending[1] if self._pending is not None else self.active_id

    def frame_character(self, client):
        if client is None:
            return
        with self._lock:
            entry = self.active_entry
            if entry is None:
                low, high = np.array((-0.6, 0., -0.4)), np.array((0.6, 1.5, 0.4))
            else:
                low, high = entry.asset.bounds
            target = (low + high) / 2
            if entry is not None and entry.retargeter is not None:
                rest_root = entry.retargeter.retarget(*entry.retargeter.neutral_source_pose()).root_position
                target += self.actor_root() - rest_root
            elif entry is not None:
                target[1] += entry.ground_offset
            radius = max(float(np.linalg.norm(high - low)) / 2, .25)
            distance = radius / np.sin(np.deg2rad(21.)) * 1.2
            direction = np.array((.55, .25, 1.))
            direction /= np.linalg.norm(direction)
            client.camera.position = target + direction * distance
            client.camera.look_at = target
            client.camera.up_direction = (0., 1., 0.)
            client.camera.fov = np.deg2rad(42.)

    def build_gui(self, gui):
        gui.add_html('<div class="sz-section">Character<small>Choose a saved person or create one from a description.</small></div>')
        self._preset_gui = gui
        presets = gui.add_button('Browse ready-made characters')
        choose = gui.add_dropdown('Character', ('G1 robot',))
        from character_creation import CharacterCreation
        self.creation = CharacterCreation(self.server, self)
        self.creation.build_gui(gui)
        with gui.add_folder('Import a rigged model', expand_by_default=False):
            upload = gui.add_upload_button('Load GLB', mime_type='.glb')
            add_character_guide(gui)
        self._mapping_gui = gui
        self._mapping_folder = gui.add_folder('Rig mapping', visible=False)
        with self._mapping_folder:
            mapping = gui.add_upload_button('Load rig mapping', mime_type='.json', visible=False)
        preview = gui.add_button('Open static preview', visible=False)
        another = gui.add_button('Choose another file', visible=False)
        frame = gui.add_button('Frame character')
        status = gui.add_html('')
        with gui.add_folder('Rig diagnostics', expand_by_default=False):
            self._diagnostics = gui.add_html('')
        gui.add_markdown('Body motion is approximate; fingers and faces stay still. Character selection is shared between viewers.')
        self._controls = (choose, mapping, status, preview, another)
        self.upload_limits = acquire_scoped_upload_limits(self.server, gui=gui,
                                                           factory=ScopedUploadLimits)
        self._upload_gui = gui
        self._upload_handle = upload
        self.upload_limits.register(upload, max_bytes=32 * 1024 * 1024, on_error=self._set_error)
        self.upload_limits.register(mapping, max_bytes=1024 * 1024, on_error=self._set_error)
        self._bind_mapping_upload(mapping, None, self._ticket)

        @presets.on_click
        def browse_presets(event):
            client = getattr(event, 'client', None)
            if client is not None:
                self.open_presets(client)

        @upload.on_upload
        def uploaded(event):
            if event.client is None:
                return
            with self._lock:
                self._ticket += 1
                ticket = self._ticket
                self._cancel_pending()
                self._candidate_id = None
                self._requested_id = self.active_id
            try:
                asset_id = self.add_file(event.file.content, event.file.name, ticket=ticket)
                with self._lock:
                    if ticket == self._ticket:
                        self.select(asset_id, event.client.client_id)
            except (ValueError, OSError) as exc:
                with self._lock:
                    if ticket == self._ticket:
                        self.compatibility = unsupported_character(exc)
                        self._set_error(exc)

        @choose.on_update
        def chosen(event):
            if event.client is not None:
                with self._lock:
                    asset_id = self._options().get(choose.value)
                    self.select(asset_id, event.client.client_id)

        @preview.on_click
        def previewed(event):
            if event.client is not None:
                self.open_static_preview(event.client.client_id)

        @another.on_click
        def dismissed(event):
            if event.client is not None:
                self.choose_another_file()

        @frame.on_click
        def framed(event):
            self.frame_character(event.client)

    def open_presets(self, client):
        """Show reference cards without replacing the current actor on open."""
        from character_presets import PRESETS, preset_reference
        gui = getattr(client, 'gui', None) or self._preset_gui
        if not hasattr(gui, 'add_modal'):
            self._set_error('Character gallery is unavailable in this client')
            return
        modal = gui.add_modal('Choose your character', size='lg', show_close_button=True)
        with modal:
            gallery_status = gui.add_html('<div class="sz-note">Ready to animate · choose a character.</div>')
            for preset in PRESETS:
                with Image.open(BytesIO(preset_reference(preset['key']))) as image:
                    image.thumbnail((256, 384))
                    gui.add_image(np.asarray(image.convert('RGB')), label=preset['name'])
                gui.add_html('<div class="sz-note">' + escape(preset['detail']) + '</div>')
                button = gui.add_button('Use ' + preset['name'])

                @button.on_click
                def selected(event, key=preset['key']):
                    try:
                        self.use_preset(key, event.client.client_id)
                        self._close_modal(modal)
                    except (ValueError, OSError) as exc:
                        gallery_status.content = ('<div class="sz-note">Could not load character: '
                                                  + escape(str(exc)[:180]) + '</div>')
            close = gui.add_button('Back to scene')
            close.on_click(lambda _: self._close_modal(modal))

    @staticmethod
    def _close_modal(modal):
        try:
            modal.close()
        except KeyError:
            pass

    def _bind_mapping_upload(self, handle, asset_id, ticket):
        """Bind an immutable upload handle to one asset and selection epoch."""
        self._mapping_binding = (asset_id, ticket)

        @handle.on_upload
        def mapped(event):
            if event.client is None:
                return
            with self._lock:
                if ticket != self._ticket or asset_id != self.mapping_asset_id:
                    return
                if asset_id is None:
                    self._set_error('Load a rigged GLB before its mapping')
                    return
                try:
                    self.apply_mapping(asset_id, event.file.content, event.client.client_id)
                except (ValueError, OSError) as exc:
                    self._set_error(exc)

    def _mapping_control(self, target):
        choose, mapping, status, preview, another = self._controls
        if self._mapping_binding != (target, self._ticket):
            # Remove the old GUI UUID and its buffers together. A callback
            # already queued by Viser still fails its captured epoch check.
            self.upload_limits.unregister(mapping, remove=True)
            with self._mapping_folder:
                mapping = self._mapping_gui.add_upload_button('Load rig mapping', mime_type='.json', visible=False)
            self.upload_limits.register(mapping, max_bytes=1024 * 1024, on_error=self._set_error)
            self._bind_mapping_upload(mapping, target, self._ticket)
            self._controls = (choose, mapping, status, preview, another)
        return mapping

    def _options(self):
        return {'G1 robot': None, **{f'{entry.asset.display_name} · {asset_id[:8]}': asset_id
                for asset_id, entry in self.entries.items()}}

    def _update_gui(self):
        if self._controls is None:
            return
        choose, mapping, status, preview, another = self._controls
        options = self._options()
        if choose.options != tuple(options):
            choose.options = tuple(options)
        target = self.mapping_asset_id
        mapping = self._mapping_control(target)
        label = next((label for label, value in options.items() if value == target), 'G1 robot')
        if choose.value != label:
            choose.value = label
        entry = self.entries.get(target)
        mapping.visible = (entry is not None and entry.compatibility.can_map
                           and not self._generated_origin(target)
                           and (self.compatibility is None or self.compatibility.status != 'unsupported'))
        mapping.disabled = not mapping.visible
        self._mapping_folder.visible = mapping.visible
        preview.visible = self._candidate_id is not None
        preview.disabled = not preview.visible
        another.visible = self._candidate_id is not None or self._pending is not None or (
            self.compatibility is not None and self.compatibility.status == 'unsupported')
        content = '<div class="sz-note">' + escape(self.status) + '</div>'
        if status.content != content:
            status.content = content
        if self._diagnostics is not None:
            report = self.compatibility
            details = 'Built-in G1 rig' if report is None else report.title + ' · ' + ' '.join(report.reasons)
            if entry is not None:
                details += f' · {entry.asset.triangle_count:,} triangles'
            if report is not None:
                details += ' · Checked: ' + ', '.join(report.checks) if report.checks else ''
                details += ' · ' + ' '.join(report.warnings)
                details += ' · Actions: ' + ', '.join(report.actions)
            details = '<div class="sz-note">' + escape(details) + '</div>'
            if self._diagnostics.content != details:
                self._diagnostics.content = details

    def close(self):
        if self.creation is not None:
            self.creation.stop()
        if self.upload_limits is not None:
            if self._upload_handle is not None:
                self.upload_limits.unregister(self._upload_handle, remove=True)
            if self._controls is not None:
                self.upload_limits.unregister(self._controls[1], remove=True)
            release_scoped_upload_limits(self._upload_gui)
            self.upload_limits = None
        self.renderer.restore_g1()
