"""Character start placement controls for the StageZero studio."""

from __future__ import annotations

import math
from html import escape


class CharacterPlacement:
    """Keep the visible floor handle and the saved take's start pose in sync."""

    def __init__(self, server, session, camera):
        self.server = server
        self.session = session
        self.camera = camera
        self.active = False
        self._placing_take = None
        self.last_pose = None
        self.handle = server.scene.add_transform_controls(
            '/character-start-placement',
            scale=.72,
            active_axes=(True, False, True),
            disable_rotations=True,
            translation_limits=((-100., 100.), (0., 0.), (-100., 100.)),
            visible=False,
        )

        @self.handle.on_update
        def moved(event):
            if not self.active or getattr(event, 'client', None) is None:
                return
            if self.session.playing:
                self.active = False
                self.update()
                return
            x, _, z = self.handle.position
            self._place(float(x), float(z), float(self.facing.value))

    def build(self, gui):
        gui.add_html('<div class="sz-note">Set where your character begins before you generate an action. You can move a saved take later too.</div>')
        self.toggle = gui.add_button('Move character start')
        self.undo = gui.add_button('Undo move', color='gray', visible=False)
        self.status = gui.add_html('')
        with gui.add_folder('Exact start position', expand_by_default=False):
            gui.add_html('<div class="sz-note">Drag the red and blue arrows in the scene, or enter a position and facing direction here.</div>')
            x, z, heading = self._pose()
            self.x = gui.add_number('Left / right (m)', initial_value=x, min=-100., max=100., step=.1)
            self.z = gui.add_number('Forward / back (m)', initial_value=z, min=-100., max=100., step=.1)
            self.facing = gui.add_number('Facing (degrees)', initial_value=heading, min=-180., max=180., step=5.)
            self.center = gui.add_button('Center on stage', color='gray')

        @self.toggle.on_click
        def toggle(event):
            with self.session.lock:
                if self.session.busy:
                    return
                if self.active:
                    self.active = False
                else:
                    if self.session.mode != 'Live ARDY':
                        self.session.set_mode('Live ARDY')
                    self.session.seek(0)
                    self.active = True
                    self._placing_take = self.session.active_take
                    if getattr(event, 'client', None) is not None:
                        self.camera.reset(event.client)
                self.update()

        for control in (self.x, self.z, self.facing):
            @control.on_update
            def exact(event):
                if getattr(event, 'client', None) is None:
                    return
                self._place(float(self.x.value), float(self.z.value), float(self.facing.value))

        @self.undo.on_click
        def undo(_):
            with self.session.lock:
                if not self._can_undo_move():
                    return
                self.session.undo_edit()
                self.update()

        @self.center.on_click
        def center(_):
            self._place(0., 0., float(self.facing.value))

        self.update()

    def _pose(self):
        with self.session.lock:
            x, z, heading = self.session.get_start_pose()
        return float(x), float(z), float(heading)

    def _place(self, x, z, heading):
        if not all(math.isfinite(value) for value in (x, z, heading)):
            self._status('Enter a valid position and facing direction.')
            return
        if abs(x) > 100. or abs(z) > 100. or abs(heading) > 180.:
            self._status('Keep position within 100 m and facing within 180°.')
            return
        with self.session.lock:
            if self.session.busy:
                self._status('Wait for generation to finish before moving the character.')
                return
            if self.session.mode != 'Live ARDY':
                self.session.set_mode('Live ARDY')
            self.session.seek(0)
            try:
                self.session.set_start_pose(x, z, heading)
            except ValueError as exc:
                self._status(str(exc))
                return
            self.update()

    def _can_undo_move(self):
        snapshot = getattr(self.session, '_edit_undo', None) or {}
        return (not self.session.busy and bool(getattr(self.session, 'can_undo_edit', False))
                and snapshot.get('label') == 'move character start')

    def _status(self, message):
        content = '<div class="sz-note" style="white-space:normal;overflow-wrap:anywhere">' + escape(message) + '</div>'
        if hasattr(self, 'status') and self.status.content != content:
            self.status.content = content

    def update(self):
        """Refresh controls after take changes and after a drag, with no idle writes."""
        if not hasattr(self, 'toggle'):
            return
        x, z, heading = self._pose()
        busy = self.session.busy
        if self.active and (busy or self.session.playing or self.session.mode != 'Live ARDY'
                            or self.session.active_take != self._placing_take):
            self.active = False
        can_undo = self._can_undo_move()
        if self.undo.visible != can_undo:
            self.undo.visible = can_undo
        label = 'Done placing' if self.active else 'Move character start'
        if self.toggle.label != label:
            self.toggle.label = label
        if self.toggle.disabled != busy:
            self.toggle.disabled = busy
        for control, value in ((self.x, x), (self.z, z), (self.facing, heading)):
            if abs(float(control.value) - value) > 1e-4:
                control.value = value
            if control.disabled != busy:
                control.disabled = busy
        if self.center.disabled != busy:
            self.center.disabled = busy
        if self.handle.visible != self.active:
            self.handle.visible = self.active
        if self.active and (self.last_pose is None or (x, z) != self.last_pose[:2]):
            self.handle.position = (x, 0., z)
        self.last_pose = (x, z, heading)
        text = ('Drag the red and blue arrows to choose the start. Press Done placing when ready.'
                if self.active else
                f'Start: {x:.1f} m left/right · {z:.1f} m forward/back · facing {heading:.0f}°. Press Move character start to change it.')
        self._status(text)
