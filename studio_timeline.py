"""Read-only Viser timeline for stored takes and recorded preview playback.

The prompt blocks describe motion that already exists in a take. Changing a
block's text or bounds would not change those frames, so Viser's constraint
editing is disabled while its playhead remains available for scrubbing.
"""

from __future__ import annotations


SEGMENT_COLORS = (
    (72, 202, 183),   # teal
    (96, 165, 250),   # blue
    (192, 132, 252),  # violet
    (251, 191, 36),   # amber
    (251, 146, 60),   # orange
    (244, 114, 182),  # pink
)


class StudioTimeline:
    """Synchronize DirectorSession playback with Viser's bottom timeline.

    Call :meth:`update` from the viewer loop after ``session.tick()``. Viser
    dispatches playhead clicks through ``on_frame_change`` on its own worker.
    """

    def __init__(self, server, session):
        self.timeline = server.timeline
        self.session = session
        self._layout = None
        self._frame = None
        self._visible = None
        self.timeline.disable_constraints()
        self.timeline.on_frame_change(self._seek)
        self.update()

    def _seek(self, frame: int) -> None:
        # The native ruler can scroll past the take. DirectorSession.seek()
        # clamps to the actual final frame and pauses playback.
        with self.session.lock:
            if self.session.kind in ("recorded", "generated") and not self.session.busy:
                self.session.seek(frame)
                # The browser ruler can request a frame beyond the clip, so
                # publish the clamped position even if it matches our cache.
                self._frame = None

    def update(self) -> None:
        """Publish only changed layout or playhead values (safe at 10 Hz)."""
        with self.session.lock:
            kind = self.session.kind
            visible = kind in ("recorded", "generated")
            length = len(self.session.positions)
            fps = float(self.session.fps)
            frame = max(0, min(int(self.session.frame), length - 1))
            take = self.session.takes.get(self.session.active_take) if kind == "generated" else None
            segments = tuple(
                (int(s["start"]), int(s["end"]), str(s["prompt"]))
                for s in (take.segments if take is not None else ())
            )
            layout = (kind, self.session.active_take if take is not None else None,
                      length, fps, segments)

        if layout != self._layout:
            self.timeline.clear_prompts()
            for index, (start, end, prompt) in enumerate(segments):
                self.timeline.add_prompt(
                    prompt, start, end,
                    color=SEGMENT_COLORS[index % len(SEGMENT_COLORS)],
                    uuid=f"stagezero-segment-{index}",
                )
            self.timeline.set_fps(fps)
            self.timeline.set_zoom_settings(
                default_num_frames_zoom=max(1, length),
                max_frames_zoom=max(1, length),
            )
            # Viser's zoom setter can expand end_frame to max_frames_zoom.
            # Restore the actual last pose after configuring the viewport.
            self.timeline.set_frame_range(0, max(0, length - 1))
            self._layout = layout

        if visible != self._visible:
            self.timeline.set_visible(visible)
            self._visible = visible

        if visible and frame != self._frame:
            self.timeline.set_current_frame(frame)
            self._frame = frame
        elif not visible:
            self._frame = None
