# Fixed cameras and camera cuts

The Studio supports project-wide perspective cameras and a separate hard-cut
sequence for each saved take. Camera edits and switching never request motion
generation. This feature implements the approved fixed-camera plan; smooth
transitions and video export are outside this version.

## Create and edit cameras

Open **View → Camera** in the inspector. In the camera
panel, **Add camera** captures the current viewport. Cameras are named Camera 1,
Camera 2, and so on. Select a camera to rename it, change its world position,
rotation or vertical field of view, duplicate it, or delete it.

In free view, camera outlines and labels are visible in the scene. Click an
outline to select it; its transform controls edit position and rotation.
Selecting a camera does not change the viewport. **View camera** switches to it
instantly without pausing playback. Navigation is locked while viewing a fixed
camera, and camera helpers are hidden. **Free view** restores the previous editor
view. Existing manual camera buttons leave fixed/sequence view before acting.

Camera definitions are shared by all connected viewers. Each viewer independently
chooses free view, a fixed camera or sequence preview. Character following never
moves a locked camera. Changing a shared camera updates viewers looking through it.

## Edit the cut sequence

The timeline has a **Cameras** row immediately below the motion actions. Select a
camera in its toolbar and choose **Add at playhead**. The first insertion also
creates a frame-zero starting shot if necessary. Each shot continues until the
next cut; the final shot continues to the end of the take.

Select a shot to change its camera or time, drag it to a new frame, or delete it.
Times snap to the take's 25 FPS frame grid. Assigning a cut at an occupied frame
replaces the existing assignment. The first cut must stay at frame zero; change
its camera or use **Clear cuts** to remove the entire sequence.

**Preview cuts** follows the camera sequence during playback, seeking, frame
stepping and looping. Manually viewing another camera leaves sequence preview.
Press **Preview cuts** again to use the shot at the current frame.

## Saving and take editing

**Save project + download** stores every camera and each take's cut sequence in
project format version 2. Version 1 files still open with empty camera libraries.
Invalid camera geometry, cut times or references reject the incoming project
before changing the current session. New project clears cameras; changing the
scene's props or lighting preserves them.

Duplicating a take copies its cuts. Trimming or changing motion preserves cuts
at their absolute frame positions and removes those outside the new duration.
Undoing a motion edit restores its camera sequence as well. Editing cuts after
a motion edit supersedes that edit's Undo snapshot, preventing later Undo from
silently discarding the new camera work.

A camera referenced by a take or an available Undo snapshot cannot be deleted.
The error names the affected takes. Reassign or clear their cuts first; restore
a removed take before editing its retained sequence.

## Implementation contract

- Camera records contain stable `id`, `name`, world `position`, normalized
  `wxyz` orientation and vertical `fov` in radians. The orientation follows
  Viser/OpenCV axes: +Z forward, +Y down. The scene's world up is +Y.
- `Take.camera_cuts` contains ordered `{id, frame, camera_id}` records. Nonempty
  sequences start at zero, have unique frames and reference project cameras.
- Session methods validate under the session lock and mark project changes
  unsaved. New/open project rotates a camera project identifier.
- `CameraStudioStateMessage` carries the authoritative library, take cuts and
  viewer mode. `CameraStudioCommandMessage` carries project identifier, revision
  and take identifier, so delayed edits cannot affect a different project/take.
- The server evaluates the last cut at or before the current frame; the browser
  never advances a separate camera clock. Camera endpoints and FOV are applied
  atomically. Unchanged locked views do not generate repeated camera writes.
- Browser stores are viewer-specific and reset on reconnect. The standalone
  camera row shares the timeline's existing frame/zoom coordinate system.

Verification covers camera CRUD, save/open and v1 compatibility, frame-zero and
collision rules, trim/copy/Undo behavior, fixed/free/sequence modes, two viewers,
stale commands, frontend coordinates and private synthetic browser playback.
