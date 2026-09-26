// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useMemo, useRef, useState } from "react";
import type { ViewerContextContents } from "./ViewerContext";
import type { CameraCut } from "./cameraProtocol";
import type { CameraStoreState } from "./cameraStore";
import { getCameraStore, sendCameraCommand } from "./cameraStore";
import { cameraCutPreviewSegments, frameAfterCutDrag, boundedCutFrame, boundedPlayheadFrame, visibleCameraBlock } from "./cameraTimelineMath";
import { FRAME_LABELS_HEIGHT, TRACK_HEIGHT } from "./TimelineConstants";
import { TRACK_LABEL_WIDTH } from "./timeline/constants";
import type { TimelineCoordinates } from "./timeline/types";
import "./CameraTimeline.css";

type SharedProps = {
  viewer: ViewerContextContents;
  state: CameraStoreState;
  darkMode: boolean;
  selectedCutId: string | null;
  onSelectCut: (id: string | null) => void;
};

type TrackProps = SharedProps & {
  coords: TimelineCoordinates;
  width: number;
};

function cutName(state: CameraStoreState, cut: CameraCut): string {
  return state.cameras.find((camera) => camera.id === cut.camera_id)?.name ?? "Missing camera";
}

export function CameraTimelineToolbar({ viewer, state, darkMode, selectedCutId, onSelectCut }: SharedProps) {
  const selectedCut = state.cuts.find((cut) => cut.id === selectedCutId);
  const [addCameraId, setAddCameraId] = useState("");
  const [draftCameraId, setDraftCameraId] = useState("");
  const [draftSeconds, setDraftSeconds] = useState("");
  const fps = Math.max(1, state.fps || 30);
  const addCamera = state.cameras.find((camera) => camera.id === addCameraId)
    ?? state.cameras.find((camera) => camera.id === state.selected_camera_id)
    ?? state.cameras.find((camera) => camera.id === state.active_camera_id)
    ?? state.cameras[0];
  const canEdit = state.take_id !== null && !state.busy;

  useEffect(() => {
    setDraftCameraId(selectedCut?.camera_id ?? "");
    setDraftSeconds(selectedCut ? (selectedCut.frame / fps).toFixed(6) : "");
  }, [selectedCut?.id, selectedCut?.frame, selectedCut?.camera_id, fps]);

  const applySelectedCut = () => {
    if (!selectedCut || !canEdit) return;
    const parsedSeconds = Number(draftSeconds);
    if (!Number.isFinite(parsedSeconds) || parsedSeconds < 0) return;
    const frame = boundedCutFrame(parsedSeconds * fps, state.length, selectedCut.frame === 0);
    if (!draftCameraId || !state.cameras.some((camera) => camera.id === draftCameraId)) return;
    if (frame === selectedCut.frame && draftCameraId === selectedCut.camera_id) return;
    sendCameraCommand(viewer, "cut_update", {
      cut_id: selectedCut.id,
      frame,
      camera_id: draftCameraId,
    });
  };

  if (!state.received) return null;
  return <div className="sz-camera-toolbar" data-theme={darkMode ? "dark" : "light"} aria-label="Camera cuts controls">
    <span className="sz-camera-toolbar-title">Cameras <span>· {state.cuts.length} {state.cuts.length === 1 ? "cut" : "cuts"}</span></span>
    <label className="sz-camera-field">
      <span>Camera</span>
      <select aria-label="Camera for new cut" value={addCamera?.id ?? ""}
        onChange={(event) => setAddCameraId(event.target.value)} disabled={!canEdit || state.cameras.length === 0}>
        {state.cameras.length === 0 && <option value="">Add a camera first</option>}
        {state.cameras.map((camera) => <option key={camera.id} value={camera.id}>{camera.name}</option>)}
      </select>
    </label>
    <button type="button" disabled={!canEdit || !addCamera || state.length <= 0}
      title={state.cuts.length === 0 ? "Add at the playhead; an initial frame-0 cut is created automatically" : "Add or replace a camera cut at the playhead"}
      onClick={() => sendCameraCommand(viewer, "cut_add", {
        frame: boundedPlayheadFrame(state.frame, state.length),
        camera_id: addCamera.id,
      })}>Add at playhead</button>
    <span className="sz-camera-toolbar-divider" aria-hidden="true" />
    <button type="button" aria-pressed={state.mode === "sequence"}
      disabled={!state.take_id || state.cuts.length === 0}
      onClick={() => sendCameraCommand(viewer, "sequence")}>Preview cuts</button>
    <button type="button" aria-pressed={state.mode === "free"}
      onClick={() => sendCameraCommand(viewer, "free")}>Free view</button>
    <button type="button" disabled={!canEdit || state.cuts.length === 0}
      onClick={() => { sendCameraCommand(viewer, "cuts_clear"); onSelectCut(null); }}>Clear cuts</button>
    {selectedCut && <div className="sz-camera-cut-editor" role="group" aria-label="Selected camera cut">
      <span className="sz-camera-cut-frame">F {selectedCut.frame}</span>
      <label className="sz-camera-field"><span>Cut camera</span>
        <select aria-label="Selected cut camera" value={draftCameraId}
          onChange={(event) => setDraftCameraId(event.target.value)} disabled={!canEdit}>
          {state.cameras.map((camera) => <option key={camera.id} value={camera.id}>{camera.name}</option>)}
        </select>
      </label>
      <label className="sz-camera-field sz-camera-time"><span>Time (s)</span>
        <input aria-label="Selected cut time in seconds" type="number" min="0" max={Math.max(0, (state.length - 1) / fps)} step="any"
          value={draftSeconds} onChange={(event) => setDraftSeconds(event.target.value)}
          onKeyDown={(event) => { if (event.key === "Enter") applySelectedCut(); }}
          disabled={!canEdit || selectedCut.frame === 0} />
      </label>
      <button type="button" disabled={!canEdit} onClick={applySelectedCut}>Apply cut</button>
      <button type="button" disabled={!canEdit || selectedCut.frame === 0}
        onClick={() => { sendCameraCommand(viewer, "cut_delete", { cut_id: selectedCut.id }); onSelectCut(null); }}>Delete cut</button>
    </div>}
    {state.error && <span className="sz-camera-toolbar-error" role="alert">{state.error}</span>}
  </div>;
}

export function CameraTimelineTrack({ viewer, state, darkMode, selectedCutId, onSelectCut, coords, width }: TrackProps) {
  const [dragPreview, setDragPreview] = useState<{ id: string; frame: number } | null>(null);
  const dragRef = useRef<{ id: string; frame: number; clientX: number; pointerId: number; currentFrame: number; projectId: string; takeId: string | null; revision: number } | null>(null);
  useEffect(() => {
    const drag = dragRef.current;
    if (drag && (drag.projectId !== state.project_id || drag.takeId !== state.take_id || drag.revision !== state.revision)) {
      dragRef.current = null;
      setDragPreview(null);
    }
  }, [state.project_id, state.take_id, state.revision]);
  const segments = useMemo(() => cameraCutPreviewSegments(state.cuts, state.length, dragPreview),
    [state.cuts, state.length, dragPreview]);
  const segmentById = useMemo(() => new Map(segments.map((segment) => [segment.cut.id, segment])), [segments]);
  const cameraNameById = useMemo(() => new Map(state.cameras.map((camera) => [camera.id, camera.name])), [state.cameras]);
  const frameX = (frame: number) => coords.timelineStartX + (frame - coords.viewStartFrame) * coords.pixelRange;

  if (!state.received) return null;
  return <div className="sz-camera-track" data-theme={darkMode ? "dark" : "light"}
    style={{ top: FRAME_LABELS_HEIGHT + TRACK_HEIGHT, height: TRACK_HEIGHT }} aria-label="Camera cuts track">
    <div className="sz-camera-track-label" style={{ width: TRACK_LABEL_WIDTH }}>Cameras</div>
    {segments.length === 0 && <span className="sz-camera-track-empty" style={{ left: coords.timelineStartX }}>
      Add a camera cut to preview fixed shots
    </span>}
    {state.cuts.map((sourceCut) => {
      const segment = segmentById.get(sourceCut.id);
      if (!segment) return null;
      const { cut, endFrame, firstCut } = segment;
      const bounds = visibleCameraBlock(
        frameX(cut.frame), frameX(endFrame), coords.timelineStartX, width,
        dragPreview?.id === cut.id,
      );
      if (!bounds) return null;
      const duration = Math.max(0, endFrame - cut.frame) / Math.max(1, state.fps);
      const cameraName = cameraNameById.get(cut.camera_id) ?? cutName(state, cut);
      const selected = selectedCutId === cut.id;
      return <button key={cut.id} type="button" className="sz-camera-cut"
        data-selected={selected ? "true" : "false"} data-fixed={firstCut ? "true" : "false"}
        aria-label={`${cameraName}, starts at frame ${cut.frame}, lasts ${duration.toFixed(2)} seconds${firstCut ? ", fixed at frame 0" : ""}`}
        aria-pressed={selected} title={`${cameraName} · ${duration.toFixed(2)} s · F ${cut.frame}${firstCut ? " · first cut stays at frame 0" : " · drag to move"}`}
        style={bounds}
        onClick={() => onSelectCut(cut.id)}
        onPointerDown={(event) => {
          if (event.button !== 0) return;
          onSelectCut(cut.id);
          if (firstCut || state.busy || state.take_id === null) return;
          dragRef.current = { id: cut.id, frame: cut.frame, clientX: event.clientX, pointerId: event.pointerId, currentFrame: cut.frame,
            projectId: state.project_id, takeId: state.take_id, revision: state.revision };
          event.currentTarget.setPointerCapture(event.pointerId);
        }}
        onPointerMove={(event) => {
          const drag = dragRef.current;
          if (!drag || drag.id !== cut.id || drag.pointerId !== event.pointerId) return;
          const frame = frameAfterCutDrag(drag.frame, event.clientX - drag.clientX, coords.pixelRange, state.length);
          drag.currentFrame = frame;
          setDragPreview({ id: cut.id, frame });
        }}
        onPointerUp={(event) => {
          const drag = dragRef.current;
          if (!drag || drag.id !== cut.id || drag.pointerId !== event.pointerId) return;
          const latest = getCameraStore(viewer).getState();
          if (drag.projectId === latest.project_id && drag.takeId === latest.take_id && drag.revision === latest.revision &&
              drag.currentFrame !== drag.frame) {
            sendCameraCommand(viewer, "cut_update", { cut_id: cut.id, frame: drag.currentFrame });
          }
          dragRef.current = null;
          setDragPreview(null);
        }}
        onPointerCancel={() => { dragRef.current = null; setDragPreview(null); }}>
        <span className="sz-camera-cut-name">{cameraName}</span>
        <span className="sz-camera-cut-duration">{duration.toFixed(1)} s</span>
      </button>;
    })}
    {frameX(state.frame) >= coords.timelineStartX && frameX(state.frame) <= width &&
      <span className="sz-camera-track-playhead" aria-hidden="true" style={{ left: frameX(state.frame) }} />}
  </div>;
}
