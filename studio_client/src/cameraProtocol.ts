/** Fixed camera records use Viser/OpenCV axes: +Z forward and +Y down. */
export type CameraRecord = {
  id: string;
  name: string;
  position: [number, number, number];
  wxyz: [number, number, number, number];
  fov: number; // Vertical field of view, radians.
};

export type CameraCut = {
  id: string;
  frame: number;
  camera_id: string;
};

export type CameraMode = "free" | "fixed" | "sequence";

export type CameraStudioStateMessage = {
  type: "CameraStudioStateMessage";
  project_id: string;
  revision: number;
  take_id: string | null;
  frame: number;
  length: number;
  fps: number;
  cameras: CameraRecord[];
  cuts: CameraCut[];
  mode: CameraMode;
  active_camera_id: string | null;
  selected_camera_id: string | null;
  busy: boolean;
  error: string | null;
};

export type CameraAction =
  | "add"
  | "select"
  | "update"
  | "duplicate"
  | "delete"
  | "view"
  | "free"
  | "sequence"
  | "cut_add"
  | "cut_update"
  | "cut_delete"
  | "cuts_clear";

export type CameraCommandPayload = {
  camera_id?: string;
  cut_id?: string;
  frame?: number;
  camera?: Partial<Pick<CameraRecord, "name" | "position" | "wxyz" | "fov">>;
};

export type CameraStudioCommandMessage = CameraCommandPayload & {
  type: "CameraStudioCommandMessage";
  project_id: string;
  revision: number;
  take_id: string | null;
  action: CameraAction;
};
