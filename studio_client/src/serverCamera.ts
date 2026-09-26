import type CameraControls from "camera-controls";
import { Vector3 } from "three";

/** Apply an authoritative server target without polar clamping against the old up vector. */
export function setServerCameraTarget(
  controls: Pick<CameraControls, "getPosition" | "setLookAt">,
  target: Vector3,
): void {
  const position = controls.getPosition(new Vector3());
  controls.setLookAt(
    position.x, position.y, position.z,
    target.x, target.y, target.z,
    false,
  );
}
