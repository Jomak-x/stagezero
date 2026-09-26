import * as THREE from "three";

export const MIN_PITCH_MARGIN = 0.06;
export const DRAG_THRESHOLD_PX = 4;

export function isDragGesture(startX: number, startY: number, x: number, y: number): boolean {
  return Math.hypot(x - startX, y - startY) >= DRAG_THRESHOLD_PX;
}

export function panTranslation(
  cameraPosition: THREE.Vector3,
  target: THREE.Vector3,
  cameraQuaternion: THREE.Quaternion,
  fovDegrees: number,
  viewportHeight: number,
  dx: number,
  dy: number,
): THREE.Vector3 {
  const distance = Math.max(0.1, cameraPosition.distanceTo(target));
  const unitsPerPixel =
    (2 * distance * Math.tan((fovDegrees * Math.PI) / 360)) /
    Math.max(1, viewportHeight);
  const right = new THREE.Vector3(1, 0, 0).applyQuaternion(cameraQuaternion);
  const up = new THREE.Vector3(0, 1, 0).applyQuaternion(cameraQuaternion);
  return right.multiplyScalar(-dx * unitsPerPixel).addScaledVector(up, dy * unitsPerPixel);
}

export function lookTargetAfterDrag(
  cameraPosition: THREE.Vector3,
  target: THREE.Vector3,
  worldUp: THREE.Vector3,
  dx: number,
  dy: number,
  radiansPerPixel = 0.005,
): THREE.Vector3 {
  const offset = target.clone().sub(cameraPosition);
  const distance = Math.max(0.1, offset.length());
  const up = worldUp.clone().normalize();
  const vertical = THREE.MathUtils.clamp(offset.dot(up) / distance, -1, 1);
  const currentPitch = Math.asin(vertical);
  const pitch = THREE.MathUtils.clamp(
    currentPitch - dy * radiansPerPixel,
    -Math.PI / 2 + MIN_PITCH_MARGIN,
    Math.PI / 2 - MIN_PITCH_MARGIN,
  );
  const horizontal = offset.addScaledVector(up, -offset.dot(up));
  if (horizontal.lengthSq() < 1e-8) {
    horizontal.set(Math.abs(up.x) < 0.9 ? 1 : 0, Math.abs(up.x) < 0.9 ? 0 : 1, 0);
    horizontal.addScaledVector(up, -horizontal.dot(up)).normalize();
  } else {
    horizontal.normalize();
  }
  horizontal.applyAxisAngle(up, -dx * radiansPerPixel);
  return cameraPosition.clone().addScaledVector(horizontal, distance * Math.cos(pitch))
    .addScaledVector(up, distance * Math.sin(pitch));
}

export function walkTranslation(
  cameraPosition: THREE.Vector3,
  target: THREE.Vector3,
  worldUp: THREE.Vector3,
  forwardAmount: number,
  rightAmount: number,
  upAmount: number,
): THREE.Vector3 {
  const up = worldUp.clone().normalize();
  const forward = target.clone().sub(cameraPosition);
  forward.addScaledVector(up, -forward.dot(up));
  if (forward.lengthSq() < 1e-8) {
    forward.set(Math.abs(up.z) < 0.9 ? 0 : 1, 0, Math.abs(up.z) < 0.9 ? -1 : 0);
    forward.addScaledVector(up, -forward.dot(up)).normalize();
  } else {
    forward.normalize();
  }
  const right = new THREE.Vector3().crossVectors(forward, up).normalize();
  return forward.multiplyScalar(forwardAmount)
    .addScaledVector(right, rightAmount)
    .addScaledVector(up, upAmount);
}

export function wheelDollyDistance(distance: number, deltaY: number): number {
  return THREE.MathUtils.clamp(distance * Math.exp(THREE.MathUtils.clamp(deltaY, -200, 200) * 0.004), 0.1, 1000);
}
