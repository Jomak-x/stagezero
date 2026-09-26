// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import { createMessageThrottle } from "./MessageThrottle";
import * as THREE from "three";
import { Message } from "./WebsocketMessages";
import { ViewerContext, ViewerContextContents } from "./ViewerContext";

/** Easier, hook version of makeThrottledMessageSender. */
export function useThrottledMessageSender(throttleMilliseconds: number) {
  const viewer = React.useContext(ViewerContext)!;
  const sender = React.useMemo(
    () => makeThrottledMessageSender(viewer, throttleMilliseconds),
    [viewer, throttleMilliseconds],
  );
  React.useEffect(() => () => sender.dispose(), [sender]);
  return sender;
}

/** Returns a function for sending messages, with automatic throttling. */
export function makeThrottledMessageSender(
  viewer: ViewerContextContents,
  throttleMilliseconds: number,
) {
  return createMessageThrottle<Message>(message => {
    viewer.mutable.current.sendMessage?.(message);
  }, throttleMilliseconds);
}

/** Type guard for threejs textures. Meant to be used with `scene.background`. */
export function isTexture(
  background:
    | THREE.Color
    | THREE.Texture
    | THREE.CubeTexture
    | null
    | undefined,
): background is THREE.Texture {
  return (
    background !== null &&
    background !== undefined &&
    (background as THREE.Texture).isTexture !== undefined
  );
}
