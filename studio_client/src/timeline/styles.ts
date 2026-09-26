// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export interface TimelineTheme {
  timeCursorColor: string;
  backgroundColor: string;
  frameLabelColor: string;
  frameTickColor: string;
  promptBorderColor: string;
  promptTextColor: string;
  trackLabelColor: string;
  trackSeparatorColor: string;
  labelOverlayColor: string;
  keyframeBorderColor: string;
  intervalBorderColor: string;
  headerShadowColor: string;
  topBorderColor: string;
}

export const DARK_THEME: TimelineTheme = {
  timeCursorColor: "#A1E5D0",
  backgroundColor: "#1C2028",
  frameLabelColor: "#929AA7",
  frameTickColor: "rgba(255, 255, 255, 0.07)",
  promptBorderColor: "rgba(12, 18, 24, 0.7)",
  promptTextColor: "#FFFFFF",
  trackLabelColor: "#AAB3C0",
  trackSeparatorColor: "rgba(255, 255, 255, 0.1)",
  labelOverlayColor: "#1C2028",
  keyframeBorderColor: "rgba(255, 255, 255, 0.72)",
  intervalBorderColor: "rgba(255, 255, 255, 0.54)",
  headerShadowColor: "rgba(0, 0, 0, 0.12)",
  topBorderColor: "rgba(255, 255, 255, 0.1)",
};

export const LIGHT_THEME: TimelineTheme = {
  timeCursorColor: "#2563EB",
  backgroundColor: "#F8F9FA",
  frameLabelColor: "#495057",
  frameTickColor: "#CED4DA",
  promptBorderColor: "#495057",
  promptTextColor: "#FFFFFF",
  trackLabelColor: "#495057",
  trackSeparatorColor: "#CED4DA",
  labelOverlayColor: "#F8F9FA",
  keyframeBorderColor: "#495057",
  intervalBorderColor: "#495057",
  headerShadowColor: "rgba(0, 0, 0, 0.08)",
  topBorderColor: "rgba(0, 0, 0, 0.05)",
};

export const PROMPT_COLORS: [number, number, number][] = [
  [82, 133, 166],  // Default (teal-blue)
  [239, 68, 68],   // Red
  [249, 115, 22],  // Orange
  [234, 179, 8],   // Yellow
  [34, 197, 94],   // Green
  [59, 130, 246],  // Blue
  [168, 85, 247],  // Purple
  [236, 72, 153],  // Pink
];

export const DEFAULT_PROMPT_COLOR: [number, number, number] = [73, 118, 121];
export const PROMPT_TEXT_SHADOW = "rgba(0, 0, 0, 0.5)";

export const HIGHLIGHT_COLOR = "#A1E5D0";
export const HIGHLIGHT_BORDER_COLOR = "rgba(255, 255, 255, 0.22)";
export const HIGHLIGHT_SHADOW_COLOR = "rgba(0, 0, 0, 0.18)";
export const HIGHLIGHT_TEXT_COLOR = "#14251F";
export const HIGHLIGHT_FONT =
  "bold 10px Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
export const HIGHLIGHT_TEXT_SHADOW_COLOR = "rgba(0, 0, 0, 0)";
export const HIGHLIGHT_SHADOW_BLUR = 4;
export const HIGHLIGHT_SHADOW_OFFSET_Y = 1;
export const HIGHLIGHT_BORDER_WIDTH = 1;
export const HIGHLIGHT_TEXT_SHADOW_BLUR = 0;
