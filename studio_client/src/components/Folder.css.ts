import { style } from "@vanilla-extract/css";
import { vars } from "../AppTheme";

export const folderWrapper = style({
  marginLeft: vars.spacing.xs,
  marginRight: vars.spacing.xs,
  marginTop: vars.spacing.xs,
  borderTop: "1px solid var(--mantine-color-default-border)",
});

export const folderLabel = style({
  width: "100%",
  minHeight: "2.75rem",
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  gap: vars.spacing.xs,
  padding: "0.55rem 0.65rem",
  borderRadius: vars.radius.md,
  color: "var(--mantine-color-text)",
  fontSize: "0.875rem",
  textAlign: "left",
  userSelect: "none",
  fontWeight: 600,
  transition: "background-color 150ms ease",
  selectors: {
    "&:hover:not(:disabled)": {
      backgroundColor: "var(--mantine-color-default-hover)",
    },
    "&:focus-visible": {
      outline: "2px solid var(--mantine-primary-color-filled)",
      outlineOffset: "-2px",
    },
    "&:disabled": {
      cursor: "default",
    },
  },
});

export const folderToggleIcon = style({
  flexShrink: 0,
  width: "1rem",
  height: "1rem",
  strokeWidth: 2,
  opacity: 0.7,
  transition: "transform 150ms ease",
});
