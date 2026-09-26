// @refresh reset

import { ActionIcon, Box, Divider, Paper, ScrollArea, Tooltip } from "@mantine/core";
import React from "react";
import { useDisclosure } from "@mantine/hooks";
import { IconChevronLeft, IconChevronRight } from "@tabler/icons-react";

const SidebarPanelContext = React.createContext<null | {
  collapsible: boolean;
  toggleCollapsed: () => void;
}>(null);

/** A full-height inspector next to the viewport and timeline. */
export default function SidebarPanel({
  children,
  collapsible,
  width,
}: {
  children: string | React.ReactNode;
  collapsible: boolean;
  width: string;
}) {
  const [collapsed, { toggle: toggleCollapsed }] = useDisclosure(false);

  return (
    <SidebarPanelContext.Provider value={{ collapsible, toggleCollapsed }}>
      <Paper
        component="aside"
        data-testid="studio-inspector"
        aria-label="Inspector"
        radius={0}
        shadow="-0.25em 0 0.8em rgba(0,0,0,0.12)"
        style={{
          width: collapsed ? "2.75em" : `min(${width}, 40vw)`,
          height: "100%",
          minWidth: 0,
          minHeight: 0,
          flex: "0 0 auto",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
          boxSizing: "border-box",
          transition: "width 180ms ease",
          zIndex: 8,
        }}
      >
        {collapsed && (
          <Tooltip zIndex={100} label="Show sidebar">
            <ActionIcon
              aria-label="Show sidebar"
              m="xs"
              onClick={(evt) => {
                evt.stopPropagation();
                toggleCollapsed();
              }}
            >
              <IconChevronLeft />
            </ActionIcon>
          </Tooltip>
        )}
        <Box
          style={{
            width: "100%",
            minWidth: 0,
            minHeight: 0,
            flex: "1 1 auto",
            display: collapsed ? "none" : "flex",
            flexDirection: "column",
          }}
        >
          {children}
        </Box>
      </Paper>
    </SidebarPanelContext.Provider>
  );
}

/** Header with the control to collapse the inspector. */
SidebarPanel.Handle = function SidebarPanelHandle({
  children,
}: {
  children: string | React.ReactNode;
}) {
  const { toggleCollapsed, collapsible } = React.useContext(SidebarPanelContext)!;

  return (
    <>
      <Box
        p="xs"
        style={{
          lineHeight: "1.5em",
          fontWeight: 400,
          position: "relative",
          zIndex: 20,
          alignItems: "center",
          display: "flex",
          flexDirection: "row",
          minWidth: 0,
          flex: "0 0 auto",
        }}
      >
        {children}
        {collapsible && (
          <Tooltip zIndex={100} label="Collapse sidebar">
            <ActionIcon
              aria-label="Collapse sidebar"
              onClick={(evt) => {
                evt.stopPropagation();
                toggleCollapsed();
              }}
            >
              <IconChevronRight stroke={1.625} />
            </ActionIcon>
          </Tooltip>
        )}
      </Box>
      <Divider mx="xs" />
    </>
  );
};

/** Independently scrolling inspector controls. */
SidebarPanel.Contents = function SidebarPanelContents({
  children,
}: {
  children: string | React.ReactNode;
}) {
  return (
    <ScrollArea style={{ flex: "1 1 auto", minHeight: 0, minWidth: 0 }}>
      {children}
    </ScrollArea>
  );
};
