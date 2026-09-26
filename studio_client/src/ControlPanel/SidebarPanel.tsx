// @refresh reset

import { ActionIcon, Box, Paper, ScrollArea, Tooltip } from "@mantine/core";
import React from "react";
import { useDisclosure } from "@mantine/hooks";
import { IconChevronLeft, IconChevronRight } from "@tabler/icons-react";

const SidebarPanelContext = React.createContext<null | {
  collapsible: boolean;
  toggleCollapsed: () => void;
}>(null);

/** A floating inspector over the viewport. */
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
        data-collapsed={collapsed}
        aria-label="Inspector"
        className="sz-inspector"
        style={{ "--sz-control-width": width } as React.CSSProperties}
      >
        {collapsed && (
          <Tooltip zIndex={100} label="Show inspector">
            <button
              type="button"
              className="sz-inspector-pill"
              aria-label="Show inspector"
              onClick={(evt) => {
                evt.stopPropagation();
                toggleCollapsed();
              }}
            >
              <IconChevronLeft size={17} stroke={1.7} />
              <span>Inspector</span>
            </button>
          </Tooltip>
        )}
        <Box
          className="sz-inspector-content"
          aria-hidden={collapsed}
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
    <Box className="sz-inspector-header">
      {children}
      {collapsible && (
        <Tooltip zIndex={100} label="Collapse inspector">
          <ActionIcon
            aria-label="Collapse inspector"
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
