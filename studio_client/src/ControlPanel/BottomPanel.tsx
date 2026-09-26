import { Box, Collapse, Divider, Paper, ScrollArea } from "@mantine/core";
import React from "react";
import { useDisclosure } from "@mantine/hooks";

const BottomPanelContext = React.createContext<null | {
  expanded: boolean;
  toggleExpanded: () => void;
  contentsId: string;
}>(null);

/** A bottom panel is used to display the controls on mobile devices. */
export default function BottomPanel({
  children,
}: {
  children: string | React.ReactNode;
}) {
  const [expanded, { toggle: toggleExpanded }] = useDisclosure(false);
  const contentsId = React.useId();
  const [dockLayout, setDockLayout] = React.useState({
    bottom: 0,
    availableHeight: typeof window === "undefined" ? 0 : window.innerHeight,
    viewportHeight: typeof window === "undefined" ? 0 : window.innerHeight,
  });

  // The timeline is docked outside the viewport containing this panel. Keep
  // the mobile sheet above it as track count and window size change.
  React.useLayoutEffect(() => {
    const dock = document.querySelector<HTMLElement>("[data-studio-timeline-dock]");
    const update = () => {
      const bottom = dock && dock.getBoundingClientRect().height > 0
        ? Math.max(0, window.innerHeight - dock.getBoundingClientRect().top)
        : 0;
      setDockLayout({
        bottom,
        // Leave air above the sheet and between it and the timeline.
        availableHeight: Math.max(0, window.innerHeight - bottom - 40),
        viewportHeight: window.innerHeight,
      });
    };
    update();
    const observer = dock && typeof ResizeObserver !== "undefined" ? new ResizeObserver(update) : null;
    if (dock && observer) observer.observe(dock);
    window.addEventListener("resize", update);
    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", update);
    };
  }, []);
  return (
    <BottomPanelContext.Provider
      value={{
        expanded: expanded,
        toggleExpanded: toggleExpanded,
        contentsId,
      }}
    >
      <>
        <Paper
          className="sz-mobile-inspector"
          data-testid="studio-mobile-inspector"
          style={{
            bottom: dockLayout.bottom + 12,
            maxHeight: Math.min(dockLayout.viewportHeight * 0.58, dockLayout.availableHeight),
          }}
          component={ScrollArea.Autosize}
        >
          <Box
            style={{ width: "100%" }}
          >
            {children}
          </Box>
        </Paper>
      </>
    </BottomPanelContext.Provider>
  );
}
BottomPanel.Handle = function BottomPanelHandle({
  children,
}: {
  children: string | React.ReactNode;
}) {
  const panelContext = React.useContext(BottomPanelContext)!;
  return (
    <Box
      className="sz-inspector-header sz-mobile-inspector-header"
      style={{
        minHeight: "3.5em",
      }}
    >
      {children}
      <button
        type="button"
        aria-label={panelContext.expanded ? "Collapse controls" : "Expand controls"}
        aria-expanded={panelContext.expanded}
        aria-controls={panelContext.contentsId}
        onClick={panelContext.toggleExpanded}
        style={{
          flexShrink: 0,
          minWidth: 44,
          minHeight: 44,
          padding: "0 8px",
          border: 0,
          borderRadius: 6,
          background: "transparent",
          color: "inherit",
          font: "inherit",
          cursor: "pointer",
        }}
      >
        {panelContext.expanded ? "Close" : "Controls"}
      </button>
    </Box>
  );
};

/** Contents of a panel. */
BottomPanel.Contents = function BottomPanelContents({
  children,
}: {
  children: string | React.ReactNode;
}) {
  const panelContext = React.useContext(BottomPanelContext)!;
  // Inner generated-control collapses measure their content as data arrives.
  // Keep this wrapper in layout while closed so their height is measurable.
  return (
    <Collapse id={panelContext.contentsId} in={panelContext.expanded} keepMounted>
      <Divider mx="xs" />
      {children}
    </Collapse>
  );
};

/** Hides contents when panel is collapsed. */
BottomPanel.HideWhenCollapsed = function BottomPanelHideWhenCollapsed({
  children,
}: {
  children: React.ReactNode;
}) {
  const expanded = React.useContext(BottomPanelContext)?.expanded ?? true;
  return expanded ? children : null;
};
