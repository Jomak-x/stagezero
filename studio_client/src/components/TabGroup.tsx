import * as React from "react";
import { GuiTabGroupMessage } from "../WebsocketMessages";
import { Tabs } from "@mantine/core";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";
import { htmlIconWrapper } from "./ComponentStyles.css";

export default function TabGroupComponent({
  props: {
    _tab_labels: tab_labels,
    _tab_icons_html: tab_icons_html,
    _tab_container_ids: tab_container_ids,
    visible,
  },
}: GuiTabGroupMessage) {
  const { GuiContainer } = React.useContext(GuiComponentContext)!;
  if (!visible) return null;
  return (
    <Tabs variant="pills" radius="md" defaultValue={"0"}>
      <Tabs.List styles={{ list: { gap: 3, padding: 3, margin: "0 0.5rem 0.5rem", borderRadius: 10, background: "var(--mantine-color-default)" } }}>
        {tab_labels.map((label, index) => (
          <Tabs.Tab
            value={index.toString()}
            key={index}
            styles={{
              tabSection: { marginRight: "0.5em" },
              tab: { padding: "0.45rem 0.65rem", flexGrow: 1, textAlign: "center", fontWeight: 600 },
            }}
            leftSection={
              tab_icons_html[index] === null ? undefined : (
                <div
                  className={htmlIconWrapper}
                  dangerouslySetInnerHTML={{ __html: tab_icons_html[index]! }}
                />
              )
            }
          >
            {label}
          </Tabs.Tab>
        ))}
      </Tabs.List>
      {tab_container_ids.map((containerUuid, index) => (
        <Tabs.Panel value={index.toString()} key={containerUuid}>
          <GuiContainer containerUuid={containerUuid} />
        </Tabs.Panel>
      ))}
    </Tabs>
  );
}
