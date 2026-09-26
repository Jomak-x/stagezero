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
    _studio_tab_request: navigation,
  },
}: GuiTabGroupMessage) {
  const { GuiContainer } = React.useContext(GuiComponentContext)!;
  const [activeTab, setActiveTab] = React.useState<string | null>("0");
  React.useEffect(() => {
    if (navigation && navigation.index >= 0 && navigation.index < tab_labels.length) {
      setActiveTab(String(navigation.index));
    }
  }, [navigation?.nonce]);
  if (!visible) return null;
  return (
    <Tabs radius="xs" value={activeTab} onChange={setActiveTab} style={{ marginTop: "-0.55em" }}>
      <Tabs.List>
        {tab_labels.map((label, index) => (
          <Tabs.Tab
            value={index.toString()}
            key={index}
            styles={{
              tabSection: { marginRight: "0.5em" },
              tab: { padding: "0.75em" },
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
