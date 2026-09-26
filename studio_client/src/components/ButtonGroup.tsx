import * as React from "react";
import { Button, Flex } from "@mantine/core";
import { ViserInputComponent } from "./common";
import { GuiButtonGroupMessage } from "../WebsocketMessages";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";

export default function ButtonGroupComponent({
  uuid,
  props: { hint, label, visible, disabled, options },
}: GuiButtonGroupMessage) {
  const { messageSender } = React.useContext(GuiComponentContext)!;
  if (!visible) return null;
  return (
    <ViserInputComponent {...{ uuid, hint, label }}>
      <Flex justify="space-between" gap="xs">
        {options.map((option) => (
          <Button
            key={option}
            onClick={() =>
              messageSender({
                type: "GuiUpdateMessage",
                uuid,
                updates: { value: option },
              })
            }
            style={{ flex: "1 1 0", minWidth: 0 }}
            disabled={disabled}
            size="compact-xs"
            radius="sm"
            variant="subtle"
          >
            {option}
          </Button>
        ))}
      </Flex>
    </ViserInputComponent>
  );
}
