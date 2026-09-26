import * as React from "react";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";
import { ViserInputComponent } from "./common";
import { GuiDropdownMessage } from "../WebsocketMessages";
import { Box, Button, Flex, Select, Text } from "@mantine/core";

export default function DropdownComponent({
  uuid,
  value,
  props: { hint, label, disabled, visible, options },
}: GuiDropdownMessage) {
  const { setValue } = React.useContext(GuiComponentContext)!;
  if (!visible) return null;
  if (label === "Select action") {
    return (
      <Box px="xs" pb="xs">
        <Text size="xs" c="dimmed" mb={6}>Your sequence · click an action to edit</Text>
        <Flex role="group" aria-label="Select action" gap={6} direction="column"
          style={{ maxHeight: 180, overflowY: "auto" }}>
          {options.filter((option) => option !== "Select an action").map((option) => {
            const [number, timing, ...description] = option.split(" · ");
            return <Button key={option} onClick={() => setValue(uuid, option)} disabled={disabled}
              aria-pressed={value === option} variant={value === option ? "light" : "subtle"}
              color="teal" radius="md" fullWidth title={description.join(" · ")}
              styles={{ root: { height: "auto", minHeight: 48, padding: "7px 10px", justifyContent: "flex-start", border: value === option ? "1px solid #487e76" : "1px solid transparent" },
                inner: { width: "100%", justifyContent: "flex-start" }, label: { width: "100%", textAlign: "left", gap: 10 } }}>
              <Text component="span" size="xs" c="teal.3">{number}</Text>
              <Box style={{ minWidth: 0, flex: 1 }}>
                <Text component="span" size="xs" truncate style={{ display: "block" }}>{description.join(" · ") || option}</Text>
                <Text component="span" size="xs" c="dimmed" style={{ display: "block", fontSize: 10 }}>{timing}</Text>
              </Box>
            </Button>;
          })}
        </Flex>
      </Box>
    );
  }
  if (label === "Compose") {
    return (
      <Box px="xs" pb="md" title={hint ?? undefined}>
        <Text id={`compose-label-${uuid}`} size="xs" fw={700} c="dimmed" mb={7} tt="uppercase" style={{ letterSpacing: "0.08em" }}>
          {label}
        </Text>
        <Flex role="group" aria-labelledby={`compose-label-${uuid}`} gap="xs" wrap="nowrap">
          {options.map((option) => (
            <Button
              key={option}
              type="button"
              onClick={() => setValue(uuid, option)}
              style={{ flex: "1 1 0", minWidth: 0 }}
              disabled={disabled}
              size="sm"
              radius="md"
              variant={value === option ? "filled" : "light"}
              aria-pressed={value === option}
              styles={{
                root: { minHeight: 42, height: "auto", padding: "0.5rem 0.35rem" },
                label: { whiteSpace: "normal", textAlign: "center", lineHeight: 1.2, fontWeight: 600 },
              }}
            >
              {option}
            </Button>
          ))}
        </Flex>
      </Box>
    );
  }
  return (
    <ViserInputComponent {...{ uuid, hint, label }}>
      <Select
        id={uuid}
        radius="xs"
        value={value}
        data={options}
        onChange={(value) => value !== null && setValue(uuid, value)}
        disabled={disabled}
        searchable
        maxDropdownHeight={400}
        size="xs"
        rightSectionWidth="1.2em"
        styles={{
          input: {
            padding: "0.5em",
            letterSpacing: "-0.5px",
            minHeight: "1.625rem",
            height: "1.625rem",
          },
        }}
        // zIndex of dropdown should be >modal zIndex.
        // On edge cases: it seems like existing dropdowns are always closed when a new modal is opened.
        comboboxProps={{ zIndex: 1000 }}
      />
    </ViserInputComponent>
  );
}
