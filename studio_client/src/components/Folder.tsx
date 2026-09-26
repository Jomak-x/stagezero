import * as React from "react";
import { useDisclosure } from "@mantine/hooks";
import { GuiFolderMessage } from "../WebsocketMessages";
import { IconChevronDown } from "@tabler/icons-react";
import { Box, Collapse, UnstyledButton } from "@mantine/core";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";
import { ViewerContext } from "../ViewerContext";
import { folderLabel, folderToggleIcon, folderWrapper } from "./Folder.css";
import { shallowObjectKeysEqual } from "../utils/shallowObjectKeysEqual";

export default function FolderComponent({
  uuid,
  props: { label, visible, expand_by_default },
  nextGuiUuid,
}: GuiFolderMessage & { nextGuiUuid: string | null }) {
  const viewer = React.useContext(ViewerContext)!;
  const [opened, { toggle }] = useDisclosure(expand_by_default);
  const guiIdSet = viewer.useGui(
    (state) => state.guiUuidSetFromContainerUuid[uuid],
    shallowObjectKeysEqual,
  );
  const guiContext = React.useContext(GuiComponentContext)!;
  const isEmpty = guiIdSet === undefined || Object.keys(guiIdSet).length === 0;
  const nextGuiType = viewer.useGui((state) =>
    nextGuiUuid == null ? null : state.guiConfigFromUuid[nextGuiUuid]?.type,
  );

  if (!visible) return null;
  return (
    <Box
      className={folderWrapper}
      mb={nextGuiType === "GuiFolderMessage" ? "xs" : undefined}
    >
      <UnstyledButton
        className={folderLabel}
        onClick={isEmpty ? undefined : toggle}
        disabled={isEmpty}
        aria-expanded={isEmpty ? undefined : opened}
        aria-controls={isEmpty ? undefined : `folder-content-${uuid}`}
      >
        <span>{label}</span>
        <IconChevronDown
          className={folderToggleIcon}
          style={{
            display: isEmpty ? "none" : undefined,
            transform: opened ? "rotate(180deg)" : undefined,
          }}
        />
      </UnstyledButton>
      <Collapse in={opened && !isEmpty} id={`folder-content-${uuid}`}>
        <Box pt="xs" pb="xs">
          <GuiComponentContext.Provider
            value={{
              ...guiContext,
              folderDepth: guiContext.folderDepth + 1,
            }}
          >
            <guiContext.GuiContainer containerUuid={uuid} />
          </GuiComponentContext.Provider>
        </Box>
      </Collapse>
    </Box>
  );
}
