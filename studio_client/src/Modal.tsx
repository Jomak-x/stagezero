// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ViewerContext } from "./ViewerContext";
import { GuiModalMessage } from "./WebsocketMessages";
import GeneratedGuiContainer from "./ControlPanel/Generated";
import { Modal } from "@mantine/core";
import { useContext, useEffect, useRef } from "react";
import { shallowArrayEqual } from "./utils/shallowArrayEqual";
import { getModalChoiceSaved } from "./QuickStartPersistence";

export function ViserModal() {
  const viewer = useContext(ViewerContext)!;

  const modalList = viewer.useGui((state) => state.modals, shallowArrayEqual);
  const didRequestCloseRef = useRef<Set<string>>(new Set());

  // Auto-dismiss any "save_choice" modals that have already been acknowledged,
  // and avoid rendering them entirely to prevent flicker.
  useEffect(() => {
    for (const conf of modalList) {
      if (!conf.save_choice) continue;
      if (!getModalChoiceSaved(conf.save_choice)) continue;
      if (didRequestCloseRef.current.has(conf.uuid)) continue;
      didRequestCloseRef.current.add(conf.uuid);
      viewer.mutable.current.sendMessage({
        type: "GuiCloseModalRequestMessage",
        uuid: conf.uuid,
      });
    }
  }, [modalList, viewer.mutable]);

  const visibleModalList = modalList.filter(
    (conf) => !conf.save_choice || !getModalChoiceSaved(conf.save_choice),
  );
  const modals = visibleModalList.map((conf, index) => {
    return <GeneratedModal key={conf.uuid} conf={conf} index={index} />;
  });

  return modals;
}

function GeneratedModal({
  conf,
  index,
}: {
  conf: GuiModalMessage;
  index: number;
}) {
  const viewer = useContext(ViewerContext)!;

  const requestClose = conf.show_close_button
    ? () => {
        viewer.mutable.current.sendMessage({
          type: "GuiCloseModalRequestMessage",
          uuid: conf.uuid,
        });
      }
    : null;

  return (
    <Modal
      opened={true}
      title={conf.title}
      size={conf.size ?? "md"}
      radius="lg"
      overlayProps={{ backgroundOpacity: 0.6, blur: 4 }}
      onClose={() => {
        if (requestClose) {
          requestClose();
          return;
        }
        // To make memory management easier, we should only close modals from
        // the server.
        // Otherwise, the client would need to communicate to the server that
        // the modal was deleted and contained GUI elements were cleared.
      }}
      withCloseButton={requestClose !== null}
      closeButtonProps={{
        "aria-label": "Close",
      }}
      styles={{
        content: {
          border: "1px solid var(--mantine-color-default-border)",
          boxShadow: "0 24px 70px rgba(0, 0, 0, 0.35)",
        },
        header: { paddingBottom: "0.25rem" },
        title: { fontSize: "1.05rem", fontWeight: 700 },
        close: { color: "var(--mantine-color-dimmed)", borderRadius: 8 },
      }}
      closeOnClickOutside={false}
      closeOnEscape={requestClose !== null}
      centered
      zIndex={100 + index}
    >
      <GeneratedGuiContainer containerUuid={conf.uuid} gallery={/background|character|preset/i.test(conf.title)} />
    </Modal>
  );
}
