import { GuiHtmlMessage } from "../WebsocketMessages";
import React from "react";
import { CameraPanel } from "../CameraPanel";

const cameraMarker = "<div data-stagezero-cameras></div>";

function HtmlComponent({ props }: GuiHtmlMessage) {
  if (!props.visible) return null;
  if (!props.content.includes(cameraMarker)) {
    return <div dangerouslySetInnerHTML={{ __html: props.content }} />;
  }
  const segments = props.content.split(cameraMarker);
  return (
    <div>
      {segments.map((segment, index) => (
        <React.Fragment key={index}>
          {segment && <div style={{ display: "contents" }} dangerouslySetInnerHTML={{ __html: segment }} />}
          {index < segments.length - 1 && <CameraPanel />}
        </React.Fragment>
      ))}
    </div>
  );
}

export default HtmlComponent;
