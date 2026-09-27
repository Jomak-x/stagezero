import React from "react";
import { useVoiceView, voicePlayback } from "./VoicePlayback";

export function VoiceCaptions() {
  const caption = useVoiceView((state) => state.caption);
  const audioError = useVoiceView((state) => state.audioError);
  const canRetryAssets = useVoiceView((state) => state.canRetryAssets);

  React.useEffect(() => {
    const unlock = () => voicePlayback.unlock();
    window.addEventListener("pointerdown", unlock, { passive: true });
    window.addEventListener("keydown", unlock);
    return () => {
      window.removeEventListener("pointerdown", unlock);
      window.removeEventListener("keydown", unlock);
    };
  }, []);

  if (!caption && !audioError) return null;
  return <div role="status" aria-live="off" style={{
    position: "absolute", left: 16, right: "min(420px, 35%)", bottom: 16, zIndex: 5,
    display: "flex", alignItems: "center", flexDirection: "column", gap: 8, pointerEvents: "none",
    whiteSpace: "pre-line", textAlign: "center",
  }}>
    {caption && <span style={{
      display: "inline-block", maxWidth: "min(720px, 90%)", padding: "8px 14px",
      color: "#fff", background: "#111b24e8", borderRadius: 8,
      fontSize: "clamp(15px, 2vw, 22px)", lineHeight: 1.35,
      textShadow: "0 1px 2px #000", boxShadow: "0 3px 18px #0007",
    }}>{caption}</span>}
    {audioError && <span style={{
      color: "#fff", background: "#111b24ec", borderRadius: 8, padding: "8px 12px",
      fontSize: 13, pointerEvents: "auto",
    }}>
      {audioError}
      {canRetryAssets ? <span style={{ marginLeft: 8 }}>
        <button type="button" onClick={() => voicePlayback.retryAssets()}>Retry audio</button>{" "}
        <button type="button" onClick={() => voicePlayback.playWithoutAudio()}>Play without sound</button>
      </span> : audioError.startsWith("Click Enable") && <button type="button" style={{ marginLeft: 8 }}
        onClick={() => voicePlayback.unlock()}>Enable audio</button>}
    </span>}
  </div>;
}
