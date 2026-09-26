import { useEffect, useId, useRef, useState } from "react";
import "./WelcomeScreen.css";

const videoSource = "/welcome/victory-dance.mp4";
const videoPoster = "/welcome/victory-poster.jpg";
const reducedMotionQuery = "(prefers-reduced-motion: reduce)";

function getReducedMotionPreference() {
  return typeof window !== "undefined" && window.matchMedia(reducedMotionQuery).matches;
}

export type WelcomeScreenProps = {
  onEnter: () => void;
  connectionState: "connecting" | "connected" | "disconnected";
};

const connectionLabels: Record<WelcomeScreenProps["connectionState"], string> = {
  connecting: "Connecting to studio",
  connected: "Studio connected",
  disconnected: "Studio disconnected",
};

const guideSteps = [
  { number: "01", title: "Describe", detail: "Tell the actor what you imagine." },
  { number: "02", title: "Generate", detail: "See your direction become motion." },
  { number: "03", title: "Refine", detail: "Shape the result until it feels right." },
];

export function WelcomeScreen({
  onEnter,
  connectionState,
}: WelcomeScreenProps) {
  const [guideOpen, setGuideOpen] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(getReducedMotionPreference);
  const [isPlaying, setIsPlaying] = useState(false);
  const [videoFailed, setVideoFailed] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const pausedByUser = useRef(false);
  const guideId = useId();

  useEffect(() => {
    const preference = window.matchMedia(reducedMotionQuery);
    const onPreferenceChange = (event: MediaQueryListEvent) => {
      setReducedMotion(event.matches);
      if (event.matches) videoRef.current?.pause();
    };

    preference.addEventListener("change", onPreferenceChange);
    setReducedMotion(preference.matches);
    return () => preference.removeEventListener("change", onPreferenceChange);
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || videoFailed) return;

    if (reducedMotion) {
      video.pause();
    } else if (!pausedByUser.current) {
      void video.play().catch(() => {
        if (videoRef.current === video) setIsPlaying(!video.paused);
      });
    }
  }, [reducedMotion, videoFailed]);

  const togglePlayback = () => {
    const video = videoRef.current;
    if (!video) return;

    if (video.paused) {
      pausedByUser.current = false;
      void video.play().catch(() => {
        if (videoRef.current === video) setIsPlaying(!video.paused);
      });
    } else {
      pausedByUser.current = true;
      video.pause();
    }
  };

  return (
    <section
      className={`welcome-screen${guideOpen ? " welcome-screen--guide-open" : ""}`}
      aria-label="Welcome to StageZero"
    >
      <div className="welcome-screen__film" aria-hidden="true">
        <img className="welcome-screen__poster" src={videoPoster} alt="" />
        {!videoFailed && (
          <video
            ref={videoRef}
            className="welcome-screen__video"
            autoPlay={!reducedMotion}
            muted
            loop
            playsInline
            preload="metadata"
            poster={videoPoster}
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
            onError={() => {
              setVideoFailed(true);
              setIsPlaying(false);
            }}
          >
            <source src={videoSource} type="video/mp4" />
          </video>
        )}
      </div>
      {videoFailed ? (
        <span className="welcome-screen__animation-status" role="status">
          Animation unavailable
        </span>
      ) : (
        <button
          className="welcome-screen__animation-toggle"
          type="button"
          onClick={togglePlayback}
          aria-label={isPlaying ? "Pause animation" : "Play animation"}
        >
          <span aria-hidden="true">{isPlaying ? "Ⅱ" : "▶"}</span>
          {isPlaying ? "Pause animation" : "Play animation"}
        </button>
      )}
      <header className="welcome-screen__header">
        <div className="welcome-screen__brand" aria-label="StageZero">
          <span className="welcome-screen__brand-mark" aria-hidden="true">
            <span />
          </span>
          <span className="welcome-screen__brand-name">StageZero</span>
        </div>
        <div className="welcome-screen__hero-index" aria-hidden="true">
          <span className="welcome-screen__index-line" />
          <span>G1 / MOTION STUDY</span>
          <span className="welcome-screen__index-number">01 — 01</span>
        </div>
      </header>

      <main className="welcome-screen__main">
        <div className="welcome-screen__content">
          <p className="welcome-screen__eyebrow">
            <span className="welcome-screen__eyebrow-line" aria-hidden="true" />
            MOTION DIRECTION STUDIO
          </p>
          <h1 className="welcome-screen__title">
            <span>Motion</span>
            <span>
              starts here<span className="welcome-screen__period">.</span>
            </span>
          </h1>
          <p className="welcome-screen__description">
            Give an idea a body. Direct a performance through movement, then make
            every moment your own.
          </p>

          <div className="welcome-screen__actions">
            <button
              className="welcome-screen__enter"
              type="button"
              onClick={onEnter}
            >
              <span>Enter studio</span>
              <span className="welcome-screen__enter-arrow" aria-hidden="true">
                ↗
              </span>
            </button>
            <button
              className="welcome-screen__guide-toggle"
              type="button"
              aria-expanded={guideOpen}
              aria-controls={guideId}
              onClick={() => setGuideOpen((open) => !open)}
            >
              How it works
              <span aria-hidden="true">{guideOpen ? "−" : "+"}</span>
            </button>
          </div>

          {guideOpen && (
            <div
              className="welcome-screen__guide"
              id={guideId}
              role="region"
              aria-label="How it works"
            >
              {guideSteps.map((step) => (
                <div className="welcome-screen__guide-step" key={step.number}>
                  <span className="welcome-screen__guide-number">
                    {step.number}
                  </span>
                  <div>
                    <strong>{step.title}</strong>
                    <p>{step.detail}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </main>

      <footer className="welcome-screen__footer">
        <div className="welcome-screen__footer-left">
          <span className="welcome-screen__footer-rule" aria-hidden="true" />
          <span>YOUR CAST. YOUR DIRECTION.</span>
        </div>
        <div
          className={`welcome-screen__connection welcome-screen__connection--${connectionState}`}
          role="status"
          aria-live="polite"
        >
          <span className="welcome-screen__connection-dot" aria-hidden="true" />
          {connectionLabels[connectionState]}
        </div>
      </footer>
    </section>
  );
}

export default WelcomeScreen;
