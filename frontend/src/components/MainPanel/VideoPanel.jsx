import { usePipeline } from "../../store/PipelineContext.jsx";
import { EmptyState } from "../ui/EmptyState.jsx";

export function VideoPanel() {
  const { state } = usePipeline();
  const active = state.activeTab === "video";
  const showClips = state.clipsInited || state.clips.length > 0;
  const fv = state.finalVideo;

  return (
    <div className={`tab-panel${active ? " active" : ""}`} id="video-tab">
      <div id="video-panel">
        {!showClips && !fv ? (
          <EmptyState
            icon="🎞️"
            title="No clips yet"
            subtitle="Video clips appear here during Stage 7. The stitched final commercial appears at the bottom."
          />
        ) : null}

        {showClips ? (
          <>
            <div className="clips-header">Generated Clips</div>
            <div className="clips-grid" id="clips-grid">
              {state.clips.map((c) => (
                <div key={c.shotId} className="clip-card">
                  <video src={c.src} controls muted loop preload="metadata" />
                  <div className="clip-info">
                    <span className="clip-id">{c.shotId}</span>
                    <span className="clip-dur">{c.duration}s</span>
                    <a href={c.src} download={`${c.shotId}.mp4`} className="clip-dl">
                      ↓ Save
                    </a>
                  </div>
                </div>
              ))}
            </div>
          </>
        ) : null}

        {fv ? (
          <div className="final-section" id="final-section">
            <div className="final-hdr">
              <div className="final-title">
                {fv.withVoiceover ? "🎵" : "🏆"} <span>{fv.title}{fv.withVoiceover ? " (with Voiceover)" : ""}</span>
              </div>
              <a href={fv.downloadHref || fv.src} download="final_campaign.mp4" className="dl-btn">
                ↓ Download HD
              </a>
            </div>
            <video src={fv.src} controls autoPlay loop />
            <div className="final-meta">
              <span>⏱ {fv.duration}s runtime</span>
              <span>🎬 {fv.clips} clips stitched</span>
              <span>💰 ${fv.costUsd} total spend</span>
              {fv.withVoiceover ? <span>🔊 Voiceover included</span> : null}
            </div>
          </div>
        ) : null}

        {state.audio ? (
          <div className="audio-section" id="audio-section">
            <div className="audio-hdr">
              <div className="audio-title">
                🎙️ RunwayML TTS Voiceover — {state.audio.voice}
              </div>
              <a href={state.audio.audioSrc} download="voiceover.mp3" className="dl-btn" style={{ fontSize: ".72rem", padding: "5px 12px" }}>
                ↓ MP3
              </a>
            </div>
            <audio src={state.audio.audioSrc} controls style={{ width: "100%", marginTop: "6px" }} />
            {state.audio.hasMusic && state.audio.musicSrc ? (
              <>
                <div className="audio-hdr" style={{ marginTop: "10px" }}>
                  <div className="audio-title">🎵 Background Music — RunwayML SFX</div>
                  <a href={state.audio.musicSrc} download="music.mp3" className="dl-btn" style={{ fontSize: ".72rem", padding: "5px 12px" }}>
                    ↓ MP3
                  </a>
                </div>
                <audio src={state.audio.musicSrc} controls style={{ width: "100%", marginTop: "6px" }} />
              </>
            ) : null}
            {state.audio.script ? (
              <>
                <div className="script-label">Voiceover Script</div>
                <div className="script-box">{state.audio.script}</div>
              </>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}
