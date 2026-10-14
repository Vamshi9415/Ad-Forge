import { usePipeline } from "../../store/PipelineContext.jsx";

export function CampaignForm({ onUploadSuccess, onUploadError }) {
  const { state, dispatch } = usePipeline();
  const d = state.draft;

  function clearImage(ev) {
    ev.preventDefault();
    ev.stopPropagation();
    dispatch({
      type: "UPDATE_DRAFT",
      payload: {
        imageDataUri: null,
        uploadPreview: null,
        hasUpload: false,
      },
    });
  }

  async function onFileChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    try {
      const r = await fetch("/api/upload-image", { method: "POST", body: fd });
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      dispatch({
        type: "UPDATE_DRAFT",
        payload: {
          imageDataUri: data.data_uri,
          uploadPreview: data.data_uri,
          hasUpload: true,
        },
      });
      onUploadSuccess?.(`Reference image loaded (${(data.size / 1024).toFixed(0)}KB)`);
    } catch (err) {
      onUploadError?.(`Upload failed: ${err.message}`);
    }
    e.target.value = "";
  }

  return (
    <div className="sb-section">
      <div className="sb-title">Campaign Brief</div>

      <div className="form-g">
        <label>Advertising Concept *</label>
        <textarea
          rows={4}
          placeholder="Describe your product, target audience, emotional tone, and campaign goal...&#10;&#10;Example: A luxury cold brew coffee brand targeting urban millennials aged 25-35. Emphasise the 24-hour slow-brew process, sustainably sourced beans, and artisan craft. CTA: 'Slow is Bold.'"
          value={d.prompt}
          onChange={(e) => dispatch({ type: "UPDATE_DRAFT", payload: { prompt: e.target.value } })}
        />
      </div>

      <div className="form-g">
        <label>
          Reference Image <span style={{ color: "var(--text3)", fontWeight: 400 }}>(optional)</span>
        </label>
        <div className={`upload-wrap${d.hasUpload ? " has-img" : ""}`}>
          <input type="file" accept="image/*" onChange={onFileChange} />
          <span
            id="up-clear"
            role="button"
            tabIndex={0}
            onClick={clearImage}
            onKeyDown={(e) => e.key === "Enter" && clearImage(e)}
            style={{ display: d.hasUpload ? "block" : "none" }}
          >
            Remove
          </span>
          <div id="up-content" style={{ display: d.hasUpload ? "none" : "block" }}>
            <span className="up-ico">🖼️</span>
            <div className="up-txt">
              Drop product image or <span className="up-btn">browse</span>
            </div>
            <div style={{ fontSize: ".65rem", color: "var(--text3)", marginTop: "4px" }}>
              PNG, JPG, WEBP — max 10MB
            </div>
          </div>
          <img
            className={`upload-preview-img${d.uploadPreview ? " is-visible" : ""}`}
            src={d.uploadPreview || ""}
            alt=""
          />
        </div>
      </div>

      <div className="form-g">
        <label>Target Platform</label>
        <select
          value={d.platform}
          onChange={(e) => dispatch({ type: "UPDATE_DRAFT", payload: { platform: e.target.value } })}
        >
          <option value="youtube">YouTube (1280×720 · 16:9)</option>
          <option value="tiktok">TikTok (720×1280 · 9:16)</option>
          <option value="instagram_reels">Instagram Reels (720×1280 · 9:16)</option>
          <option value="linkedin">LinkedIn (1280×720 · 16:9)</option>
          <option value="twitter">Twitter / X (1280×720 · 16:9)</option>
        </select>
      </div>

      <div className="form-g">
        <label>Ad Duration</label>
        <div className="slider-row">
          <input
            type="range"
            min={10}
            max={60}
            step={5}
            value={d.duration}
            onChange={(e) =>
              dispatch({ type: "UPDATE_DRAFT", payload: { duration: Number(e.target.value) } })
            }
          />
          <div className="sv">
            <span id="dur-val">{d.duration}</span>s
          </div>
        </div>
      </div>

      <div className="form-g">
        <label>Video Model</label>
        <div className="model-select-wrap">
          <select
            value={d.videoModel}
            onChange={(e) =>
              dispatch({ type: "UPDATE_DRAFT", payload: { videoModel: e.target.value } })
            }
          >
            <optgroup label="RunwayML">
              <option value="gen3a_turbo">🧬 Gen3 Alpha Turbo — $0.05/s (image-driven, 5/10s)</option>
              <option value="gen4_turbo">⚡ Gen4 Turbo — $0.05/s (fast, image-driven)</option>
              <option value="gen4.5">🎬 Gen4.5 — $0.12/s (high quality)</option>
              <option value="seedance2">🎭 Seedance 2 — $0.36/s (up to 15s, reference video)</option>
            </optgroup>
            <optgroup label="Google (via RunwayML)">
              <option value="veo3.1_fast">🚀 Veo 3.1 Fast — $0.12/s (4/6/8s)</option>
              <option value="veo3.1">🌟 Veo 3.1 — $0.30/s (4/6/8s, high quality)</option>
              <option value="veo3">✨ Veo 3 — $0.40/s (8s only, premium)</option>
            </optgroup>
          </select>
        </div>
      </div>

      <div className="form-g">
        <label>Image Model</label>
        <div className="model-select-wrap">
          <select
            value={d.imageModel}
            onChange={(e) =>
              dispatch({ type: "UPDATE_DRAFT", payload: { imageModel: e.target.value } })
            }
          >
            <optgroup label="RunwayML">
              <option value="gen4_image">🎨 Gen4 Image — $0.06/frame (best quality)</option>
              <option value="gen4_image_turbo">⚡ Gen4 Image Turbo — $0.02/frame (fast)</option>
            </optgroup>
            <optgroup label="Google (via RunwayML)">
              <option value="gemini_2.5_flash">🌐 Gemini 2.5 Flash — $0.05/frame</option>
            </optgroup>
          </select>
        </div>
      </div>
    </div>
  );
}
