import { usePipeline } from "../../store/PipelineContext.jsx";
import { escapeHtml } from "../../utils/escapeHtml.js";
import { EmptyState } from "../ui/EmptyState.jsx";

export function BriefPanel() {
  const { state } = usePipeline();
  const active = state.activeTab === "brief";
  const data = state.briefMerge;

  const fieldCards = [];
  const add = (label, val, full) => {
    if (!val) return;
    fieldCards.push({ label, val: String(val), full });
  };

  add("Campaign Title", data.campaign_title, true);
  add("Visual Style", data.visual_style);
  add("Lighting", data.lighting_style);
  add("Pacing", data.pacing);
  add("Music Mood", data.music_mood);
  add("Target Audience", data.target_audience || data.demographic_insights);
  add("Narrative Arc", data.narrative_arc);
  add("Brand Safety", data.brand_safety_notes);

  const hasPalette = data.color_palette?.length;
  const hasScenes = data.key_scenes?.length;
  const hasContent = fieldCards.length || hasPalette || hasScenes;

  let empty;
  if (!hasContent) {
    if (state.running) {
      empty = (
        <EmptyState icon="✨" title="Generating creative brief..." subtitle="" />
      );
    } else {
      empty = (
        <EmptyState
          icon="✨"
          title="Enriched brief not yet generated"
          subtitle="The AI-generated creative manifesto appears here after Stages 1 & 2 complete."
        />
      );
    }
  }

  return (
    <div className={`tab-panel${active ? " active" : ""}`} id="brief-tab">
      <div id="brief-panel">
        {!hasContent ? (
          empty
        ) : (
          <div className="brief-grid">
            {fieldCards.map((f) => (
              <div
                key={f.label}
                className="brief-card"
                style={f.full ? { gridColumn: "1 / -1" } : undefined}
              >
                <div className="brief-label">{f.label}</div>
                <div className="brief-val">{f.val}</div>
              </div>
            ))}
            {hasPalette ? (
              <div className="brief-card">
                <div className="brief-label">Color Palette</div>
                <ul className="brief-list">
                  {data.color_palette.map((x) => (
                    <li key={x}>{x}</li>
                  ))}
                </ul>
              </div>
            ) : null}
            {hasScenes ? (
              <div className="brief-card" style={{ gridColumn: "1 / -1" }}>
                <div className="brief-label">Key Scenes</div>
                <ul className="brief-list">
                  {data.key_scenes.map((x) => (
                    <li key={x}>{x}</li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}
