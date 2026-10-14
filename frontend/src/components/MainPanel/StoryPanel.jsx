import { usePipeline } from "../../store/PipelineContext.jsx";
import { normalizeSpillStatus } from "../../store/pipelineReducer.js";
import { EmptyState } from "../ui/EmptyState.jsx";

export function StoryPanel() {
  const { state } = usePipeline();
  const active = state.activeTab === "story";
  const rows = state.storyboard;

  let empty;
  if (!rows.length) {
    if (state.running) {
      empty = (
        <EmptyState icon="⏳" title="Generating storyboard..." subtitle="" />
      );
    } else {
      empty = (
        <EmptyState
          icon="📖"
          title="Storyboard not yet generated"
          subtitle="The full shot breakdown appears here after Stage 3 completes."
        />
      );
    }
  }

  return (
    <div className={`tab-panel${active ? " active" : ""}`} id="story-tab">
      <div id="story-panel">
        {!rows.length ? (
          empty
        ) : (
          <table className="story-tbl" id="story-tbl">
            <thead>
              <tr>
                <th>Shot</th>
                <th>Function</th>
                <th>Subject</th>
                <th>Dur</th>
                <th>Image</th>
                <th>Video</th>
              </tr>
            </thead>
            <tbody id="story-body">
              {rows.map((s) => (
                <tr key={s.shot_id} id={`sr-${s.shot_id}`}>
                  <td className="shot-id">{s.shot_id}</td>
                  <td>
                    {s.narrative_function || ""}
                    {s.prompt_preview ? (
                      <div className="prompt-preview">{s.prompt_preview}</div>
                    ) : null}
                  </td>
                  <td style={{ maxWidth: "180px", fontSize: ".72rem" }}>
                    {(s.subject || "").slice(0, 70)}
                  </td>
                  <td style={{ fontWeight: 700, color: "var(--gold)" }}>{s.duration}s</td>
                  <td>
                    <span className={`spill sp-${normalizeSpillStatus(s.image_status)}`}>
                      {s.image_status || "pending"}
                    </span>
                  </td>
                  <td>
                    <span className={`spill sp-${normalizeSpillStatus(s.video_status)}`}>
                      {s.video_status || "pending"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
