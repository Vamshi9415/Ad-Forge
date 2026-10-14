import { escapeHtml } from "../../utils/escapeHtml.js";

export function StoryboardEditor({ shots }) {
  return (
    <>
      {shots.map((s) => (
        <div key={s.shot_id} className="approval-shot" data-shot-id={s.shot_id}>
          <div className="approval-shot-hdr">
            <span className="approval-shot-id">{s.shot_id}</span>
            <span
              className="approval-shot-func"
              dangerouslySetInnerHTML={{ __html: escapeHtml(s.narrative_function || "") }}
            />
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 80px",
              gap: "8px",
              marginBottom: "8px",
            }}
          >
            <div className="approval-field" style={{ margin: 0 }}>
              <label>Subject</label>
              <input data-f="subject" defaultValue={s.subject || ""} />
            </div>
            <div className="approval-field" style={{ margin: 0 }}>
              <label>Duration</label>
              <input data-f="duration" type="number" min={3} max={15} defaultValue={s.duration} />
            </div>
          </div>
          <div className="approval-field">
            <label>Environment</label>
            <input data-f="environment" defaultValue={s.environment || ""} />
          </div>
          <div className="approval-field">
            <label>Action</label>
            <input data-f="action" defaultValue={s.action || ""} />
          </div>
          <input type="hidden" data-f="narrative_function" defaultValue={s.narrative_function || ""} />
        </div>
      ))}
    </>
  );
}
