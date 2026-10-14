import { escapeHtml } from "../../utils/escapeHtml.js";

export function PromptsEditor({ shots }) {
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
          <div className="approval-field">
            <label>Image Prompt</label>
            <textarea data-f="image_prompt" rows={3} defaultValue={s.image_prompt || ""} />
          </div>
          <div className="approval-field">
            <label>Motion Prompt</label>
            <textarea data-f="motion_prompt" rows={2} defaultValue={s.motion_prompt || ""} />
          </div>
        </div>
      ))}
    </>
  );
}
