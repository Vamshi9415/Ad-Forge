import { escapeHtml } from "../../utils/escapeHtml.js";

export function FramesReview({ shots, onOpenLightbox }) {
  return (
    <div className="approval-frame-grid">
      {shots.map((s) => {
        const st = s.image_status || "pending";
        const stCls = st === "VERIFIED" ? "verified" : st === "FAILED" ? "failed" : "";
        return (
          <div key={s.shot_id} className="approval-frame">
            {s.image_src ? (
              <img
                src={s.image_src}
                alt=""
                loading="lazy"
                onClick={() => onOpenLightbox(s.image_src)}
              />
            ) : (
              <div
                style={{
                  aspectRatio: "16 / 9",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  background: "var(--card2)",
                  color: "var(--text3)",
                  fontSize: ".72rem",
                }}
              >
                No image
              </div>
            )}
            <div className={`approval-frame-status ${stCls}`.trim()}>{st}</div>
            <div
              className="approval-frame-label"
              dangerouslySetInnerHTML={{
                __html: `${escapeHtml(s.shot_id)} — ${escapeHtml((s.narrative_function || "").slice(0, 30))}`,
              }}
            />
          </div>
        );
      })}
    </div>
  );
}
