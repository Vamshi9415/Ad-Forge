import { usePipeline } from "../../store/PipelineContext.jsx";
import { EmptyState } from "../ui/EmptyState.jsx";

function frameBadge(status) {
  const u = String(status || "").toUpperCase();
  if (u === "VERIFIED") return { cls: "fb-ver", label: "Verified" };
  if (u === "FAILED") return { cls: "fb-fail", label: "Failed" };
  if (u === "GENERATED" || status === "generated") return { cls: "fb-gen", label: "Generated" };
  return { cls: "fb-pend", label: "Pending" };
}

export function FramesPanel({ onOpenLightbox }) {
  const { state } = usePipeline();
  const portrait = ["tiktok", "instagram_reels"].includes(state.draft.platform);
  const active = state.activeTab === "frames";

  if (!state.frameOrder.length) {
    return (
      <div className={`tab-panel${active ? " active" : ""}`} id="frames-tab">
        <div id="frames-panel">
          <div className="frames-grid" id="frames-grid">
            <EmptyState
              style={{ gridColumn: "1 / -1" }}
              icon="🎨"
              title="No frames yet"
              subtitle="Generated reference frames appear here in real-time during Stage 6."
            />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`tab-panel${active ? " active" : ""}`} id="frames-tab">
      <div id="frames-panel">
        <div className="frames-grid" id="frames-grid">
          {state.frameOrder.map((shotId) => {
            const fr = state.frames[shotId] || {};
            const wrapCls = `frame-wrap${portrait ? " portrait" : ""}`;
            const { cls, label } = frameBadge(fr.status);
            return (
              <div
                key={shotId}
                className="frame-card"
                id={`fc-${shotId}`}
                onClick={() => fr.src && onOpenLightbox(fr.src)}
                role="presentation"
              >
                <div className={wrapCls} id={`fw-${shotId}`}>
                  {!fr.src ? (
                    <div className="frame-ph">
                      <div className="fspin" />
                      <span>Queued</span>
                    </div>
                  ) : (
                    <>
                      <img
                        src={fr.src}
                        alt=""
                        loading="lazy"
                        onClick={(e) => {
                          e.stopPropagation();
                          onOpenLightbox(fr.src);
                        }}
                      />
                      <div className="fov">
                        <span className={`fbadge ${cls}`}>{label}</span>
                      </div>
                    </>
                  )}
                </div>
                <div className="finfo">
                  <div className="fid">{shotId}</div>
                  <div className="ffunc" id={`ff-${shotId}`}>
                    {fr.prompt ? `${fr.prompt.slice(0, 55)}…` : fr.func || ""}
                  </div>
                  <div className="fdur" id={`fd-${shotId}`} />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
