import { usePipeline } from "../../store/PipelineContext.jsx";
import { STAGE_ITEMS } from "../../store/constants.js";
import { stageDotLabel } from "../../store/pipelineReducer.js";

export function PipelineStages() {
  const { state } = usePipeline();

  return (
    <>
      <div className="sb-section" style={{ borderBottom: "none", paddingBottom: 0 }}>
        <div className="sb-title">Pipeline Progress</div>
      </div>
      <div className="stages" id="stages-list">
        {STAGE_ITEMS.map((st) => {
          const meta = state.stages[st.id] || {};
          const status = meta.status || "";
          const dotText = stageDotLabel(st.id, status);
          const timeCls = meta.elapsedSec != null ? "stage-time show" : "stage-time";
          return (
            <div
              key={st.id}
              id={`si-${st.id}`}
              className="stage-item"
              style={st.id === "9" ? { paddingBottom: 0 } : undefined}
            >
              <div className={`stage-dot ${status}`.trim()} id={`sd-${st.id}`}>
                {dotText}
              </div>
              <div className="stage-info">
                <div className={`stage-name ${status}`.trim()} id={`sn-${st.id}`}>
                  {st.name}
                </div>
                <div className="stage-sub">{st.sub}</div>
                <div className={timeCls} id={`st-${st.id}`}>
                  {meta.elapsedSec != null ? `${meta.elapsedSec}s` : ""}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
}
