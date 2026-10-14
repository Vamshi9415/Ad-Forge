import { useEffect, useRef } from "react";
import { escapeHtml } from "../../utils/escapeHtml.js";
import { usePipeline } from "../../store/PipelineContext.jsx";
import { ApprovalBanner } from "../Approval/ApprovalBanner.jsx";
import { EmptyState } from "../ui/EmptyState.jsx";

function logStageClass(stage) {
  let sc = `s${stage}`;
  sc = sc.replace("s6a", "s6").replace("ssys", "sys");
  return sc;
}

export function LogPanel({ sendWs }) {
  const { state } = usePipeline();
  const bottomRef = useRef(null);
  const active = state.activeTab === "log";

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [state.logs.length, state.approval]);

  const showEmpty = state.logs.length === 0;

  return (
    <div className={`tab-panel${active ? " active" : ""}`} id="log-tab">
      <div id="log-panel">
        {showEmpty && !state.approval ? (
          <EmptyState
            icon="📡"
            title="Awaiting pipeline start"
            subtitle="Configure your campaign brief on the left and click Generate. Real-time agent activity will stream here."
          />
        ) : null}
        {state.logs.map((e) => (
          <div key={e.id} className="log-entry">
            <span className="log-time">{e.time}</span>
            <span className={`log-stage ${logStageClass(e.stage)}`}>{e.stage}</span>
            <span
              className={`log-msg ${e.level || "info"}`}
              dangerouslySetInnerHTML={{ __html: escapeHtml(e.message) }}
            />
          </div>
        ))}
        {state.approval ? <ApprovalBanner sendWs={sendWs} /> : null}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
