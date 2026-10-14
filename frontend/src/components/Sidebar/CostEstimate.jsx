import { usePipeline } from "../../store/PipelineContext.jsx";

export function CostEstimate() {
  const { state } = usePipeline();
  const c = state.costEstimate;
  return (
    <div className="sb-section">
      <div className="sb-title">Cost Estimate</div>
      <div className="cost-card">
        <div className="cost-row">
          <span>Image frames</span>
          <span className="cost-val">{c.img}</span>
        </div>
        <div className="cost-row">
          <span>Video synthesis</span>
          <span className="cost-val">{c.vid}</span>
        </div>
        <div className="cost-row cost-divider">
          <span style={{ fontWeight: 700, color: "var(--text)" }}>Total estimate</span>
          <span className="cost-val">{c.total}</span>
        </div>
        <div className="cost-row live-cost">
          <span>Live spend</span>
          <span className="cost-val">{state.liveCost}</span>
        </div>
      </div>
    </div>
  );
}
