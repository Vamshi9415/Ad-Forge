import { CampaignForm } from "./CampaignForm.jsx";
import { CostEstimate } from "./CostEstimate.jsx";
import { PipelineStages } from "./PipelineStages.jsx";
import { Toggle } from "../ui/Toggle.jsx";
import { usePipeline } from "../../store/PipelineContext.jsx";

export function Sidebar({ startPipeline, cancelPipeline, onUploadSuccess, onUploadError }) {
  const { state, dispatch } = usePipeline();
  const d = state.draft;

  return (
    <div className="sidebar">
      <div className="hero">
        <h1>Generative Ad Pipeline</h1>
        <p>
          7-stage autonomous multi-agent system for broadcast-quality commercial video synthesis.
        </p>
        <div className="hero-stats">
          <div className="hstat">
            <strong>{state.costEstimate.shots}</strong>Shots
          </div>
          <div className="hstat">
            <strong>{state.costEstimate.dur}</strong>Runtime
          </div>
          <div className="hstat">
            <strong>{state.costEstimate.price}</strong>Est. Cost
          </div>
        </div>
      </div>

      <CampaignForm onUploadSuccess={onUploadSuccess} onUploadError={onUploadError} />
      <CostEstimate />

      <div className="sb-section">
        <Toggle
          checked={d.interactive}
          onChange={(interactive) => dispatch({ type: "UPDATE_DRAFT", payload: { interactive } })}
          label="Interactive mode — review & edit at each stage"
        />
        <button type="button" className="gen-btn" disabled={state.genBusy} onClick={startPipeline}>
          <span>{state.genBusy ? "⟳" : "🚀"}</span>{" "}
          <span>{state.genBusy ? "Generating..." : "Generate Ad Campaign"}</span>
        </button>
        <div style={{ height: "8px" }} />
        <button
          type="button"
          className={`cancel-btn${state.genBusy ? "" : " is-hidden"}`}
          onClick={cancelPipeline}
        >
          Stop Pipeline
        </button>
      </div>

      <PipelineStages />
    </div>
  );
}
