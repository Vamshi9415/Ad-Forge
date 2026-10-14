import { STAGE_ORDER } from "../../store/constants.js";
import { stageSegClass } from "../../store/pipelineReducer.js";
import { usePipeline } from "../../store/PipelineContext.jsx";

export function PipelineProgressBar() {
  const { state } = usePipeline();
  const doneCount = state.completedStages.length;
  const label = state.running
    ? `${doneCount}/${STAGE_ORDER.length}`
    : doneCount === STAGE_ORDER.length && doneCount > 0
      ? "Done!"
      : state.progressLabel;

  return (
    <div className="pipeline-progress" id="pp">
      <span className="pp-label" id="pp-label">
        {label}
      </span>
      {STAGE_ORDER.map((id) => (
        <div
          key={id}
          id={`pp-${id}`}
          className={`pp-seg ${stageSegClass(state.stages[id]?.status || "")}`.trim()}
        />
      ))}
    </div>
  );
}
