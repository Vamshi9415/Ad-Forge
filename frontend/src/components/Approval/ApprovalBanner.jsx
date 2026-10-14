import { escapeHtml } from "../../utils/escapeHtml.js";
import { usePipeline } from "../../store/PipelineContext.jsx";
import { useApproval } from "../../hooks/useApproval.js";
import { BriefEditor } from "./BriefEditor.jsx";
import { StoryboardEditor } from "./StoryboardEditor.jsx";
import { PromptsEditor } from "./PromptsEditor.jsx";
import { FramesReview } from "./FramesReview.jsx";

export function ApprovalBanner({ sendWs }) {
  const { state, dispatch } = usePipeline();
  const { submitApproval, submitSkipAll } = useApproval(sendWs);
  const ap = state.approval;
  if (!ap) return null;

  const { stageId, stageName, reviewData } = ap;
  const rd = reviewData || {};
  let inner = null;
  if (rd.type === "brief") inner = <BriefEditor fields={rd.fields} />;
  else if (rd.type === "storyboard") inner = <StoryboardEditor shots={rd.shots} />;
  else if (rd.type === "prompts") inner = <PromptsEditor shots={rd.shots} />;
  else if (rd.type === "frames")
    inner = (
      <FramesReview
        shots={rd.shots}
        onOpenLightbox={(src) => dispatch({ type: "SET_LIGHTBOX", payload: src })}
      />
    );

  return (
    <div className="approval-banner" id="approval-panel">
      <div className="approval-hdr">
        <div>
          <div
            className="approval-title"
            dangerouslySetInnerHTML={{ __html: `✋ Review: ${escapeHtml(stageName)}` }}
          />
          <div className="approval-subtitle">
            Review the output below. Edit fields directly, then approve to continue.
          </div>
        </div>
      </div>
      <div className="approval-body" id="approval-body">
        {inner}
      </div>
      <div className="approval-actions">
        <button type="button" className="btn-approve" onClick={() => submitApproval(stageId, "approve")}>
          Approve &amp; Continue
        </button>
        <button type="button" className="btn-edit" onClick={() => submitApproval(stageId, "edit")}>
          Save Edits &amp; Continue
        </button>
        <button type="button" className="btn-skip-all" onClick={submitSkipAll}>
          Skip All Reviews
        </button>
      </div>
    </div>
  );
}
