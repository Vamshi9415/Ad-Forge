import { useCallback } from "react";
import { usePipeline } from "../store/PipelineContext.jsx";

function collectEditsFromBody(reviewData) {
  if (!reviewData) return null;
  const rd = reviewData;
  if (rd.type === "brief") {
    const fields = {};
    document.querySelectorAll("#approval-body [data-field]").forEach((el) => {
      fields[el.dataset.field] = el.value;
    });
    return { fields };
  }
  if (rd.type === "storyboard") {
    const shots = [];
    document.querySelectorAll("#approval-body .approval-shot").forEach((el) => {
      shots.push({
        shot_id: el.dataset.shotId,
        narrative_function: el.querySelector('[data-f="narrative_function"]')?.value || "",
        duration: parseInt(el.querySelector('[data-f="duration"]')?.value, 10) || 5,
        subject: el.querySelector('[data-f="subject"]')?.value || "",
        environment: el.querySelector('[data-f="environment"]')?.value || "",
        action: el.querySelector('[data-f="action"]')?.value || "",
      });
    });
    return { shots };
  }
  if (rd.type === "prompts") {
    const shots = [];
    document.querySelectorAll("#approval-body .approval-shot").forEach((el) => {
      shots.push({
        shot_id: el.dataset.shotId,
        image_prompt: el.querySelector('[data-f="image_prompt"]')?.value || "",
        motion_prompt: el.querySelector('[data-f="motion_prompt"]')?.value || "",
      });
    });
    return { shots };
  }
  return null;
}

/**
 * Submit approval / skip-all WebSocket messages and sync UI state with the vanilla client.
 */
export function useApproval(sendWs) {
  const { dispatch, state } = usePipeline();

  const submitApproval = useCallback(
    (stageId, action) => {
      const edits = action === "edit" ? collectEditsFromBody(state.approval?.reviewData) : null;
      sendWs({ type: "approval_response", stage_id: stageId, action, edits });
      dispatch({
        type: "APPROVAL_RESOLVED",
        payload: {
          stageId,
          logMsg: action === "edit" ? "Edits saved — continuing" : "Approved — continuing",
        },
      });
    },
    [dispatch, sendWs, state.approval]
  );

  const submitSkipAll = useCallback(() => {
    sendWs({ type: "skip_all_reviews" });
    dispatch({ type: "CLEAR_APPROVAL" });
    dispatch({ type: "SKIP_ALL_APPROVALS_LOG" });
  }, [dispatch, sendWs]);

  return { submitApproval, submitSkipAll };
}
