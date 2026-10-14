import { useCallback, useEffect } from "react";
import { usePipeline } from "../store/PipelineContext.jsx";
import { useWebSocket } from "./useWebSocket.js";
import { VID_COST, IMG_COST } from "../store/constants.js";

function computeCostEstimate(draft) {
  const dur = Number(draft.duration) || 30;
  const shots = Math.max(2, Math.round(dur / 5));
  const vidModel = draft.videoModel || "gen4_turbo";
  const imgModel = draft.imageModel || "gen4_image";
  const imgC = shots * (IMG_COST[imgModel] ?? 6);
  const vidC = dur * (VID_COST[vidModel] ?? 5);
  const tot = imgC + vidC;
  return {
    img: `~${imgC}cr ($${(imgC * 0.01).toFixed(2)})`,
    vid: `~${vidC}cr ($${(vidC * 0.01).toFixed(2)})`,
    total: `~${tot}cr ($${(tot * 0.01).toFixed(2)})`,
    shots: String(shots),
    dur: `${dur}s`,
    price: `$${(tot * 0.01).toFixed(2)}`,
  };
}

/**
 * Wires WebSocket to session, exposes start/cancel and cost recompute.
 */
export function usePipelineController() {
  const { state, dispatch } = usePipeline();
  const { send: sendWs, close: closeWs } = useWebSocket(state.sessionId, dispatch);

  const recomputeCost = useCallback(() => {
    dispatch({ type: "SET_COST_ESTIMATE", payload: computeCostEstimate(state.draft) });
  }, [dispatch, state.draft]);

  useEffect(() => {
    dispatch({ type: "SET_COST_ESTIMATE", payload: computeCostEstimate(state.draft) });
  }, [
    dispatch,
    state.draft.duration,
    state.draft.platform,
    state.draft.videoModel,
    state.draft.imageModel,
  ]);

  const startPipeline = useCallback(async () => {
    const prompt = state.draft.prompt.trim();
    if (!prompt) {
      dispatch({
        type: "TOAST",
        payload: { msg: "Please enter an advertising concept", type: "error" },
      });
      return;
    }
    if (state.running) {
      dispatch({
        type: "TOAST",
        payload: { msg: "Pipeline already running", type: "warning" },
      });
      return;
    }

    dispatch({ type: "PIPELINE_RESET" });
    dispatch({ type: "SET_RUNNING", payload: true });
    dispatch({ type: "SET_LIVE_COST", payload: "$0.00" });

    const platform = state.draft.platform;
    const duration = Number(state.draft.duration) || 30;

    try {
      const r = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          image_data_uri: state.draft.imageDataUri,
          platform,
          quality_mode: "preview",
          total_duration: duration,
          video_model: state.draft.videoModel,
          image_model: state.draft.imageModel,
          interactive: state.draft.interactive,
        }),
      });
      if (!r.ok) throw new Error(`Failed to start session: ${await r.text()}`);
      const { session_id } = await r.json();
      dispatch({ type: "SET_SESSION", payload: session_id });
      dispatch({
        type: "ADD_LOG",
        payload: {
          stage: "sys",
          message: `Session ${session_id} created — pipeline starting...`,
          level: "info",
        },
      });
    } catch (e) {
      dispatch({ type: "SET_RUNNING", payload: false });
      dispatch({
        type: "TOAST",
        payload: { msg: `Error: ${e.message}`, type: "error" },
      });
    }
  }, [state.draft, state.running, dispatch]);

  const cancelPipeline = useCallback(() => {
    closeWs();
    dispatch({ type: "SET_RUNNING", payload: false });
    dispatch({
      type: "ADD_LOG",
      payload: { stage: "sys", message: "Pipeline cancelled by user", level: "warning" },
    });
  }, [dispatch, closeWs]);

  return {
    state,
    dispatch,
    sendWs,
    closeWs,
    recomputeCost,
    startPipeline,
    cancelPipeline,
  };
}
