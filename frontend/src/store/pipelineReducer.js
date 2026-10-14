import { STAGE_ORDER, nextLogId } from "./constants.js";

function emptyStages() {
  const o = {};
  for (const id of STAGE_ORDER) {
    o[id] = { status: "", elapsedSec: null, startedAt: null };
  }
  return o;
}

function logTime() {
  return new Date().toLocaleTimeString("en", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function createInitialState() {
  return {
    sessionId: null,
    running: false,
    genBusy: false,
    draft: {
      prompt: "",
      platform: "youtube",
      duration: 30,
      videoModel: "gen4_turbo",
      imageModel: "gen4_image",
      imageDataUri: null,
      interactive: true,
      uploadPreview: null,
      hasUpload: false,
    },
    stages: emptyStages(),
    completedStages: [],
    progressLabel: "Ready",
    logs: [],
    logBadge: 0,
    briefMerge: {},
    briefShown: false,
    storyboard: [],
    frames: {},
    frameOrder: [],
    clips: [],
    clipsInited: false,
    finalVideo: null,
    audio: null,
    approval: null,
    activeTab: "log",
    tabBadges: { log: 0, frames: 0, video: 0 },
    toast: { msg: "", type: "", show: false },
    lightboxSrc: null,
    liveCost: "$0.00",
    costEstimate: {
      img: "—",
      vid: "—",
      total: "—",
      shots: "—",
      dur: "—",
      price: "—",
    },
  };
}

function appendLog(state, stage, message, level = "info") {
  const entry = {
    id: nextLogId(),
    time: logTime(),
    stage: String(stage),
    message,
    level,
  };
  return {
    ...state,
    logs: [...state.logs, entry],
    logBadge: state.logBadge + 1,
    tabBadges: { ...state.tabBadges, log: state.tabBadges.log + 1 },
  };
}

function stageSegClass(status) {
  if (status === "complete") return "done";
  if (status === "running") return "active";
  if (status === "waiting") return "waiting";
  if (status === "error") return "err";
  return "";
}

function setStageStatus(state, id, status) {
  const prev = state.stages[id] || {};
  let elapsedSec = prev.elapsedSec;
  let startedAt = prev.startedAt;
  if (status === "running") {
    startedAt = Date.now();
  }
  if (status === "complete" || status === "error") {
    if (prev.startedAt != null) {
      elapsedSec = ((Date.now() - prev.startedAt) / 1000).toFixed(1);
    }
    startedAt = null;
  }
  return {
    ...state,
    stages: {
      ...state.stages,
      [id]: { ...prev, status, elapsedSec: elapsedSec ?? prev.elapsedSec, startedAt },
    },
  };
}

function mergeBrief(state, data) {
  if (!data) return state;
  return {
    ...state,
    briefMerge: { ...state.briefMerge, ...data },
    briefShown: true,
  };
}

function storyFromStageData(storyboard) {
  return storyboard.map((s) => ({
    shot_id: s.shot_id,
    narrative_function: s.function || s.narrative_function,
    subject: s.subject || "",
    duration: s.duration,
    image_status: "pending",
    video_status: "pending",
    prompt_preview: "",
  }));
}

function normalizeSpillStatus(st) {
  const x = String(st || "pending").toLowerCase();
  if (x === "verified") return "verified";
  if (x === "complete") return "complete";
  if (x === "generated") return "generated";
  if (x === "failed") return "failed";
  if (x === "generating") return "generating";
  return "pending";
}

export function pipelineReducer(state, action) {
  switch (action.type) {
    case "PIPELINE_RESET": {
      const d = state.draft;
      return { ...createInitialState(), draft: { ...d }, sessionId: null };
    }

    case "UPDATE_DRAFT": {
      return { ...state, draft: { ...state.draft, ...action.payload } };
    }

    case "SET_SESSION": {
      return { ...state, sessionId: action.payload };
    }

    case "SET_RUNNING": {
      return {
        ...state,
        running: action.payload,
        genBusy: action.payload,
      };
    }

    case "SET_TAB": {
      return { ...state, activeTab: action.payload };
    }

    case "SET_LIGHTBOX": {
      return { ...state, lightboxSrc: action.payload };
    }

    case "ADD_LOG": {
      return appendLog(
        state,
        action.payload.stage,
        action.payload.message,
        action.payload.level || "info"
      );
    }

    case "TOAST": {
      return {
        ...state,
        toast: { msg: action.payload.msg, type: action.payload.type || "info", show: true },
      };
    }

    case "CLEAR_TOAST": {
      return { ...state, toast: { ...state.toast, show: false } };
    }

    case "SET_COST_ESTIMATE": {
      return { ...state, costEstimate: { ...state.costEstimate, ...action.payload } };
    }

    case "SET_LIVE_COST": {
      return { ...state, liveCost: action.payload };
    }

    case "WS_MESSAGE": {
      return reduceWsMessage(state, action.payload);
    }

    case "APPROVAL_RESOLVED": {
      const { stageId, logMsg } = action.payload;
      let s = setStageStatus(state, stageId, "complete");
      const done = new Set(state.completedStages);
      done.add(stageId);
      s = {
        ...s,
        completedStages: [...done],
        approval: null,
        progressLabel:
          s.running && done.size < STAGE_ORDER.length
            ? `${done.size}/${STAGE_ORDER.length}`
            : done.size === STAGE_ORDER.length
              ? "Done!"
              : "Ready",
      };
      if (logMsg) s = appendLog(s, stageId, logMsg, "success");
      return s;
    }

    case "SKIP_ALL_APPROVALS_LOG": {
      return appendLog(state, "sys", "Auto-approve enabled for remaining stages", "info");
    }

    case "CLEAR_APPROVAL": {
      return { ...state, approval: null };
    }

    case "WS_CLOSED": {
      if (!state.running) return state;
      return appendLog(state, "sys", "Connection closed", "info");
    }

    default:
      return state;
  }
}

function reduceWsMessage(state, msg) {
  switch (msg.type) {
    case "stage_start": {
      let s = setStageStatus(state, msg.stage_id, "running");
      s = appendLog(s, msg.stage_id, `Started: ${msg.stage_name}`, "info");
      return { ...s, progressLabel: `Stage ${msg.stage_id}` };
    }

    case "stage_complete": {
      let s = setStageStatus(state, msg.stage_id, "complete");
      const done = new Set(state.completedStages);
      done.add(msg.stage_id);
      s = {
        ...s,
        completedStages: [...done],
        progressLabel: s.running ? `${done.size}/${STAGE_ORDER.length}` : s.progressLabel,
      };
      s = appendLog(s, msg.stage_id, `Complete: ${msg.stage_name}`, "success");
      return s;
    }

    case "stage_error": {
      let s = setStageStatus(state, msg.stage_id, "error");
      s = appendLog(s, msg.stage_id, `Error in ${msg.stage_name}: ${msg.message}`, "error");
      return {
        ...s,
        toast: { msg: `Stage ${msg.stage_id} failed`, type: "error", show: true },
      };
    }

    case "log": {
      return appendLog(state, msg.stage || "sys", msg.message, msg.level || "info");
    }

    case "stage_data": {
      let s = state;
      if (msg.stage === "3" && msg.data?.storyboard) {
        const shots = storyFromStageData(msg.data.storyboard);
        const frames = { ...s.frames };
        const frameOrder = [...s.frameOrder];
        for (const sh of shots) {
          if (!frames[sh.shot_id]) {
            frames[sh.shot_id] = {
              shotId: sh.shot_id,
              func: sh.narrative_function,
              status: "pending",
              src: null,
              prompt: null,
              placeholder: true,
            };
            frameOrder.push(sh.shot_id);
          }
        }
        s = {
          ...s,
          storyboard: shots,
          frames,
          frameOrder,
          tabBadges: { ...s.tabBadges, frames: shots.length },
        };
      }
      if ((msg.stage === "2" || msg.stage === "1") && msg.data) {
        s = mergeBrief(s, msg.data);
      }
      return s;
    }

    case "image_ready": {
      const { shot_id, src, status, prompt, cost_usd } = msg;
      const frames = {
        ...state.frames,
        [shot_id]: {
          ...(state.frames[shot_id] || {}),
          shotId: shot_id,
          status,
          src,
          prompt: prompt || state.frames[shot_id]?.prompt,
          placeholder: false,
        },
      };
      let frameOrder = state.frameOrder.includes(shot_id)
        ? state.frameOrder
        : [...state.frameOrder, shot_id];
      let s = {
        ...state,
        frames,
        frameOrder,
        storyboard: state.storyboard.map((row) =>
          row.shot_id === shot_id
            ? {
                ...row,
                image_status: status,
                prompt_preview: prompt ? prompt.slice(0, 70) + "…" : row.prompt_preview,
              }
            : row
        ),
      };
      if (cost_usd != null) s = { ...s, liveCost: `$${Number(cost_usd).toFixed(2)}` };
      return s;
    }

    case "image_validated": {
      const { shot_id, verdict } = msg;
      const prevSrc = state.frames[shot_id]?.src || "";
      const frames = {
        ...state.frames,
        [shot_id]: {
          ...(state.frames[shot_id] || { shotId: shot_id }),
          status: verdict,
          src: prevSrc,
        },
      };
      return {
        ...state,
        frames,
        storyboard: state.storyboard.map((row) =>
          row.shot_id === shot_id ? { ...row, image_status: verdict } : row
        ),
      };
    }

    case "video_ready": {
      let s = {
        ...state,
        clips: [...state.clips, { shotId: msg.shot_id, src: msg.src, duration: msg.duration }],
        clipsInited: true,
        tabBadges: { ...state.tabBadges, video: state.tabBadges.video + 1 },
        storyboard: state.storyboard.map((row) =>
          row.shot_id === msg.shot_id ? { ...row, video_status: "complete" } : row
        ),
      };
      if (msg.cost_usd != null) s = { ...s, liveCost: `$${Number(msg.cost_usd).toFixed(2)}` };
      return s;
    }

    case "audio_ready": {
      let s = {
        ...state,
        audio: {
          audioSrc: msg.audio_src,
          script: msg.script,
          voice: msg.voice || "Maya",
          musicSrc: msg.music_src,
          hasMusic: msg.has_music,
        },
      };
      if (msg.audio_video_src && s.finalVideo) {
        s = {
          ...s,
          finalVideo: {
            ...s.finalVideo,
            src: msg.audio_video_src,
            downloadHref: msg.audio_video_src,
            withVoiceover: true,
          },
        };
      }
      s = appendLog(s, "9", "Voiceover audio ready — mixed into final video", "success");
      return {
        ...s,
        toast: { msg: "Voiceover generated and mixed into final video 🎙️", type: "success", show: true },
      };
    }

    case "final_video_ready": {
      return appendLog(state, "8", "Video assembled — awaiting audio mux...", "info");
    }

    case "vlm_summary": {
      return appendLog(
        state,
        "6a",
        `VLM complete: ${msg.verified}/${msg.total} verified, ${msg.failed} failed`,
        "info"
      );
    }

    case "approval_request": {
      let s = setStageStatus(state, msg.stage_id, "waiting");
      return {
        ...s,
        activeTab: "log",
        approval: {
          stageId: msg.stage_id,
          stageName: msg.stage_name,
          reviewData: msg.review_data,
        },
        toast: { msg: "Review required — check the approval panel", type: "warning", show: true },
      };
    }

    case "pipeline_complete": {
      const clipCount = msg.individual_clips?.length ?? state.clips.length;
      let s = {
        ...state,
        running: false,
        genBusy: false,
        progressLabel: "Done!",
        finalVideo: msg.final_video
          ? {
              src: msg.final_video,
              title: msg.campaign_title || "Ad Campaign",
              duration: msg.total_duration,
              costUsd: msg.total_cost_usd?.toFixed(2) ?? "?",
              clips: clipCount,
              audioSrc: msg.audio_src || null,
              downloadHref: msg.final_video,
              withVoiceover: Boolean(msg.audio_src),
            }
          : state.finalVideo,
      };
      if (msg.total_cost_usd != null) {
        s = { ...s, liveCost: `$${Number(msg.total_cost_usd).toFixed(2)}` };
      }
      if (msg.voiceover_script && !s.audio) {
        s = {
          ...s,
          audio: {
            audioSrc: msg.audio_src,
            script: msg.voiceover_script,
            voice: "Maya",
            musicSrc: null,
            hasMusic: false,
          },
        };
      }
      s = appendLog(
        s,
        "sys",
        `Pipeline complete — ${msg.total_duration}s runtime, $${(msg.total_cost_usd || 0).toFixed(2)} spend`,
        "success"
      );
      return {
        ...s,
        toast: {
          msg: `Done! ${msg.total_duration}s commercial generated — $${(msg.total_cost_usd || 0).toFixed(2)}`,
          type: "success",
          show: true,
        },
      };
    }

    case "pipeline_error": {
      return {
        ...appendLog(state, "sys", `Pipeline error: ${msg.message}`, "error"),
        running: false,
        genBusy: false,
        toast: { msg: `Pipeline error: ${msg.message}`, type: "error", show: true },
      };
    }

    default:
      return state;
  }
}

export function stageDotLabel(id, status) {
  if (status === "complete") return "✓";
  if (status === "error") return "✗";
  if (status === "waiting") return "✎";
  return id;
}

export { stageSegClass, normalizeSpillStatus };
