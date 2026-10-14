import { TabBar } from "./TabBar.jsx";
import { PipelineProgressBar } from "./PipelineProgressBar.jsx";
import { LogPanel } from "./LogPanel.jsx";
import { FramesPanel } from "./FramesPanel.jsx";
import { VideoPanel } from "./VideoPanel.jsx";
import { StoryPanel } from "./StoryPanel.jsx";
import { BriefPanel } from "./BriefPanel.jsx";
import { usePipeline } from "../../store/PipelineContext.jsx";

export function MainPanel({ sendWs }) {
  const { dispatch } = usePipeline();

  return (
    <div className="main-panel">
      <PipelineProgressBar />
      <TabBar />
      <LogPanel sendWs={sendWs} />
      <FramesPanel
        onOpenLightbox={(src) => dispatch({ type: "SET_LIGHTBOX", payload: src })}
      />
      <VideoPanel />
      <StoryPanel />
      <BriefPanel />
    </div>
  );
}
