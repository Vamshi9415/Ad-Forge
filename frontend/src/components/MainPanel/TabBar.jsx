import { usePipeline } from "../../store/PipelineContext.jsx";
import { Badge } from "../ui/Badge.jsx";

const TAB_DEFS = [
  { id: "log", label: "📋 Live Log", badgeKey: "log" },
  { id: "frames", label: "🖼️ Frames", badgeKey: "frames" },
  { id: "video", label: "🎬 Video", badgeKey: "video" },
  { id: "story", label: "📖 Storyboard", badgeKey: null },
  { id: "brief", label: "✨ Brief", badgeKey: null },
];

export function TabBar() {
  const { state, dispatch } = usePipeline();

  return (
    <div className="tabs">
      {TAB_DEFS.map((t) => (
        <div
          key={t.id}
          role="tab"
          tabIndex={0}
          className={`tab${state.activeTab === t.id ? " active" : ""}`}
          onClick={() => dispatch({ type: "SET_TAB", payload: t.id })}
          onKeyDown={(e) => e.key === "Enter" && dispatch({ type: "SET_TAB", payload: t.id })}
        >
          {t.label}{" "}
          {t.badgeKey != null ? (
            <Badge>{state.tabBadges[t.badgeKey] ?? 0}</Badge>
          ) : null}
        </div>
      ))}
    </div>
  );
}
