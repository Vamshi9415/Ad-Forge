import { Header } from "./components/Header.jsx";
import { Sidebar } from "./components/Sidebar/Sidebar.jsx";
import { MainPanel } from "./components/MainPanel/MainPanel.jsx";
import { Toast } from "./components/ui/Toast.jsx";
import { Lightbox } from "./components/ui/Lightbox.jsx";
import { usePipelineController } from "./hooks/usePipeline.js";

function AppShell() {
  const { state, dispatch, sendWs, startPipeline, cancelPipeline } = usePipelineController();

  return (
    <>
      <Header sessionId={state.sessionId} />
      <div className="app">
        <Sidebar
          startPipeline={startPipeline}
          cancelPipeline={cancelPipeline}
          onUploadSuccess={(msg) =>
            dispatch({ type: "TOAST", payload: { msg, type: "success" } })
          }
          onUploadError={(msg) => dispatch({ type: "TOAST", payload: { msg, type: "error" } })}
        />
        <MainPanel sendWs={sendWs} />
      </div>
      <Toast
        msg={state.toast.msg}
        type={state.toast.type}
        show={state.toast.show}
        onClear={() => dispatch({ type: "CLEAR_TOAST" })}
      />
      <Lightbox src={state.lightboxSrc} onClose={() => dispatch({ type: "SET_LIGHTBOX", payload: null })} />
    </>
  );
}

export default function App() {
  return <AppShell />;
}
