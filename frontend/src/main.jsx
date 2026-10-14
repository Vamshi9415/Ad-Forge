import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import { PipelineProvider } from "./store/PipelineContext.jsx";
import "./App.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <PipelineProvider>
      <App />
    </PipelineProvider>
  </React.StrictMode>
);
