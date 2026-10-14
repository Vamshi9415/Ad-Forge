import React, { createContext, useContext, useReducer, useMemo } from "react";
import { createInitialState, pipelineReducer } from "./pipelineReducer.js";

const PipelineContext = createContext(null);

export function PipelineProvider({ children }) {
  const [state, dispatch] = useReducer(pipelineReducer, undefined, createInitialState);
  const value = useMemo(() => ({ state, dispatch }), [state, dispatch]);
  return <PipelineContext.Provider value={value}>{children}</PipelineContext.Provider>;
}

export function usePipeline() {
  const ctx = useContext(PipelineContext);
  if (!ctx) throw new Error("usePipeline must be used within PipelineProvider");
  return ctx;
}
