import { useEffect, useRef, useCallback } from "react";

/**
 * Connects when sessionId is set; dispatches { type: 'WS_MESSAGE', payload } for each event.
 */
export function useWebSocket(sessionId, dispatch) {
  const wsRef = useRef(null);
  const firstFrameRef = useRef(true);
  const intentionalCloseRef = useRef(false);

  const send = useCallback((obj) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(obj));
    }
  }, []);

  const close = useCallback(() => {
    intentionalCloseRef.current = true;
    try {
      wsRef.current?.close();
    } catch {
      /* ignore */
    }
    wsRef.current = null;
  }, []);

  useEffect(() => {
    firstFrameRef.current = true;
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId) {
      intentionalCloseRef.current = true;
      try {
        wsRef.current?.close();
      } catch {
        /* ignore */
      }
      wsRef.current = null;
      return;
    }

    intentionalCloseRef.current = false;
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${window.location.host}/api/ws/${sessionId}`);
    wsRef.current = ws;

    ws.onmessage = (e) => {
      let msg;
      try {
        msg = JSON.parse(e.data);
      } catch {
        return;
      }
      if (msg.type === "ping") return;
      if (msg.type === "done") {
        intentionalCloseRef.current = true;
        ws.close();
        return;
      }

      dispatch({ type: "WS_MESSAGE", payload: msg });

      if (msg.type === "image_ready" && firstFrameRef.current) {
        firstFrameRef.current = false;
        dispatch({ type: "SET_TAB", payload: "frames" });
      }
      if (msg.type === "pipeline_complete") {
        dispatch({ type: "SET_TAB", payload: "video" });
      }
    };

    ws.onerror = () => {
      intentionalCloseRef.current = true;
      dispatch({
        type: "TOAST",
        payload: { msg: "WebSocket error", type: "error" },
      });
      dispatch({ type: "SET_RUNNING", payload: false });
    };

    ws.onclose = () => {
      if (!intentionalCloseRef.current) {
        dispatch({ type: "WS_CLOSED" });
      }
      intentionalCloseRef.current = false;
      if (wsRef.current === ws) wsRef.current = null;
    };

    return () => {
      intentionalCloseRef.current = true;
      ws.close();
    };
  }, [sessionId, dispatch]);

  return { send, close };
}
