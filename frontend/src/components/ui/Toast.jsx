import { useEffect } from "react";

export function Toast({ msg, type, show, onClear }) {
  useEffect(() => {
    if (!show) return;
    const t = setTimeout(() => onClear?.(), 3500);
    return () => clearTimeout(t);
  }, [show, onClear, msg]);

  if (!msg && !show) return null;
  return (
    <div id="toast" className={`${show ? "show " : ""}${type || ""}`.trim()}>
      {msg}
    </div>
  );
}
