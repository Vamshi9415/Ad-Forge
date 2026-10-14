export function Header({ sessionId }) {
  return (
    <header>
      <div className="logo">
        <div className="logo-mark">🎬</div>
        Ad<span>Forge</span>
      </div>
      <div className="hdr-right">
        <span className={`session-id${sessionId ? " is-visible" : ""}`}>
          {sessionId ? `Session: ${sessionId}` : ""}
        </span>
        <span className="pill pill-pur">RunwayML Gen-4.5</span>
        <span className="pill pill-em">GPT-4o</span>
      </div>
    </header>
  );
}
