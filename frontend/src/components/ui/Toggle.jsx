export function Toggle({ checked, onChange, label }) {
  return (
    <div className="toggle-row">
      <label className="toggle-switch">
        <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
        <span className="toggle-track" />
      </label>
      {label ? <span className="toggle-label">{label}</span> : null}
    </div>
  );
}
