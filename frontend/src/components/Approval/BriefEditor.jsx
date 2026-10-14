const LABELS = {
  campaign_title: "Campaign Title",
  tagline: "Tagline",
  tone: "Emotional Tone",
  visual_style: "Visual Style",
  pacing: "Pacing",
  music_mood: "Music Mood",
  target_audience: "Target Audience",
  narrative_arc: "Narrative Arc",
  creative_direction: "Creative Direction",
  color_grade: "Color Grade",
};

export function BriefEditor({ fields }) {
  const keys = Object.keys(fields || {});
  return (
    <>
      {keys.map((key) => {
        if (!Object.prototype.hasOwnProperty.call(fields, key)) return null;
        const val = fields[key];
        const lbl = LABELS[key] || key;
        const isLong =
          String(val || "").length > 60 ||
          key === "creative_direction" ||
          key === "target_audience";
        if (isLong) {
          return (
            <div key={key} className="approval-field">
              <label htmlFor={`ap-${key}`}>{lbl}</label>
              <textarea id={`ap-${key}`} data-field={key} rows={2} defaultValue={val ?? ""} />
            </div>
          );
        }
        return (
          <div key={key} className="approval-field">
            <label htmlFor={`ap-${key}`}>{lbl}</label>
            <input id={`ap-${key}`} data-field={key} defaultValue={val ?? ""} />
          </div>
        );
      })}
    </>
  );
}
