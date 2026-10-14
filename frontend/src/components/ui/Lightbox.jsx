export function Lightbox({ src, onClose }) {
  if (!src) return null;
  return (
    <div className="lightbox" onClick={onClose} role="presentation">
      <img src={src} alt="" />
    </div>
  );
}
