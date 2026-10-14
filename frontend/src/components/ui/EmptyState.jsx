export function EmptyState({ icon, title, subtitle, style }) {
  return (
    <div className="empty" style={style}>
      <div className="empty-ico">{icon}</div>
      <div className="empty-ttl">{title}</div>
      <div className="empty-sub">{subtitle}</div>
    </div>
  );
}
