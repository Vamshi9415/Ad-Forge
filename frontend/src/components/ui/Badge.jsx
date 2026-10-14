export function Badge({ children, className = "" }) {
  return <span className={`tbadge ${className}`.trim()}>{children}</span>;
}
