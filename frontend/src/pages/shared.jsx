export const COLORS = { positive: "#16a34a", neutral: "#f59e0b", negative: "#dc2626" };
export const Pill = ({ label }) => <span className="pill" style={{ background: COLORS[label] }}>{label}</span>;
export const Err = ({ msg }) => (msg ? <div className="error">⚠ {msg}</div> : null);
export const pct = (x) => `${(x * 100).toFixed(1)}%`;
