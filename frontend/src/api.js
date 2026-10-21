const BASE = (import.meta.env.VITE_API_URL || "") + "/api";
const KEY = import.meta.env.VITE_API_KEY || "";
const H = KEY ? { "X-API-Key": KEY } : {};

async function req(path, opts = {}) {
  const r = await fetch(BASE + path, { ...opts, headers: { ...H, ...(opts.headers || {}) } });
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {}
    throw new Error(msg);
  }
  return r;
}
const json = async (p, o) => (await req(p, o)).json();

export const api = {
  health: () => json("/health"),
  predict: (text) => json("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) }),
  batch: (file, textColumn) => {
    const fd = new FormData(); fd.append("file", file);
    if (textColumn) fd.append("text_column", textColumn);
    return json("/batch", { method: "POST", body: fd });
  },
  history: (page, label) => json(`/history?page=${page}&page_size=15${label ? `&label=${label}` : ""}`),
  analytics: () => json("/analytics"),
  metrics: () => json("/metrics"),
  jobs: () => json("/jobs"),
  // authenticated download -> blob (works even when an API key header is required)
  download: async (id) => {
    const blob = await (await req(`/jobs/${id}/download`)).blob();
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
    a.download = `sentilytics_${id.slice(0, 8)}.csv`; a.click(); URL.revokeObjectURL(a.href);
  },
};
