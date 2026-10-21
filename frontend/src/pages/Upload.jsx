import { useState } from "react";
import { api } from "../api";
import { Err, COLORS } from "./shared.jsx";
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from "recharts";

export default function Upload() {
  const [file, setFile] = useState(null); const [col, setCol] = useState("");
  const [job, setJob] = useState(null); const [err, setErr] = useState(""); const [busy, setBusy] = useState(false);
  const run = async () => {
    setBusy(true); setErr("");
    try { setJob(await api.batch(file, col.trim())); } catch (e) { setErr(e.message); setJob(null); }
    setBusy(false);
  };
  const data = job ? Object.entries(job.counts).map(([name, value]) => ({ name, value })) : [];
  return (
    <section className="card">
      <h2>Batch analysis (CSV)</h2>
      <p className="muted">Upload a CSV with a review/feedback column (auto-detected). Max 5 MB.</p>
      <div className="row">
        <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files[0])} />
        <input placeholder="Text column name (optional)" value={col} onChange={(e) => setCol(e.target.value)} />
        <button className="primary" disabled={!file || busy} onClick={run}>{busy ? "Processing…" : "Run analysis"}</button>
      </div>
      <Err msg={err} />
      {job && (
        <div className="result">
          <p><b>{job.total_rows}</b> rows analysed from column “{job.text_column}” · avg confidence <b>{(job.avg_confidence * 100).toFixed(1)}%</b> · via {job.source}
            {job.truncated && " · (truncated to row limit)"}</p>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart><Pie data={data} dataKey="value" nameKey="name" outerRadius={90} label>
              {data.map((d) => <Cell key={d.name} fill={COLORS[d.name]} />)}</Pie><Tooltip /><Legend /></PieChart>
          </ResponsiveContainer>
          <button className="primary" onClick={() => api.download(job.id).catch((e) => setErr(e.message))}>⬇ Download processed CSV</button>
        </div>
      )}
    </section>
  );
}
