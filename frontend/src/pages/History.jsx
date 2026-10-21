import { useEffect, useState } from "react";
import { api } from "../api";
import { Pill, Err, pct } from "./shared.jsx";

export default function History() {
  const [page, setPage] = useState(1); const [label, setLabel] = useState("");
  const [data, setData] = useState({ items: [], total: 0 }); const [err, setErr] = useState("");
  useEffect(() => { api.history(page, label).then(setData).catch((e) => setErr(e.message)); }, [page, label]);
  const pages = Math.max(1, Math.ceil(data.total / 15));
  return (
    <section className="card">
      <h2>Analysis history <small>({data.total})</small></h2>
      <select value={label} onChange={(e) => { setLabel(e.target.value); setPage(1); }}>
        <option value="">All sentiments</option><option>positive</option><option>neutral</option><option>negative</option>
      </select>
      <Err msg={err} />
      <table><thead><tr><th>Text</th><th>Sentiment</th><th>Confidence</th><th>Source</th><th>Time</th></tr></thead>
        <tbody>{data.items.map((r) => (
          <tr key={r.id}><td className="txt">{r.text}</td><td><Pill label={r.label} /></td><td>{pct(r.confidence)}</td>
            <td>{r.source}</td><td>{new Date(r.created_at).toLocaleString()}</td></tr>))}</tbody></table>
      <div className="row"><button disabled={page <= 1} onClick={() => setPage(page - 1)}>← Prev</button>
        <span>Page {page} / {pages}</span><button disabled={page >= pages} onClick={() => setPage(page + 1)}>Next →</button></div>
    </section>
  );
}
