import { useEffect, useState } from "react";
import { api } from "../api";
import { Err } from "./shared.jsx";

export default function Model() {
  const [ms, setMs] = useState(null); const [err, setErr] = useState("");
  useEffect(() => { api.metrics().then(setMs).catch((e) => setErr(e.message)); }, []);
  if (err) return <Err msg={err} />;
  if (!ms) return <p>Loading…</p>;
  if (!ms.length) return <section className="card"><p>No trained models found. Run the training pipeline first (see README).</p></section>;
  return (
    <section>
      <div className="card"><h2>Model comparison</h2>
        <table><thead><tr><th>Model</th><th>Accuracy</th><th>Macro-F1</th><th>Weighted-F1</th><th>Train / Test</th></tr></thead>
          <tbody>{ms.map((m) => <tr key={m.model_name}><td>{m.model_name}</td><td>{(m.accuracy * 100).toFixed(2)}%</td>
            <td>{m.macro_f1}</td><td>{m.weighted_f1}</td><td>{m.train_size} / {m.test_size}</td></tr>)}</tbody></table></div>
      {ms.map((m) => (
        <div className="card" key={m.model_name + "cm"}><h3>{m.model_name}</h3>
          <div className="grid2">
            <table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
              <tbody>{Object.entries(m.per_class).map(([c, v]) => <tr key={c}><td>{c}</td><td>{v.precision}</td><td>{v.recall}</td><td>{v["f1-score"]}</td><td>{v.support}</td></tr>)}</tbody></table>
            <table className="cm"><thead><tr><th>Actual ↓ / Pred →</th>{m.confusion_matrix.labels.map((l) => <th key={l}>{l}</th>)}</tr></thead>
              <tbody>{m.confusion_matrix.matrix.map((row, i) => <tr key={i}><th>{m.confusion_matrix.labels[i]}</th>
                {row.map((v, j) => <td key={j} style={{ background: i === j ? "#c7d2fe" : "#fff" }}>{v}</td>)}</tr>)}</tbody></table>
          </div></div>))}
    </section>
  );
}
