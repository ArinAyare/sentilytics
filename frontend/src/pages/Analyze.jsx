import { useState } from "react";
import { api } from "../api";
import { Pill, Err, pct, COLORS } from "./shared.jsx";

const SAMPLES = ["The delivery was fast and the support team was incredibly helpful!", "It's okay, nothing special.", "Terrible experience. The app keeps crashing and nobody replies."];

export default function Analyze() {
  const [text, setText] = useState(""); const [res, setRes] = useState(null);
  const [err, setErr] = useState(""); const [busy, setBusy] = useState(false);
  const run = async () => {
    setBusy(true); setErr("");
    try { setRes(await api.predict(text)); } catch (e) { setErr(e.message); setRes(null); }
    setBusy(false);
  };
  return (
    <section className="card">
      <h2>Analyze a single review</h2>
      <textarea rows={5} maxLength={5000} value={text} onChange={(e) => setText(e.target.value)} placeholder="Type or paste customer feedback…" />
      <div className="row">
        <button className="primary" disabled={busy || !text.trim()} onClick={run}>{busy ? "Analyzing…" : "Analyze sentiment"}</button>
        {SAMPLES.map((s, i) => <button key={i} className="ghost" onClick={() => setText(s)}>Sample {i + 1}</button>)}
      </div>
      <Err msg={err} />
      {res && (
        <div className="result">
          <div><Pill label={res.label} /> <b>{pct(res.confidence)}</b> confidence <small>· served by {res.source}</small></div>
          {["positive", "neutral", "negative"].map((l) => (
            <div key={l} className="bar"><span>{l}</span>
              <div><i style={{ width: pct(res.probabilities[l]), background: COLORS[l] }} /></div><em>{pct(res.probabilities[l])}</em></div>
          ))}
        </div>
      )}
    </section>
  );
}
