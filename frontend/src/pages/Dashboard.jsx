import { useEffect, useState } from "react";
import { api } from "../api";
import { Err, COLORS } from "./shared.jsx";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid, LineChart, Line } from "recharts";

export default function Dashboard() {
  const [a, setA] = useState(null); const [err, setErr] = useState("");
  useEffect(() => { api.analytics().then(setA).catch((e) => setErr(e.message)); }, []);
  if (err) return <Err msg={err} />;
  if (!a) return <p>Loading…</p>;
  const bars = ["positive", "neutral", "negative"].map((l) => ({ name: l, count: a.counts[l] }));
  return (
    <section>
      <div className="kpis">
        <div className="card kpi"><b>{a.total}</b><span>Total analyses</span></div>
        {["positive", "neutral", "negative"].map((l) => <div key={l} className="card kpi" style={{ borderTop: `4px solid ${COLORS[l]}` }}><b>{a.percentages[l]}%</b><span>{l}</span></div>)}
        <div className="card kpi"><b>{(a.avg_confidence * 100).toFixed(1)}%</b><span>Avg confidence</span></div>
      </div>
      <div className="card"><h3>Sentiment distribution</h3>
        <ResponsiveContainer width="100%" height={240}><BarChart data={bars}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" /><YAxis allowDecimals={false} /><Tooltip />
          <Bar dataKey="count" fill="#4f46e5" /></BarChart></ResponsiveContainer></div>
      <div className="card"><h3>Sentiment trend (last 30 days)</h3>
        {a.trend.length === 0 ? <p className="muted">No data yet — analyse some text first.</p> :
          <ResponsiveContainer width="100%" height={280}><LineChart data={a.trend}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="date" /><YAxis allowDecimals={false} /><Tooltip /><Legend />
            {["positive", "neutral", "negative"].map((l) => <Line key={l} dataKey={l} stroke={COLORS[l]} strokeWidth={2} />)}</LineChart></ResponsiveContainer>}</div>
    </section>
  );
}
