import { useEffect, useState } from "react";
import { api } from "./api";
import Analyze from "./pages/Analyze.jsx";
import Upload from "./pages/Upload.jsx";
import History from "./pages/History.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Model from "./pages/Model.jsx";
import Pipeline from "./pages/Pipeline.jsx";

const TABS = [["analyze", "Analyze", Analyze], ["upload", "Batch CSV", Upload], ["dashboard", "Dashboard", Dashboard],
  ["history", "History", History], ["model", "Model Performance", Model], ["pipeline", "ML Pipeline", Pipeline]];

export default function App() {
  const [tab, setTab] = useState("analyze");
  const [health, setHealth] = useState(null);
  useEffect(() => { api.health().then(setHealth).catch(() => setHealth({ status: "down" })); }, []);
  const Page = TABS.find((t) => t[0] === tab)[2];
  const ok = health?.status === "ok";
  return (
    <div className="app">
      <header>
        <h1>Senti<span>lytics</span></h1>
        <p>Cloud-based sentiment analysis powered by Azure Machine Learning</p>
        <div className={`badge ${ok ? "ok" : "bad"}`}>
          {ok ? `API online · ${health.azure_endpoint_configured ? "Azure ML endpoint" : "local model"} · ${health.storage}` : "API offline"}
        </div>
      </header>
      <nav>{TABS.map(([k, label]) => <button key={k} className={tab === k ? "active" : ""} onClick={() => setTab(k)}>{label}</button>)}</nav>
      <main><Page /></main>
      <footer>Sentilytics · B.Tech Computer Engineering Final Year Project</footer>
    </div>
  );
}
