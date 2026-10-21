const STEPS = [
  ["1. Data collection", "Labelled reviews/tweets (positive · neutral · negative)."],
  ["2. Preprocessing", "Lower-casing, URL/mention/HTML removal, repeated-char normalisation. Negations & punctuation kept."],
  ["3. Split", "Stratified 70 / 15 / 15 train / validation / test."],
  ["4. Feature extraction", "TF-IDF with uni+bi-grams (baseline) or sub-word tokenisation (DistilBERT)."],
  ["5. Training on Azure ML", "Command job on an auto-scaling compute cluster; metrics tracked with MLflow."],
  ["6. Evaluation", "Accuracy, macro-F1, per-class precision/recall, confusion matrix on the unseen test set."],
  ["7. Model registry", "Best model registered as a versioned asset in the Azure ML workspace."],
  ["8. Managed online endpoint", "Model served as an authenticated REST API (score.py)."],
  ["9. Backend integration", "FastAPI calls the endpoint; transparent fallback to a local copy if Azure is unreachable."],
  ["10. Analytics", "Predictions stored in MongoDB → dashboard, trends, CSV export."],
];
export default function Pipeline() {
  return (
    <section className="card"><h2>End-to-end ML pipeline</h2>
      <ol className="steps">{STEPS.map(([t, d]) => <li key={t}><b>{t}</b><span>{d}</span></li>)}</ol></section>
  );
}
