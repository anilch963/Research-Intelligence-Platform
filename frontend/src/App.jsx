import { useEffect, useState } from "react";
import { getHealth, query } from "./api.js";

export default function App() {
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);
  const [question, setQuestion] = useState("");
  const [retriever, setRetriever] = useState("hybrid");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getHealth().then(setHealth).catch((e) => setHealthError(e.message));
  }, []);

  async function onSubmit(e) {
    e.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await query(question, retriever));
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="container">
      <header>
        <h1>Research Intelligence</h1>
        <p className="status">
          {healthError ? (
            <span className="bad">Backend unreachable: {healthError}</span>
          ) : health ? (
            <span className="ok">
              Backend OK · {health.chunks_indexed} chunks · {health.llm_model}
            </span>
          ) : (
            "Checking backend…"
          )}
        </p>
      </header>

      <form onSubmit={onSubmit}>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about your documents…"
          rows={3}
        />
        <div className="row">
          <label>
            Retriever{" "}
            <select value={retriever} onChange={(e) => setRetriever(e.target.value)}>
              <option value="vector">Vector only</option>
              <option value="hybrid">Hybrid (BM25 + vector)</option>
            </select>
          </label>
          <button type="submit" disabled={loading || !question.trim()}>
            {loading ? "Thinking…" : "Ask"}
          </button>
        </div>
      </form>

      {error && <p className="bad">{error}</p>}

      {result && (
        <section>
          <h2>Answer <small>({result.retriever})</small></h2>
          <p className="answer">{result.answer}</p>
          <h3>Sources</h3>
          <ol className="sources">
            {result.sources.map((s, i) => (
              <li key={i}>
                <strong>
                  {s.source}
                  {s.page != null && ` · p.${s.page + 1}`}
                </strong>
                <p>{s.snippet}…</p>
              </li>
            ))}
          </ol>
        </section>
      )}
    </main>
  );
}
