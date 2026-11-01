import { useState, type FormEvent } from "react";

import { DOMAIN_COLORS, sendChat, shortModel, type ChatResponse } from "./api";

interface Exchange {
  id: number;
  prompt: string;
  response?: ChatResponse;
  error?: string;
}

function AnswerMeta({ response }: { response: ChatResponse }) {
  const s = response.speculative;
  return (
    <div className="meta">
      <span className="badge draft" title={s.draft_model ?? undefined}>
        Draft model: {s.draft_model ? shortModel(s.draft_model) : "none (target only)"}
      </span>
      <span className="badge" style={{ borderColor: DOMAIN_COLORS[s.domain], color: DOMAIN_COLORS[s.domain] }}>
        {s.domain}
      </span>
      <span>K={s.k}</span>
      {s.drafted_tokens > 0 && (
        <span>
          accepted {s.accepted_tokens}/{s.drafted_tokens} ({Math.round(s.acceptance_rate * 100)}%)
        </span>
      )}
      <span>{s.tokens_per_second.toFixed(1)} tok/s</span>
      {s.speedup !== null && <span>{s.speedup.toFixed(2)}× speedup</span>}
    </div>
  );
}

export function Chat({ onAnswer }: { onAnswer: () => void }) {
  const [prompt, setPrompt] = useState("");
  const [temperature, setTemperature] = useState(0);
  const [maxTokens, setMaxTokens] = useState(256);
  const [json, setJson] = useState(false);
  const [busy, setBusy] = useState(false);
  const [history, setHistory] = useState<Exchange[]>([]);

  async function submit(e: FormEvent) {
    e.preventDefault();
    const text = prompt.trim();
    if (!text || busy) return;
    const id = Date.now();
    setHistory((h) => [{ id, prompt: text }, ...h]);
    setPrompt("");
    setBusy(true);
    try {
      const response = await sendChat(text, { temperature, maxTokens, json });
      setHistory((h) => h.map((x) => (x.id === id ? { ...x, response } : x)));
      onAnswer();
    } catch (err) {
      setHistory((h) => h.map((x) => (x.id === id ? { ...x, error: String(err) } : x)));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card chat">
      <h2>Chat</h2>
      <form onSubmit={submit}>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Ask something: code, JSON, or prose. The router picks the draft model."
          rows={3}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit(e);
          }}
        />
        <div className="controls">
          <label>
            Temperature
            <input type="number" min={0} max={2} step={0.1} value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} />
          </label>
          <label>
            Max tokens
            <input type="number" min={1} max={4096} value={maxTokens} onChange={(e) => setMaxTokens(Number(e.target.value))} />
          </label>
          <label className="check">
            <input type="checkbox" checked={json} onChange={(e) => setJson(e.target.checked)} />
            JSON mode
          </label>
          <button type="submit" disabled={busy || !prompt.trim()}>
            {busy ? "Generating…" : "Send (⌘↵)"}
          </button>
        </div>
      </form>

      <ul className="history">
        {history.map((x) => (
          <li key={x.id}>
            <p className="prompt">{x.prompt}</p>
            {x.response && (
              <>
                <AnswerMeta response={x.response} />
                <pre className="answer">{x.response.choices[0].message.content}</pre>
              </>
            )}
            {x.error && <p className="error">{x.error}</p>}
            {!x.response && !x.error && <p className="muted">Generating…</p>}
          </li>
        ))}
      </ul>
    </div>
  );
}
