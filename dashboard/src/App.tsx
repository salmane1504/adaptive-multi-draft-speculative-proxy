import { useCallback, useEffect, useState } from "react";

import { DOMAIN_COLORS, fetchMetrics, shortModel, type DomainStatus, type Metrics, type RequestMetrics } from "./api";
import { AcceptanceChart, DepthChart, SpeedupChart } from "./Charts";
import { Chat } from "./Chat";

const POLL_MS = 2000;

function DomainCard({ status }: { status: DomainStatus }) {
  return (
    <div className="card domain" style={{ borderTopColor: DOMAIN_COLORS[status.domain] }}>
      <div className="domain-head">
        <h3>{status.domain}</h3>
        <span className={status.speculation_enabled ? "status on" : "status off"}>
          {status.speculation_enabled ? "speculating" : "disabled"}
        </span>
      </div>
      <p className="model" title={status.draft_model}>
        {shortModel(status.draft_model)}
      </p>
      <div className="stats">
        <div>
          <strong>{status.k}</strong>
          <span>K</span>
        </div>
        <div>
          <strong>{status.acceptance_rate === null ? "—" : `${Math.round(status.acceptance_rate * 100)}%`}</strong>
          <span>α (window)</span>
        </div>
        <div>
          <strong>{status.samples}</strong>
          <span>samples</span>
        </div>
      </div>
    </div>
  );
}

function fmt(v: number | null, digits = 1, suffix = "") {
  return v === null ? "—" : `${v.toFixed(digits)}${suffix}`;
}

function RecentRequests({ requests }: { requests: RequestMetrics[] }) {
  const rows = [...requests].reverse().slice(0, 12);
  return (
    <div className="card table">
      <h2>Recent requests</h2>
      <table>
        <thead>
          <tr>
            <th>Time</th>
            <th>Domain</th>
            <th>Draft model</th>
            <th>K</th>
            <th>α</th>
            <th>Spec tok/s</th>
            <th>Baseline tok/s</th>
            <th>Speedup</th>
            <th>Dispatch ms</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.timestamp}>
              <td>{new Date(r.timestamp * 1000).toLocaleTimeString()}</td>
              <td style={{ color: DOMAIN_COLORS[r.domain] }}>{r.domain}</td>
              <td title={r.draft_model ?? undefined}>{shortModel(r.draft_model)}</td>
              <td>{r.k}</td>
              <td>{r.acceptance_rate === null ? "—" : `${Math.round(r.acceptance_rate * 100)}%`}</td>
              <td>{fmt(r.tokens_per_second)}</td>
              <td>{fmt(r.baseline_tokens_per_second)}</td>
              <td>{fmt(r.speedup, 2, "×")}</td>
              <td>{fmt(r.dispatch_latency_ms)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length === 0 && <p className="muted">No requests yet.</p>}
    </div>
  );
}

export default function App() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setMetrics(await fetchMetrics());
      setError(null);
    } catch (err) {
      setError(String(err));
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, POLL_MS);
    return () => clearInterval(id);
  }, [refresh]);

  const requests = metrics?.requests ?? [];

  return (
    <main>
      <header>
        <h1>Adaptive Multi-Draft Speculative Proxy</h1>
        <div className="header-info">
          <span>Target: {metrics ? shortModel(metrics.target_model) : "…"}</span>
          <span>Baseline: {fmt(metrics?.baseline_tokens_per_second ?? null, 1, " tok/s")}</span>
          {error ? <span className="error">API unreachable</span> : <span className="live">● live</span>}
        </div>
      </header>

      <section className="domains">
        {metrics?.domains.map((d) => <DomainCard key={d.domain} status={d} />)}
      </section>

      <section className="charts">
        <SpeedupChart requests={requests} />
        <AcceptanceChart requests={requests} />
        <DepthChart requests={requests} />
      </section>

      <section className="bottom">
        <Chat onAnswer={refresh} />
        <RecentRequests requests={requests} />
      </section>
    </main>
  );
}
