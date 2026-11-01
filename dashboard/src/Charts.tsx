import type { ReactNode } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { DOMAIN_COLORS, DOMAINS, type RequestMetrics } from "./api";

interface DomainChartProps {
  title: string;
  requests: RequestMetrics[];
  value: (r: RequestMetrics) => number | null;
  format: (v: number) => string;
  yDomain?: [number, number];
  step?: boolean;
  children?: ReactNode;
}

// One row per request with a value only in its domain's column; lines connect per domain.
function toRows(requests: RequestMetrics[], value: DomainChartProps["value"]) {
  return requests.map((r, i) => ({
    n: i + 1,
    time: new Date(r.timestamp * 1000).toLocaleTimeString(),
    [r.domain]: value(r),
  }));
}

function DomainChart({ title, requests, value, format, yDomain, step, children }: DomainChartProps) {
  const rows = toRows(requests, value);
  return (
    <div className="card chart">
      <h2>{title}</h2>
      {rows.length === 0 ? (
        <p className="muted">No requests yet. Send a prompt below.</p>
      ) : (
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="n" tick={{ fontSize: 12 }} />
            <YAxis domain={yDomain ?? ["auto", "auto"]} tickFormatter={format} tick={{ fontSize: 12 }} width={48} />
            <Tooltip
              labelFormatter={(_, payload) => payload?.[0]?.payload?.time ?? ""}
              formatter={(v) => (typeof v === "number" ? format(v) : String(v))}
            />
            <Legend />
            {children}
            {DOMAINS.map((d) => (
              <Line
                key={d}
                dataKey={d}
                name={d}
                type={step ? "stepAfter" : "monotone"}
                stroke={DOMAIN_COLORS[d]}
                strokeWidth={2}
                dot={{ r: 3 }}
                connectNulls
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

const times = (v: number) => `${v.toFixed(2)}×`;
const percent = (v: number) => `${Math.round(v * 100)}%`;

export function SpeedupChart({ requests }: { requests: RequestMetrics[] }) {
  return (
    <DomainChart
      title="Token speedup vs target-only baseline"
      requests={requests}
      value={(r) => r.speedup}
      format={times}
      yDomain={[0, 3]}
    >
      <ReferenceArea y1={1.5} y2={2.8} fill="#22c55e" fillOpacity={0.08} label={{ value: "1.5×–2.8× goal", position: "insideTopRight", fontSize: 11 }} />
      <ReferenceLine y={1} stroke="#9ca3af" strokeDasharray="4 4" />
    </DomainChart>
  );
}

export function AcceptanceChart({ requests }: { requests: RequestMetrics[] }) {
  return (
    <DomainChart
      title="Draft acceptance ratio (α)"
      requests={requests}
      value={(r) => r.acceptance_rate}
      format={percent}
      yDomain={[0, 1]}
    >
      <ReferenceLine y={0.6} stroke="#16a34a" strokeDasharray="4 4" label={{ value: "K+1 above", position: "insideTopLeft", fontSize: 11 }} />
      <ReferenceLine y={0.35} stroke="#dc2626" strokeDasharray="4 4" label={{ value: "K−1 below", position: "insideBottomLeft", fontSize: 11 }} />
    </DomainChart>
  );
}

export function DepthChart({ requests }: { requests: RequestMetrics[] }) {
  return (
    <DomainChart title="Draft depth K used" requests={requests} value={(r) => r.k} format={(v) => String(v)} yDomain={[0, 8]} step />
  );
}
