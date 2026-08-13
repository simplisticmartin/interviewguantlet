import type { ReactNode } from "react";

export function Card({
  title,
  action,
  children,
  className = "",
}: {
  title?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      {(title || action) && (
        <div className="card-title">
          {typeof title === "string" ? <h3>{title}</h3> : title}
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({
  label,
  value,
  sub,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
}) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

/** Colour follows the score, so a wall of bars is readable at a glance. */
export function scoreColor(value: number): string {
  if (value >= 0.75) return "var(--good)";
  if (value >= 0.5) return "var(--warn)";
  return "var(--bad)";
}

export function Bar({
  label,
  value,
  max = 100,
  hint,
}: {
  label: string;
  value: number;
  max?: number;
  hint?: string;
}) {
  const ratio = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
  return (
    <div className="bar-row" title={hint}>
      <span className="muted" style={{ overflow: "hidden", textOverflow: "ellipsis" }}>
        {label}
      </span>
      <span className="bar-track">
        <span
          className="bar-fill"
          style={{ width: `${ratio * 100}%`, background: scoreColor(ratio) }}
        />
      </span>
      <span className="bar-value">{Math.round(value)}</span>
    </div>
  );
}

export function Badge({
  children,
  tone = "default",
}: {
  children: ReactNode;
  tone?: "default" | "good" | "warn" | "bad" | "info";
}) {
  const cls = tone === "default" ? "badge" : `badge badge-${tone}`;
  return <span className={cls}>{children}</span>;
}

export function recommendationTone(recommendation?: string | null) {
  switch (recommendation) {
    case "STRONG_HIRE":
    case "HIRE":
      return "good" as const;
    case "LEAN_HIRE":
      return "warn" as const;
    case "LEAN_NO_HIRE":
    case "NO_HIRE":
      return "bad" as const;
    default:
      return "default" as const;
  }
}

export function Empty({ title, hint, action }: { title: string; hint?: string; action?: ReactNode }) {
  return (
    <div className="empty">
      <p style={{ fontWeight: 600, color: "var(--text)" }}>{title}</p>
      {hint && <p className="small">{hint}</p>}
      {action}
    </div>
  );
}

export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div className="thinking" style={{ padding: "20px 0" }}>
      <span className="spinner" />
      {label}
    </div>
  );
}

export function ErrorNote({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : "Something went wrong.";
  return <div className="callout callout-bad">{message}</div>;
}

export function PageHeader({
  title,
  subtitle,
  action,
}: {
  title: string;
  subtitle?: string;
  action?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        <h1>{title}</h1>
        {subtitle && <p className="page-subtitle">{subtitle}</p>}
      </div>
      {action}
    </header>
  );
}

export function formatDuration(seconds: number): string {
  const safe = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(safe / 60);
  const rest = safe % 60;
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}

export function formatDate(value: string | null): string {
  if (!value) return "—";
  return new Date(value).toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}
