import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import {
  Badge,
  Bar,
  Card,
  Empty,
  ErrorNote,
  Loading,
  PageHeader,
  Stat,
  formatDate,
  recommendationTone,
} from "../components/ui";
import { api, getIdentity } from "../lib/api";

export default function Dashboard() {
  const identity = getIdentity();
  const analytics = useQuery({ queryKey: ["analytics"], queryFn: api.analytics });
  const interviews = useQuery({ queryKey: ["interviews"], queryFn: api.listInterviews });

  if (analytics.isLoading) return <Loading label="Loading your readiness" />;
  if (analytics.error) return <ErrorNote error={analytics.error} />;

  const data = analytics.data;
  const recent = interviews.data?.slice(0, 5) ?? [];
  const hasHistory = (data?.interviews_completed ?? 0) > 0;

  return (
    <>
      <PageHeader
        title={`Welcome back, ${identity?.displayName ?? "there"}`}
        subtitle="Interview readiness is measured from what you actually demonstrated, not from topics you have read about."
        action={
          <Link className="btn btn-primary" to="/practice">
            Start an interview
          </Link>
        }
      />

      {!hasHistory ? (
        <Empty
          title="No interviews yet"
          hint="Upload a resume and paste a job description — Gauntlet works out what a real loop would probe, then adapts as you answer."
          action={
            <Link className="btn btn-primary" to="/practice" style={{ marginTop: 12 }}>
              Start your first interview
            </Link>
          }
        />
      ) : (
        <>
          <div className="grid grid-4" style={{ marginBottom: 16 }}>
            <Stat label="Interviews" value={data!.interviews_completed} sub="completed" />
            <Stat
              label="Average score"
              value={data!.average_overall ?? "—"}
              sub="out of 100"
            />
            <Stat
              label="Open misconceptions"
              value={data!.open_misconceptions.length}
              sub="confidently incorrect"
            />
            <Stat
              label="Due for review"
              value={data!.due_for_review.length}
              sub="spaced repetition"
            />
          </div>

          <div className="grid grid-2">
            <Card title="Interview readiness">
              {data!.readiness.length === 0 ? (
                <p className="muted small">Nothing measured yet.</p>
              ) : (
                data!.readiness.map((item) => (
                  <Bar
                    key={item.concept_key}
                    label={item.display_name}
                    value={item.mastery * 100}
                    hint={`${item.evidence_count} question(s) of evidence`}
                  />
                ))
              )}
            </Card>

            <Card title="Progress over time">
              {data!.improvement.length < 2 ? (
                <p className="muted small">
                  Complete another interview to see whether you are actually improving.
                </p>
              ) : (
                <Sparkline points={data!.improvement} />
              )}
            </Card>
          </div>

          {data!.open_misconceptions.length > 0 && (
            <Card
              title="Fix these first"
              action={<span className="small faint">Wrong and sure of it</span>}
            >
              {data!.open_misconceptions.slice(0, 4).map((item, index) => (
                <div key={index} style={{ marginBottom: 12 }}>
                  <div className="small faint">
                    {item.display_name}
                    {item.times_observed > 1 ? ` · seen ${item.times_observed}x` : ""}
                  </div>
                  <div style={{ fontWeight: 560 }}>{item.belief}</div>
                  <div className="small muted">{item.correction}</div>
                </div>
              ))}
            </Card>
          )}

          <Card
            title="Recent interviews"
            action={
              <Link className="small" to="/history">
                See all
              </Link>
            }
          >
            {recent.length === 0 ? (
              <p className="muted small">Nothing yet.</p>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Role</th>
                      <th>Date</th>
                      <th>Score</th>
                      <th>Outcome</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {recent.map((row) => (
                      <tr key={row.id}>
                        <td>{row.target_role}</td>
                        <td className="muted">{formatDate(row.ended_at ?? row.started_at)}</td>
                        <td>{row.overall ?? "—"}</td>
                        <td>
                          {row.recommendation ? (
                            <Badge tone={recommendationTone(row.recommendation)}>
                              {row.recommendation.replace(/_/g, " ")}
                            </Badge>
                          ) : (
                            <span className="faint small">In progress</span>
                          )}
                        </td>
                        <td>
                          <Link
                            className="small"
                            to={
                              row.status === "completed"
                                ? `/report/${row.id}`
                                : `/interview/${row.id}`
                            }
                          >
                            {row.status === "completed" ? "Scorecard" : "Resume"}
                          </Link>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </>
      )}
    </>
  );
}

function Sparkline({ points }: { points: Array<{ label: string; overall: number }> }) {
  const width = 520;
  const height = 130;
  const padding = 22;
  const max = 100;

  const step = points.length > 1 ? (width - padding * 2) / (points.length - 1) : 0;
  const coords = points.map((point, index) => ({
    x: padding + index * step,
    y: height - padding - (point.overall / max) * (height - padding * 2),
    ...point,
  }));
  const path = coords.map((c, i) => `${i === 0 ? "M" : "L"} ${c.x} ${c.y}`).join(" ");

  return (
    <svg viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", height: "auto" }}>
      <line
        x1={padding}
        y1={height - padding}
        x2={width - padding}
        y2={height - padding}
        stroke="var(--border)"
      />
      <path d={path} fill="none" stroke="var(--accent)" strokeWidth={2} />
      {coords.map((c, i) => (
        <g key={i}>
          <circle cx={c.x} cy={c.y} r={3.5} fill="var(--accent)" />
          <text x={c.x} y={height - 6} fontSize={9} fill="var(--text-faint)" textAnchor="middle">
            {c.label}
          </text>
          <text x={c.x} y={c.y - 8} fontSize={10} fill="var(--text-muted)" textAnchor="middle">
            {c.overall}
          </text>
        </g>
      ))}
    </svg>
  );
}
