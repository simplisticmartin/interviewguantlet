import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";

import {
  Badge,
  Bar,
  Card,
  Empty,
  ErrorNote,
  Loading,
  PageHeader,
  Stat,
  recommendationTone,
} from "../components/ui";
import { api } from "../lib/api";

const SUPPORT_LABEL: Record<string, string> = {
  well_supported: "Well supported",
  partially_supported: "Partly supported",
  thinly_supported: "Thin",
  little_evidence: "Little evidence",
  not_tested: "Not tested",
};

export default function Report() {
  const { sessionId = "" } = useParams();
  const { data, isLoading, error } = useQuery({
    queryKey: ["interview", sessionId],
    queryFn: () => api.getInterview(sessionId),
  });

  if (isLoading) return <Loading label="Loading your scorecard" />;
  if (error) return <ErrorNote error={error} />;
  if (!data) return <Empty title="Interview not found." />;

  const card = data.scorecard;
  if (!card) {
    return (
      <Empty
        title="This interview has no scorecard yet."
        hint="Finish the interview to generate one."
        action={
          <Link className="btn btn-primary" to={`/interview/${sessionId}`}>
            Resume interview
          </Link>
        }
      />
    );
  }

  const committee = card.committee;

  return (
    <>
      <PageHeader
        title="Interview scorecard"
        subtitle={`${data.target_role} · ${data.target_level}${data.company ? ` · ${data.company} simulation` : ""}`}
        action={
          <Link className="btn" to="/practice">
            Practice again
          </Link>
        }
      />

      <div className="grid grid-4" style={{ marginBottom: 16 }}>
        <Stat label="Overall" value={`${card.overall}`} sub="out of 100" />
        <Stat
          label="Simulated outcome"
          value={
            <Badge tone={recommendationTone(committee.recommendation)}>
              {committee.recommendation.replace(/_/g, " ")}
            </Badge>
          }
          sub="Gauntlet simulation, not a real decision"
        />
        <Stat label="Questions" value={card.questions_asked} sub={`${card.duration_minutes} min`} />
        <Stat
          label="Misconceptions"
          value={card.misconceptions.length}
          sub="confidently incorrect"
        />
      </div>

      <div className="grid grid-2">
        <Card title="By area">
          {Object.keys(card.category_scores).length === 0 ? (
            <p className="muted small">No areas scored.</p>
          ) : (
            Object.entries(card.category_scores)
              .sort((a, b) => b[1] - a[1])
              .map(([label, value]) => <Bar key={label} label={label} value={value} />)
          )}
        </Card>

        <Card title="Most likely rejection reason">
          {committee.most_likely_rejection_reason ? (
            <div className="callout callout-bad">{committee.most_likely_rejection_reason}</div>
          ) : (
            <p className="muted small">Nothing stood out as disqualifying.</p>
          )}

          {committee.strengths.length > 0 && (
            <>
              <h4 className="stat-label" style={{ marginTop: 16 }}>
                Strengths
              </h4>
              <ul className="small muted" style={{ paddingLeft: 18, margin: "6px 0 0" }}>
                {committee.strengths.map((item, i) => (
                  <li key={i}>{item}</li>
                ))}
              </ul>
            </>
          )}

          {committee.risks.length > 0 && (
            <>
              <h4 className="stat-label" style={{ marginTop: 14 }}>
                Risks
              </h4>
              <ul className="small muted" style={{ paddingLeft: 18, margin: "6px 0 0" }}>
                {committee.risks.map((item, i) => (
                  <li key={i}>{item}</li>
                ))}
              </ul>
            </>
          )}
        </Card>
      </div>

      {card.misconceptions.length > 0 && (
        <Card
          title="Confidently incorrect"
          action={<span className="small faint">Highest-value things to fix</span>}
        >
          {card.misconceptions.map((item, index) => (
            <div
              key={index}
              style={{
                borderLeft: "3px solid var(--misconception)",
                paddingLeft: 14,
                marginBottom: 16,
              }}
            >
              <div className="small faint">{item.concept_key}</div>
              <p style={{ margin: "3px 0", fontWeight: 600 }}>You said: {item.belief}</p>
              <p className="small muted" style={{ margin: 0 }}>
                <strong>Actually:</strong> {item.correction}
              </p>
              {item.evidence_quote && (
                <p className="small faint mono" style={{ marginTop: 6 }}>
                  “{item.evidence_quote}”
                </p>
              )}
            </div>
          ))}
        </Card>
      )}

      <div className="grid grid-2">
        <Card title="Strongest">
          {card.strongest_areas.length === 0 ? (
            <p className="muted small">Not enough evidence.</p>
          ) : (
            card.strongest_areas.map((item) => (
              <Bar
                key={item.concept_key}
                label={item.display_name}
                value={item.mastery * 100}
                hint={`${item.evidence_count} question(s)`}
              />
            ))
          )}
        </Card>

        <Card title="Weakest">
          {card.weakest_areas.length === 0 ? (
            <p className="muted small">Not enough evidence.</p>
          ) : (
            card.weakest_areas.map((item) => (
              <Bar
                key={item.concept_key}
                label={item.display_name}
                value={item.mastery * 100}
                hint={`${item.evidence_count} question(s)`}
              />
            ))
          )}
        </Card>
      </div>

      {card.resume_claims_tested.length > 0 && (
        <Card
          title="Resume claims"
          action={
            <span className="small faint">
              Evidence depth only — not a judgement about accuracy
            </span>
          }
        >
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Claim</th>
                  <th style={{ width: 150 }}>Support</th>
                </tr>
              </thead>
              <tbody>
                {card.resume_claims_tested.map((row, index) => (
                  <tr key={index}>
                    <td>{row.claim}</td>
                    <td>
                      <Badge
                        tone={
                          row.support === "well_supported"
                            ? "good"
                            : row.support === "not_tested"
                              ? "default"
                              : row.support === "partially_supported"
                                ? "warn"
                                : "bad"
                        }
                      >
                        {SUPPORT_LABEL[row.support] ?? row.support}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {card.study_plan.items.length > 0 && (
        <Card
          title="Your study plan"
          action={
            <Link className="small" to="/study-plan">
              Open full plan
            </Link>
          }
        >
          <p className="small muted">{card.study_plan.summary}</p>
          {card.study_plan.items.map((item) => (
            <div key={item.priority} style={{ marginBottom: 16 }}>
              <h4>
                {item.priority}. {item.title}
              </h4>
              <p className="small muted" style={{ margin: "4px 0" }}>
                {item.rationale}
              </p>
              {item.learn_items.length > 0 && (
                <div className="chips" style={{ marginTop: 6 }}>
                  {item.learn_items.map((learn, i) => (
                    <span key={i} className="chip">
                      {learn}
                    </span>
                  ))}
                </div>
              )}
              {item.reattempt_prompt && (
                <p className="small faint" style={{ marginTop: 8 }}>
                  <strong>Re-attempt (reworded):</strong> {item.reattempt_prompt}
                </p>
              )}
            </div>
          ))}
        </Card>
      )}

      {(card.communication_notes.length > 0 || card.missed_opportunities.length > 0) && (
        <div className="grid grid-2">
          {card.communication_notes.length > 0 && (
            <Card title="Communication">
              <ul className="small muted" style={{ paddingLeft: 18, margin: 0 }}>
                {card.communication_notes.map((note, i) => (
                  <li key={i}>{note}</li>
                ))}
              </ul>
            </Card>
          )}
          {card.missed_opportunities.length > 0 && (
            <Card title="Missed opportunities">
              <ul className="small muted" style={{ paddingLeft: 18, margin: 0 }}>
                {card.missed_opportunities.map((note, i) => (
                  <li key={i}>{note}</li>
                ))}
              </ul>
            </Card>
          )}
        </div>
      )}

      {card.replay_moments.length > 0 && (
        <Card
          title="Moments worth another attempt"
          action={<span className="small faint">Replay lands in a later release</span>}
        >
          {card.replay_moments.map((moment) => (
            <div key={moment.ordinal} style={{ marginBottom: 12 }}>
              <div className="small faint">
                {moment.at_minute.toFixed(1)} min · Q{moment.ordinal} ·{" "}
                {Math.round(moment.score * 100)}%
              </div>
              <div className="small">{moment.prompt_text}</div>
              <div className="small muted">{moment.note}</div>
            </div>
          ))}
        </Card>
      )}

      <Card title="Transcript">
        {data.transcript.map((entry) => (
          <div key={entry.ordinal} style={{ marginBottom: 18 }}>
            <div className="small faint">
              Q{entry.ordinal}
              {entry.is_followup ? " · follow-up" : ""}
              {entry.score !== null ? ` · ${Math.round(entry.score * 100)}%` : ""}
            </div>
            <p style={{ margin: "3px 0", fontWeight: 560 }}>{entry.prompt_text}</p>
            {entry.answer_text && (
              <p className="small muted" style={{ whiteSpace: "pre-wrap", margin: 0 }}>
                {entry.answer_text}
              </p>
            )}
          </div>
        ))}
      </Card>
    </>
  );
}
