import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { Card, Empty, ErrorNote, Loading, PageHeader } from "../components/ui";
import { api } from "../lib/api";

export default function StudyPlanPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["study-plan"],
    queryFn: api.studyPlan,
  });

  if (isLoading) return <Loading label="Loading your plan" />;
  if (error) return <ErrorNote error={error} />;

  if (!data || data.items.length === 0) {
    return (
      <>
        <PageHeader title="Study plan" />
        <Empty
          title="No plan yet"
          hint="A plan is generated from what an interview actually measured, so it targets your real gaps rather than a generic syllabus."
          action={
            <Link className="btn btn-primary" to="/practice" style={{ marginTop: 12 }}>
              Take an interview
            </Link>
          }
        />
      </>
    );
  }

  return (
    <>
      <PageHeader
        title="Study plan"
        subtitle={data.summary}
        action={
          <Link className="btn" to="/practice">
            Practice again
          </Link>
        }
      />

      {data.items.map((item) => (
        <Card
          key={`${item.priority}-${item.concept_key}`}
          title={`${item.priority}. ${item.title}`}
          action={<span className="small faint mono">{item.concept_key}</span>}
        >
          <p className="small muted">{item.rationale}</p>

          {item.learn_items.length > 0 && (
            <>
              <h4 className="stat-label" style={{ marginTop: 12 }}>
                Learn
              </h4>
              <ul className="small" style={{ paddingLeft: 18, margin: "6px 0 0" }}>
                {item.learn_items.map((learn, index) => (
                  <li key={index}>{learn}</li>
                ))}
              </ul>
            </>
          )}

          {item.practice_items.length > 0 && (
            <>
              <h4 className="stat-label" style={{ marginTop: 12 }}>
                Then answer these
              </h4>
              <ul className="small" style={{ paddingLeft: 18, margin: "6px 0 0" }}>
                {item.practice_items.map((practice, index) => (
                  <li key={index}>{practice.prompt}</li>
                ))}
              </ul>
              <p className="small faint" style={{ marginTop: 8, marginBottom: 0 }}>
                These are reworded versions of the same underlying knowledge — you can't
                pass by memorising one phrasing.
              </p>
            </>
          )}
        </Card>
      ))}
    </>
  );
}
