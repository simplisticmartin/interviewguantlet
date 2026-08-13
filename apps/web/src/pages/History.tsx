import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import {
  Badge,
  Card,
  Empty,
  ErrorNote,
  Loading,
  PageHeader,
  formatDate,
  recommendationTone,
} from "../components/ui";
import { api } from "../lib/api";

export default function History() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["interviews"],
    queryFn: api.listInterviews,
  });

  if (isLoading) return <Loading />;
  if (error) return <ErrorNote error={error} />;

  return (
    <>
      <PageHeader
        title="History"
        subtitle="Every interview, including the ones you abandoned. Unfinished sessions can be resumed exactly where you stopped."
        action={
          <Link className="btn btn-primary" to="/practice">
            New interview
          </Link>
        }
      />

      {!data || data.length === 0 ? (
        <Empty title="No interviews yet" hint="Your first one takes about 20 minutes." />
      ) : (
        <Card>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Role</th>
                  <th>Company</th>
                  <th>Mode</th>
                  <th>Date</th>
                  <th>Qs</th>
                  <th>Score</th>
                  <th>Outcome</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {data.map((row) => (
                  <tr key={row.id}>
                    <td>
                      {row.target_role}
                      <div className="small faint">{row.target_level}</div>
                    </td>
                    <td className="muted">{row.company ?? "—"}</td>
                    <td className="muted">{row.mode.replace(/_/g, " ")}</td>
                    <td className="muted">{formatDate(row.ended_at ?? row.started_at)}</td>
                    <td className="muted">{row.questions_asked}</td>
                    <td>{row.overall ?? "—"}</td>
                    <td>
                      {row.recommendation ? (
                        <Badge tone={recommendationTone(row.recommendation)}>
                          {row.recommendation.replace(/_/g, " ")}
                        </Badge>
                      ) : (
                        <Badge>{row.status.replace(/_/g, " ")}</Badge>
                      )}
                    </td>
                    <td>
                      <Link
                        className="small"
                        to={row.status === "completed" ? `/report/${row.id}` : `/interview/${row.id}`}
                      >
                        {row.status === "completed" ? "Scorecard" : "Resume"}
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </>
  );
}
