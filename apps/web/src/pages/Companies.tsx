import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { Bar, Card, ErrorNote, Loading, PageHeader, Stat } from "../components/ui";
import { api } from "../lib/api";

export default function Companies() {
  const [selected, setSelected] = useState<string>("");

  const companies = useQuery({ queryKey: ["companies"], queryFn: api.companies });
  const patterns = useQuery({
    queryKey: ["company-patterns", selected],
    queryFn: () => api.companyPatterns(selected),
    enabled: Boolean(selected),
  });

  if (companies.isLoading) return <Loading />;
  if (companies.error) return <ErrorNote error={companies.error} />;

  const bySector = new Map<string, typeof companies.data>();
  for (const company of companies.data ?? []) {
    const bucket = bySector.get(company.sector) ?? [];
    bucket.push(company);
    bySector.set(company.sector, bucket);
  }

  return (
    <>
      <PageHeader
        title="Company simulation"
        subtitle="Configure an interview in the shape of a kind of engineering organisation, and see how your measured skills compare."
      />

      <Card title="Pick a company">
        {[...bySector.entries()].map(([sector, list]) => (
          <div key={sector} style={{ marginBottom: 14 }}>
            <div className="stat-label" style={{ marginBottom: 6 }}>
              {sector}
            </div>
            <div className="chips">
              {(list ?? []).map((company) => (
                <span
                  key={company.slug}
                  className={`chip${selected === company.slug ? " selected" : ""}`}
                  onClick={() => setSelected(company.slug)}
                >
                  {company.name}
                </span>
              ))}
            </div>
          </div>
        ))}
      </Card>

      {patterns.isLoading && selected && <Loading label="Loading patterns" />}
      {patterns.error && <ErrorNote error={patterns.error} />}

      {patterns.data && (
        <>
          <div className="callout callout-warn" style={{ marginBottom: 16 }}>
            <strong>{patterns.data.evidence === "estimated" ? "Estimated" : "Observed"}.</strong>{" "}
            {patterns.data.disclaimer}
          </div>

          <div className="grid grid-2">
            <Card title={`${patterns.data.name} — estimated interview mix`}>
              {Object.entries(patterns.data.distribution)
                .sort((a, b) => b[1] - a[1])
                .map(([area, weight]) => (
                  <Bar
                    key={area}
                    label={area.replace(/_/g, " ")}
                    value={weight * 100}
                    hint={`${Math.round(weight * 100)}% of the estimated loop`}
                  />
                ))}
              <p className="small faint" style={{ marginTop: 10, marginBottom: 0 }}>
                Basis: {patterns.data.basis}
              </p>
            </Card>

            <Card title="Your readiness">
              {!patterns.data.readiness ? (
                <p className="muted small">
                  Complete an interview first — readiness is computed from measured skills.
                </p>
              ) : (
                <>
                  <Stat
                    label="Estimated readiness"
                    value={
                      patterns.data.readiness.estimated_readiness !== null
                        ? `${patterns.data.readiness.estimated_readiness}%`
                        : "—"
                    }
                    sub={`across ${Math.round(patterns.data.readiness.coverage * 100)}% of the loop you've been assessed on`}
                  />
                  <div style={{ marginTop: 14 }}>
                    {patterns.data.readiness.areas.map((area) => (
                      <div key={area.interview_type}>
                        {area.measured ? (
                          <Bar
                            label={area.interview_type.replace(/_/g, " ")}
                            value={area.score ?? 0}
                            hint={`${area.evidence_count} question(s)`}
                          />
                        ) : (
                          <div className="bar-row">
                            <span className="muted">{area.interview_type.replace(/_/g, " ")}</span>
                            <span className="small faint">not yet assessed</span>
                            <span />
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                  <p className="small faint" style={{ marginTop: 12, marginBottom: 0 }}>
                    {patterns.data.readiness.caveat}
                  </p>
                </>
              )}
            </Card>
          </div>
        </>
      )}
    </>
  );
}
