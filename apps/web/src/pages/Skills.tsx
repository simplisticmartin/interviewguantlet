import { useQuery } from "@tanstack/react-query";

import { Badge, Card, Empty, ErrorNote, Loading, PageHeader } from "../components/ui";
import { api, type SkillView } from "../lib/api";

const CALIBRATION: Record<string, { label: string; tone: "good" | "warn" | "bad" | "default"; hint: string }> = {
  mastery: { label: "Mastery", tone: "good", hint: "Knows it and knows they know it." },
  confidence_deficit: {
    label: "Confidence deficit",
    tone: "warn",
    hint: "Knows it but doubts themselves — needs rehearsal, not study.",
  },
  known_weakness: {
    label: "Known gap",
    tone: "warn",
    hint: "Doesn't know it and is aware — the easiest kind to fix.",
  },
  misconception: {
    label: "Misconception",
    tone: "bad",
    hint: "Wrong and confident. The costliest quadrant.",
  },
  unknown: { label: "Unrated", tone: "default", hint: "No self-confidence rating captured." },
};

export default function Skills() {
  const { data, isLoading, error } = useQuery({ queryKey: ["skills"], queryFn: api.skills });

  if (isLoading) return <Loading label="Loading your skill graph" />;
  if (error) return <ErrorNote error={error} />;
  if (!data || data.length === 0) {
    return (
      <>
        <PageHeader title="Skill graph" />
        <Empty
          title="No skills measured yet"
          hint="Your skill graph is built from demonstrated answers, so it fills in as you interview."
        />
      </>
    );
  }

  const byDomain = new Map<string, SkillView[]>();
  for (const skill of data) {
    const root = skill.concept_key.split(".")[0]!;
    const bucket = byDomain.get(root) ?? [];
    bucket.push(skill);
    byDomain.set(root, bucket);
  }

  return (
    <>
      <PageHeader
        title="Skill graph"
        subtitle="Mastery is weighted by question difficulty, recency, independence, and how sure the grader was — not a simple average."
      />

      {[...byDomain.entries()].map(([domain, skills]) => (
        <Card key={domain} title={domain.replace(/_/g, " ")}>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Concept</th>
                  <th style={{ width: 90 }}>Mastery</th>
                  <th style={{ width: 90 }}>Certainty</th>
                  <th style={{ width: 80 }}>Evidence</th>
                  <th style={{ width: 170 }}>Calibration</th>
                </tr>
              </thead>
              <tbody>
                {skills
                  .sort((a, b) => b.mastery - a.mastery)
                  .map((skill) => {
                    const cal = CALIBRATION[skill.calibration] ?? CALIBRATION.unknown!;
                    return (
                      <tr key={skill.concept_key}>
                        <td>
                          {skill.display_name}
                          <div className="small faint mono">{skill.concept_key}</div>
                        </td>
                        <td>{Math.round(skill.mastery * 100)}%</td>
                        <td className="muted">{Math.round(skill.confidence * 100)}%</td>
                        <td className="muted">{skill.evidence_count}</td>
                        <td>
                          <span title={cal.hint}>
                            <Badge tone={cal.tone}>{cal.label}</Badge>
                          </span>
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>
        </Card>
      ))}
    </>
  );
}
