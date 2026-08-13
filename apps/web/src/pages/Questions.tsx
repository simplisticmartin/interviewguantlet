import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { Badge, Card, Empty, ErrorNote, Loading, PageHeader } from "../components/ui";
import { api } from "../lib/api";

const TYPES = [
  "",
  "java",
  "spring",
  "database",
  "distributed",
  "system_design",
  "dsa",
  "cloud",
  "behavioral",
  "hiring_manager",
];

export default function Questions() {
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [type, setType] = useState("");
  const [minDifficulty, setMinDifficulty] = useState(1);

  const { data, isLoading, error } = useQuery({
    queryKey: ["questions", submitted, type, minDifficulty],
    queryFn: () =>
      api.searchQuestions({
        q: submitted,
        ...(type ? { interview_type: type } : {}),
        min_difficulty: String(minDifficulty),
      }),
  });

  return (
    <>
      <PageHeader
        title="Question bank"
        subtitle="Every question here is authored by Gauntlet and tagged with its provenance. Nothing is scraped, and nothing is attributed to a company."
      />

      <Card>
        <form
          style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "flex-end" }}
          onSubmit={(e) => {
            e.preventDefault();
            setSubmitted(query);
          }}
        >
          <div className="field" style={{ flex: "2 1 260px", marginBottom: 0 }}>
            <label htmlFor="q">Search</label>
            <input
              id="q"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="kafka ordering, transaction rollback, cache stampede…"
            />
          </div>
          <div className="field" style={{ flex: "1 1 150px", marginBottom: 0 }}>
            <label htmlFor="type">Area</label>
            <select id="type" value={type} onChange={(e) => setType(e.target.value)}>
              {TYPES.map((item) => (
                <option key={item} value={item}>
                  {item ? item.replace(/_/g, " ") : "All areas"}
                </option>
              ))}
            </select>
          </div>
          <div className="field" style={{ flex: "1 1 130px", marginBottom: 0 }}>
            <label htmlFor="diff">Min difficulty: {minDifficulty}</label>
            <input
              id="diff"
              type="range"
              min={1}
              max={5}
              value={minDifficulty}
              onChange={(e) => setMinDifficulty(Number(e.target.value))}
            />
          </div>
          <button className="btn-primary" type="submit">
            Search
          </button>
        </form>
      </Card>

      {isLoading && <Loading />}
      {error && <ErrorNote error={error} />}

      {data && data.length === 0 && (
        <Empty
          title="Nothing matched"
          hint="Try a broader search, or lower the difficulty floor. The corpus is seeded from the shipped catalogue — run the seeder if the database is empty."
        />
      )}

      {data?.map((item, index) => (
        <Card key={item.id ?? index}>
          <p style={{ fontWeight: 560, marginBottom: 8 }}>{item.question}</p>
          <div className="chips">
            <Badge tone="info">{item.interview_type.replace(/_/g, " ")}</Badge>
            <Badge>difficulty {item.difficulty}</Badge>
            {item.concept_keys.slice(0, 4).map((concept) => (
              <span key={concept} className="chip">
                {concept}
              </span>
            ))}
          </div>
          <p className="small faint" style={{ marginTop: 8, marginBottom: 0 }}>
            Origin: {item.question_origin} · source: {item.source_type}
            {item.score !== null ? ` · relevance ${item.score.toFixed(2)}` : ""}
          </p>
        </Card>
      ))}
    </>
  );
}
