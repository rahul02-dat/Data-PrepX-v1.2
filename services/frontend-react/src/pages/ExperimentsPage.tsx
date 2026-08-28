import React, { useState } from "react";
import { RunDetail } from "../api/types";
import { RunHistoryTable } from "../components/history/RunHistoryTable";
import { RunComparator } from "../components/history/RunComparator";

interface ExperimentsPageProps {
  runs: RunDetail[];
  onSelectRun: (runId: string) => void;
}

export const ExperimentsPage: React.FC<ExperimentsPageProps> = ({
  runs,
  onSelectRun,
}) => {
  const [selectedForCompare, setSelectedForCompare] = useState<string[]>([]);
  const [isComparing, setIsComparing] = useState(false);

  const toggleCompare = (runId: string) => {
    setSelectedForCompare((prev) =>
      prev.includes(runId)
        ? prev.filter((id) => id !== runId)
        : prev.length < 2
        ? [...prev, runId]
        : [prev[1], runId]
    );
  };

  const runA = runs.find((r) => r.id === selectedForCompare[0]);
  const runB = runs.find((r) => r.id === selectedForCompare[1]);

  if (isComparing && runA && runB) {
    return (
      <RunComparator
        runA={runA}
        runB={runB}
        onBack={() => setIsComparing(false)}
      />
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
      <div>
        <h2 style={{ fontSize: "20px", fontWeight: 800, color: "var(--text-primary)" }}>
          Lineage & Experiment Registry
        </h2>
        <p style={{ fontSize: "13px", color: "var(--text-muted)" }}>
          Query, inspect, and benchmark autonomous runs across model families and datasets
        </p>
      </div>

      <RunHistoryTable
        runs={runs}
        onSelectRun={onSelectRun}
        selectedRunsForCompare={selectedForCompare}
        onToggleCompareRun={toggleCompare}
        onLaunchComparator={() => setIsComparing(true)}
      />
    </div>
  );
};
