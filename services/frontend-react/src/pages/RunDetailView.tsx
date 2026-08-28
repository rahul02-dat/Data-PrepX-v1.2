import React, { useEffect, useState } from "react";
import { RunDetail } from "../api/types";
import { fetchRunDetail } from "../api/client";
import { ResultsDashboard } from "../components/results/ResultsDashboard";
import { Button } from "../components/common/Button";

interface RunDetailViewProps {
  runId: string;
  onBack: () => void;
  onReplayRun: (run: RunDetail) => void;
}

export const RunDetailView: React.FC<RunDetailViewProps> = ({
  runId,
  onBack,
  onReplayRun,
}) => {
  const [run, setRun] = useState<RunDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetchRunDetail(runId)
      .then((data) => {
        setRun(data);
        setLoading(false);
      })
      .catch(() => {
        setLoading(false);
      });
  }, [runId]);

  if (loading) {
    return (
      <div style={{ padding: "48px", textAlign: "center", color: "var(--text-muted)" }}>
        Loading research run details & lineage DAG...
      </div>
    );
  }

  if (!run) {
    return (
      <div style={{ padding: "48px", textAlign: "center" }}>
        <h3>Run not found ({runId})</h3>
        <Button variant="secondary" onClick={onBack} style={{ marginTop: "16px" }}>
          ← Back to Experiments
        </Button>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <Button variant="ghost" onClick={onBack}>
          ← Back to History
        </Button>
      </div>

      <ResultsDashboard run={run} onReplayRun={() => onReplayRun(run)} />
    </div>
  );
};
