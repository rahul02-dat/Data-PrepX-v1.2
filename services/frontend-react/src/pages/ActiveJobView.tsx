import React from "react";
import { JobTracker } from "../components/tracker/JobTracker";
import { Button } from "../components/common/Button";

interface ActiveJobViewProps {
  jobId: string;
  onViewResults: (jobId: string) => void;
  onCancel: () => void;
}

export const ActiveJobView: React.FC<ActiveJobViewProps> = ({
  jobId,
  onViewResults,
  onCancel,
}) => {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h2 style={{ fontSize: "20px", fontWeight: 800, color: "var(--text-primary)" }}>
            Live Pipeline Execution
          </h2>
          <p style={{ fontSize: "13px", color: "var(--text-muted)" }}>
            Tracking Celery distributed task graph & WebSocket transitions in real time
          </p>
        </div>
        <Button variant="ghost" onClick={onCancel}>
          ← Back to Studio
        </Button>
      </div>

      <JobTracker jobId={jobId} onViewResults={onViewResults} />
    </div>
  );
};
