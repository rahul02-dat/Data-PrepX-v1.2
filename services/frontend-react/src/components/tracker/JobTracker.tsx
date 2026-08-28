import React from "react";
import { JobStatus } from "../../api/types";
import { useJobSocket } from "../../hooks/useJobSocket";
import { Card } from "../common/Card";
import { Badge } from "../common/Badge";
import { ProgressRing } from "./ProgressRing";
import { LiveLogFeed } from "./LiveLogFeed";
import { Button } from "../common/Button";

interface JobTrackerProps {
  jobId: string;
  onViewResults: (jobId: string) => void;
}

const STAGES: { key: JobStatus; label: string }[] = [
  { key: "queued", label: "Queued" },
  { key: "running", label: "Lineage Ingest" },
  { key: "gate-check", label: "Validation Gates" },
  { key: "optimizing", label: "RL & Optuna HPO" },
  { key: "done", label: "Completed" },
];

export const JobTracker: React.FC<JobTrackerProps> = ({ jobId, onViewResults }) => {
  const { status, logs, isConnected } = useJobSocket(jobId);

  const getStageIndex = (st: JobStatus) => {
    switch (st) {
      case "queued": return 0;
      case "running": return 1;
      case "gate-check": return 2;
      case "optimizing": return 3;
      case "done": return 4;
      case "failed": return 4;
      default: return 0;
    }
  };

  const currentIndex = getStageIndex(status);
  const progressPercent = status === "done" ? 100 : status === "failed" ? 100 : ((currentIndex + 1) / STAGES.length) * 100;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
      <Card
        title={
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <span>Live Job Execution Tracker</span>
            <Badge variant={status === "done" ? "green" : status === "failed" ? "red" : "blue"} dot>
              {status.toUpperCase()}
            </Badge>
          </div>
        }
        subtitle={`Run ID: ${jobId}`}
        action={
          status === "done" && (
            <Button variant="primary" onClick={() => onViewResults(jobId)}>
              View Full Results Dashboard →
            </Button>
          )
        }
      >
        <div style={{ display: "grid", gridTemplateColumns: "1fr 200px", gap: "24px", alignItems: "center" }}>
          {/* Stepper tracker */}
          <div className="stepper-container">
            <div className="stepper-track">
              <div
                className="stepper-track-progress"
                style={{ width: `${(currentIndex / (STAGES.length - 1)) * 100}%` }}
              />
            </div>
            {STAGES.map((s, idx) => {
              const isCompleted = currentIndex > idx || status === "done";
              const isActive = currentIndex === idx && status !== "done";
              return (
                <div
                  key={s.key}
                  className={`step-node ${isCompleted ? "completed" : ""} ${isActive ? "active" : ""}`}
                >
                  <div className="step-circle">
                    {isCompleted ? "✓" : idx + 1}
                  </div>
                  <div className="step-label">{s.label}</div>
                </div>
              );
            })}
          </div>

          {/* Radial progress ring */}
          <div>
            <ProgressRing progress={progressPercent} status={status} />
          </div>
        </div>
      </Card>

      <LiveLogFeed logs={logs} isConnected={isConnected} />
    </div>
  );
};
