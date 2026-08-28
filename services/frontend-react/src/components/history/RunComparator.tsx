import React from "react";
import { RunDetail } from "../../api/types";
import { Card } from "../common/Card";
import { Button } from "../common/Button";
import { Badge } from "../common/Badge";

interface RunComparatorProps {
  runA: RunDetail;
  runB: RunDetail;
  onBack: () => void;
}

export const RunComparator: React.FC<RunComparatorProps> = ({ runA, runB, onBack }) => {
  const metricDelta = runB.stacking_result.ensemble_score - runA.stacking_result.ensemble_score;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
      <Card
        title="Side-by-Side Run & Ablation Comparator"
        subtitle="Compare hyperparameters, validation gate outcomes, and downstream metrics between two runs"
        action={
          <Button variant="secondary" onClick={onBack}>
            ← Back to Run History
          </Button>
        }
      >
        <div style={{ display: "grid", gridTemplateColumns: "200px 1fr 1fr", gap: "16px", fontSize: "13px" }}>
          {/* Header Row */}
          <div style={{ fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase" }}>
            Attribute
          </div>
          <div style={{ background: "rgba(15, 23, 42, 0.8)", padding: "12px", borderRadius: "8px", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontWeight: 700, color: "var(--accent-primary)" }}>Run A</div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-muted)" }}>{runA.id.substring(0, 16)}</div>
          </div>
          <div style={{ background: "rgba(15, 23, 42, 0.8)", padding: "12px", borderRadius: "8px", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontWeight: 700, color: "var(--accent-secondary)" }}>Run B</div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-muted)" }}>{runB.id.substring(0, 16)}</div>
          </div>

          {/* Dataset */}
          <div style={{ color: "var(--text-secondary)", fontWeight: 600 }}>Dataset</div>
          <div>{runA.dataset_name}</div>
          <div>{runB.dataset_name}</div>

          {/* Task Type */}
          <div style={{ color: "var(--text-secondary)", fontWeight: 600 }}>Task Type</div>
          <div><Badge variant="blue">{runA.task_type}</Badge></div>
          <div><Badge variant="blue">{runB.task_type}</Badge></div>

          {/* Imputation Action */}
          <div style={{ color: "var(--text-secondary)", fontWeight: 600 }}>Imputation Strategy</div>
          <div>{runA.config.imputation_method.toUpperCase()}</div>
          <div>{runB.config.imputation_method.toUpperCase()}</div>

          {/* Outlier Action */}
          <div style={{ color: "var(--text-secondary)", fontWeight: 600 }}>Outlier Detector</div>
          <div>{runA.config.outlier_method.toUpperCase()}</div>
          <div>{runB.config.outlier_method.toUpperCase()}</div>

          {/* Final Ensemble Metric */}
          <div style={{ color: "var(--text-secondary)", fontWeight: 600 }}>Ensemble Metric</div>
          <div style={{ fontFamily: "var(--font-mono)", fontWeight: 700 }}>
            {runA.stacking_result.ensemble_score.toFixed(4)}
          </div>
          <div style={{ fontFamily: "var(--font-mono)", fontWeight: 700, color: metricDelta >= 0 ? "var(--accent-emerald)" : "var(--accent-rose)" }}>
            {runB.stacking_result.ensemble_score.toFixed(4)} ({metricDelta >= 0 ? "+" : ""}{metricDelta.toFixed(4)})
          </div>

          {/* RL Delta Reward */}
          <div style={{ color: "var(--text-secondary)", fontWeight: 600 }}>RL Δ Reward</div>
          <div style={{ fontFamily: "var(--font-mono)" }}>+{runA.rl_result.delta_reward.toFixed(3)}</div>
          <div style={{ fontFamily: "var(--font-mono)" }}>+{runB.rl_result.delta_reward.toFixed(3)}</div>

          {/* Git SHA */}
          <div style={{ color: "var(--text-secondary)", fontWeight: 600 }}>Git Commit SHA</div>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px" }}>{runA.git_sha}</div>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px" }}>{runB.git_sha}</div>
        </div>
      </Card>
    </div>
  );
};
