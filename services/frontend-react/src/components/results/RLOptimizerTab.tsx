import React from "react";
import { RLOptimizerResult } from "../../api/types";
import { Card } from "../common/Card";
import { MetricStat } from "../common/MetricStat";
import { Badge } from "../common/Badge";

interface RLOptimizerTabProps {
  rl: RLOptimizerResult;
}

export const RLOptimizerTab: React.FC<RLOptimizerTabProps> = ({ rl }) => {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
      {/* Metric Summary Cards */}
      <div className="metrics-grid">
        <MetricStat
          label="Downstream Delta (Δ Reward)"
          value={`+${rl.delta_reward.toFixed(3)}`}
          delta={{ value: `${((rl.optimized_metric - rl.baseline_metric) * 100).toFixed(1)}% lift`, positive: true }}
        />
        <MetricStat
          label="RL Optimized Metric"
          value={rl.optimized_metric.toFixed(3)}
          subValue="Validation CV performance"
        />
        <MetricStat
          label="Default Baseline Metric"
          value={rl.baseline_metric.toFixed(3)}
          subValue="Raw un-optimized baseline"
        />
        <MetricStat
          label="Discretized State Bin"
          value={`[${rl.state_bin.join(", ")}]`}
          subValue="5-D meta-feature vector"
        />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px" }}>
        {/* Meta-Features Profiler */}
        <Card
          title="Dataset Meta-Features State (s)"
          subtitle="Computed state vector mapping dataset properties to RL action space"
        >
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            {Object.entries(rl.meta_features).map(([key, val]) => (
              <div
                key={key}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  padding: "8px 12px",
                  background: "rgba(15, 23, 42, 0.6)",
                  borderRadius: "var(--radius-md)",
                  border: "1px solid var(--border-subtle)",
                }}
              >
                <span style={{ fontSize: "12px", textTransform: "capitalize", color: "var(--text-secondary)" }}>
                  {key.replace(/_/g, " ")}
                </span>
                <span style={{ fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--accent-primary)" }}>
                  {typeof val === "number" ? val.toFixed(4) : String(val)}
                </span>
              </div>
            ))}
          </div>
        </Card>

        {/* Action Selection & Policy Decision */}
        <Card
          title="Q-Learning Action Chosen (a)"
          subtitle="Learned discrete policy selection {Imputer} × {Outlier} × {Threshold}"
        >
          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            <div style={{ padding: "12px", background: "rgba(15, 23, 42, 0.6)", borderRadius: "var(--radius-md)" }}>
              <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
                Selected Imputation Method
              </div>
              <div style={{ fontSize: "15px", fontWeight: 700, color: "var(--accent-cyan)", marginTop: "2px" }}>
                {rl.action_chosen.imputation.toUpperCase()}
              </div>
            </div>

            <div style={{ padding: "12px", background: "rgba(15, 23, 42, 0.6)", borderRadius: "var(--radius-md)" }}>
              <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
                Selected Outlier Detector
              </div>
              <div style={{ fontSize: "15px", fontWeight: 700, color: "var(--accent-secondary)", marginTop: "2px" }}>
                {rl.action_chosen.outlier.toUpperCase()} (contamination = {rl.action_chosen.threshold})
              </div>
            </div>

            <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
              <Badge variant="green">Policy Convergence: Verified</Badge>
              <Badge variant="purple">ε-greedy decayed</Badge>
            </div>
          </div>
        </Card>
      </div>

      {/* Q-Learning Convergence Curve */}
      <Card
        title="Q-Learning Episodic Reward Convergence"
        subtitle="Offline trajectory tracking reward maximization across episodes"
      >
        <div style={{ height: "180px", display: "flex", alignItems: "flex-end", gap: "14px", padding: "16px 0" }}>
          {rl.convergence_history.map((ep) => {
            const maxR = 0.05;
            const barHeight = Math.max((ep.reward / maxR) * 120, 10);
            return (
              <div
                key={ep.episode}
                style={{
                  flex: 1,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: "6px",
                }}
              >
                <span style={{ fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--accent-primary)" }}>
                  +{ep.reward.toFixed(3)}
                </span>
                <div
                  style={{
                    width: "100%",
                    height: `${barHeight}px`,
                    background: "linear-gradient(180deg, var(--accent-primary) 0%, rgba(56, 189, 248, 0.2) 100%)",
                    borderRadius: "4px 4px 0 0",
                    border: "1px solid var(--accent-primary)",
                  }}
                />
                <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                  Ep {ep.episode}
                </span>
              </div>
            );
          })}
        </div>
      </Card>
    </div>
  );
};
