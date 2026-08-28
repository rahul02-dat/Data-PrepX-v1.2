import React from "react";
import { StackingResult } from "../../api/types";
import { Card } from "../common/Card";
import { MetricStat } from "../common/MetricStat";

interface StackingTabProps {
  stacking: StackingResult;
}

export const StackingTab: React.FC<StackingTabProps> = ({ stacking }) => {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
      {/* Metric Summary */}
      <div className="metrics-grid">
        <MetricStat
          label={`Ensemble Score (${stacking.metric_name})`}
          value={stacking.ensemble_score.toFixed(4)}
          subValue="Cross-validated stacked score"
        />
        <MetricStat
          label={stacking.task_type === "classification" ? "Macro F1 Score" : "RMSE Loss"}
          value={stacking.metrics.f1_or_rmse.toFixed(4)}
          subValue="Primary optimization metric"
        />
        <MetricStat
          label={stacking.task_type === "classification" ? "Accuracy" : "R² Determination"}
          value={stacking.metrics.accuracy_or_r2.toFixed(4)}
        />
        <MetricStat
          label="Meta-Learner Architecture"
          value={stacking.meta_learner_type.split(" ")[0]}
          subValue={stacking.meta_learner_type}
        />
      </div>

      {/* Ensemble Composition & Weight Distribution */}
      <Card
        title="Stacked Ensemble Composition & Base Model Weights"
        subtitle="Meta-learner blending coefficients assigned across best-of-family estimators"
      >
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          {Object.entries(stacking.base_family_weights).map(([family, weight]) => {
            const pct = Math.round(weight * 100);
            return (
              <div key={family} style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "13px" }}>
                  <span style={{ fontWeight: 600, textTransform: "capitalize", color: "var(--text-primary)" }}>
                    {family.replace(/_/g, " ")}
                  </span>
                  <span style={{ fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--accent-primary)" }}>
                    {(weight * 100).toFixed(1)}% weight
                  </span>
                </div>
                <div
                  style={{
                    width: "100%",
                    height: "10px",
                    background: "rgba(15, 23, 42, 0.8)",
                    borderRadius: "9999px",
                    overflow: "hidden",
                    border: "1px solid var(--border-subtle)",
                  }}
                >
                  <div
                    style={{
                      width: `${pct}%`,
                      height: "100%",
                      background: "linear-gradient(90deg, var(--accent-primary), var(--accent-secondary))",
                      borderRadius: "9999px",
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </Card>
    </div>
  );
};
