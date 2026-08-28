import React from "react";
import { MAMLAdaptationStep } from "../../api/types";
import { Card } from "../common/Card";
import { Badge } from "../common/Badge";

interface MAMLTimelineTabProps {
  timeline: MAMLAdaptationStep[];
}

export const MAMLTimelineTab: React.FC<MAMLTimelineTabProps> = ({ timeline }) => {
  if (timeline.length === 0) {
    return (
      <Card
        title="MAML Continuous Meta-Learning & Adaptive Feature Engineering"
        subtitle="Batch-level adaptation timeline tracking drift detection and fast gradient updates"
      >
        <div style={{ padding: "24px", textAlign: "center", color: "var(--text-muted)", fontSize: "13px" }}>
          No stream/batch drift events recorded for this static run. Run an adaptive continuous stream to view MAML gradient steps.
        </div>
      </Card>
    );
  }

  return (
    <Card
      title="MAML Continuous Meta-Learning & Adaptive Feature Engineering"
      subtitle="Drift-triggered inner gradient loop restoring performance without full retrains"
    >
      <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
        {timeline.map((step) => (
          <div
            key={step.batch_id}
            style={{
              background: "rgba(15, 23, 42, 0.7)",
              border: `1px solid ${step.drift_detected ? "rgba(245, 158, 11, 0.3)" : "var(--border-subtle)"}`,
              borderRadius: "var(--radius-md)",
              padding: "16px",
              display: "flex",
              flexDirection: "column",
              gap: "10px",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <span style={{ fontSize: "14px", fontWeight: 700, color: "var(--text-primary)" }}>
                  Data Batch #{step.batch_id}
                </span>
                <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>[{step.timestamp}]</span>
              </div>
              <div style={{ display: "flex", gap: "8px" }}>
                <Badge variant={step.drift_detected ? "yellow" : "green"}>
                  {step.drift_detected ? "DRIFT DETECTED" : "STABLE"}
                </Badge>
                {step.adapted && <Badge variant="purple">{step.inner_gradient_steps} MAML Steps</Badge>}
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "10px", fontSize: "12px", fontFamily: "var(--font-mono)" }}>
              <div style={{ background: "#080c14", padding: "8px", borderRadius: "6px" }}>
                <span style={{ color: "var(--text-muted)" }}>Drift PSI: </span>
                <span style={{ color: step.drift_detected ? "var(--accent-amber)" : "var(--accent-emerald)", fontWeight: 700 }}>
                  {step.drift_psi.toFixed(3)}
                </span>
              </div>

              <div style={{ background: "#080c14", padding: "8px", borderRadius: "6px" }}>
                <span style={{ color: "var(--text-muted)" }}>Pre-Adapt Score: </span>
                <span style={{ color: "var(--accent-rose)", fontWeight: 700 }}>
                  {step.pre_adaptation_score.toFixed(3)}
                </span>
              </div>

              <div style={{ background: "#080c14", padding: "8px", borderRadius: "6px" }}>
                <span style={{ color: "var(--text-muted)" }}>Post-Adapt Score: </span>
                <span style={{ color: "var(--accent-emerald)", fontWeight: 700 }}>
                  {step.post_adaptation_score.toFixed(3)}
                </span>
              </div>

              <div style={{ background: "#080c14", padding: "8px", borderRadius: "6px" }}>
                <span style={{ color: "var(--text-muted)" }}>Active Features: </span>
                <span style={{ color: "var(--accent-cyan)", fontWeight: 700 }}>
                  {step.selected_feature_count}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
};
