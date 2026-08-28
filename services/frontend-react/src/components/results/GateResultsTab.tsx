import React from "react";
import { GateCheckResult } from "../../api/types";
import { Card } from "../common/Card";
import { Badge } from "../common/Badge";

interface GateResultsTabProps {
  gates: GateCheckResult[];
}

export const GateResultsTab: React.FC<GateResultsTabProps> = ({ gates }) => {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <Card
        title="Fail-Closed Validation Gate Evaluations"
        subtitle="Hard gate criteria ensuring data quality and distribution stability before modeling"
      >
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "16px" }}>
          {gates.map((gate, i) => (
            <div
              key={i}
              style={{
                background: "rgba(15, 23, 42, 0.7)",
                border: `1px solid ${gate.passed ? "rgba(16, 185, 129, 0.25)" : "rgba(244, 63, 94, 0.3)"}`,
                borderRadius: "var(--radius-md)",
                padding: "16px",
                display: "flex",
                flexDirection: "column",
                gap: "10px",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: "14px", fontWeight: 700, color: "var(--text-primary)" }}>
                  {gate.gate_name}
                </span>
                <Badge variant={gate.passed ? "green" : "red"}>
                  {gate.passed ? "PASSED" : "FAILED"}
                </Badge>
              </div>

              <div style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
                {gate.details}
              </div>

              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: "12px",
                  fontFamily: "var(--font-mono)",
                  padding: "8px",
                  background: "#080c14",
                  borderRadius: "6px",
                  border: "1px solid var(--border-subtle)",
                }}
              >
                <span style={{ color: "var(--text-muted)" }}>Computed: {gate.score.toFixed(3)}</span>
                <span style={{ color: "var(--accent-cyan)" }}>Threshold: {gate.threshold.toFixed(3)}</span>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};
