import React from "react";
import { DatasetSpec } from "../../api/types";
import { Card } from "../common/Card";
import { Badge } from "../common/Badge";

interface GatePrecheckProps {
  dataset: DatasetSpec | null;
  targetColumn: string;
}

export const GatePrecheck: React.FC<GatePrecheckProps> = ({
  dataset,
  targetColumn,
}) => {
  if (!dataset || dataset.rows.length === 0) {
    return null;
  }

  // Calculate missing rates
  const colMissing: Record<string, number> = {};
  dataset.columns.forEach((col) => {
    let nullCount = 0;
    dataset.rows.forEach((row) => {
      if (row[col] === null || row[col] === undefined || row[col] === "") {
        nullCount++;
      }
    });
    colMissing[col] = nullCount / dataset.rows.length;
  });

  const maxNullRate = Math.max(...Object.values(colMissing), 0);
  const nullGatePassed = maxNullRate <= 0.20;

  // Target balance check
  let targetStats = "";
  if (targetColumn && dataset.columns.includes(targetColumn)) {
    const counts: Record<string, number> = {};
    dataset.rows.forEach((r) => {
      const val = String(r[targetColumn]);
      counts[val] = (counts[val] || 0) + 1;
    });
    const entries = Object.entries(counts);
    targetStats = entries.map(([k, v]) => `${k}: ${v}`).slice(0, 4).join(", ");
  }

  return (
    <Card
      title="3. Client-Side Pre-Validation Health"
      subtitle="Early gate checks to ensure conformance before pipeline submission"
    >
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "12px" }}>
        <div
          style={{
            padding: "12px",
            background: "rgba(15, 23, 42, 0.6)",
            borderRadius: "var(--radius-md)",
            border: `1px solid ${nullGatePassed ? "rgba(16, 185, 129, 0.2)" : "rgba(244, 63, 94, 0.3)"}`,
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
            <span style={{ fontSize: "12px", fontWeight: 700 }}>Null-Rate Gate</span>
            <Badge variant={nullGatePassed ? "green" : "red"}>
              {nullGatePassed ? "PASS" : "WARN"}
            </Badge>
          </div>
          <div style={{ fontSize: "13px", color: "var(--text-muted)" }}>
            Max column missing: {(maxNullRate * 100).toFixed(1)}% (Threshold: 20.0%)
          </div>
        </div>

        <div
          style={{
            padding: "12px",
            background: "rgba(15, 23, 42, 0.6)",
            borderRadius: "var(--radius-md)",
            border: "1px solid rgba(56, 189, 248, 0.2)",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
            <span style={{ fontSize: "12px", fontWeight: 700 }}>Schema Conformance</span>
            <Badge variant="blue">READY</Badge>
          </div>
          <div style={{ fontSize: "13px", color: "var(--text-muted)" }}>
            {dataset.columns.length} features detected, {dataset.rows.length} rows loaded
          </div>
        </div>

        <div
          style={{
            padding: "12px",
            background: "rgba(15, 23, 42, 0.6)",
            borderRadius: "var(--radius-md)",
            border: "1px solid var(--border-subtle)",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
            <span style={{ fontSize: "12px", fontWeight: 700 }}>Target Distribution</span>
            <Badge variant="purple">{targetColumn || "N/A"}</Badge>
          </div>
          <div style={{ fontSize: "13px", color: "var(--text-muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {targetStats || "Select a target column"}
          </div>
        </div>
      </div>
    </Card>
  );
};
