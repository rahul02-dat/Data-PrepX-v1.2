import React, { useState } from "react";
import { RunDetail } from "../../api/types";
import { TabItem, TabNav } from "../common/TabNav";
import { MetricStat } from "../common/MetricStat";
import { Badge } from "../common/Badge";
import { LineageDAG } from "../lineage/LineageDAG";
import { GateResultsTab } from "./GateResultsTab";
import { RLOptimizerTab } from "./RLOptimizerTab";
import { OptunaHPOTab } from "./OptunaHPOTab";
import { StackingTab } from "./StackingTab";
import { MAMLTimelineTab } from "./MAMLTimelineTab";
import { BoundedSummaryTab } from "./BoundedSummaryTab";

interface ResultsDashboardProps {
  run: RunDetail;
  onReplayRun: () => void;
}

export const ResultsDashboard: React.FC<ResultsDashboardProps> = ({ run, onReplayRun }) => {
  const [activeTab, setActiveTab] = useState<string>("summary");

  const tabs: TabItem[] = [
    { id: "summary", label: "Bounded Summary", badge: `${run.bounded_summary.claims.length} Claims` },
    { id: "gates", label: "Validation Gates", badge: `${run.gates.length} Gates` },
    { id: "rl", label: "RL Preprocessing", badge: `+${run.rl_result.delta_reward.toFixed(3)}` },
    { id: "optuna", label: "Optuna HPO", badge: `${run.optuna_trials.length} Trials` },
    { id: "stacking", label: "Stacked Ensemble", badge: `${(run.stacking_result.ensemble_score * 100).toFixed(1)}%` },
    { id: "maml", label: "MAML Adaptation", badge: `${run.maml_timeline.length} Batches` },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* Top Run Header Banner */}
      <div
        style={{
          background: "linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.8))",
          border: "1px solid var(--border-subtle)",
          borderRadius: "var(--radius-lg)",
          padding: "20px 24px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "16px",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "6px" }}>
            <h2 style={{ fontSize: "20px", fontWeight: 800, color: "var(--text-primary)" }}>
              Run Results: {run.dataset_name}
            </h2>
            <Badge variant="green" dot>
              {run.status.toUpperCase()}
            </Badge>
            <Badge variant="blue">{run.task_type}</Badge>
          </div>
          <div style={{ display: "flex", gap: "16px", fontSize: "12px", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
            <span>Run ID: {run.id.substring(0, 16)}...</span>
            <span>Git SHA: {run.git_sha}</span>
            <span>Duration: {run.duration_seconds}s</span>
          </div>
        </div>

        <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
          <MetricStat
            label="Final Model Metric"
            value={run.stacking_result.ensemble_score.toFixed(4)}
            subValue={run.stacking_result.metric_name}
            style={{ padding: "8px 16px", minWidth: "180px", margin: 0 }}
          />
        </div>
      </div>

      {/* Lineage Graph Section */}
      <LineageDAG
        lineage={run.lineage}
        configHash={run.config_hash}
        config={run.config}
        onReplay={onReplayRun}
      />

      {/* Tab Navigation */}
      <TabNav tabs={tabs} activeTab={activeTab} onChange={setActiveTab} />

      {/* Tab Content Panels */}
      <div>
        {activeTab === "summary" && <BoundedSummaryTab summary={run.bounded_summary} />}
        {activeTab === "gates" && <GateResultsTab gates={run.gates} />}
        {activeTab === "rl" && <RLOptimizerTab rl={run.rl_result} />}
        {activeTab === "optuna" && (
          <OptunaHPOTab trials={run.optuna_trials} familyBests={run.family_best_models} />
        )}
        {activeTab === "stacking" && <StackingTab stacking={run.stacking_result} />}
        {activeTab === "maml" && <MAMLTimelineTab timeline={run.maml_timeline} />}
      </div>
    </div>
  );
};
