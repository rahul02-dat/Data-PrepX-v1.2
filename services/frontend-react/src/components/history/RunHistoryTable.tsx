import React from "react";
import { RunDetail } from "../../api/types";
import { Card } from "../common/Card";
import { Badge } from "../common/Badge";
import { Button } from "../common/Button";

interface RunHistoryTableProps {
  runs: RunDetail[];
  onSelectRun: (runId: string) => void;
  selectedRunsForCompare: string[];
  onToggleCompareRun: (runId: string) => void;
  onLaunchComparator: () => void;
}

export const RunHistoryTable: React.FC<RunHistoryTableProps> = ({
  runs,
  onSelectRun,
  selectedRunsForCompare,
  onToggleCompareRun,
  onLaunchComparator,
}) => {
  return (
    <Card
      title="Experiment Run History & Lineage Archive"
      subtitle="Complete ledger of executed pipelines with immutable content hashes and model artifacts"
      action={
        selectedRunsForCompare.length === 2 && (
          <Button variant="primary" onClick={onLaunchComparator}>
            Compare 2 Selected Runs →
          </Button>
        )
      }
    >
      <div className="data-table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th style={{ width: "40px" }}>Diff</th>
              <th>Run ID</th>
              <th>Dataset</th>
              <th>Task</th>
              <th>Status</th>
              <th>Score</th>
              <th>Duration</th>
              <th>Git SHA</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => {
              const isChecked = selectedRunsForCompare.includes(r.id);
              return (
                <tr key={r.id}>
                  <td>
                    <input
                      type="checkbox"
                      checked={isChecked}
                      onChange={() => onToggleCompareRun(r.id)}
                    />
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)", fontSize: "12px", color: "var(--accent-primary)" }}>
                    {r.id.substring(0, 12)}...
                  </td>
                  <td style={{ fontWeight: 600 }}>{r.dataset_name}</td>
                  <td>
                    <Badge variant="blue">{r.task_type}</Badge>
                  </td>
                  <td>
                    <Badge variant="green" dot>
                      {r.status}
                    </Badge>
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--accent-emerald)" }}>
                    {r.stacking_result.ensemble_score.toFixed(4)}
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                    {r.duration_seconds}s
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-muted)" }}>
                    {r.git_sha}
                  </td>
                  <td>
                    <Button variant="secondary" style={{ padding: "4px 10px", fontSize: "12px" }} onClick={() => onSelectRun(r.id)}>
                      Inspect
                    </Button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
};
