import React from "react";
import { FamilyBestModel, OptunaTrial } from "../../api/types";
import { Card } from "../common/Card";
import { Badge } from "../common/Badge";

interface OptunaHPOTabProps {
  trials: OptunaTrial[];
  familyBests: FamilyBestModel[];
}

export const OptunaHPOTab: React.FC<OptunaHPOTabProps> = ({ trials, familyBests }) => {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
      {/* Family Best Cards */}
      <Card
        title="Best Model Candidates by Family (Optuna TPE Sampler)"
        subtitle="Independent Bayesian hyperparameter optimization studies evaluated across families"
      >
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "14px" }}>
          {familyBests.map((fam) => (
            <div
              key={fam.family}
              style={{
                background: "rgba(15, 23, 42, 0.7)",
                border: "1px solid var(--border-subtle)",
                borderRadius: "var(--radius-md)",
                padding: "14px",
                display: "flex",
                flexDirection: "column",
                gap: "8px",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: "13px", fontWeight: 700, textTransform: "uppercase", color: "var(--accent-cyan)" }}>
                  {fam.family.replace(/_/g, " ")}
                </span>
                <Badge variant="blue">{fam.n_trials_evaluated} Trials</Badge>
              </div>

              <div style={{ display: "flex", alignItems: "baseline", gap: "6px" }}>
                <span style={{ fontSize: "22px", fontWeight: 800, fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>
                  {fam.best_score.toFixed(4)}
                </span>
                <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>CV metric</span>
              </div>

              <div style={{ marginTop: "4px" }}>
                <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 600 }}>
                  Optimal Hyperparameters:
                </div>
                <pre
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "10px",
                    background: "#080c14",
                    padding: "6px",
                    borderRadius: "4px",
                    marginTop: "4px",
                    color: "var(--text-secondary)",
                    overflowX: "auto",
                  }}
                >
                  {JSON.stringify(fam.best_params, null, 2)}
                </pre>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Trial Execution History */}
      <Card
        title="Optuna Trial Log"
        subtitle="Individual trial evaluations with parameters, objective scores, and durations"
      >
        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Trial #</th>
                <th>Family</th>
                <th>Validation Score</th>
                <th>Duration (s)</th>
                <th>State</th>
                <th>Parameters</th>
              </tr>
            </thead>
            <tbody>
              {trials.map((t) => (
                <tr key={t.trial_number}>
                  <td style={{ fontFamily: "var(--font-mono)", fontWeight: 700 }}>#{t.trial_number}</td>
                  <td style={{ textTransform: "capitalize" }}>{t.family.replace(/_/g, " ")}</td>
                  <td style={{ fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--accent-emerald)" }}>
                    {t.score.toFixed(4)}
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                    {t.duration_seconds.toFixed(2)}s
                  </td>
                  <td>
                    <Badge variant={t.state === "COMPLETE" ? "green" : "yellow"}>
                      {t.state}
                    </Badge>
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-muted)" }}>
                    {JSON.stringify(t.params)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
};
