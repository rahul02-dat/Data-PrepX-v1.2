import React from "react";
import { ImputationMethod, OutlierMethod, PipelineConfig, TaskType } from "../../api/types";
import { Card } from "../common/Card";

interface ConfigPanelProps {
  columns: string[];
  config: PipelineConfig;
  onChange: (config: PipelineConfig) => void;
}

export const ConfigPanel: React.FC<ConfigPanelProps> = ({
  columns,
  config,
  onChange,
}) => {
  const update = (partial: Partial<PipelineConfig>) => {
    onChange({ ...config, ...partial });
  };

  return (
    <Card
      title="2. Pipeline Configuration"
      subtitle="Autonomous preprocessing and Bayesian estimation hyperparameters"
    >
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "16px" }}>
        {/* Task Type & Target Column */}
        <div className="form-group">
          <label className="form-label">Task Type</label>
          <select
            className="form-select"
            value={config.task_type}
            onChange={(e) => update({ task_type: e.target.value as TaskType })}
          >
            <option value="classification">Classification (Multi-class / Binary)</option>
            <option value="regression">Regression (Continuous Target)</option>
          </select>
        </div>

        <div className="form-group">
          <label className="form-label">Target Column (y)</label>
          <select
            className="form-select"
            value={config.target_column}
            onChange={(e) => update({ target_column: e.target.value })}
          >
            {columns.length === 0 && <option value="">No dataset loaded</option>}
            {columns.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>

        {/* Preprocessing Actions (RL Action Space) */}
        <div className="form-group">
          <label className="form-label">Imputation Strategy (RL Action)</label>
          <select
            className="form-select"
            value={config.imputation_method}
            onChange={(e) => update({ imputation_method: e.target.value as ImputationMethod })}
          >
            <option value="mice">MICE (Multivariate Chained Equations)</option>
            <option value="knn">KNN (K-Nearest Neighbors)</option>
          </select>
        </div>

        <div className="form-group">
          <label className="form-label">Outlier Detection (RL Action)</label>
          <select
            className="form-select"
            value={config.outlier_method}
            onChange={(e) => update({ outlier_method: e.target.value as OutlierMethod })}
          >
            <option value="isolation_forest">Isolation Forest (Multivariate Trees)</option>
            <option value="lof">Local Outlier Factor (Density-based)</option>
            <option value="none">None (Preserve All Data Points)</option>
          </select>
        </div>

        {/* Bayesian HPO & Stacking Options */}
        <div className="form-group">
          <label className="form-label">Optuna Trials Budget</label>
          <input
            type="number"
            className="form-input"
            value={config.n_trials}
            min={5}
            max={200}
            onChange={(e) => update({ n_trials: parseInt(e.target.value, 10) || 20 })}
          />
        </div>

        <div className="form-group">
          <label className="form-label">Cross-Validation Folds</label>
          <input
            type="number"
            className="form-input"
            value={config.cv_folds}
            min={2}
            max={10}
            onChange={(e) => update({ cv_folds: parseInt(e.target.value, 10) || 5 })}
          />
        </div>

        <div className="form-group">
          <label className="form-label">Stacking Meta-CV Folds</label>
          <input
            type="number"
            className="form-input"
            value={config.stacking_cv_folds}
            min={2}
            max={10}
            onChange={(e) => update({ stacking_cv_folds: parseInt(e.target.value, 10) || 5 })}
          />
        </div>

        <div className="form-group">
          <label className="form-label">Deterministic Random Seed</label>
          <input
            type="number"
            className="form-input"
            value={config.seed}
            onChange={(e) => update({ seed: parseInt(e.target.value, 10) || 42 })}
          />
        </div>
      </div>
    </Card>
  );
};
