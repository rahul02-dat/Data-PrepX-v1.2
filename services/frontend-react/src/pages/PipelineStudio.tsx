import React, { useState } from "react";
import { DatasetSpec, PipelineConfig } from "../api/types";
import { DatasetUpload } from "../components/upload/DatasetUpload";
import { ConfigPanel } from "../components/upload/ConfigPanel";
import { GatePrecheck } from "../components/upload/GatePrecheck";
import { Button } from "../components/common/Button";

interface PipelineStudioProps {
  onSubmitJob: (dataset: DatasetSpec, config: PipelineConfig, filename: string) => void;
  initialConfig?: PipelineConfig;
}

export const PipelineStudio: React.FC<PipelineStudioProps> = ({
  onSubmitJob,
  initialConfig,
}) => {
  const [dataset, setDataset] = useState<DatasetSpec | null>(null);
  const [filename, setFilename] = useState<string>("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const [config, setConfig] = useState<PipelineConfig>(
    initialConfig || {
      task_type: "classification",
      target_column: "",
      imputation_method: "mice",
      outlier_method: "isolation_forest",
      seed: 42,
      n_trials: 20,
      cv_folds: 5,
      stacking_cv_folds: 5,
    }
  );

  const handleDatasetLoaded = (ds: DatasetSpec, fn: string) => {
    setDataset(ds);
    setFilename(fn);
    if (ds.columns.length > 0 && !config.target_column) {
      // Default to last column or target
      const target = ds.columns.find((c) => c.toLowerCase() === "target") || ds.columns[ds.columns.length - 1];
      setConfig((prev) => ({ ...prev, target_column: target }));
    }
  };

  const handleSubmit = async () => {
    if (!dataset || !config.target_column) return;
    setIsSubmitting(true);
    try {
      onSubmitJob(dataset, config, filename || "custom_dataset.csv");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* Studio Header Banner */}
      <div
        style={{
          background: "linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(30, 41, 59, 0.6))",
          border: "1px solid var(--border-subtle)",
          borderRadius: "var(--radius-lg)",
          padding: "24px",
        }}
      >
        <h2 style={{ fontSize: "22px", fontWeight: 800, color: "var(--text-primary)", marginBottom: "4px" }}>
          Autonomous Pipeline Studio
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "14px" }}>
          Upload tabular datasets, configure validation gates, and initiate learned RL preprocessing with Bayesian HPO and stacked ensembles.
        </p>
      </div>

      <DatasetUpload
        dataset={dataset}
        filename={filename}
        onDatasetLoaded={handleDatasetLoaded}
      />

      {dataset && (
        <>
          <GatePrecheck dataset={dataset} targetColumn={config.target_column} />

          <ConfigPanel
            columns={dataset.columns}
            config={config}
            onChange={setConfig}
          />

          <div style={{ display: "flex", justifyContent: "flex-end", marginTop: "8px" }}>
            <Button
              variant="primary"
              disabled={isSubmitting || !config.target_column}
              onClick={handleSubmit}
              style={{ padding: "12px 28px", fontSize: "15px" }}
            >
              {isSubmitting ? "Submitting Pipeline..." : "🚀 Launch Autonomous Pipeline"}
            </Button>
          </div>
        </>
      )}
    </div>
  );
};
