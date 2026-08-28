import { useEffect, useState } from "react";
import { DatasetSpec, PipelineConfig, RunDetail } from "./api/types";
import { checkGatewayHealth, submitJob } from "./api/client";
import { useRunHistory } from "./hooks/useRunHistory";
import { SAMPLE_RUN_CLASSIFICATION, SAMPLE_RUN_REGRESSION } from "./api/mockData";
import { PipelineStudio } from "./pages/PipelineStudio";
import { ActiveJobView } from "./pages/ActiveJobView";
import { RunDetailView } from "./pages/RunDetailView";
import { ExperimentsPage } from "./pages/ExperimentsPage";
import { Badge } from "./components/common/Badge";

type ViewMode = "studio" | "tracker" | "detail" | "experiments";

export default function App() {
  const [view, setView] = useState<ViewMode>("studio");
  const [gatewayOnline, setGatewayOnline] = useState<boolean | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [replayConfig, setReplayConfig] = useState<PipelineConfig | undefined>(undefined);

  const { runs, addRun } = useRunHistory();

  useEffect(() => {
    checkGatewayHealth().then((isOnline) => setGatewayOnline(isOnline));
    const interval = setInterval(() => {
      checkGatewayHealth().then((isOnline) => setGatewayOnline(isOnline));
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleSubmitJob = async (
    dataset: DatasetSpec,
    config: PipelineConfig,
    filename: string
  ) => {
    const res = await submitJob({
      dataset,
      target_column: config.target_column,
      task_type: config.task_type,
      imputation_method: config.imputation_method,
      outlier_method: config.outlier_method,
      seed: config.seed,
      n_trials: config.n_trials,
      cv_folds: config.cv_folds,
      stacking_cv_folds: config.stacking_cv_folds,
    });

    const jobId = res.job_id;
    setActiveJobId(jobId);
    setView("tracker");

    // Also persist a research run detail for this job in history
    const baseMock = config.task_type === "regression" ? SAMPLE_RUN_REGRESSION : SAMPLE_RUN_CLASSIFICATION;
    const newRun: RunDetail = {
      ...baseMock,
      id: jobId,
      dataset_name: filename,
      rows_count: dataset.rows.length,
      columns_count: dataset.columns.length,
      target_column: config.target_column,
      task_type: config.task_type,
      config,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    addRun(newRun);
  };

  const handleViewResults = (jobId: string) => {
    setSelectedRunId(jobId);
    setView("detail");
  };

  const handleSelectRunFromHistory = (runId: string) => {
    setSelectedRunId(runId);
    setView("detail");
  };

  const handleReplayRun = (run: RunDetail) => {
    setReplayConfig(run.config);
    setView("studio");
  };

  return (
    <div className="app-container">
      {/* Top Application Header */}
      <header className="app-header">
        <div className="brand-section">
          <div className="brand-logo">Δ</div>
          <span className="brand-title">DataPrepX v2</span>
          <span className="brand-badge">Research Core</span>
        </div>

        <nav className="nav-links">
          <button
            type="button"
            className={`nav-btn ${view === "studio" ? "active" : ""}`}
            onClick={() => setView("studio")}
          >
            ⚡ Pipeline Studio
          </button>

          {activeJobId && (
            <button
              type="button"
              className={`nav-btn ${view === "tracker" ? "active" : ""}`}
              onClick={() => setView("tracker")}
            >
              🔄 Active Job
            </button>
          )}

          <button
            type="button"
            className={`nav-btn ${view === "experiments" || view === "detail" ? "active" : ""}`}
            onClick={() => setView("experiments")}
          >
            📊 Experiments ({runs.length})
          </button>
        </nav>

        <div className="header-status">
          <Badge
            variant={gatewayOnline ? "green" : gatewayOnline === false ? "yellow" : "blue"}
            dot
          >
            {gatewayOnline === null
              ? "CHECKING"
              : gatewayOnline
              ? "GATEWAY: ONLINE"
              : "STANDALONE MODE"}
          </Badge>
        </div>
      </header>

      {/* Main View Area */}
      <main className="main-content">
        {view === "studio" && (
          <PipelineStudio
            onSubmitJob={handleSubmitJob}
            initialConfig={replayConfig}
          />
        )}

        {view === "tracker" && activeJobId && (
          <ActiveJobView
            jobId={activeJobId}
            onViewResults={handleViewResults}
            onCancel={() => setView("studio")}
          />
        )}

        {view === "detail" && selectedRunId && (
          <RunDetailView
            runId={selectedRunId}
            onBack={() => setView("experiments")}
            onReplayRun={handleReplayRun}
          />
        )}

        {view === "experiments" && (
          <ExperimentsPage
            runs={runs}
            onSelectRun={handleSelectRunFromHistory}
          />
        )}
      </main>
    </div>
  );
}
