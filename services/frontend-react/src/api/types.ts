// Canonical TypeScript Types for DataPrepX v2

export type JobStatus =
  | "queued"
  | "running"
  | "gate-check"
  | "optimizing"
  | "done"
  | "failed";

export type TaskType = "classification" | "regression";
export type ImputationMethod = "mice" | "knn";
export type OutlierMethod = "isolation_forest" | "lof" | "none";

export interface DatasetSpec {
  rows: Record<string, any>[];
  columns: string[];
}

export interface PipelineConfig {
  task_type: TaskType;
  target_column: string;
  imputation_method: ImputationMethod;
  outlier_method: OutlierMethod;
  seed: number;
  n_trials: number;
  cv_folds: number;
  stacking_cv_folds: number;
  reference_dataset?: DatasetSpec | null;
}

export interface JobSubmitRequest {
  dataset: DatasetSpec;
  target_column: string;
  task_type: TaskType;
  imputation_method?: ImputationMethod;
  outlier_method?: OutlierMethod;
  seed?: number;
  n_trials?: number;
  cv_folds?: number;
  stacking_cv_folds?: number;
  reference_dataset?: DatasetSpec | null;
}

export interface JobResponse {
  id?: string;
  job_id: string;
  celery_task_id: string;
  status: JobStatus;
  config_hash?: string;
  created_at?: string;
  updated_at?: string;
}

export interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
  celery_task_id?: string | null;
  last_step?: string | null;
}

// Gate Results
export interface GateCheckResult {
  gate_name: string;
  passed: boolean;
  score: number;
  threshold: number;
  details: string;
  columns_flagged?: string[];
}

// RL Optimizer Results
export interface MetaFeatures {
  missing_rate: number;
  skewness: number;
  cardinality_mean: number;
  class_imbalance_ratio: number;
  drift_score: number;
}

export interface RLOptimizerResult {
  meta_features: MetaFeatures;
  state_bin: number[];
  action_chosen: {
    imputation: ImputationMethod;
    outlier: OutlierMethod;
    threshold: number;
  };
  delta_reward: number;
  baseline_metric: number;
  optimized_metric: number;
  convergence_history: { episode: number; reward: number; epsilon: number }[];
}

// Optuna Search Results
export interface OptunaTrial {
  trial_number: number;
  family: string;
  score: number;
  duration_seconds: number;
  params: Record<string, any>;
  state: "COMPLETE" | "PRUNED";
}

export interface FamilyBestModel {
  family: "random_forest" | "xgboost" | "lightgbm" | "linear";
  best_score: number;
  best_params: Record<string, any>;
  n_trials_evaluated: number;
}

// Stacking Ensemble Results
export interface StackingResult {
  task_type: TaskType;
  ensemble_score: number;
  base_family_weights: Record<string, number>;
  meta_learner_type: string;
  metric_name: string;
  metrics: {
    accuracy_or_r2: number;
    f1_or_rmse: number;
    roc_auc_or_mae?: number;
  };
}

// MAML Adaptation Timeline
export interface MAMLAdaptationStep {
  batch_id: number;
  drift_detected: boolean;
  drift_psi: number;
  adapted: boolean;
  inner_gradient_steps: number;
  pre_adaptation_score: number;
  post_adaptation_score: number;
  selected_feature_count: number;
  timestamp: string;
}

// Bounded RAG Summarizer
export type ConfidenceLevel = "HIGH" | "MODERATE" | "FLAGGED";

export interface BoundedClaim {
  id: string;
  text: string;
  verifiable_metric: string;
  computed_value: number | string;
  confidence_level: ConfidenceLevel;
  confidence_score: number;
  p_value_or_ci?: string;
  verification_passed: boolean;
}

export interface BoundedSummaryReport {
  summary_text: string;
  claims: BoundedClaim[];
  grounding_stats: Record<string, any>;
  generated_at: string;
  model_name: string;
}

// Lineage DAG Graph
export interface LineageNode {
  id: string;
  label: string;
  type: "dataset" | "gate" | "transformation" | "rl_policy" | "hpo_trial" | "ensemble" | "summary";
  content_hash: string;
  status: "completed" | "active" | "failed" | "skipped";
  metadata: Record<string, any>;
  timestamp: string;
}

export interface LineageEdge {
  from: string;
  to: string;
}

export interface LineageDAGData {
  run_id: string;
  git_sha: string;
  dataset_id: string;
  nodes: LineageNode[];
  edges: LineageEdge[];
}

// Complete Full Research Run View Model
export interface RunDetail {
  id: string;
  status: JobStatus;
  task_type: TaskType;
  target_column: string;
  dataset_name: string;
  rows_count: number;
  columns_count: number;
  config_hash: string;
  dataset_hash: string;
  git_sha: string;
  created_at: string;
  updated_at: string;
  duration_seconds: number;
  gates: GateCheckResult[];
  rl_result: RLOptimizerResult;
  optuna_trials: OptunaTrial[];
  family_best_models: FamilyBestModel[];
  stacking_result: StackingResult;
  maml_timeline: MAMLAdaptationStep[];
  bounded_summary: BoundedSummaryReport;
  lineage: LineageDAGData;
  config: PipelineConfig;
}
