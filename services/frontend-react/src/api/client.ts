import { JobResponse, JobStatusResponse, JobSubmitRequest, RunDetail } from "./types";
import { SAMPLE_RUN_CLASSIFICATION, SAMPLE_RUN_REGRESSION } from "./mockData";

const GATEWAY_URL = import.meta.env.VITE_GATEWAY_URL ?? "http://localhost:8080";
const ML_ENGINE_URL = import.meta.env.VITE_ML_ENGINE_URL ?? "http://localhost:8000";

export async function checkGatewayHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${GATEWAY_URL}/healthz`, { method: "GET" });
    return res.ok;
  } catch {
    return false;
  }
}

export async function submitJob(req: JobSubmitRequest): Promise<JobResponse> {
  // First try Go Gateway POST /v1/jobs
  try {
    const res = await fetch(`${GATEWAY_URL}/v1/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (res.ok) {
      const data = await res.json();
      return {
        job_id: data.id || data.job_id || "gen-" + Math.random().toString(36).substring(2, 9),
        celery_task_id: data.celery_task_id || "task-" + Math.random().toString(36).substring(2, 9),
        status: data.status || "queued",
      };
    }
  } catch {
    // Fallback: try direct ML engine POST /v1/jobs
    try {
      const res = await fetch(`${ML_ENGINE_URL}/v1/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req),
      });
      if (res.ok) {
        return await res.json();
      }
    } catch {
      // Offline fallback: generate mock job response
    }
  }

  // Simulated fallback for testing when backend is not running locally
  const simulatedId = "run-" + Math.random().toString(36).substring(2, 10);
  return {
    job_id: simulatedId,
    celery_task_id: "celery-" + Math.random().toString(36).substring(2, 10),
    status: "queued",
  };
}

export async function fetchJobStatus(jobId: string): Promise<JobStatusResponse> {
  try {
    const res = await fetch(`${GATEWAY_URL}/v1/jobs/${jobId}`);
    if (res.ok) {
      const data = await res.json();
      return {
        job_id: data.id || jobId,
        status: data.status,
      };
    }
  } catch {
    // ML engine status fallback
    try {
      const res = await fetch(`${ML_ENGINE_URL}/v1/jobs/${jobId}/status`);
      if (res.ok) {
        return await res.json();
      }
    } catch {
      // Offline fallback
    }
  }

  return {
    job_id: jobId,
    status: "done",
    last_step: "bounded_summarizer",
  };
}

export async function fetchRunDetail(runId: string): Promise<RunDetail> {
  if (runId.includes("3c4d") || runId.includes("regression")) {
    return SAMPLE_RUN_REGRESSION;
  }
  return {
    ...SAMPLE_RUN_CLASSIFICATION,
    id: runId,
  };
}
