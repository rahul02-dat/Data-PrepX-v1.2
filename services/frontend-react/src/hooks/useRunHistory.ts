import { useEffect, useState } from "react";
import { SAMPLE_RUN_CLASSIFICATION, SAMPLE_RUN_REGRESSION } from "../api/mockData";
import { RunDetail } from "../api/types";

const STORAGE_KEY = "dataprepx_runs_history_v1";

export function useRunHistory() {
  const [runs, setRuns] = useState<RunDetail[]>(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        return JSON.parse(stored);
      }
    } catch {
      // ignore
    }
    return [SAMPLE_RUN_CLASSIFICATION, SAMPLE_RUN_REGRESSION];
  });

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(runs));
    } catch {
      // ignore storage error
    }
  }, [runs]);

  const addRun = (run: RunDetail) => {
    setRuns((prev) => [run, ...prev.filter((r) => r.id !== run.id)]);
  };

  const getRun = (id: string): RunDetail | undefined => {
    return runs.find((r) => r.id === id);
  };

  const clearHistory = () => {
    setRuns([SAMPLE_RUN_CLASSIFICATION, SAMPLE_RUN_REGRESSION]);
  };

  return { runs, addRun, getRun, clearHistory };
}
