import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ResultsDashboard } from "./ResultsDashboard";
import { SAMPLE_RUN_CLASSIFICATION } from "../../api/mockData";

describe("ResultsDashboard", () => {
  it("renders run header, lineage graph, tabs, and bounded summary", () => {
    render(
      <ResultsDashboard
        run={SAMPLE_RUN_CLASSIFICATION}
        onReplayRun={vi.fn()}
      />
    );

    expect(screen.getByText(/Run Results: iris_multivariate_drift.csv/)).toBeDefined();
    expect(screen.getByText("Bounded Reasoning Natural-Language Report")).toBeDefined();
    expect(screen.getByText("Verifiable Assertions & Confidence Badges")).toBeDefined();
  });

  it("switches to Validation Gates tab when clicked", () => {
    render(
      <ResultsDashboard
        run={SAMPLE_RUN_CLASSIFICATION}
        onReplayRun={vi.fn()}
      />
    );

    const gatesTab = screen.getByText("Validation Gates");
    fireEvent.click(gatesTab);

    expect(screen.getByText("Fail-Closed Validation Gate Evaluations")).toBeDefined();
    expect(screen.getByText("MaxNullRateGate")).toBeDefined();
  });

  it("switches to RL Preprocessing tab and renders meta-features", () => {
    render(
      <ResultsDashboard
        run={SAMPLE_RUN_CLASSIFICATION}
        onReplayRun={vi.fn()}
      />
    );

    const rlTab = screen.getByText("RL Preprocessing");
    fireEvent.click(rlTab);

    expect(screen.getByText("Dataset Meta-Features State (s)")).toBeDefined();
    expect(screen.getByText("Q-Learning Action Chosen (a)")).toBeDefined();
  });
});
