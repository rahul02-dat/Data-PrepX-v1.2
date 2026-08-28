import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { JobTracker } from "./JobTracker";

describe("JobTracker", () => {
  it("renders stages stepper and radial progress ring", () => {
    render(<JobTracker jobId="test-run-123" onViewResults={vi.fn()} />);

    expect(screen.getByText("Live Job Execution Tracker")).toBeDefined();
    expect(screen.getByText(/Run ID: test-run-123/)).toBeDefined();
    expect(screen.getByText("Validation Gates")).toBeDefined();
    expect(screen.getByText("RL & Optuna HPO")).toBeDefined();
  });
});
