import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { RunComparator } from "./RunComparator";
import { SAMPLE_RUN_CLASSIFICATION, SAMPLE_RUN_REGRESSION } from "../../api/mockData";

describe("RunComparator", () => {
  it("renders side-by-side run attributes and back button", () => {
    render(
      <RunComparator
        runA={SAMPLE_RUN_CLASSIFICATION}
        runB={SAMPLE_RUN_REGRESSION}
        onBack={vi.fn()}
      />
    );

    expect(screen.getByText("Side-by-Side Run & Ablation Comparator")).toBeDefined();
    expect(screen.getByText("Run A")).toBeDefined();
    expect(screen.getByText("Run B")).toBeDefined();
    expect(screen.getByText("iris_multivariate_drift.csv")).toBeDefined();
    expect(screen.getByText("california_housing_sample.csv")).toBeDefined();
  });
});
