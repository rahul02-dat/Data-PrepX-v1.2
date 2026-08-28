import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { LineageDAG } from "./LineageDAG";
import { SAMPLE_RUN_CLASSIFICATION } from "../../api/mockData";

describe("LineageDAG", () => {
  it("renders DAG graph and nodes", () => {
    render(
      <LineageDAG
        lineage={SAMPLE_RUN_CLASSIFICATION.lineage}
        configHash={SAMPLE_RUN_CLASSIFICATION.config_hash}
        config={SAMPLE_RUN_CLASSIFICATION.config}
        onReplay={vi.fn()}
      />
    );

    expect(screen.getByText("Immutable Lineage DAG (Content-Addressed)")).toBeDefined();
    expect(screen.getByText("Node Inspector")).toBeDefined();
    expect(screen.getByText(/Replay Run/)).toBeDefined();
  });

  it("opens replay modal when Replay button is clicked", () => {
    render(
      <LineageDAG
        lineage={SAMPLE_RUN_CLASSIFICATION.lineage}
        configHash={SAMPLE_RUN_CLASSIFICATION.config_hash}
        config={SAMPLE_RUN_CLASSIFICATION.config}
        onReplay={vi.fn()}
      />
    );

    const replayBtn = screen.getByText(/Replay Run/);
    fireEvent.click(replayBtn);

    expect(screen.getByText("Deterministic Run Replay")).toBeDefined();
    expect(screen.getByText(/Load into Pipeline Studio/)).toBeDefined();
  });
});
