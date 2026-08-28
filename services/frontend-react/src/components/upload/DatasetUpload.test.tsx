import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DatasetUpload } from "./DatasetUpload";

describe("DatasetUpload", () => {
  it("renders upload dropzone and quick benchmark buttons", () => {
    const handleLoaded = vi.fn();
    render(
      <DatasetUpload
        dataset={null}
        filename=""
        onDatasetLoaded={handleLoaded}
      />
    );

    expect(screen.getByText("1. Dataset Ingestion")).toBeDefined();
    expect(screen.getByText(/Drop CSV or JSON file here/)).toBeDefined();
    expect(screen.getByText("Iris Classification (150 rows)")).toBeDefined();
  });

  it("loads iris benchmark dataset when clicked", () => {
    const handleLoaded = vi.fn();
    render(
      <DatasetUpload
        dataset={null}
        filename=""
        onDatasetLoaded={handleLoaded}
      />
    );

    const btn = screen.getByText("Iris Classification (150 rows)");
    fireEvent.click(btn);

    expect(handleLoaded).toHaveBeenCalled();
    const args = handleLoaded.mock.calls[0];
    expect(args[1]).toBe("iris_benchmark.csv");
    expect(args[0].columns).toContain("target");
  });

  it("displays preview table when dataset is provided", () => {
    const sampleDataset = {
      columns: ["col1", "col2"],
      rows: [{ col1: "valA", col2: 10 }],
    };
    render(
      <DatasetUpload
        dataset={sampleDataset}
        filename="test.csv"
        onDatasetLoaded={vi.fn()}
      />
    );

    expect(screen.getByText(/Data Preview/)).toBeDefined();
    expect(screen.getByText("valA")).toBeDefined();
  });
});
