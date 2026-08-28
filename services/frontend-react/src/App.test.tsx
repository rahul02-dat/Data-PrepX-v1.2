import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import App from "./App";

describe("App", () => {
  it("renders DataPrepX v2 branding and header navigation", () => {
    render(<App />);
    expect(screen.getByText("DataPrepX v2")).toBeDefined();
    expect(screen.getAllByText(/Pipeline Studio/).length).toBeGreaterThan(0);
    expect(screen.getByText(/Experiments/)).toBeDefined();
  });

  it("navigates to Experiments page when clicked", () => {
    render(<App />);
    const expNav = screen.getByText(/Experiments/);
    fireEvent.click(expNav);

    expect(screen.getByText("Lineage & Experiment Registry")).toBeDefined();
    expect(screen.getByText("iris_multivariate_drift.csv")).toBeDefined();
  });
});
