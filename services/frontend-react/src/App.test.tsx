import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import App from "./App";

describe("App", () => {
  it("renders the DataPrepX heading", () => {
    render(<App />);
    expect(screen.getByText("DataPrepX v2")).toBeDefined();
  });

  it("renders an initial checking status before the health fetch resolves", () => {
    render(<App />);
    expect(screen.getByText(/Gateway status:/)).toBeDefined();
  });
});
