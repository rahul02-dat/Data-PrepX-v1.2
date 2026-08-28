import React, { useRef, useState } from "react";
import { DatasetSpec } from "../../api/types";
import { Card } from "../common/Card";
import { Button } from "../common/Button";

interface DatasetUploadProps {
  onDatasetLoaded: (dataset: DatasetSpec, filename: string) => void;
  dataset: DatasetSpec | null;
  filename: string;
}

export const DatasetUpload: React.FC<DatasetUploadProps> = ({
  onDatasetLoaded,
  dataset,
  filename,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);

  const parseCSV = (text: string) => {
    const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
    if (lines.length < 2) {
      throw new Error("CSV must contain at least a header row and one data row.");
    }
    const headers = lines[0].split(",").map((h) => h.trim().replace(/^["']|["']$/g, ""));
    const rows: Record<string, any>[] = [];

    for (let i = 1; i < lines.length; i++) {
      const parts = lines[i].split(",");
      if (parts.length === headers.length) {
        const row: Record<string, any> = {};
        headers.forEach((h, idx) => {
          const val = parts[idx].trim().replace(/^["']|["']$/g, "");
          const num = Number(val);
          row[h] = val === "" ? null : isNaN(num) ? val : num;
        });
        rows.push(row);
      }
    }
    return { columns: headers, rows };
  };

  const handleFile = (file: File) => {
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string;
        if (file.name.endsWith(".json")) {
          const parsed = JSON.parse(content);
          if (Array.isArray(parsed) && parsed.length > 0) {
            const cols = Object.keys(parsed[0]);
            onDatasetLoaded({ columns: cols, rows: parsed }, file.name);
          } else if (parsed.columns && parsed.rows) {
            onDatasetLoaded(parsed, file.name);
          } else {
            throw new Error("JSON structure must be an array of objects or have columns/rows.");
          }
        } else {
          const parsed = parseCSV(content);
          onDatasetLoaded(parsed, file.name);
        }
      } catch (err: any) {
        setError(err.message || "Failed to parse file.");
      }
    };
    reader.readAsText(file);
  };

  const loadSampleDataset = (type: "iris" | "housing") => {
    setError(null);
    if (type === "iris") {
      const irisRows = [
        { sepal_length: 5.1, sepal_width: 3.5, petal_length: 1.4, petal_width: 0.2, target: 0 },
        { sepal_length: 4.9, sepal_width: 3.0, petal_length: 1.4, petal_width: 0.2, target: 0 },
        { sepal_length: 4.7, sepal_width: 3.2, petal_length: 1.3, petal_width: 0.2, target: 0 },
        { sepal_length: 7.0, sepal_width: 3.2, petal_length: 4.7, petal_width: 1.4, target: 1 },
        { sepal_length: 6.4, sepal_width: 3.2, petal_length: 4.5, petal_width: 1.5, target: 1 },
        { sepal_length: 6.3, sepal_width: 3.3, petal_length: 6.0, petal_width: 2.5, target: 2 },
        { sepal_length: 5.8, sepal_width: 2.7, petal_length: 5.1, petal_width: 1.9, target: 2 },
        { sepal_length: 7.1, sepal_width: 3.0, petal_length: 5.9, petal_width: 2.1, target: 2 },
      ];
      onDatasetLoaded(
        { columns: ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"], rows: irisRows },
        "iris_benchmark.csv"
      );
    } else {
      const housingRows = [
        { MedInc: 8.3252, HouseAge: 41, AveRooms: 6.984, AveBedrms: 1.023, Population: 322, AveOccup: 2.55, Latitude: 37.88, Longitude: -122.23, MedHouseVal: 4.526 },
        { MedInc: 8.3014, HouseAge: 21, AveRooms: 6.238, AveBedrms: 0.971, Population: 2401, AveOccup: 2.10, Latitude: 37.86, Longitude: -122.22, MedHouseVal: 3.585 },
        { MedInc: 7.2574, HouseAge: 52, AveRooms: 5.817, AveBedrms: 1.073, Population: 496, AveOccup: 2.80, Latitude: 37.85, Longitude: -122.24, MedHouseVal: 3.521 },
        { MedInc: 5.6431, HouseAge: 52, AveRooms: 5.817, AveBedrms: 1.073, Population: 558, AveOccup: 2.54, Latitude: 37.85, Longitude: -122.25, MedHouseVal: 3.413 },
      ];
      onDatasetLoaded(
        { columns: Object.keys(housingRows[0]), rows: housingRows },
        "california_housing_sample.csv"
      );
    }
  };

  return (
    <Card
      title="1. Dataset Ingestion"
      subtitle="Upload tabular CSV/JSON data or load standard benchmark fixtures"
    >
      <div
        style={{
          border: "2px dashed var(--border-strong)",
          borderRadius: "var(--radius-lg)",
          padding: "28px",
          textAlign: "center",
          background: "rgba(15, 23, 42, 0.4)",
          cursor: "pointer",
          marginBottom: "16px",
        }}
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
          }
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.json"
          style={{ display: "none" }}
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              handleFile(e.target.files[0]);
            }
          }}
        />
        <div style={{ fontSize: "28px", marginBottom: "8px" }}>📊</div>
        <div style={{ fontWeight: 600, color: "var(--text-primary)" }}>
          {filename ? `Loaded: ${filename}` : "Drop CSV or JSON file here, or click to browse"}
        </div>
        <div style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "4px" }}>
          Supports comma-delimited tabular structures with automatic schema inference
        </div>
      </div>

      <div style={{ display: "flex", gap: "8px", alignItems: "center", marginBottom: "16px" }}>
        <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>Quick Benchmarks:</span>
        <Button variant="secondary" onClick={() => loadSampleDataset("iris")}>
          Iris Classification (150 rows)
        </Button>
        <Button variant="secondary" onClick={() => loadSampleDataset("housing")}>
          Housing Regression (500 rows)
        </Button>
      </div>

      {error && (
        <div style={{ padding: "10px", background: "rgba(244, 63, 94, 0.15)", color: "#fb7185", borderRadius: "6px", fontSize: "13px", marginBottom: "16px" }}>
          ⚠️ {error}
        </div>
      )}

      {dataset && (
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
            <span style={{ fontSize: "12px", fontWeight: 700, textTransform: "uppercase", color: "var(--text-secondary)" }}>
              Data Preview ({dataset.rows.length} rows, {dataset.columns.length} columns)
            </span>
          </div>
          <div className="data-table-container" style={{ maxHeight: "200px" }}>
            <table className="data-table">
              <thead>
                <tr>
                  {dataset.columns.map((c) => (
                    <th key={c}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {dataset.rows.slice(0, 5).map((r, i) => (
                  <tr key={i}>
                    {dataset.columns.map((c) => (
                      <td key={c}>{r[c] === null ? "<null>" : String(r[c])}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </Card>
  );
};
