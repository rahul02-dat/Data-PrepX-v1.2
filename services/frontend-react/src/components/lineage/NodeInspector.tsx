import React from "react";
import { LineageNode } from "../../api/types";
import { Badge } from "../common/Badge";

interface NodeInspectorProps {
  node: LineageNode | null;
  onClose: () => void;
}

export const NodeInspector: React.FC<NodeInspectorProps> = ({ node, onClose }) => {
  if (!node) return null;

  return (
    <div
      style={{
        background: "rgba(15, 23, 42, 0.95)",
        borderLeft: "1px solid var(--border-strong)",
        padding: "20px",
        width: "340px",
        display: "flex",
        flexDirection: "column",
        gap: "16px",
        fontSize: "13px",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h4 style={{ fontSize: "14px", fontWeight: 700, color: "var(--text-primary)" }}>
          Node Inspector
        </h4>
        <button
          onClick={onClose}
          style={{ background: "none", border: "none", color: "var(--text-muted)", cursor: "pointer", fontSize: "16px" }}
        >
          ✕
        </button>
      </div>

      <div>
        <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
          Stage Label
        </div>
        <div style={{ fontSize: "15px", fontWeight: 600, color: "var(--text-primary)", marginTop: "2px" }}>
          {node.label}
        </div>
      </div>

      <div>
        <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
          Type & Status
        </div>
        <div style={{ display: "flex", gap: "8px", marginTop: "4px" }}>
          <Badge variant="blue">{node.type}</Badge>
          <Badge variant={node.status === "completed" ? "green" : "yellow"}>
            {node.status}
          </Badge>
        </div>
      </div>

      <div>
        <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
          Content-Addressed SHA-256 Hash
        </div>
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "11px",
            background: "#080c14",
            padding: "8px",
            borderRadius: "6px",
            wordBreak: "break-all",
            marginTop: "4px",
            border: "1px solid var(--border-subtle)",
            color: "var(--accent-cyan)",
          }}
        >
          {node.content_hash}
        </div>
      </div>

      <div>
        <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
          Execution Metadata
        </div>
        <pre
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "11px",
            background: "#080c14",
            padding: "8px",
            borderRadius: "6px",
            marginTop: "4px",
            overflowX: "auto",
            border: "1px solid var(--border-subtle)",
            color: "var(--text-secondary)",
          }}
        >
          {JSON.stringify(node.metadata, null, 2)}
        </pre>
      </div>

      <div style={{ fontSize: "11px", color: "var(--text-muted)" }}>
        Executed at: {node.timestamp}
      </div>
    </div>
  );
};
