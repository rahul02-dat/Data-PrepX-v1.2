import React from "react";
import { PipelineConfig } from "../../api/types";
import { Modal } from "../common/Modal";
import { Button } from "../common/Button";

interface ReplayModalProps {
  isOpen: boolean;
  onClose: () => void;
  runId: string;
  configHash: string;
  gitSha: string;
  config: PipelineConfig;
  onConfirmReplay: () => void;
}

export const ReplayModal: React.FC<ReplayModalProps> = ({
  isOpen,
  onClose,
  runId,
  configHash,
  gitSha,
  config,
  onConfirmReplay,
}) => {
  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Deterministic Run Replay">
      <div style={{ display: "flex", flexDirection: "column", gap: "16px", fontSize: "13px" }}>
        <p style={{ color: "var(--text-secondary)" }}>
          DataPrepX content-addressing guarantees deterministic re-execution of this exact pipeline configuration.
        </p>

        <div style={{ background: "rgba(15, 23, 42, 0.6)", padding: "12px", borderRadius: "var(--radius-md)", border: "1px solid var(--border-subtle)" }}>
          <div style={{ display: "grid", gridTemplateColumns: "140px 1fr", gap: "8px", fontFamily: "var(--font-mono)", fontSize: "12px" }}>
            <span style={{ color: "var(--text-muted)" }}>Run ID:</span>
            <span style={{ color: "var(--text-primary)" }}>{runId}</span>

            <span style={{ color: "var(--text-muted)" }}>Config Hash:</span>
            <span style={{ color: "var(--accent-cyan)", wordBreak: "break-all" }}>{configHash}</span>

            <span style={{ color: "var(--text-muted)" }}>Git Commit SHA:</span>
            <span style={{ color: "var(--accent-emerald)" }}>{gitSha}</span>
          </div>
        </div>

        <div>
          <h4 style={{ fontSize: "12px", fontWeight: 700, textTransform: "uppercase", color: "var(--text-muted)", marginBottom: "6px" }}>
            Replay Hyperparameters
          </h4>
          <pre style={{ background: "#080c14", padding: "10px", borderRadius: "6px", fontSize: "11px", color: "var(--text-secondary)", border: "1px solid var(--border-subtle)" }}>
            {JSON.stringify(config, null, 2)}
          </pre>
        </div>

        <div style={{ display: "flex", justifyContent: "flex-end", gap: "10px", marginTop: "8px" }}>
          <Button variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="primary"
            onClick={() => {
              onClose();
              onConfirmReplay();
            }}
          >
            Load into Pipeline Studio →
          </Button>
        </div>
      </div>
    </Modal>
  );
};
