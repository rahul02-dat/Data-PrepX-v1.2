import React from "react";
import { SocketLog } from "../../hooks/useJobSocket";
import { Card } from "../common/Card";

interface LiveLogFeedProps {
  logs: SocketLog[];
  isConnected: boolean;
}

export const LiveLogFeed: React.FC<LiveLogFeedProps> = ({ logs, isConnected }) => {
  return (
    <Card
      title="Live Execution Stream"
      subtitle={isConnected ? "Streaming via WebSocket gateway" : "Polling & task execution log"}
      action={
        <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px" }}>
          <span
            style={{
              width: "8px",
              height: "8px",
              borderRadius: "50%",
              backgroundColor: isConnected ? "var(--accent-emerald)" : "var(--accent-amber)",
              boxShadow: isConnected ? "0 0 8px rgba(16, 185, 129, 0.8)" : "none",
            }}
          />
          <span style={{ color: isConnected ? "var(--accent-emerald)" : "var(--text-muted)" }}>
            {isConnected ? "WS Connected" : "Sync Active"}
          </span>
        </div>
      }
    >
      <div
        style={{
          background: "#080c14",
          border: "1px solid var(--border-subtle)",
          borderRadius: "var(--radius-md)",
          padding: "12px",
          height: "220px",
          overflowY: "auto",
          fontFamily: "var(--font-mono)",
          fontSize: "12px",
          display: "flex",
          flexDirection: "column",
          gap: "6px",
        }}
      >
        {logs.length === 0 ? (
          <div style={{ color: "var(--text-muted)", margin: "auto" }}>Waiting for task graph updates...</div>
        ) : (
          logs.map((log, i) => (
            <div key={i} style={{ display: "flex", gap: "8px", lineHeight: "1.4" }}>
              <span style={{ color: "var(--text-muted)" }}>[{log.timestamp}]</span>
              <span style={{ color: "var(--accent-cyan)", fontWeight: 600 }}>[{log.stage}]</span>
              <span style={{ color: log.status === "failed" ? "var(--accent-rose)" : "var(--text-primary)" }}>
                {log.message}
              </span>
            </div>
          ))
        )}
      </div>
    </Card>
  );
};
