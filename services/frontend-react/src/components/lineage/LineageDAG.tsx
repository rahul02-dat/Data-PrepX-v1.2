import React, { useState } from "react";
import { LineageDAGData, LineageNode, PipelineConfig } from "../../api/types";
import { Card } from "../common/Card";
import { Button } from "../common/Button";
import { NodeInspector } from "./NodeInspector";
import { ReplayModal } from "./ReplayModal";

interface LineageDAGProps {
  lineage: LineageDAGData;
  configHash: string;
  config: PipelineConfig;
  onReplay: () => void;
}

export const LineageDAG: React.FC<LineageDAGProps> = ({
  lineage,
  configHash,
  config,
  onReplay,
}) => {
  const [selectedNode, setSelectedNode] = useState<LineageNode | null>(lineage.nodes[0] || null);
  const [isReplayOpen, setIsReplayOpen] = useState(false);

  // Position nodes horizontally
  const nodeWidth = 180;
  const nodeHeight = 64;
  const gapX = 60;
  const startX = 30;
  const startY = 40;

  const totalWidth = lineage.nodes.length * (nodeWidth + gapX) + 60;
  const totalHeight = 160;

  return (
    <Card
      title="Immutable Lineage DAG (Content-Addressed)"
      subtitle="Cryptographically verified DAG tracking every data, policy, and model transformation"
      action={
        <Button variant="secondary" onClick={() => setIsReplayOpen(true)}>
          🔄 Replay Run
        </Button>
      }
    >
      <div style={{ display: "flex", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius-lg)", overflow: "hidden" }}>
        {/* SVG Graph Viewport */}
        <div style={{ flex: 1, overflowX: "auto", background: "#080c14", padding: "16px" }}>
          <svg width={Math.max(totalWidth, 700)} height={totalHeight}>
            <defs>
              <marker
                id="arrowhead"
                markerWidth="8"
                markerHeight="6"
                refX="7"
                refY="3"
                orient="auto"
              >
                <polygon points="0 0, 8 3, 0 6" fill="rgba(56, 189, 248, 0.6)" />
              </marker>
            </defs>

            {/* Connecting Edges */}
            {lineage.nodes.map((_node, i) => {
              if (i === lineage.nodes.length - 1) return null;
              const x1 = startX + i * (nodeWidth + gapX) + nodeWidth;
              const y1 = startY + nodeHeight / 2;
              const x2 = startX + (i + 1) * (nodeWidth + gapX);
              const y2 = startY + nodeHeight / 2;

              return (
                <line
                  key={`edge-${i}`}
                  x1={x1}
                  y1={y1}
                  x2={x2 - 8}
                  y2={y2}
                  stroke="rgba(56, 189, 248, 0.4)"
                  strokeWidth="2"
                  strokeDasharray="4 3"
                  markerEnd="url(#arrowhead)"
                />
              );
            })}

            {/* Nodes */}
            {lineage.nodes.map((node, i) => {
              const x = startX + i * (nodeWidth + gapX);
              const y = startY;
              const isSelected = selectedNode?.id === node.id;

              return (
                <g
                  key={node.id}
                  transform={`translate(${x}, ${y})`}
                  onClick={() => setSelectedNode(node)}
                  style={{ cursor: "pointer" }}
                >
                  <rect
                    width={nodeWidth}
                    height={nodeHeight}
                    rx="8"
                    fill={isSelected ? "rgba(30, 58, 138, 0.5)" : "rgba(15, 23, 42, 0.8)"}
                    stroke={isSelected ? "var(--accent-primary)" : "rgba(255, 255, 255, 0.15)"}
                    strokeWidth={isSelected ? 2 : 1}
                  />
                  <text
                    x="12"
                    y="22"
                    fill="var(--accent-cyan)"
                    fontSize="10"
                    fontWeight="700"
                    fontFamily="var(--font-sans)"
                    style={{ textTransform: "uppercase" }}
                  >
                    {node.type}
                  </text>
                  <text
                    x="12"
                    y="38"
                    fill="#f8fafc"
                    fontSize="12"
                    fontWeight="600"
                    fontFamily="var(--font-sans)"
                  >
                    {node.label.length > 20 ? node.label.substring(0, 18) + "..." : node.label}
                  </text>
                  <text
                    x="12"
                    y="52"
                    fill="var(--text-muted)"
                    fontSize="9"
                    fontFamily="var(--font-mono)"
                  >
                    {node.content_hash.substring(0, 16)}...
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* Side Node Inspector */}
        <NodeInspector node={selectedNode} onClose={() => setSelectedNode(null)} />
      </div>

      <ReplayModal
        isOpen={isReplayOpen}
        onClose={() => setIsReplayOpen(false)}
        runId={lineage.run_id}
        configHash={configHash}
        gitSha={lineage.git_sha}
        config={config}
        onConfirmReplay={onReplay}
      />
    </Card>
  );
};
