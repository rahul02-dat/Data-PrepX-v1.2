import React from "react";

interface ProgressRingProps {
  size?: number;
  strokeWidth?: number;
  progress: number; // 0 to 100
  status: string;
}

export const ProgressRing: React.FC<ProgressRingProps> = ({
  size = 120,
  strokeWidth = 8,
  progress,
  status,
}) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (progress / 100) * circumference;

  const isDone = status === "done";
  const isFailed = status === "failed";
  const color = isDone
    ? "var(--accent-emerald)"
    : isFailed
    ? "var(--accent-rose)"
    : "var(--accent-primary)";

  return (
    <div style={{ position: "relative", width: size, height: size, margin: "0 auto" }}>
      <svg width={size} height={size}>
        <circle
          stroke="rgba(255, 255, 255, 0.08)"
          fill="transparent"
          strokeWidth={strokeWidth}
          r={radius}
          cx={size / 2}
          cy={size / 2}
        />
        <circle
          stroke={color}
          fill="transparent"
          strokeWidth={strokeWidth}
          strokeDasharray={`${circumference} ${circumference}`}
          style={{
            strokeDashoffset: offset,
            transition: "stroke-dashoffset 0.5s ease-in-out, stroke 0.3s ease",
            transform: "rotate(-90deg)",
            transformOrigin: "50% 50%",
          }}
          strokeLinecap="round"
          r={radius}
          cx={size / 2}
          cy={size / 2}
        />
      </svg>
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <span style={{ fontSize: "20px", fontWeight: 800, fontFamily: "var(--font-mono)", color }}>
          {Math.round(progress)}%
        </span>
        <span style={{ fontSize: "11px", textTransform: "uppercase", color: "var(--text-muted)", fontWeight: 600 }}>
          {status}
        </span>
      </div>
    </div>
  );
};
