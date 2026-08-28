import React from "react";

interface MetricStatProps {
  label: string;
  value: string | number;
  subValue?: string;
  delta?: { value: string | number; positive?: boolean };
  className?: string;
  style?: React.CSSProperties;
}

export const MetricStat: React.FC<MetricStatProps> = ({
  label,
  value,
  subValue,
  delta,
  className = "",
  style,
}) => {
  return (
    <div className={`metric-stat-card ${className}`} style={style}>
      <div className="metric-stat-label">{label}</div>
      <div style={{ display: "flex", alignItems: "baseline", gap: "8px" }}>
        <div className="metric-stat-value">{value}</div>
        {delta && (
          <span
            style={{
              fontSize: "12px",
              fontWeight: 700,
              color: delta.positive ? "var(--accent-emerald)" : "var(--accent-rose)",
            }}
          >
            {delta.positive ? "▲" : "▼"} {delta.value}
          </span>
        )}
      </div>
      {subValue && <div className="metric-stat-sub">{subValue}</div>}
    </div>
  );
};
