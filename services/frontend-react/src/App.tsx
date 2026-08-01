// Phase 0 stub. Real pages (dataset upload, live job progress, lineage
// viewer, RL/Optuna/MAML dashboards, confidence-annotated summary) land in
// Phase 9 per docs/01_IMPLEMENTATION_PLANNER.md. This component only proves
// the frontend builds and can reach the gateway's /healthz endpoint.
import { useEffect, useState } from "react";

type HealthState = "checking" | "ok" | "unreachable";

export default function App() {
  const [health, setHealth] = useState<HealthState>("checking");

  useEffect(() => {
    const gatewayUrl = import.meta.env.VITE_GATEWAY_URL ?? "http://localhost:8080";
    fetch(`${gatewayUrl}/healthz`)
      .then((res) => setHealth(res.ok ? "ok" : "unreachable"))
      .catch(() => setHealth("unreachable"));
  }, []);

  return (
    <main>
      <h1>DataPrepX v2</h1>
      <p>Gateway status: {health}</p>
    </main>
  );
}
