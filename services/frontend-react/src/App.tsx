import { useEffect, useState } from "react";

type HealthState = "checking" | "ok" | "unreachable";

// Main application entry component
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
