import { useEffect, useState } from "react";
import Upload from "./components/Upload";
import Chat from "./components/Chat";
import { healthCheck } from "./services/api";

export default function App() {
  const [backendStatus, setBackendStatus] = useState("Checking backend...");

  useEffect(() => {
    let mounted = true;

    async function check() {
      try {
        const result = await healthCheck();
        if (mounted) {
          setBackendStatus(result?.status === "ok" ? "Backend: online" : "Backend: unknown");
        }
      } catch {
        if (mounted) {
          setBackendStatus("Backend: offline");
        }
      }
    }

    check();
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <main className="app-shell">
      <header>
        <h1>RAG Full Stack Interface</h1>
        <p>{backendStatus}</p>
      </header>
      <Upload />
      <Chat />
    </main>
  );
}
