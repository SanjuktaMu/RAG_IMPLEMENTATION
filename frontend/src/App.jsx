import { useEffect, useState } from "react";
import Upload from "./components/Upload";
import Chat from "./components/Chat";
import { healthCheck } from "./services/api";

export default function App() {
  const [backendStatus, setBackendStatus] = useState("Checking...");

  useEffect(() => {
    async function check() {
      try {
        const result = await healthCheck();
        setBackendStatus(result?.status === "ok" ? "🟢 Online" : "⚠️ Unknown");
      } catch {
        setBackendStatus("🔴 Offline");
      }
    }
    check();
  }, []);

  return (
    <main className="app-shell">
      <header className="header">
        <div>
          <h1>📄 Intelligent Document Assistant</h1>
          <p className="subtitle">
            Ask questions from PDFs using advanced RAG pipelines
          </p>
        </div>
        <span className="status-badge">{backendStatus}</span>
      </header>

      <Upload />
      <Chat />
    </main>
  );
}