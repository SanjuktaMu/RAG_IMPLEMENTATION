import { useState } from "react";
import { askQuestion } from "../services/api";
import Message from "./Message";

export default function Chat() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState([]);
  const [latestResult, setLatestResult] = useState(null);

  async function handleAsk() {
    const cleanQuestion = question.trim();
    if (!cleanQuestion) return;

    setLoading(true);
    setMessages((prev) => [...prev, { role: "user", content: cleanQuestion }]);
    setQuestion("");

    try {
      const result = await askQuestion(cleanQuestion);
      setLatestResult(result);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: result.answer || "Not found in document" },
      ]);
    } catch (error) {
      const errMessage =
        error?.response?.data?.detail || error.message || "Query failed";
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${errMessage}` },
      ]);
      setLatestResult(null);
    } finally {
      setLoading(false);
    }
  }

  function onEnter(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  }

  return (
    <section className="card chat-card">
      <h2>💬 Ask Questions</h2>

      <div className="chat-box">
        {messages.length === 0 && (
          <p className="muted">
            Upload a document and start asking questions ✨
          </p>
        )}

        {messages.map((msg, i) => (
          <Message key={i} role={msg.role} content={msg.content} />
        ))}
      </div>

      <div className="chat-input-row">
        <textarea
          value={question}
          placeholder="Type your question here..."
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={onEnter}
          rows={2}
        />
        <button onClick={handleAsk} disabled={loading}>
          {loading ? "Thinking..." : "Ask"}
        </button>
      </div>

      {latestResult && (
        <div className="result-panel">
          <h3>📊 Retrieved Evidence</h3>
          {(latestResult.context || []).slice(0, 5).map((chunk, idx) => (
            <details key={idx}>
              <summary>Chunk {idx + 1}</summary>
              <pre>{chunk}</pre>
            </details>
          ))}
        </div>
      )}
    </section>
  );
}