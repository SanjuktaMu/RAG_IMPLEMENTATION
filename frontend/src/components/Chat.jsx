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
      setMessages((prev) => [...prev, { role: "assistant", content: result.answer || "" }]);
    } catch (error) {
      const errMessage = error?.response?.data?.detail || error.message || "Query failed";
      setMessages((prev) => [...prev, { role: "assistant", content: `Error: ${errMessage}` }]);
      setLatestResult(null);
    } finally {
      setLoading(false);
    }
  }

  function onEnter(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleAsk();
    }
  }

  return (
    <section className="card">
      <h2>Chat</h2>
      <div className="chat-box">
        {messages.length === 0 && <p className="muted">Ask a question after uploading a PDF.</p>}
        {messages.map((msg, index) => (
          <Message key={`${msg.role}-${index}`} role={msg.role} content={msg.content} />
        ))}
      </div>

      <div className="chat-input-row">
        <textarea
          value={question}
          placeholder="Ask a question about the uploaded document"
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={onEnter}
          rows={3}
        />
        <button onClick={handleAsk} disabled={loading}>
          {loading ? "Thinking..." : "Send"}
        </button>
      </div>

      {latestResult && (
        <div className="result-panel">
          <h3>Retrieved Context</h3>
          {(latestResult.context || []).map((chunk, idx) => (
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
