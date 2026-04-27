import { useState } from "react";
import { uploadFile } from "../services/api";

export default function Upload({ onUploaded }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  async function handleUpload() {
    if (!selectedFile) {
      setMessage("Please select a PDF first.");
      return;
    }

    setLoading(true);
    setMessage("");

    try {
      const result = await uploadFile(selectedFile);
      setMessage("✅ Document processed successfully!");
      onUploaded?.(result);
    } catch (error) {
      const errMessage =
        error?.response?.data?.detail || error.message || "Upload failed";
      setMessage(errMessage);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="card upload-card">
      <h2>📤 Upload Document</h2>

      <div className="upload-row">
        <input
          type="file"
          accept=".pdf,application/pdf"
          onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
        />

        <button onClick={handleUpload} disabled={loading}>
          {loading ? "Processing..." : "Upload & Analyze"}
        </button>
      </div>

      {message && <p className="status">{message}</p>}
    </section>
  );
}