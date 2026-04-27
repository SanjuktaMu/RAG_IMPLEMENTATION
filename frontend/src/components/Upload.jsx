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
      const successMessage = result?.message || "Upload and processing complete.";
      setMessage(successMessage);
      onUploaded?.(result);
    } catch (error) {
      const errMessage = error?.response?.data?.detail || error.message || "Upload failed";
      setMessage(errMessage);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="card">
      <h2>Upload PDF</h2>
      <input
        type="file"
        accept=".pdf,application/pdf"
        onChange={(event) => setSelectedFile(event.target.files?.[0] || null)}
      />
      <button onClick={handleUpload} disabled={loading}>
        {loading ? "Processing..." : "Upload & Process"}
      </button>
      {message && <p className="status">{message}</p>}
    </section>
  );
}
