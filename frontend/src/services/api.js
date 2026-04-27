import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const client = axios.create({
  baseURL: API_BASE,
  timeout: 120000
});

export async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const { data } = await client.post("/api/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" }
  });
  return data;
}

export async function askQuestion(question) {
  const { data } = await client.post("/api/query", { question });
  return data;
}

export async function healthCheck() {
  const { data } = await client.get("/health");
  return data;
}
