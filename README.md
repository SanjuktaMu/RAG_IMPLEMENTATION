# 🔥 RAG Benchmarking System (10 RAG Variants)

This project implements and evaluates multiple Retrieval-Augmented Generation (RAG) techniques on a single dataset using a local LLM.

## Full-Stack Layout

- `backend/` - FastAPI API layer (upload + query endpoints)
- `frontend/` - React (Vite) client
- `src/` - Core RAG engine and evaluation pipelines
- `data/` - Source and uploaded PDFs
- `db/` - Chroma persisted vector store

### Run Full-Stack App

1. Backend:

```bash
uvicorn backend.main:app --reload --port 8000
```

2. Frontend:

```bash
cd frontend
npm install
npm run dev
```

3. Open UI at `http://localhost:5173`

---

## 🧠 Objective

To compare different RAG strategies based on:

- Retrieval Quality
- Answer Accuracy
- Faithfulness (no hallucination)
- Latency (performance)

---

## 📊 Implemented RAG Types

1. Simple RAG
2. HyDE RAG
3. Multi-Query RAG
4. Fusion RAG
5. Contextual RAG
6. Self-Query RAG
7. Re-ranking RAG
8. Agentic RAG
9. Graph RAG
10. Adaptive RAG

---

## 🏗️ Project Structure
rag-system/
│
├── data/ # Input PDFs
├── db/ # ChromaDB
├── evaluation/
│ ├── dataset.json # Questions + ground truth
│ ├── results/ # Outputs per RAG type
│ └── scores/ # RAGAS evaluation scores
│
├── src/
│ ├── core/ # Shared components
│ ├── rag_types/ # RAG implementations
│ ├── pipelines/ # Execution scripts
│ └── evaluation/ # Evaluation scripts
│
├── requirements.txt
└── README.md


---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt


2. Start Ollama
ollama serve
3. Pull models
ollama pull qwen2.5:3b-instruct
ollama pull nomic-embed-text
🚀 Running the Pipeline
Run Simple RAG
python src/pipelines/run_simple.py
Run HyDE RAG
python src/pipelines/run_hyde.py
Evaluate with RAGAS
python src/evaluation/ragas_eval.py
📊 Metrics Used
Faithfulness (hallucination detection)
Answer Relevancy
Context Precision
Latency
🧠 Key Insights
Better retrieval ≠ better answers
Context quality is more important than quantity
Advanced RAG improves reasoning but increases latency
🚀 Future Improvements
Add re-ranking models (CrossEncoder)
Add Streamlit dashboard
Add hybrid search (BM25 + vector)
Optimize chunking strategies
👩‍💻 Author

Sanjukta Mukherjee
BTech CSE | AI/ML Enthusiast

⭐ Notes

This project is designed as a research + engineering system to understand real-world RAG performance trade-offs.


---




