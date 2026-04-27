# 🚀 RAG Benchmarking & Agentic Multimodal RAG System (12 Variants)

🚀 Built with 12 RAG variants | Multimodal | Agentic | Local LLM | Full-stack deployment

---

## 🧠 Objective

This project implements and evaluates **12 different Retrieval-Augmented Generation (RAG) techniques** using a **local LLM setup**, and integrates them into a **full-stack application (FastAPI + React)**.

The goal is to understand:

- 📌 Retrieval Quality  
- 📌 Answer Accuracy  
- 📌 Faithfulness (hallucination reduction)  
- 📌 Latency & performance trade-offs  

---

## 🏗️ Full-Stack Architecture


React Frontend → FastAPI Backend → RAG Engine (src/) → LLM + Vector DB


---

## 📁 Project Structure


RAG_PROJECT/
│
├── backend/ # FastAPI API (upload + query)
├── frontend/ # React (Vite) UI
├── src/ # Core RAG engine
│ ├── core/ # Shared components
│ ├── rag_types/ # All RAG implementations
│ ├── pipelines/ # Execution scripts
│ └── evaluation/ # Evaluation logic
│
├── data/ # Input & uploaded PDFs
├── db/ # Chroma vector store
├── final_evaluation/ # Results + scores
├── requirements.txt
└── README.md


---

## 📊 Implemented RAG Techniques (12 Variants)

### 🔹 Core
- Simple RAG  

### 🔹 Query Enhancement
- HyDE RAG  
- Multi-Query RAG  

### 🔹 Retrieval Optimization
- Fusion RAG (Hybrid search)  
- Re-ranking RAG  

### 🔹 Context-Aware Systems
- Contextual RAG  
- Self RAG  

### 🔹 Advanced Reasoning
- Agentic RAG  
- Adaptive RAG  

### 🔹 Specialized Architectures
- Graph RAG  
- Memo RAG  

### 🔹 Multimodal Capability
- Multimodal RAG (text + tables + images)

---

## ⚙️ Tech Stack

### 🔹 Backend
- FastAPI  
- LangChain  
- ChromaDB  

### 🔹 Frontend
- React (Vite)  

### 🔹 Models
- Ollama (local LLM)
- qwen2.5 (generation)
- all-MiniLM-L6-v2 / nomic-embed-text (embeddings)

### 🔹 Document Processing
- Unstructured  
- PyMuPDF  
- Camelot (tables)  
- BLIP (image captions)

---

## 🚀 Run Full-Stack Application

### 🟦 Backend

```bash
uvicorn backend.main:app --reload --port 8000
🟩 Frontend
cd frontend
npm install
npm run dev

👉 Open UI:

http://localhost:5173
📥 How It Works
1. Upload PDF
Extracts:
Text
Tables
Images
2. Processing Pipeline
PDF → Chunking → Embedding → Vector DB
3. Query Flow
User Query → Agentic Routing → Retrieval → Reranking → LLM → Answer
📊 Evaluation Framework
Metrics Used
✅ Faithfulness
✅ Answer Relevancy
✅ Context Precision
✅ Latency
Run Evaluation
python src/evaluation/ragas_eval.py
🧠 Key Insights
🔥 Better retrieval ≠ better answers
🔥 Context quality matters more than quantity
🔥 Advanced RAG improves reasoning but increases latency
🔥 Multimodal handling improves real-world performance
🚀 Final System

This project culminates in:

Agentic Multimodal Hybrid RAG System

✔ Combines multiple RAG strategies
✔ Handles complex PDFs (tables + images)
✔ Uses intelligent routing
✔ Fully deployed as a full-stack application
🔮 Future Improvements
Add CrossEncoder re-ranking
Improve chunking strategies
Integrate LangGraph agents
Enhance multimodal reasoning
Deploy using Docker / cloud
👩‍💻 Author

Sanjukta Mukherjee
BTech CSE | AI/ML Enthusiast

⭐ Notes

This project is designed as both:

🧠 Research system (RAG comparison)
⚙️ Engineering system (production-ready application)
