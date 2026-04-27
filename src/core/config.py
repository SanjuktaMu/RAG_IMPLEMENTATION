import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths
DATA_PATH = os.path.join(BASE_DIR, "data")
PDF_PATH = os.path.join(DATA_PATH, "BTL AR SS 09-10-2025 (c2c).pdf")
DB_DIR = os.path.join(BASE_DIR, "db")
EVAL_DIR = os.path.join(BASE_DIR, "final_evaluation")

# LLM (REMOTE - OLLAMA)
MODEL_NAME = "qwen2.5-coder:32b"
OLLAMA_BASE_URL = "http://192.168.19.21:11434"

# Embeddings (LOCAL)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking (IMPORTANT)
CHUNK_SIZE = 600
CHUNK_OVERLAP = 150

# Retrieval
TOP_K = 4
SCORE_THRESHOLD = 0.3   # use only if threshold retrieval enabled

# LLM behavior
TEMPERATURE = 0

# Vector DB
COLLECTION_NAME = "rag_pdf_collection"