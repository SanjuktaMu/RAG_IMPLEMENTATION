import os
import shutil
from langchain_chroma import Chroma
from src.core.embeddings import LocalEmbedding
from src.core.config import DB_DIR, COLLECTION_NAME
from src.core.document_loader import load_documents, split_documents

MIN_EXPECTED_CHUNKS = 10


def create_vector_store():
    documents = load_documents()
    chunks = split_documents(documents)

    os.makedirs(DB_DIR, exist_ok=True)

    print("Creating embeddings + storing in ChromaDB...")

    embedding = LocalEmbedding()

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=DB_DIR,
        collection_name=COLLECTION_NAME
    )

    print("✅ Chroma DB created successfully!")

    return db


def load_vector_store():
    os.makedirs(DB_DIR, exist_ok=True)
    embedding = LocalEmbedding()

    try:
        db = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embedding,
            collection_name=COLLECTION_NAME
        )
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ Failed to load existing Chroma DB ({exc}). Rebuilding from source PDF...")
        if os.path.isdir(DB_DIR):
            shutil.rmtree(DB_DIR, ignore_errors=True)
        os.makedirs(DB_DIR, exist_ok=True)
        db = create_vector_store()

    current_count = db._collection.count()
    if current_count == 0:
        print("⚠️ Vector store is empty. Building it from the source PDF...")
        db = create_vector_store()
    elif current_count < MIN_EXPECTED_CHUNKS:
        print(
            f"⚠️ Vector store has only {current_count} chunks "
            f"(< {MIN_EXPECTED_CHUNKS}). Rebuilding for better retrieval quality..."
        )
        if os.path.isdir(DB_DIR):
            shutil.rmtree(DB_DIR, ignore_errors=True)
        os.makedirs(DB_DIR, exist_ok=True)
        db = create_vector_store()

    print("✅ Chroma DB loaded!")

    return db