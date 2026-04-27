import argparse
from importlib import import_module

from src.core.vector_store import load_vector_store

TEST_QUERIES = [
    "What is total revenue?",
    "Which segment performed best?",
    "What trends are shown in the report?",
]


def _resolve_multimodal_chain(db, top_k: int | None):
    module = import_module("src.rag_types.multimodal_rag")
    get_multimodal_rag_chain = getattr(module, "get_multimodal_rag_chain")
    return get_multimodal_rag_chain(db, top_k=top_k)

def run_query(rag, query: str):
    print("\n" + "=" * 72)
    print(f"Query: {query}")

    answer, docs, _context, _latency = rag(query)

    print("\nRetrieved context:")
    if not docs:
        print("- No documents retrieved")
    for i, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        doc_type = metadata.get("type", "unknown")
        page = metadata.get("page", "unknown")
        print(f"\nChunk {i} | type={doc_type} | page={page}")
        print(doc.page_content[:500])

    print("\nFinal answer:")
    print(answer)


def parse_args():
    parser = argparse.ArgumentParser(description="Test multimodal LangChain RAG retrieval and grounded answers")
    parser.add_argument("--top-k", type=int, default=8, help="Retriever top-k")
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Optional query override. Can be used multiple times.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    db = load_vector_store()
    if db is None:
        raise ValueError("Vector store not loaded properly")

    rag = _resolve_multimodal_chain(db, top_k=args.top_k)

    queries = args.query if args.query else TEST_QUERIES
    for query in queries:
        run_query(rag, query)


if __name__ == "__main__":
    main()
