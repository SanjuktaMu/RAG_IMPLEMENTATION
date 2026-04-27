from __future__ import annotations

import heapq
import json
import re
import time
from typing import Any, cast

import networkx as nx
import numpy as np
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from sklearn.metrics.pairwise import cosine_similarity

from src.core.config import MODEL_NAME, OLLAMA_BASE_URL, TOP_K
from src.core.embeddings import LocalEmbedding


FINAL_PROMPT = """
You are a document QA assistant.

Rules:
- Answer ONLY from the provided context.
- If partial information is available, answer as much as possible.
- If the answer is completely missing, return exactly: Not found in document
- Do NOT use external knowledge.
- Prefer concise, fact-only answers.

Context:
{context}

Question:
{question}

Answer:
""".strip()


STRICT_FINAL_PROMPT = """
You are a strict information extractor.

Rules:
- Use ONLY the context.
- For numeric or named-entity questions, return exact values/names from context.
- Keep answer very short.
- If exact value/name is not present, return exactly: Not found in document

Context:
{context}

Question:
{question}

Answer:
""".strip()


STOP_WORDS = {
	"the",
	"is",
	"a",
	"an",
	"of",
	"in",
	"on",
	"to",
	"for",
	"and",
	"or",
	"by",
	"with",
	"from",
	"at",
	"what",
	"which",
	"who",
	"when",
	"where",
	"how",
	"does",
	"did",
	"do",
	"are",
	"was",
	"were",
	"be",
	"it",
	"its",
	"this",
	"that",
	"these",
	"those",
	"as",
	"into",
	"than",
	"also",
	"about",
}

NUMERIC_ENTITY_HINTS = {
	"how many",
	"total",
	"revenue",
	"income",
	"expenses",
	"profit",
	"difference",
	"value",
	"amount",
	"fy",
	"year",
	"age",
	"strength",
	"tax",
	"crore",
	"lacs",
	"who",
	"which",
	"name",
	"client",
	"company",
	"director",
	"founded",
	"location",
	"sector",
}


def _tokenize(text: str) -> list[str]:
	return re.findall(r"[a-zA-Z0-9]+", (text or "").lower())


def _normalize_query(question: str) -> str:
	tokens = _tokenize(question)
	filtered = [token for token in tokens if token not in STOP_WORDS]
	if not filtered:
		return question
	return " ".join(filtered)


def _concepts_from_text(text: str, max_terms: int = 14) -> list[str]:
	tokens = [tok for tok in _tokenize(text) if tok not in STOP_WORDS and len(tok) > 2]
	counts: dict[str, int] = {}
	for token in tokens:
		counts[token] = counts.get(token, 0) + 1
	ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
	return [token for token, _ in ranked[:max_terms]]


def _doc_key(doc: Document) -> str:
	content = (doc.page_content or "").strip()
	metadata = doc.metadata or {}
	return f"{content}\n{json.dumps(metadata, sort_keys=True, ensure_ascii=False)}"


def _is_numeric_or_entity_question(question: str) -> bool:
	lower = question.lower()
	return any(h in lower for h in NUMERIC_ENTITY_HINTS)


def _extract_documents_from_db(db) -> list[Document]:
	try:
		raw = db.get(include=["documents", "metadatas"])
		documents = raw.get("documents", []) or []
		metadatas = raw.get("metadatas", []) or []
		result: list[Document] = []
		for index, text in enumerate(documents):
			metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
			result.append(Document(page_content=text or "", metadata=metadata))
		if result:
			return result
	except Exception:  # noqa: BLE001
		pass

	count = 0
	collection = getattr(db, "_collection", None)
	if collection is not None:
		count_fn = getattr(collection, "count", None)
		if callable(count_fn):
			try:
				count_value = count_fn()
				count = int(cast(Any, count_value))
			except Exception:  # noqa: BLE001
				count = 0

	if count <= 0:
		return []

	try:
		return db.similarity_search(" ", k=count)
	except Exception:  # noqa: BLE001
		return []


def _build_knowledge_graph(docs: list[Document], similarity_threshold: float = 0.38, max_neighbors: int = 8):
	graph = nx.Graph()
	if not docs:
		return graph, np.zeros((0, 0), dtype=float), []

	embedding_model = LocalEmbedding()
	texts = [doc.page_content for doc in docs]
	embeddings = np.array(embedding_model.embed_documents(texts), dtype=float)
	similarity = cosine_similarity(embeddings)

	for index, doc in enumerate(docs):
		graph.add_node(
			index,
			content=doc.page_content,
			metadata=doc.metadata,
			concepts=_concepts_from_text(doc.page_content),
		)

	for i in range(len(docs)):
		neighbor_indices = np.argsort(similarity[i])[::-1]
		added = 0
		for j in neighbor_indices:
			if i == j:
				continue
			if j < i:
				continue

			sim_score = float(similarity[i][j])
			concept_i = set(graph.nodes[i]["concepts"])
			concept_j = set(graph.nodes[j]["concepts"])
			shared = concept_i & concept_j
			max_possible_shared = max(1, min(len(concept_i), len(concept_j)))
			normalized_shared = len(shared) / max_possible_shared
			edge_weight = 0.75 * sim_score + 0.25 * normalized_shared

			if sim_score < similarity_threshold and len(shared) == 0:
				continue

			graph.add_edge(
				i,
				j,
				weight=edge_weight,
				similarity=sim_score,
				shared_concepts=list(shared),
			)
			added += 1
			if added >= max_neighbors:
				break

	return graph, similarity, texts


def _query_variants(question: str) -> list[str]:
	base = _normalize_query(question)
	lower = question.lower()
	variants = [base if base.strip() else question]

	if "relationship" in lower or "between" in lower:
		variants.append(f"{base} relation dependency")
	if "who" in lower or "director" in lower:
		variants.append(f"{base} managing director board leadership")
	if "founded" in lower or "when" in lower:
		variants.append(f"{base} founded year")
	if "client" in lower or "marquee" in lower:
		variants.append(f"{base} clients ntpc adani bhel")

	seen: set[str] = set()
	unique: list[str] = []
	for query in variants:
		cleaned = query.strip()
		if not cleaned or cleaned in seen:
			continue
		seen.add(cleaned)
		unique.append(cleaned)
	return unique


def _traverse_graph(
	graph: nx.Graph,
	start_nodes: list[int],
	max_nodes: int = 10,
) -> list[int]:
	if not start_nodes:
		return []

	priority_queue: list[tuple[float, int]] = []
	distances: dict[int, float] = {}
	visited: set[int] = set()
	path: list[int] = []

	for rank, node in enumerate(start_nodes, start=1):
		initial_dist = 1.0 / (rank + 1)
		distances[node] = initial_dist
		heapq.heappush(priority_queue, (initial_dist, node))

	while priority_queue and len(path) < max_nodes:
		current_dist, node = heapq.heappop(priority_queue)
		if node in visited:
			continue
		visited.add(node)
		path.append(node)

		for neighbor in graph.neighbors(node):
			edge_weight = float(graph[node][neighbor].get("weight", 0.1))
			new_dist = current_dist + (1.0 / max(edge_weight, 1e-6))
			if new_dist < distances.get(neighbor, float("inf")):
				distances[neighbor] = new_dist
				heapq.heappush(priority_queue, (new_dist, neighbor))

	return path


def get_graph_rag_chain(db, top_k: int | None = None, max_traversal_nodes: int = 10):
	k = top_k if top_k is not None else TOP_K
	docs = _extract_documents_from_db(db)
	graph, _similarity_matrix, _texts = _build_knowledge_graph(docs)
	key_to_node: dict[str, int] = {}
	for idx, doc in enumerate(docs):
		key_to_node[_doc_key(doc)] = idx

	llm = ChatOllama(
		model=MODEL_NAME,
		base_url=OLLAMA_BASE_URL,
		temperature=0,
	)
	rag_handle: Any = None

	def rag(question: str):
		start = time.perf_counter()
		if not docs:
			return "Not found in document", [], "", time.perf_counter() - start

		variants = _query_variants(question)
		initial_docs: list[Document] = []
		seen: set[str] = set()
		for variant in variants:
			retrieved = db.similarity_search(variant, k=max(4, k * 2))
			for doc in retrieved:
				key = _doc_key(doc)
				if key in seen:
					continue
				seen.add(key)
				initial_docs.append(doc)

		start_nodes: list[int] = []
		for doc in initial_docs:
			node = key_to_node.get(_doc_key(doc))
			if node is None:
				continue
			if node not in start_nodes:
				start_nodes.append(node)

		path = _traverse_graph(graph, start_nodes, max_nodes=max_traversal_nodes)
		if not path:
			final_docs = initial_docs[: max(1, k)]
		else:
			final_docs = [docs[node] for node in path[: max(1, k * 2)]]

		start_nodes = [int(node) for node in start_nodes]
		path = [int(node) for node in path]

		context = "\n\n".join(doc.page_content for doc in final_docs if doc.page_content.strip())
		if not context.strip():
			answer_text = "Not found in document"
		else:
			prompt = FINAL_PROMPT.format(context=context, question=question)
			response = llm.invoke(prompt)
			content = response.content if hasattr(response, "content") else response
			if isinstance(content, str):
				answer_text = content
			elif isinstance(content, list):
				answer_text = "\n".join(str(item) for item in content)
			else:
				answer_text = str(content)

			if _is_numeric_or_entity_question(question):
				if answer_text.lower().strip() in {"not found in document", "not found in document."}:
					strict_prompt = STRICT_FINAL_PROMPT.format(context=context, question=question)
					strict_response = llm.invoke(strict_prompt)
					strict_content = strict_response.content if hasattr(strict_response, "content") else strict_response
					if isinstance(strict_content, str):
						answer_text = strict_content
					elif isinstance(strict_content, list):
						answer_text = "\n".join(str(item) for item in strict_content)
					else:
						answer_text = str(strict_content)

		latency = time.perf_counter() - start
		rag_handle.last_debug = {
			"question": question,
			"query_variants": variants,
			"start_nodes": start_nodes,
			"traversal_path": path,
			"retrieved_doc_count": len(initial_docs),
			"final_doc_count": len(final_docs),
		}
		return answer_text, final_docs, context, latency

	rag_handle = rag
	rag_handle.last_debug = {
		"query_variants": [],
		"traversal_path": [],
		"retrieved_doc_count": 0,
		"final_doc_count": 0,
	}
	return rag
