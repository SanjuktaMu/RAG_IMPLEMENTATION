import re
import time
from types import SimpleNamespace
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.core.config import MODEL_NAME, OLLAMA_BASE_URL, TOP_K
from src.core.retriever import get_retriever


QUERY_REWRITE_PROMPT = """
You rewrite the user's question into a standalone retrieval query.

Rules:
- Use chat memory only to resolve references
- Keep the original intent
- Do not answer
- Return one line only

Chat memory:
{chat_memory}

Question:
{question}

Standalone query:
""".strip()


FINAL_PROMPT = """
You are a document QA assistant.

Rules:
- Answer ONLY from the provided context and memory snippets
- If partial information is available, answer as much as possible
- If completely missing, return exactly: Not found in document
- Do NOT use external knowledge
- Prefer concise, fact-only answers

Context:
{context}

Question:
{question}

Answer:
""".strip()


CONTEXT_REF_HINTS = {
	"it",
	"this",
	"that",
	"they",
	"them",
	"their",
	"its",
	"he",
	"she",
	"his",
	"her",
	"these",
	"those",
	"former",
	"latter",
	"above",
	"previous",
}


def _normalize_query(question: str) -> str:
	query = re.sub(r"\bbtl\s*epc\s*(limited|ltd\.?)\b", "", question, flags=re.IGNORECASE)
	query = re.sub(r"\s+", " ", query).strip(" ?")
	return query if query else question


def _is_boilerplate_chunk(text: str) -> bool:
	lower = text.lower()
	signals = [
		"registered office",
		"scan code to visit",
		"cin:",
		"ph.no",
		"corporate & marketing office",
	]
	hits = sum(1 for signal in signals if signal in lower)
	return hits >= 2


def _has_context_reference(question: str) -> bool:
	lower = question.lower()
	return any(token in lower.split() for token in CONTEXT_REF_HINTS)


def _format_memory(history: list[dict[str, str]], max_turns: int = 4) -> str:
	if not history:
		return "No prior turns."

	window = history[-max_turns:]
	lines: list[str] = []
	for turn in window:
		user = str(turn.get("user", "")).strip()
		assistant = str(turn.get("assistant", "")).strip()
		if user:
			lines.append(f"User: {user}")
		if assistant:
			lines.append(f"Assistant: {assistant}")
	return "\n".join(lines) if lines else "No prior turns."


def _tokenize(text: str) -> set[str]:
	return set(re.findall(r"[a-zA-Z0-9]+", (text or "").lower()))


def _memory_score(question: str, memory_item: dict[str, Any]) -> float:
	question_tokens = _tokenize(question)
	if not question_tokens:
		return 0.0

	q_tokens = _tokenize(str(memory_item.get("question", "")))
	a_tokens = _tokenize(str(memory_item.get("answer", "")))
	overlap = len(question_tokens.intersection(q_tokens.union(a_tokens)))

	if overlap == 0:
		return 0.0

	denominator = max(1, len(question_tokens))
	return overlap / denominator


def _select_memory_snippets(
	question: str,
	memory_bank: list[dict[str, Any]],
	limit: int,
) -> list[SimpleNamespace]:
	if not memory_bank:
		return []

	ranked = sorted(
		memory_bank,
		key=lambda item: _memory_score(question, item),
		reverse=True,
	)

	selected = []
	for item in ranked[:limit]:
		snippet = str(item.get("memory_snippet", "")).strip()
		if not snippet:
			continue
		selected.append(
			SimpleNamespace(
				page_content=snippet,
				metadata={"source": "memo_history", "question": item.get("question", "")},
			)
		)
	return selected


def _response_text(response: Any) -> str:
	content = response.content if hasattr(response, "content") else response
	if isinstance(content, str):
		return content
	if isinstance(content, list):
		return "\n".join(str(item) for item in content)
	return str(content)


def get_memo_rag_chain(
	db,
	top_k: int | None = None,
	history_turns: int = 4,
	memory_snippet_count: int = 2,
):
	k = top_k if top_k is not None else TOP_K
	retriever = get_retriever(db, top_k=k)

	llm = ChatOllama(
		model=MODEL_NAME,
		base_url=OLLAMA_BASE_URL,
		temperature=0,
	)

	rewrite_prompt = ChatPromptTemplate.from_template(QUERY_REWRITE_PROMPT)
	answer_prompt = ChatPromptTemplate.from_template(FINAL_PROMPT)

	chat_history: list[dict[str, str]] = []
	memory_bank: list[dict[str, Any]] = []
	rag_handle: Any = None

	def rag(question: str):
		start = time.perf_counter()
		normalized_question = _normalize_query(question)

		rewritten_query = normalized_question
		if chat_history and _has_context_reference(question):
			memory_text = _format_memory(chat_history, max_turns=history_turns)
			messages = rewrite_prompt.format_messages(chat_memory=memory_text, question=question)
			rewritten_query = _normalize_query(_response_text(llm.invoke(messages)).strip())
			if not rewritten_query:
				rewritten_query = normalized_question

		retrieved_docs = retriever.invoke(rewritten_query)
		retrieved_docs = [d for d in retrieved_docs if d.page_content and not _is_boilerplate_chunk(d.page_content)]
		if not retrieved_docs:
			fallback_docs = db.similarity_search(rewritten_query, k=max(10, k * 3))
			retrieved_docs = [
				d
				for d in fallback_docs
				if d.page_content and not _is_boilerplate_chunk(d.page_content)
			][:k]

		memory_docs = _select_memory_snippets(
			question=question,
			memory_bank=memory_bank,
			limit=max(0, memory_snippet_count),
		)

		docs = [*retrieved_docs, *memory_docs]
		context = "\n\n".join(doc.page_content for doc in docs if doc.page_content.strip())

		if not context.strip():
			answer_text = "Not found in document"
			latency = time.perf_counter() - start
			rag_handle.last_debug = {
				"question": question,
				"rewritten_query": rewritten_query,
				"retrieved_doc_count": 0,
				"memory_doc_count": len(memory_docs),
			}
			return answer_text, docs, context, latency

		messages = answer_prompt.format_messages(context=context, question=question)
		answer_text = _response_text(llm.invoke(messages)).strip() or "Not found in document"

		chat_history.append({"user": question, "assistant": answer_text})
		if history_turns > 0 and len(chat_history) > history_turns:
			del chat_history[:-history_turns]

		memory_bank.append(
			{
				"question": question,
				"answer": answer_text,
				"memory_snippet": "\n".join(
					[
						f"Q: {question}",
						f"A: {answer_text}",
						"Evidence:",
						*(doc.page_content[:240] for doc in retrieved_docs[:2]),
					]
				),
			}
		)
		# Keep memory bounded for long runs.
		if len(memory_bank) > 50:
			del memory_bank[:-50]

		latency = time.perf_counter() - start
		rag_handle.last_debug = {
			"question": question,
			"rewritten_query": rewritten_query,
			"retrieved_doc_count": len(retrieved_docs),
			"memory_doc_count": len(memory_docs),
			"memory_bank_size": len(memory_bank),
		}
		return answer_text, docs, context, latency

	def reset_history():
		chat_history.clear()
		memory_bank.clear()

	rag_handle = rag
	rag_handle.reset_history = reset_history
	rag_handle.last_debug = {
		"retrieved_doc_count": 0,
		"memory_doc_count": 0,
		"memory_bank_size": 0,
	}

	return rag
