import re
import time
from importlib import import_module
from typing import Any, Callable

from src.core.config import TOP_K
from src.rag_types.contextual_rag import get_contextual_rag_chain
from src.rag_types.multi_query_rag import get_multi_query_rag_chain
from src.rag_types.simple_rag import get_simple_rag_chain

RouteInfo = dict[str, Any]
RagCallable = Callable[[str], tuple[Any, Any, str, float]]


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

GRAPH_HINTS = {
	"relationship",
	"relate",
	"linked",
	"link",
	"dependency",
	"depends",
	"network",
	"graph",
	"between",
}

FUSION_HINTS = {
	"compare",
	"comparison",
	"difference",
	"different",
	"versus",
	"vs",
	"architectures",
	"in detail",
	"detailed",
	"comprehensive",
	"pros and cons",
}


def _clean_query(question: str) -> str:
	return re.sub(r"\s+", " ", question or "").strip()


def _tokenize(question: str) -> list[str]:
	return re.findall(r"[a-zA-Z0-9]+", (question or "").lower())


def _contains_hint(question_lower: str, hints: set[str]) -> bool:
	return any(h in question_lower for h in hints)


def _looks_contextual(question: str, has_history: bool) -> bool:
	if not has_history:
		return False

	q = question.lower()
	if _contains_hint(q, CONTEXT_REF_HINTS):
		return True

	# Follow-up style short queries usually need previous turn context.
	tokens = _tokenize(question)
	return len(tokens) <= 6 and ("?" in question or q.startswith("and "))


def _looks_graph_query(question: str) -> bool:
	q = question.lower()
	if _contains_hint(q, GRAPH_HINTS):
		return True

	# Heuristic for "relationship between X and Y" patterns.
	return "between" in q and "and" in q


def _looks_fusion_query(question: str) -> bool:
	q = question.lower()
	if _contains_hint(q, FUSION_HINTS):
		return True

	tokens = _tokenize(question)
	return len(tokens) >= 12


def _looks_simple_query(question: str) -> bool:
	tokens = _tokenize(question)
	q = question.lower()
	if len(tokens) <= 5:
		return True

	starters = ("what is", "define", "meaning of")
	return any(q.startswith(s) for s in starters) and len(tokens) <= 8


def _select_route(question: str, has_history: bool) -> RouteInfo:
	if _looks_contextual(question, has_history):
		return {
			"route": "contextual",
			"reason": "Detected follow-up/context-dependent question.",
		}

	if _looks_graph_query(question):
		return {
			"route": "graph",
			"reason": "Detected relationship/structured reasoning intent.",
		}

	if _looks_fusion_query(question):
		return {
			"route": "fusion",
			"reason": "Detected comparison or broad multi-aspect question.",
		}

	if _looks_simple_query(question):
		return {
			"route": "simple",
			"reason": "Detected short/basic definition-style question.",
		}

	return {
		"route": "fusion",
		"reason": "Defaulted to broad retrieval for general query.",
	}


def _resolve_fusion_chain(db, top_k: int | None) -> tuple[RagCallable, str]:
	try:
		module = import_module("src.rag_types.fusion_rag")
		get_fusion_rag_chain = getattr(module, "get_fusion_rag_chain")

		return get_fusion_rag_chain(db, top_k=top_k), "fusion"
	except Exception:
		# Fallback to multi-query retrieval when dedicated Fusion RAG is unavailable.
		return get_multi_query_rag_chain(db, top_k=top_k, query_count=4), "multi_query_fallback"


def _resolve_graph_chain(db, top_k: int | None, history_turns: int) -> tuple[RagCallable, str]:
	try:
		module = import_module("src.rag_types.graph_rag")
		get_graph_rag_chain = getattr(module, "get_graph_rag_chain")

		return get_graph_rag_chain(db, top_k=top_k), "graph"
	except Exception:
		# Fallback to contextual reasoning when dedicated Graph RAG is unavailable.
		return get_contextual_rag_chain(db, top_k=top_k, history_turns=history_turns), "contextual_fallback"


def get_adaptive_rag_chain(db, top_k: int | None = None, history_turns: int = 3):
	k = top_k if top_k is not None else TOP_K

	simple_chain = get_simple_rag_chain(db, top_k=k)
	contextual_chain = get_contextual_rag_chain(db, top_k=k, history_turns=history_turns)
	fusion_chain, fusion_backend = _resolve_fusion_chain(db, top_k=k)
	graph_chain, graph_backend = _resolve_graph_chain(db, top_k=k, history_turns=history_turns)

	chain_map: dict[str, RagCallable] = {
		"simple": simple_chain,
		"contextual": contextual_chain,
		"fusion": fusion_chain,
		"graph": graph_chain,
	}

	chat_history: list[dict[str, str]] = []
	rag_handle: Any = None

	def rag(question: str):
		clean_question = _clean_query(question)
		start = time.perf_counter()

		route_info = _select_route(clean_question, has_history=bool(chat_history))
		requested_route = route_info["route"]
		selected_chain = chain_map.get(requested_route, fusion_chain)

		answer, docs, context, inner_latency = selected_chain(clean_question)

		total_latency = time.perf_counter() - start
		answer_text = answer if isinstance(answer, str) else str(answer)

		chat_history.append({"user": clean_question, "assistant": answer_text})
		if history_turns > 0 and len(chat_history) > history_turns * 2:
			del chat_history[:-history_turns * 2]

		rag_handle.last_debug = {
			"question": clean_question,
			"selected_route": requested_route,
			"selection_reason": route_info["reason"],
			"fusion_backend": fusion_backend,
			"graph_backend": graph_backend,
			"has_history": bool(chat_history),
			"inner_latency": inner_latency,
			"total_latency": total_latency,
		}

		return answer_text, docs, context, total_latency

	def reset_history():
		chat_history.clear()
		contextual_reset = getattr(contextual_chain, "reset_history", None)
		if callable(contextual_reset):
			contextual_reset()
		graph_reset = getattr(graph_chain, "reset_history", None)
		if callable(graph_reset):
			graph_reset()

	rag_handle = rag
	rag_handle.reset_history = reset_history
	rag_handle.last_debug = {
		"fusion_backend": fusion_backend,
		"graph_backend": graph_backend,
	}
	return rag
