from __future__ import annotations

import io
import importlib.util
import os
import re
from collections import defaultdict
from typing import Any

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import CHUNK_OVERLAP, CHUNK_SIZE, PDF_PATH

_BLIP_MODEL: Any = None
_BLIP_PROCESSOR: Any = None
IMAGE_CAPTION_INSTRUCTION = (
    "Describe the key business insight from this chart, including trends, "
    "comparisons, and important numerical information."
)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _safe_page_number(raw_page: Any, default_page: int) -> int:
    if raw_page is None:
        return default_page
    try:
        page = int(raw_page)
        return page + 1 if page == default_page - 1 else page
    except Exception:  # noqa: BLE001
        return default_page


def _ensure_required_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    meta = dict(metadata or {})
    doc_type = str(meta.get("type", "")).strip().lower()
    if doc_type not in {"text", "table", "image"}:
        doc_type = "text"

    page = meta.get("page", meta.get("page_number", -1))
    try:
        page = int(page)
    except Exception:  # noqa: BLE001
        page = -1

    meta["type"] = doc_type
    meta["page"] = page
    return meta


def _extract_text_documents(pdf_path: str) -> list[Document]:
    text_docs: list[Document] = []

    try:
        if importlib.util.find_spec("pi_heif") is None:
            raise ModuleNotFoundError("No module named 'pi_heif'")

        loader = UnstructuredPDFLoader(pdf_path, mode="elements")
        raw_docs = loader.load()

        for doc in raw_docs:
            cleaned = _clean_text(doc.page_content)
            if not cleaned:
                continue

            meta = doc.metadata or {}
            page = _safe_page_number(meta.get("page_number") or meta.get("page"), default_page=1)
            text_docs.append(
                Document(
                    page_content=cleaned,
                    metadata={
                        "type": "text",
                        "page": page,
                        "source": meta.get("source", pdf_path),
                        "layout": meta.get("category") or meta.get("element_type") or "unknown",
                    },
                )
            )

        if text_docs:
            return text_docs
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ UnstructuredPDFLoader failed ({exc}); falling back to PyPDFLoader.")

    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()
    for idx, doc in enumerate(raw_docs, start=1):
        cleaned = _clean_text(doc.page_content)
        if not cleaned:
            continue

        page = _safe_page_number((doc.metadata or {}).get("page"), default_page=idx)
        text_docs.append(
            Document(
                page_content=cleaned,
                metadata={
                    "type": "text",
                    "page": page,
                    "source": (doc.metadata or {}).get("source", pdf_path),
                    "layout": "page_text",
                },
            )
        )

    return text_docs


def _safe_float(text: str) -> float | None:
    raw = _clean_text(text).replace(",", "")
    try:
        return float(raw)
    except Exception:  # noqa: BLE001
        return None


def _table_unit_hint(header: list[str], row: list[str]) -> str:
    combined = " ".join([*header, *row]).lower()
    if "lakh" in combined or "lac" in combined or "lacs" in combined:
        return " lacs"
    if "crore" in combined:
        return " crores"
    if "million" in combined:
        return " million"
    return ""


def _build_row_sentence(header: list[str], row: list[str], unit_hint: str) -> str:
    pairs = []
    for key, value in zip(header, row):
        col = _clean_text(key)
        val = _clean_text(value)
        if not col or not val:
            continue
        pairs.append((col, val))

    if not pairs:
        return ""

    segment_key = next((col for col, _ in pairs if "segment" in col.lower()), "")
    revenue_key = next((col for col, _ in pairs if "revenue" in col.lower()), "")
    segment_val = next((val for col, val in pairs if col == segment_key), "")
    revenue_val = next((val for col, val in pairs if col == revenue_key), "")

    if segment_val and revenue_val:
        return f"{segment_val} segment generated revenue of {revenue_val}{unit_hint}."

    if revenue_key and revenue_val:
        subject = segment_val or "This entry"
        return f"{subject} has {revenue_key.lower()} of {revenue_val}{unit_hint}."

    facts = [f"{col} is {val}" for col, val in pairs]
    return "; ".join(facts) + "."


def _build_table_summary(header: list[str], body: list[list[str]], unit_hint: str) -> str:
    if not body:
        return ""

    revenue_idx = next((i for i, col in enumerate(header) if "revenue" in col.lower()), None)
    segment_idx = next((i for i, col in enumerate(header) if "segment" in col.lower()), None)

    if revenue_idx is not None:
        revenue_values = []
        for row in body:
            if revenue_idx < len(row):
                parsed = _safe_float(row[revenue_idx])
                if parsed is not None:
                    revenue_values.append(parsed)

        if revenue_values:
            total_revenue = sum(revenue_values)
            summary = [f"Total revenue is {total_revenue:,.2f}{unit_hint}."]

            if segment_idx is not None:
                best_row = None
                best_val = None
                for row in body:
                    if revenue_idx >= len(row) or segment_idx >= len(row):
                        continue
                    parsed = _safe_float(row[revenue_idx])
                    if parsed is None:
                        continue
                    if best_val is None or parsed > best_val:
                        best_val = parsed
                        best_row = row

                if best_row is not None and best_val is not None:
                    segment_name = _clean_text(best_row[segment_idx]) or "Unknown segment"
                    summary.append(
                        f"{segment_name} is the top-performing segment with revenue of {best_val:,.2f}{unit_hint}."
                    )

            return " ".join(summary)

    sample_sentences = []
    for row in body[:3]:
        sentence = _build_row_sentence(header, row, unit_hint)
        if sentence:
            sample_sentences.append(sentence)
    return " ".join(sample_sentences)


def _extract_table_documents(pdf_path: str) -> list[Document]:
    try:
        import camelot  # type: ignore
    except Exception:
        print("⚠️ Camelot not available; skipping table extraction.")
        return []

    table_docs: list[Document] = []
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ Camelot table extraction failed ({exc}).")
        return []

    for table_index, table in enumerate(tables, start=1):
        df = table.df
        if df.empty or len(df.columns) == 0:
            continue

        rows = [[_clean_text(str(cell)) for cell in row] for row in df.values.tolist()]
        rows = [row for row in rows if any(row)]
        if len(rows) < 2:
            continue

        header = rows[0]
        body = rows[1:]
        unit_hint = _table_unit_hint(header, rows[1] if len(rows) > 1 else [])

        page = 1
        try:
            page = int(getattr(table, "page", "1") or "1")
        except Exception:  # noqa: BLE001
            pass

        summary_parts = []
        for row in body:
            if not row:
                continue

            subject = next((cell for cell in row if cell), "This row")
            sentence = _build_row_sentence(header, row, unit_hint)
            if not sentence:
                continue

            summary_parts.append(f"{subject}: {sentence}")
            table_docs.append(
                Document(
                    page_content=sentence,
                    metadata=_ensure_required_metadata(
                        {
                            "type": "table",
                            "page": page,
                            "table_index": table_index,
                            "representation": "row",
                        }
                    ),
                )
            )

        if summary_parts:
            summary_text = _build_table_summary(header, body, unit_hint) or " ".join(summary_parts)
            table_docs.append(
                Document(
                    page_content=f"Table {table_index} summary: {summary_text}",
                    metadata=_ensure_required_metadata(
                        {
                            "type": "table",
                            "page": page,
                            "table_index": table_index,
                            "representation": "summary",
                        }
                    ),
                )
            )

    return table_docs


def _get_blip_components() -> tuple[Any, Any] | tuple[None, None]:
    global _BLIP_MODEL, _BLIP_PROCESSOR

    if _BLIP_MODEL is not None and _BLIP_PROCESSOR is not None:
        return _BLIP_MODEL, _BLIP_PROCESSOR

    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torchvision") is None:
        return None, None

    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor  # type: ignore
    except Exception:
        return None, None

    try:
        _BLIP_PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _BLIP_MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        return _BLIP_MODEL, _BLIP_PROCESSOR
    except Exception:  # noqa: BLE001
        return None, None


def _caption_with_blip(image_bytes: bytes) -> str:
    model, processor = _get_blip_components()
    if model is None or processor is None:
        return ""

    try:
        from PIL import Image  # type: ignore
    except Exception:
        return ""

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=image, text=IMAGE_CAPTION_INSTRUCTION, return_tensors="pt")
        generated = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(generated[0], skip_special_tokens=True)
        return _clean_text(caption)
    except Exception:  # noqa: BLE001
        return ""


def _is_informative_image(width: int, height: int) -> bool:
    if width <= 80 or height <= 80:
        return False
    return (width * height) >= 25000


def _extract_caption_fallback(blocks: list[Any], image_rect: Any) -> str:
    best = ""
    best_distance = float("inf")

    for block in blocks:
        if len(block) < 5:
            continue

        x0, y0, x1, y1, text = block[:5]
        _ = x0, x1
        cleaned = _clean_text(str(text))
        if not cleaned:
            continue

        lower = cleaned.lower()
        if not any(token in lower for token in ("figure", "fig.", "chart", "graph", "trend", "table")):
            continue

        center_y = (y0 + y1) / 2.0
        image_center_y = (image_rect.y0 + image_rect.y1) / 2.0
        distance = abs(center_y - image_center_y)

        if distance < best_distance:
            best = cleaned
            best_distance = distance

    return best


def _extract_image_documents(pdf_path: str) -> list[Document]:
    try:
        import fitz  # type: ignore
    except Exception:
        print("⚠️ PyMuPDF not available; skipping image extraction.")
        return []

    image_docs: list[Document] = []

    try:
        pdf = fitz.open(pdf_path)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ Failed to open PDF for image extraction ({exc}).")
        return []

    for page_index, page in enumerate(pdf, start=1):
        blocks = page.get_text("blocks")
        page_images = page.get_images(full=True)

        for image_index, img in enumerate(page_images, start=1):
            xref = img[0]

            try:
                image_data = pdf.extract_image(xref)
                image_bytes = image_data.get("image", b"")
                width = int(image_data.get("width", 0) or 0)
                height = int(image_data.get("height", 0) or 0)
            except Exception:  # noqa: BLE001
                continue

            if not image_bytes or not _is_informative_image(width, height):
                continue

            caption = _caption_with_blip(image_bytes)
            if not caption:
                try:
                    rects = page.get_image_rects(xref)
                    rect = rects[0] if rects else None
                except Exception:  # noqa: BLE001
                    rect = None

                if rect is not None:
                    caption = _extract_caption_fallback(blocks, rect)

            if not caption:
                caption = "Informative chart or graph detected, but no explicit caption text was extracted."

            image_docs.append(
                Document(
                    page_content=caption,
                    metadata=_ensure_required_metadata(
                        {
                            "type": "image",
                            "page": page_index,
                            "image_index": image_index,
                        }
                    ),
                )
            )

    pdf.close()
    return image_docs


def export_documents_markdown(documents: list[Document], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, doc in enumerate(documents, start=1):
            metadata = doc.metadata or {}
            f.write(f"# Chunk {idx}\n\n")
            f.write(f"Metadata: {metadata}\n\n")
            f.write("```text\n")
            f.write((doc.page_content or "").strip() + "\n")
            f.write("```\n\n")


def load_documents(pdf_path: str | None = None, export_markdown: bool = True) -> list[Document]:
    resolved_pdf_path = pdf_path or PDF_PATH

    text_docs = _extract_text_documents(resolved_pdf_path)
    table_docs = _extract_table_documents(resolved_pdf_path)
    image_docs = _extract_image_documents(resolved_pdf_path)

    documents = [*text_docs, *table_docs, *image_docs]
    documents = [
        Document(page_content=doc.page_content, metadata=_ensure_required_metadata(doc.metadata))
        for doc in documents
        if _clean_text(doc.page_content)
    ]

    counts = defaultdict(int)
    for doc in documents:
        counts[(doc.metadata or {}).get("type", "unknown")] += 1

    print(
        "Loaded multimodal documents: "
        f"total={len(documents)} | text={counts['text']} | table={counts['table']} | image={counts['image']}"
    )

    if export_markdown and documents:
        markdown_path = os.path.join(os.path.dirname(os.path.dirname(PDF_PATH)), "data", "processed_chunks.md")
        export_documents_markdown(documents, markdown_path)
        print(f"✅ Exported processed chunks to markdown: {markdown_path}")

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = []
    for parent_doc in documents:
        parent_meta = _ensure_required_metadata(parent_doc.metadata)
        split_parts = splitter.split_documents([parent_doc])
        for chunk in split_parts:
            chunk.metadata.update(parent_meta)
            chunk.metadata = _ensure_required_metadata(chunk.metadata)
            chunks.append(chunk)

    print(f"Created {len(chunks)} chunks")
    return chunks
