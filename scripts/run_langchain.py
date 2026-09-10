"""
LangChain Pipeline — Framework Comparison Runner
------------------------------------------------
Uses LangChain's RecursiveCharacterTextSplitter (document-oriented text splitting)
over flattened chat text. Demonstrates how a general-purpose framework handles
conversational data out-of-the-box.

Usage:
    python scripts/run_langchain.py
"""

import os
import json
from typing import List, Dict, Any
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 60
TOP_K = 3


def find_file(filename: str, search_dirs: List[str] = None) -> str:
    if search_dirs is None:
        search_dirs = [".", "data/benchmarks", "data/processed", "data/raw"]
    for d in search_dirs:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return filename


def load_json(path: str) -> Any:
    real_path = find_file(path)
    with open(real_path, "r", encoding="utf-8") as f:
        return json.load(f)


def messages_to_flat_text(messages: List[Dict[str, Any]]) -> str:
    lines = [f"[{m.get('date', '')} {m.get('time', '')}] {m.get('sender', '')}: {m.get('text', '')}" for m in messages]
    return "\n".join(lines)


def build_langchain_index(chat_name: str, model: SentenceTransformer):
    parsed_file = find_file(f"{chat_name}_parsed.json", [".", "data/processed", "data/raw"])
    messages = load_json(parsed_file)
    flat_text = messages_to_flat_text(messages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(flat_text)
    print(f"  [LangChain] {chat_name}: {len(messages)} messages -> {len(chunks)} chunks")

    embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings, chunks


def search(query: str, model: SentenceTransformer, embeddings: np.ndarray, chunks: List[str], top_k: int = TOP_K):
    qvec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = np.dot(embeddings, qvec)
    top_indices = np.argsort(sims)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        results.append({
            "rank": rank,
            "score": float(sims[idx]),
            "chunk_text": chunks[idx],
            "chunk_id": f"LC-{idx:03d}"
        })
    return results


def auto_score(retrieved: List[Dict[str, Any]], ground_truth: str, snippet: str, is_neg: bool) -> str:
    if is_neg:
        top_score = retrieved[0]["score"] if retrieved else 0.0
        return "hit" if top_score < 0.45 else "partial"

    target = (snippet or ground_truth or "").lower()
    combined_top = " ".join([r["chunk_text"] for r in retrieved[:TOP_K]]).lower()
    top1_text = retrieved[0]["chunk_text"].lower() if retrieved else ""

    if target and target in top1_text:
        return "hit"
    elif target and target in combined_top:
        return "hit"

    keywords = [w for w in target.replace("/", " ").replace("|", " ").split() if len(w) > 4]
    matched = sum(1 for kw in keywords if kw in combined_top)
    if keywords and (matched / len(keywords)) >= 0.5:
        return "partial"
    return "miss"


def main():
    eval_file = find_file("evaluation_set.json", [".", "data/benchmarks"])
    print(f"[*] Loading evaluation queries from: {eval_file}")
    eval_set = load_json(eval_file)
    print(f"[+] Loaded {len(eval_set)} benchmark questions")

    print(f"[*] Initializing embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    indexes = {}
    chunks_by_chat = {}
    results = []

    for q in eval_set:
        chat = q.get("chat", "sample")
        if chat not in indexes:
            print(f"[*] Building LangChain index for chat: {chat}")
            embeddings, chunks = build_langchain_index(chat, model)
            indexes[chat] = embeddings
            chunks_by_chat[chat] = chunks

        retrieved = search(q["question"], model, indexes[chat], chunks_by_chat[chat])
        is_neg = q.get("question_type") == "negative-control"
        score = auto_score(retrieved, q.get("ground_truth_answer", ""), q.get("source_text_snippet", ""), is_neg)

        results.append({
            "id": q["id"],
            "chat": chat,
            "question": q["question"],
            "question_type": q.get("question_type", "general"),
            "ground_truth_answer": q.get("ground_truth_answer"),
            "source_text_snippet": q.get("source_text_snippet"),
            "num_chunks_in_chat": len(chunks_by_chat[chat]),
            "retrieved_chunks": retrieved,
            "auto_score": score,
            "manual_score": "",  # <-- fill this in yourself by reading retrieved_chunks against ground_truth_answer
            "notes": q.get("notes", "")
        })

    os.makedirs("eval_results", exist_ok=True)
    out_path = "eval_results/results_langchain.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Successfully saved {len(results)} LangChain evaluated queries to: {out_path}")


if __name__ == "__main__":
    main()
