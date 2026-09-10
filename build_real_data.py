"""
Build Real Data — one-time setup script
-------------------------------------------
Runs the repo's OWN parser and chunker modules (src/parser, src/chunking)
on the three real chat exports, and saves the output into data/processed/
in exactly the format the evaluation scripts expect.

This replaces the fictional sample_chat.txt with real data, and produces
BOTH the naive time-gap chunks (for run_naive_baseline.py) and confirms
the parsed message files (used as fallback input by run_langchain.py,
run_llamaindex.py, and run_context_aware.py).

Usage (run from the project root):
    python build_real_data.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.parser.parse_whatsapp import parse_whatsapp_chat
from src.chunking.chunk_messages import naive_time_gap_chunking, context_aware_chunking

CHATS = {
    "yaoi": "data/raw_real/yaoi.txt",
    "diddy": "data/raw_real/diddy.txt",
    "nikhil": "data/raw_real/nikhil.txt",
}

os.makedirs("data/processed", exist_ok=True)

for chat_name, filepath in CHATS.items():
    print(f"=== {chat_name} ===")
    records = parse_whatsapp_chat(filepath, filter_system=True)
    print(f"  Parsed {len(records)} real, cleaned messages")

    parsed_out = f"data/processed/{chat_name}_parsed.json"
    with open(parsed_out, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in records], f, indent=2, ensure_ascii=False)
    print(f"  Saved: {parsed_out}")

    naive_chunks = naive_time_gap_chunking(records)
    naive_out = f"data/processed/{chat_name}_parsed_chunks.json"
    with open(naive_out, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in naive_chunks], f, indent=2, ensure_ascii=False)
    print(f"  Saved: {naive_out}  ({len(naive_chunks)} naive chunks)")

    ctx_chunks = context_aware_chunking(records)
    ctx_out = f"data/processed/{chat_name}_context_chunks.json"
    with open(ctx_out, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in ctx_chunks], f, indent=2, ensure_ascii=False)
    print(f"  Saved: {ctx_out}  ({len(ctx_chunks)} context-aware chunks)")
    print()

print("Done. Real data is now in data/processed/, ready for the evaluation scripts.")
