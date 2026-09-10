"""
Module 1: WhatsApp Chat Parser & Normalizer.
Handles 12-hour and 24-hour timestamps, multi-line continuations,
and filters noise (encryption notices, media omitted).
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional
import re
import json

# Regex patterns covering common WhatsApp export variants (Android & iOS)
TIMESTAMP_PATTERNS = [
    # 10/01/2026, 09:30 - Sender: Message
    re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[apAP][mM])?)\s*-\s*(.+)$"),
    # [10/01/2026, 09:30:15] Sender: Message
    re.compile(r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[apAP][mM])?)\]\s*(.+)$"),
]

SYSTEM_PATTERNS = [
    re.compile(r"Messages and calls are end-to-end encrypted", re.IGNORECASE),
    re.compile(r"added\s+", re.IGNORECASE),
    re.compile(r"left\s*$", re.IGNORECASE),
    re.compile(r"created group", re.IGNORECASE),
    re.compile(r"changed the subject", re.IGNORECASE),
    re.compile(r"^You deleted this message$", re.IGNORECASE),
    re.compile(r"^This message was deleted$", re.IGNORECASE),
]

MEDIA_OMITTED_PATTERN = re.compile(r"<Media omitted>", re.IGNORECASE)


@dataclass
class MessageRecord:
    message_id: str
    timestamp: str
    date: str
    time: str
    sender: str
    text: str
    is_system: bool
    has_media: bool
    char_count: int
    word_count: int

    def to_dict(self):
        return asdict(self)


def parse_whatsapp_chat(filepath: str, filter_system: bool = True) -> List[MessageRecord]:
    """
    Parses a raw WhatsApp .txt export into structured MessageRecords.
    Correctly merges multi-line messages without timestamps into the parent turn.
    """
    records: List[MessageRecord] = []
    current_record: Optional[dict] = None
    msg_counter = 0

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        raw_line = line.rstrip("\r\n")
        if not raw_line.strip():
            continue

        matched = False
        for pat in TIMESTAMP_PATTERNS:
            match = pat.match(raw_line)
            if match:
                date_str, time_str, body = match.groups()
                matched = True

                # Flush previous record
                if current_record is not None:
                    records.append(_finalize_record(current_record))

                msg_counter += 1
                sender = "System"
                text = body.strip()
                is_system = False

                if ": " in body:
                    sender_part, text_part = body.split(": ", 1)
                    sender = sender_part.strip()
                    text = text_part.strip()
                else:
                    is_system = True

                for sys_pat in SYSTEM_PATTERNS:
                    if sys_pat.search(text):
                        is_system = True
                        break

                has_media = bool(MEDIA_OMITTED_PATTERN.search(text))

                current_record = {
                    "id": f"msg_{msg_counter:05d}",
                    "date": date_str,
                    "time": time_str,
                    "sender": sender,
                    "text": text,
                    "is_system": is_system,
                    "has_media": has_media,
                }
                break

        if not matched and current_record is not None:
            # Continuation line of previous message
            current_record["text"] += "\n" + raw_line

    if current_record is not None:
        records.append(_finalize_record(current_record))

    if filter_system:
        records = [r for r in records if not r.is_system]
        # Also drop messages that are PURELY a media placeholder (no other text).
        # A message that mentions media alongside real text (e.g. a caption) is kept.
        records = [
            r for r in records
            if not (r.has_media and MEDIA_OMITTED_PATTERN.sub("", r.text).strip() == "")
        ]

    return records


def _finalize_record(data: dict) -> MessageRecord:
    text = data["text"]
    dt_str = f"{data['date']} {data['time']}"
    iso_ts = dt_str  # fallback if nothing matches
    formats = (
        "%d/%m/%Y, %H:%M", "%d/%m/%Y %H:%M", "%d/%m/%y, %H:%M", "%d/%m/%y %H:%M",
        # 12-hour AM/PM variants (both with and without a space before AM/PM,
        # since different WhatsApp export locales format this differently)
        "%d/%m/%Y, %I:%M %p", "%d/%m/%Y %I:%M %p", "%d/%m/%y, %I:%M %p", "%d/%m/%y %I:%M %p",
        "%d/%m/%Y, %I:%M%p", "%d/%m/%Y %I:%M%p", "%d/%m/%y, %I:%M%p", "%d/%m/%y %I:%M%p",
    )
    for fmt in formats:
        try:
            dt = datetime.strptime(dt_str, fmt)
            iso_ts = dt.isoformat()
            break
        except ValueError:
            continue

    return MessageRecord(
        message_id=data["id"],
        timestamp=iso_ts,
        date=data["date"],
        time=data["time"],
        sender=data["sender"],
        text=text,
        is_system=data["is_system"],
        has_media=data["has_media"],
        char_count=len(text),
        word_count=len(text.split()),
    )
