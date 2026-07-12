"""
Summarize a Zotero item's PDF fulltext using a local Ollama model and
save the result back into Zotero as a child note.

Setup:
    pip install pyzotero requests pymupdf

    Copy config.example.ini to config.ini next to this script and fill in
    your Zotero credentials (get an API key at
    https://www.zotero.org/settings/keys). Environment variables with the
    same names override config.ini values.

Usage:
    python zotero_ollama_summarize.py ABCD1234
    python zotero_ollama_summarize.py "partial title of the paper"
    python zotero_ollama_summarize.py --collection "Thesis Reading"
    python zotero_ollama_summarize.py --collection WXYZ9876 --force
    python zotero_ollama_summarize.py --collection "Thesis Reading" --dry-run
"""

import argparse
import configparser
import os
import re
import sys
from pathlib import Path

import requests
from pyzotero import zotero

CONFIG_PATH = Path(__file__).resolve().parent / "config.ini"


def _load_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    if CONFIG_PATH.exists():
        config.read(CONFIG_PATH, encoding="utf-8")
    return config


def _setting(section: str, key: str, default: str = "") -> str:
    """Resolve a setting: environment variable first, then config.ini, then default."""
    env_value = os.environ.get(key.upper())
    if env_value:
        return env_value
    return _CONFIG.get(section, key, fallback=default)


_CONFIG = _load_config()

ZOTERO_LIBRARY_ID = _setting("zotero", "zotero_library_id")
ZOTERO_LIBRARY_TYPE = _setting("zotero", "zotero_library_type", "user")
ZOTERO_API_KEY = _setting("zotero", "zotero_api_key")
ZOTERO_STORAGE_DIR = _setting(
    "zotero", "zotero_storage_dir", str(Path.home() / "Zotero" / "storage")
)

OLLAMA_URL = _setting("ollama", "ollama_url", "http://localhost:11434")
OLLAMA_MODEL = _setting("ollama", "ollama_model", "gemma4:26b-a4b-it-q4_K_M")

# Chunking thresholds for map-reduce summarization of long papers.
CHUNK_CHARS = 24000
CHUNK_OVERLAP = 500

ITEM_KEY_RE = re.compile(r"^[A-Z0-9]{8}$")
SUMMARY_MARKER = "AI Summary:"


class ProcessingError(Exception):
    """Raised for per-item failures that shouldn't abort a whole collection run."""


def build_client() -> zotero.Zotero:
    if not (ZOTERO_LIBRARY_ID and ZOTERO_API_KEY):
        sys.exit(
            "Missing Zotero credentials. Copy config.example.ini to config.ini and "
            "fill in zotero_library_id and zotero_api_key (get a key at "
            "https://www.zotero.org/settings/keys), or set the ZOTERO_LIBRARY_ID "
            "and ZOTERO_API_KEY environment variables."
        )
    return zotero.Zotero(ZOTERO_LIBRARY_ID, ZOTERO_LIBRARY_TYPE, ZOTERO_API_KEY)


def resolve_item(zot: zotero.Zotero, query: str) -> dict:
    if ITEM_KEY_RE.match(query):
        return zot.item(query)["data"] | {"key": query}

    matches = zot.items(q=query, qmode="titleCreatorYear", itemType="-attachment")
    if not matches:
        sys.exit(f"No Zotero items matched: {query!r}")
    if len(matches) > 1:
        print(f"Multiple matches for {query!r}, pick one and rerun with its key:")
        for m in matches:
            print(f"  {m['key']}  {m['data'].get('title', '(no title)')}")
        sys.exit(1)
    item = matches[0]
    return item["data"] | {"key": item["key"]}


def resolve_collection(zot: zotero.Zotero, query: str) -> str:
    if ITEM_KEY_RE.match(query):
        return query

    collections = zot.collections()
    matches = [c for c in collections if c["data"]["name"].lower() == query.lower()]
    if not matches:
        matches = [c for c in collections if query.lower() in c["data"]["name"].lower()]
    if not matches:
        sys.exit(f"No collection matched: {query!r}")
    if len(matches) > 1:
        print(f"Multiple collections matched {query!r}, pick one and rerun with its key:")
        for c in matches:
            print(f"  {c['key']}  {c['data']['name']}")
        sys.exit(1)
    return matches[0]["key"]


def get_collection_papers(zot: zotero.Zotero, collection_key: str) -> list[dict]:
    items = zot.everything(zot.collection_items_top(collection_key))
    papers = []
    for it in items:
        data = it["data"]
        if data.get("itemType") in ("attachment", "note"):
            continue
        papers.append({"key": it["key"], "title": data.get("title", "Untitled")})
    return papers


def _summary_has_body(note_html: str) -> bool:
    """True if a summary note contains real text beyond the AI Summary header."""
    without_header = re.sub(r"<h1>.*?</h1>", " ", note_html, flags=re.DOTALL)
    body_text = re.sub(r"<[^>]+>", " ", without_header)
    return bool(body_text.strip())


def has_existing_summary(zot: zotero.Zotero, parent_key: str) -> bool:
    for child in zot.children(parent_key):
        data = child["data"]
        if data.get("itemType") != "note":
            continue
        note_html = data.get("note", "")
        if SUMMARY_MARKER not in note_html:
            continue
        # Blank summaries (header but no body) left behind by earlier runs
        # where the model returned an empty response don't count, so a plain
        # rerun retries them instead of skipping.
        if _summary_has_body(note_html):
            return True
    return False


def delete_blank_summary_notes(zot: zotero.Zotero, parent_key: str) -> None:
    """Remove leftover AI Summary notes that have a header but no body."""
    for child in zot.children(parent_key):
        data = child["data"]
        if data.get("itemType") != "note":
            continue
        note_html = data.get("note", "")
        if SUMMARY_MARKER in note_html and not _summary_has_body(note_html):
            try:
                zot.delete_item(child)
                print(f"  deleted blank summary note {child['key']}")
            except Exception as exc:
                print(f"  warning: could not delete blank summary note {child['key']}: {exc}")


def find_pdf_attachment(zot: zotero.Zotero, parent_key: str) -> dict:
    children = zot.children(parent_key)
    for child in children:
        data = child["data"]
        if data.get("itemType") == "attachment" and data.get("contentType") == "application/pdf":
            return data | {"key": child["key"]}
    raise ProcessingError(f"No PDF attachment found under item {parent_key}")


def get_fulltext(zot: zotero.Zotero, attachment_key: str, attachment_filename: str) -> str:
    try:
        content = zot.fulltext_item(attachment_key).get("content", "")
        if content and content.strip():
            return content
    except Exception:
        pass

    # Fall back to reading the PDF directly from local Zotero storage.
    local_path = Path(ZOTERO_STORAGE_DIR) / attachment_key / attachment_filename
    if not local_path.exists():
        raise ProcessingError(
            f"No indexed fulltext on the server and no local file at {local_path}. "
            "Set ZOTERO_STORAGE_DIR to your Zotero storage folder, or sync the PDF locally."
        )
    import fitz  # pymupdf

    doc = fitz.open(local_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def chunk_text(text: str) -> list[str]:
    if len(text) <= CHUNK_CHARS:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_CHARS
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return chunks


def ollama_chat(prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise research assistant. Summarize academic "
                    "papers accurately, preserving key findings, methods, and limitations. "
                    "Do not invent information not present in the text.",
                },
                {"role": "user", "content": prompt},
            ],
            # Disable extended reasoning: this model supports "thinking", and on long
            # prompts it can spend its whole output budget on hidden thinking tokens
            # and return an empty final answer otherwise.
            "think": False,
            "stream": False,
        },
        timeout=600,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"].strip()
    if not content:
        raise ProcessingError(
            "Ollama returned an empty response (model may have exhausted its output "
            "budget on hidden thinking tokens without producing a final answer)."
        )
    return content


def summarize(fulltext: str) -> str:
    chunks = chunk_text(fulltext)
    if len(chunks) == 1:
        return ollama_chat(
            "Write a structured summary (background, methods, key findings, "
            "limitations) of the following paper:\n\n" + chunks[0]
        )

    partials = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  summarizing chunk {i}/{len(chunks)}...")
        partials.append(
            ollama_chat(
                f"This is part {i} of {len(chunks)} of a longer paper. "
                "Summarize the key points in this excerpt:\n\n" + chunk
            )
        )

    combined = "\n\n".join(partials)
    return ollama_chat(
        "Below are partial summaries of consecutive sections of one paper. "
        "Combine them into a single coherent structured summary (background, "
        "methods, key findings, limitations):\n\n" + combined
    )


def save_note(zot: zotero.Zotero, parent_key: str, title: str, summary: str) -> None:
    # Built by hand rather than via zot.item_template("note"): pyzotero caches that
    # template and, once it's over an hour old, revalidates it with a request that
    # (due to a pyzotero bug) omits the required itemType param, causing a 400.
    note = {
        "itemType": "note",
        "note": f"<h1>AI Summary: {title}</h1><p>{summary.replace(chr(10), '<br>')}</p>",
        "tags": [],
        "collections": [],
        "relations": {},
        "parentItem": parent_key,
    }
    result = zot.create_items([note])
    if result.get("failed"):
        print(f"  warning: failed to create Zotero note: {result['failed']}")
    else:
        print("  saved summary as a Zotero note.")


def process_item(zot: zotero.Zotero, key: str, title: str) -> None:
    attachment = find_pdf_attachment(zot, key)
    print(f"  extracting fulltext from attachment {attachment['key']}...")
    fulltext = get_fulltext(zot, attachment["key"], attachment.get("filename", ""))
    print(f"  fulltext length: {len(fulltext)} chars")

    print(f"  summarizing with Ollama model {OLLAMA_MODEL}...")
    summary = summarize(fulltext)

    save_note(zot, key, title, summary)
    delete_blank_summary_notes(zot, key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "item", nargs="?", help="Zotero item key (8 chars) or a title search string"
    )
    parser.add_argument(
        "--collection", "-c", help="Collection name or key: summarize every paper in it"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-summarize items that already have an AI Summary note",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed (and why items would be skipped) without "
        "calling Ollama or writing to Zotero",
    )
    args = parser.parse_args()

    if bool(args.item) == bool(args.collection):
        parser.error("provide exactly one of: item, or --collection")

    zot = build_client()

    if args.collection:
        collection_key = resolve_collection(zot, args.collection)
        papers = get_collection_papers(zot, collection_key)
        print(f"Found {len(papers)} papers in collection {collection_key}.")

        processed = skipped = failed = 0
        for i, paper in enumerate(papers, 1):
            print(f"[{i}/{len(papers)}] {paper['title']} ({paper['key']})")
            if not args.force and has_existing_summary(zot, paper["key"]):
                print("  already summarized, skipping (use --force to redo)")
                skipped += 1
                continue

            if args.dry_run:
                try:
                    attachment = find_pdf_attachment(zot, paper["key"])
                    print(f"  would summarize (PDF attachment {attachment['key']} found)")
                    processed += 1
                except ProcessingError as exc:
                    print(f"  would fail: {exc}")
                    failed += 1
                continue

            try:
                process_item(zot, paper["key"], paper["title"])
                processed += 1
            except Exception as exc:
                print(f"  ERROR: {exc}")
                failed += 1

        verb = "would summarize" if args.dry_run else "summarized"
        print(f"Done. {processed} {verb}, {skipped} skipped, {failed} failed.")
    else:
        print(f"Resolving item: {args.item}")
        item = resolve_item(zot, args.item)
        title = item.get("title", "Untitled")
        key = item["key"]
        print(f"Found: {title} ({key})")

        if not args.force and has_existing_summary(zot, key):
            print("Already summarized (use --force to redo). Exiting.")
            return

        try:
            if args.dry_run:
                attachment = find_pdf_attachment(zot, key)
                print(f"Would summarize (PDF attachment {attachment['key']} found).")
                return
            process_item(zot, key, title)
        except ProcessingError as exc:
            sys.exit(str(exc))
        print("Done.")


if __name__ == "__main__":
    main()
