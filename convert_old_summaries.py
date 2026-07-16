"""
One-time migration: re-render old AI Summary notes as proper HTML.

Notes created by early versions of zotero_ollama_summarize.py stored the
model's Markdown output as literal text (one <p> with <br> line breaks), so
they display raw artifacts like ### headings and **bold** in Zotero. This
script finds those notes, reconstructs the original Markdown, re-renders it
through the same pipeline the summarizer now uses (LaTeX cleanup + Markdown
to HTML), and updates each note in place. Already-converted notes are
detected and skipped, so the script is safe to rerun.

Usage:
    python convert_old_summaries.py --dry-run     # inspect whole library
    python convert_old_summaries.py               # convert whole library
    python convert_old_summaries.py ABCD1234      # single item (key or title)
    python convert_old_summaries.py --collection "Thesis Reading"
"""

import argparse
import html
import re
import sys
from pathlib import Path

import markdown

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zotero_ollama_summarize as zos

# Literal Markdown syntax visible in the note's text content (tags stripped).
_MD_ARTIFACTS = re.compile(
    r"(?m)"
    r"(?:^#{1,6}\s)"          # heading lines: ### Methods
    r"|(?:\*\*[^*\n]+\*\*)"   # bold spans: **key findings**
    r"|(?:^\s*[*+-]\s{1,4}\S)"  # bullet lines: * item / - item
)


def _text_content(note_html: str) -> str:
    """The note's visible text, with <br> kept as newlines so ^-anchored
    Markdown patterns (headings, bullets) can match line starts."""
    text = re.sub(r"<br\s*/?>", "\n", note_html)
    text = re.sub(r"</(?:p|div|h[1-6]|li)>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text)


def needs_conversion(note_html: str) -> bool:
    body = re.sub(r"<h1>.*?</h1>", "", note_html, count=1, flags=re.DOTALL)
    return bool(_MD_ARTIFACTS.search(_text_content(body)))


def convert_note_html(note_html: str) -> str:
    """Rebuild an old-format note: same <h1> header, body re-rendered from
    the Markdown that the old code stored as literal text."""
    header_match = re.match(r"\s*(<h1>.*?</h1>)", note_html, flags=re.DOTALL)
    header = header_match.group(1) if header_match else ""
    body = note_html[header_match.end():] if header_match else note_html

    md = _text_content(body).strip()
    body_html = markdown.markdown(
        zos.strip_latex(md), extensions=["sane_lists", "tables", "nl2br"]
    )
    return f"{header}{body_html}"


def all_summary_notes(zot) -> list[dict]:
    """Every AI Summary note in the library, as full API objects."""
    notes = zot.everything(zot.items(itemType="note"))
    return [n for n in notes if zos.SUMMARY_MARKER in n["data"].get("note", "")]


def notes_for_items(zot, item_keys: list[str]) -> list[dict]:
    notes = []
    for key in item_keys:
        notes.extend(zos.find_summary_notes(zot, key))
    return notes


def note_label(note: dict) -> str:
    text = _text_content(note["data"].get("note", ""))
    first_line = text.strip().splitlines()[0] if text.strip() else "(empty)"
    return f"{note['key']}  {first_line[:70]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "item", nargs="?", help="Limit to one item: Zotero item key or title search"
    )
    parser.add_argument(
        "--collection", "-c", help="Limit to a collection (name or key)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List notes that would be converted without changing anything",
    )
    args = parser.parse_args()

    if args.item and args.collection:
        parser.error("provide at most one of: item, --collection")

    zot = zos.build_client()

    if args.item:
        item = zos.resolve_item(zot, args.item)
        notes = notes_for_items(zot, [item["key"]])
    elif args.collection:
        collection_key = zos.resolve_collection(zot, args.collection)
        papers = zos.get_collection_papers(zot, collection_key)
        notes = notes_for_items(zot, [p["key"] for p in papers])
    else:
        print("Scanning all notes in the library...")
        notes = all_summary_notes(zot)

    print(f"Found {len(notes)} AI Summary note(s).")

    converted = skipped = failed = 0
    for note in notes:
        note_html = note["data"].get("note", "")
        if not needs_conversion(note_html):
            skipped += 1
            continue

        if args.dry_run:
            print(f"  would convert: {note_label(note)}")
            converted += 1
            continue

        try:
            note["data"]["note"] = convert_note_html(note_html)
            zot.update_item(note["data"])
            print(f"  converted: {note_label(note)}")
            converted += 1
        except Exception as exc:
            print(f"  ERROR converting {note['key']}: {exc}")
            failed += 1

    verb = "would convert" if args.dry_run else "converted"
    print(
        f"Done. {converted} {verb}, {skipped} already clean, {failed} failed."
    )


if __name__ == "__main__":
    main()
