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


def _strip_h1(note_html: str) -> str:
    return re.sub(r"<h1>.*?</h1>", "", note_html, count=1, flags=re.DOTALL)


# Tags that only a Markdown render produces; old raw notes contain just <p>/<br>.
_RENDERED_TAG = re.compile(
    r"<(?:h[2-6]|ul|ol|li|strong|em|sub|sup|table|code)\b", re.IGNORECASE
)


def is_rendered(note_html: str) -> bool:
    """True if the note body has already been rendered from Markdown to HTML."""
    return bool(_RENDERED_TAG.search(_strip_h1(note_html)))


def needs_conversion(note_html: str) -> bool:
    return bool(_MD_ARTIFACTS.search(_text_content(_strip_h1(note_html))))


_BULLET_START = re.compile(r"^[*+-]\s+(?=\S)")


def fix_literal_bullets(note_html: str) -> str:
    """Turn literal '*  item<br />' lines inside <p> blocks into real <ul> lists.

    Markdown's sane_lists needs a blank line before a list, so summaries where
    bullets follow a text line directly were rendered with the bullets left as
    literal '*' text. Rebuilding just those paragraphs (rather than re-rendering
    the note from its stripped text) keeps the inline <strong>/<sub>/<em> markup
    the render already produced.
    """
    def fix_block(match: re.Match) -> str:
        parts = re.split(r"<br\s*/?>", match.group(1))
        if not any(_BULLET_START.match(p.strip()) for p in parts):
            return match.group(0)

        blocks: list[str] = []
        text_run: list[str] = []
        list_run: list[str] = []

        def flush_text() -> None:
            if text_run:
                blocks.append("<p>" + "<br />".join(text_run) + "</p>")
                text_run.clear()

        def flush_list() -> None:
            if list_run:
                blocks.append(
                    "<ul>" + "".join(f"<li>{item}</li>" for item in list_run) + "</ul>"
                )
                list_run.clear()

        for part in parts:
            line = part.strip()
            if not line:
                continue
            if _BULLET_START.match(line):
                flush_text()
                list_run.append(_BULLET_START.sub("", line, count=1))
            else:
                flush_list()
                text_run.append(line)
        flush_text()
        flush_list()
        return "".join(blocks)

    return re.sub(r"<p>(.*?)</p>", fix_block, note_html, flags=re.DOTALL)


def fix_bullets_inside_list_items(note_html: str) -> str:
    """Turn a literal '*  item' line inside an <li> into a nested sub-list.

    Same cause as fix_literal_bullets, but for bullets the model nested under
    an existing list item; those live inside <li>...</li> rather than a <p>.
    Only innermost list items are matched (the content may not contain further
    <li> tags), so nested lists are rewritten from the inside out.
    """
    def fix(match: re.Match) -> str:
        parts = re.split(r"<br\s*/?>", match.group(1))
        if not any(_BULLET_START.match(p.strip()) for p in parts):
            return match.group(0)

        head: list[str] = []
        items: list[str] = []
        for part in parts:
            line = part.strip()
            if not line:
                continue
            if _BULLET_START.match(line):
                items.append(_BULLET_START.sub("", line, count=1))
            elif items:
                # Continuation text after a sub-bullet belongs to that bullet.
                items[-1] += "<br />" + line
            else:
                head.append(line)
        if not items or not head:
            return match.group(0)

        nested = "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"
        return "<li>" + "<br />".join(head) + nested + "</li>"

    return re.sub(r"<li>((?:(?!</?li\b).)*?)</li>", fix, note_html, flags=re.DOTALL)


def join_math_split_by_br(note_html: str) -> str:
    """Rejoin a $...$ span that an earlier render split with a line break.

    Rendering wrapped long lines mid-formula, leaving notes with markup like
    "$\\text<br />{H}^+$", which no longer looks like math to strip_latex. Only
    spans whose joined content is LaTeX-ish (\\, ^ or _) are rejoined, so an
    ordinary "$5<br />and $10" is left as it is.
    """
    def join(match: re.Match) -> str:
        inner = match.group(1) + match.group(2)
        if re.search(r"[\\^_]", inner) and inner == inner.strip():
            return f"${inner}$"
        return match.group(0)

    return re.sub(r"\$([^$<>]*)<br\s*/?>\s*\n?([^$<>]*)\$", join, note_html)


def repair_rendered_note(note_html: str) -> str:
    """Fix leftover artifacts in an already-rendered note, keeping its markup."""
    fixed = fix_bullets_inside_list_items(fix_literal_bullets(note_html))
    return zos.strip_latex(join_math_split_by_br(fixed))


def convert_note_html(note_html: str) -> str:
    """Rebuild an old-format note: same <h1> header, body re-rendered from
    the Markdown that the old code stored as literal text."""
    header_match = re.match(r"\s*(<h1>.*?</h1>)", note_html, flags=re.DOTALL)
    header = header_match.group(1) if header_match else ""
    body = note_html[header_match.end():] if header_match else note_html

    md = _text_content(body).strip()
    return f"{header}{zos.render_markdown(md)}"


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

        if is_rendered(note_html):
            # Already HTML: repair leftover artifacts in place, so the inline
            # markup from the earlier render survives.
            new_html = repair_rendered_note(note_html)
            action = "repaired"
        elif needs_conversion(note_html):
            # Raw Markdown stored as text: render the whole body.
            new_html = convert_note_html(note_html)
            action = "converted"
        else:
            skipped += 1
            continue

        if new_html == note_html:
            skipped += 1
            continue

        if args.dry_run:
            print(f"  would be {action}: {note_label(note)}")
            converted += 1
            continue

        try:
            note["data"]["note"] = new_html
            zot.update_item(note["data"])
            print(f"  {action}: {note_label(note)}")
            converted += 1
        except Exception as exc:
            print(f"  ERROR for {note['key']}: {exc}")
            failed += 1

    verb = "would fix" if args.dry_run else "fixed"
    print(
        f"Done. {converted} {verb}, {skipped} already clean, {failed} failed."
    )


if __name__ == "__main__":
    main()
