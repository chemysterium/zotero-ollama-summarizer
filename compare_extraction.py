"""
Compare plain-text (PyMuPDF) vs structure-aware Markdown (pymupdf4llm)
extraction of one Zotero PDF, each summarized by the same Ollama model,
to see whether markdown extraction produces a better summary.

Read-only: does not write anything to Zotero.

Requires: pip install pymupdf4llm

Usage:
    python compare_extraction.py ABCD1234
    python compare_extraction.py "partial title"
"""

import sys
from pathlib import Path

import fitz
import pymupdf4llm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zotero_ollama_summarize as zos


def local_pdf_path(attachment_key: str, filename: str) -> Path:
    path = Path(zos.ZOTERO_STORAGE_DIR) / attachment_key / filename
    if not path.exists():
        sys.exit(f"Local PDF not found at {path}. Make sure it's synced locally.")
    return path


def extract_plain(path: Path) -> str:
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def extract_markdown(path: Path) -> str:
    return pymupdf4llm.to_markdown(str(path))


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python compare_extraction.py <item-key-or-title>")

    zot = zos.build_client()
    item = zos.resolve_item(zot, sys.argv[1])
    title = item.get("title", "Untitled")
    key = item["key"]
    print(f"Item: {title} ({key})")

    attachment = zos.find_pdf_attachment(zot, key)
    path = local_pdf_path(attachment["key"], attachment.get("filename", ""))
    print(f"PDF: {path}")

    plain_text = extract_plain(path)
    md_text = extract_markdown(path)
    print(f"Plain text length: {len(plain_text)} chars")
    print(f"Markdown length:   {len(md_text)} chars")

    print("\nSummarizing plain-text extraction...")
    plain_summary = zos.summarize(plain_text)

    print("Summarizing markdown extraction...")
    md_summary = zos.summarize(md_text)

    out_dir = Path(__file__).resolve().parent / "comparisons"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{key}_plain.txt").write_text(plain_summary, encoding="utf-8")
    (out_dir / f"{key}_markdown.txt").write_text(md_summary, encoding="utf-8")
    (out_dir / f"{key}_plain_extracted.txt").write_text(plain_text, encoding="utf-8")
    (out_dir / f"{key}_markdown_extracted.md").write_text(md_text, encoding="utf-8")

    print(f"\nSaved to {out_dir}/")
    print("\n" + "=" * 70)
    print("PLAIN-TEXT SUMMARY")
    print("=" * 70)
    print(plain_summary)
    print("\n" + "=" * 70)
    print("MARKDOWN (pymupdf4llm) SUMMARY")
    print("=" * 70)
    print(md_summary)


if __name__ == "__main__":
    main()
