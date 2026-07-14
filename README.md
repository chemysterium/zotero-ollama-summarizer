# Zotero → Ollama Summarizer

Summarize the PDF fulltext of papers in your [Zotero](https://www.zotero.org/)
library with a local LLM served by [Ollama](https://ollama.com/), and save each
summary back into Zotero as a child note attached to the paper.

- Works on a single item or a whole collection
- Skips papers that already have a summary note (rerun-friendly)
- Long papers are summarized chunk-by-chunk, then combined (map-reduce)
- Everything runs locally except the Zotero Web API calls — the paper text
  never leaves your machine

## Requirements

- Python 3.10+
- A [Zotero account](https://www.zotero.org/) with your library synced
- [Ollama](https://ollama.com/) running locally with a model pulled
  (e.g. `ollama pull gemma4:26b-a4b-it-q4_K_M`)

## Setup

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Create your config:

   ```
   cp config.example.ini config.ini
   ```

   Fill in `zotero_library_id` and `zotero_api_key` — both from
   <https://www.zotero.org/settings/keys> (the key needs read/write access).
   `config.ini` is gitignored, so your credentials stay out of the repository.

   Every setting can also be provided as an environment variable with the same
   name uppercased (e.g. `ZOTERO_API_KEY`), which takes precedence over
   `config.ini`.

## Usage

Summarize a single paper by its Zotero item key, or by a title search:

```
python zotero_ollama_summarize.py ABCD1234
python zotero_ollama_summarize.py "partial title of the paper"
```

Summarize every paper in a collection (by name or collection key):

```
python zotero_ollama_summarize.py --collection "Thesis Reading"
python zotero_ollama_summarize.py --collection WXYZ9876
```

Options:

| Flag | Effect |
| --- | --- |
| `--force` | Re-summarize items that already have an AI Summary note |
| `--dry-run` | Show what would be processed, without calling Ollama or writing to Zotero |

Papers that already have a note starting with `AI Summary:` are skipped, so you
can rerun the collection command whenever you add new papers.

## How it works

1. Finds the item's PDF attachment via the Zotero Web API
2. Gets the fulltext from Zotero's server-side index, falling back to
   extracting it from the local PDF as structure-aware Markdown (via
   [pymupdf4llm](https://pypi.org/project/pymupdf4llm/)) if the index is empty
3. Sends the text to Ollama for summarization — long papers are split into
   overlapping chunks, summarized separately, then combined into one summary
4. Creates a child note (`AI Summary: <title>`) on the Zotero item

## Notes

- If your Zotero library uses WebDAV storage (e.g. Koofr), file attachments
  can't be uploaded through the Web API — that's why summaries are saved as
  notes rather than `.txt` attachments.
- The Ollama request disables extended "thinking" so reasoning-capable models
  don't spend their whole output budget on hidden reasoning tokens.
- Markdown extraction was chosen over plain-text extraction after a side-by-side
  comparison: plain text mangled unicode (units, superscripts) and lost some
  content the markdown extraction preserved (see `compare_extraction.py`).

## Comparing extraction methods

`compare_extraction.py` is a read-only tool (no writes to Zotero) that extracts
one paper both ways — plain text and Markdown — summarizes each with the same
Ollama model, and saves both for comparison:

```
python compare_extraction.py ABCD1234
```

Output goes to `comparisons/<item-key>_plain.txt` and `_markdown.txt` (plus the
raw extracted text/markdown for inspection).
