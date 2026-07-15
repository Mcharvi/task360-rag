"""
ONE-TIME RECOVERY SCRIPT.

Pulls existing chunks out of chroma_db for the given source file(s),
merges the (overlapping) chunk texts back into whole-page text using a
prefix/suffix overlap trim, and writes them out as ocr_overrides/*.txt
files. Once written, load_page_override() in ingest.py will pick these up
and SKIP OCR entirely for those pages — so your manually-corrected
currency figures survive the next FORCE_REPROCESS_ALL run instead of
getting wiped and re-OCR'd from scratch.

IMPORTANT: this is a best-effort merge, not a guaranteed-exact
reconstruction. Chunk order is read back from Chroma in whatever order
.get() returns (typically insertion order for a local persistent store,
but not contractually guaranteed). ALWAYS eyeball the output files in
ocr_overrides/ before trusting them — specifically grep for the currency
values you fixed and confirm they look right and aren't duplicated or
truncated at a merge boundary.

Usage:
    python recover_overrides.py "docs\\IT,ITes and ESDM Promotion Policy.pdf"

Run once per source file you manually corrected. Back up chroma_db before
running ingest.py again, regardless.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = "chroma_db"
OVERRIDES_DIR = "ocr_overrides"
MAX_OVERLAP_CHECK = 200  # a bit above your ingest.py's chunk_overlap=150, for safety


def longest_overlap(a: str, b: str, max_check: int = MAX_OVERLAP_CHECK) -> int:
    """Return the length of the longest suffix of `a` that matches a prefix of `b`."""
    max_check = min(max_check, len(a), len(b))
    for size in range(max_check, 0, -1):
        if a[-size:] == b[:size]:
            return size
    return 0


def merge_chunks(chunks: list[str]) -> str:
    if not chunks:
        return ""
    merged = chunks[0]
    for nxt in chunks[1:]:
        overlap = longest_overlap(merged, nxt)
        merged += nxt[overlap:]
    return merged


def main():
    if len(sys.argv) < 2:
        print('Usage:')
        print('  python recover_overrides.py "docs\\\\<filename>.pdf" [more files...]')
        print('  python recover_overrides.py --all      (exports every file currently in chroma_db)')
        sys.exit(1)

    os.makedirs(OVERRIDES_DIR, exist_ok=True)

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)

    if sys.argv[1] == "--all":
        # Don't rely on remembering which files were manually corrected —
        # just export every distinct source currently sitting in Chroma.
        # Read-only, no OCR, no embedding calls beyond client init, so this
        # is cheap and safe to run as a blanket safety net.
        all_result = vectorstore.get(include=["metadatas"])
        source_paths = sorted({m.get("source", "") for m in all_result.get("metadatas", []) if m.get("source")})
        print(f"--all: found {len(source_paths)} distinct source file(s) in chroma_db:")
        for sp in source_paths:
            print(f"  - {sp}")
        print()
    else:
        source_paths = sys.argv[1:]

    for source_path in source_paths:
        print(f"\nRecovering: {source_path}")
        result = vectorstore.get(
            where={"source": source_path},
            include=["metadatas", "documents"],
        )
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])

        if not docs:
            print(f"  → No chunks found in chroma_db for source={source_path!r}. "
                  f"Check the exact source string matches what's stored "
                  f"(case, slashes, etc.) — try inspecting one metadata entry "
                  f"from vectorstore.get() with no filter if unsure.")
            continue

        by_page: dict[int, list[str]] = {}
        for text, meta in zip(docs, metas):
            page = meta.get("page", 0)
            by_page.setdefault(page, []).append(text)

        print(f"  → Found {len(docs)} chunk(s) across {len(by_page)} page(s)")

        stem = os.path.splitext(os.path.basename(source_path))[0]
        for page_num, page_chunks in sorted(by_page.items()):
            merged_text = merge_chunks(page_chunks)
            out_path = os.path.join(OVERRIDES_DIR, f"{stem}__page_{page_num}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(merged_text)
            print(f"    page {page_num + 1}: {len(page_chunks)} chunk(s) → {out_path} "
                  f"({len(merged_text)} chars)")

    print(f"\nDone. Now MANUALLY review the files in {OVERRIDES_DIR}/ before "
          f"running ingest.py — grep for the currency values you corrected "
          f"(e.g. '₹') and confirm they look intact, not duplicated or cut "
          f"off at a merge boundary.")


if __name__ == "__main__":
    main()