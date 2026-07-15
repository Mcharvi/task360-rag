"""
bulk_fix_ocr.py

Reads ocr_currency_review.txt and automatically fixes every entry where a
clearly bogus character (%, *, =, ¥, §, ®, £, €, $, #, ~, ', +) is glued
directly in front of a number, immediately before "lakh"/"crore" — e.g.
"%10 Crore" or "$25 Lakh". No real policy document prints a percent sign
or dollar sign directly in front of a rupee amount like that, so these are
safe to correct without eyeballing each one: the stray character gets
replaced with "Rs." (matching the "Rs." convention already used elsewhere
in your documents, e.g. "Rs.1 lakh" in AVGC-XR Policy.pdf).

It does NOT touch:
  - Lines where the number is pure digits with no stray symbol (some of
    these are fine, a handful are genuinely wrong — those need a human
    glance, see the "manual check" list your assistant gave you).
  - Lines that are just OCR gibberish, not a real number at all (printed
    out separately below for you to check by hand).

HOW TO USE:

1. Put this file in the same folder as ingest.py / api.py / chroma_db.
2. Put your ocr_currency_review.txt in that same folder (or pass its path
   with --review-file).
3. First do a DRY RUN (the default) to see what it WOULD change, without
   touching anything:

     python bulk_fix_ocr.py

4. If the planned changes look right, actually apply them:

     python bulk_fix_ocr.py --apply

That's it — it goes through every safe line by itself; you don't need to
run this once per line.
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import os
import re

CHROMA_DIR = "chroma_db"
STRAY_SYMBOLS = set("%*=¥§®£€$#~'+")

LINE_RE = re.compile(
    r'^(.*?)\s*\|\s*page (\d+)\s*\|\s*low-confidence currency figure:\s*"(.*)"\s*\(number confidence (\d+)\)'
)


def classify(text: str):
    stripped = text.strip()
    if not re.search(r"\d", stripped):
        return "garbage"
    # gibberish check: too many consecutive letters once the unit words are stripped
    letters_only = re.sub(r"lakh|crore|lakhs|crores", "", stripped, flags=re.IGNORECASE)
    if re.search(r"[a-zA-Z]{4,}", letters_only):
        return "garbage"
    if stripped[0] in STRAY_SYMBOLS:
        return "auto_fix"
    return "manual"


def build_correction(text: str) -> str:
    """Replace the leading stray symbol with 'Rs.' — e.g. '%10 Crore' -> 'Rs.10 Crore'."""
    return "Rs." + text[1:].lstrip()


def main():
    parser = argparse.ArgumentParser(description="Bulk-fix safe OCR currency-symbol corruptions.")
    parser.add_argument("--review-file", default="ocr_currency_review.txt")
    parser.add_argument("--apply", action="store_true", help="Actually write changes (default is dry-run).")
    args = parser.parse_args()

    if not os.path.exists(args.review_file):
        print(f"Could not find {args.review_file} in this folder.")
        return

    lines = open(args.review_file, encoding="utf-8").read().splitlines()

    auto_fix_entries = []
    garbage_entries = []
    manual_entries = []

    for line in lines:
        m = LINE_RE.match(line)
        if not m:
            continue
        fname, page, text, conf = m.groups()
        kind = classify(text)
        if kind == "auto_fix":
            auto_fix_entries.append((fname, int(page), text, int(conf)))
        elif kind == "garbage":
            garbage_entries.append((fname, int(page), text, int(conf)))
        else:
            manual_entries.append((fname, int(page), text, int(conf)))

    print(f"Parsed {len(lines)} lines from {args.review_file}")
    print(f"  → {len(auto_fix_entries)} safe to auto-fix")
    print(f"  → {len(garbage_entries)} are OCR gibberish (not a real number) — check these by hand")
    print(f"  → {len(manual_entries)} are plain numbers — most are fine, a few need a glance\n")

    if garbage_entries:
        print("=== GIBBERISH ENTRIES (not auto-fixed, check by hand) ===")
        for fname, page, text, conf in garbage_entries:
            print(f"  {fname} | page {page} | \"{text}\" (conf {conf})")
        print()

    if not auto_fix_entries:
        print("Nothing to auto-fix.")
        return

    if not args.apply:
        print("=== DRY RUN — nothing has been changed yet ===\n")
        for fname, page, text, conf in auto_fix_entries:
            corrected = build_correction(text)
            print(f"  {fname} | page {page} | \"{text}\" -> \"{corrected}\"")
        print(f"\n{len(auto_fix_entries)} correction(s) would be applied.")
        print("Re-run with --apply to actually make these changes.")
        return

    # --apply: connect to chroma_db and make the corrections for real.
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)

    applied = 0
    not_found = 0

    for fname, page, text, conf in auto_fix_entries:
        # fname in the review file looks like "docs\Something.pdf" or
        # "docs/Something.pdf" depending on OS — try both forms since we
        # don't know for certain which one is stored as metadata.
        candidates = [fname, fname.replace("\\", "/"), fname.replace("/", "\\")]
        page_index = page - 1

        found_chunk = None
        for candidate in candidates:
            results = vectorstore.get(where={"$and": [{"source": candidate}, {"page": page_index}]})
            ids = results.get("ids", [])
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])
            for chunk_id, content, meta in zip(ids, docs, metas):
                if text in content:
                    found_chunk = (chunk_id, content, meta)
                    break
            if found_chunk:
                break

        if not found_chunk:
            print(f"  ⚠ Could not find stored chunk for {fname} page {page} containing \"{text}\" — skipped.")
            not_found += 1
            continue

        chunk_id, content, meta = found_chunk
        corrected_text = build_correction(text)
        corrected_content = content.replace(text, corrected_text)
        vectorstore.delete(ids=[chunk_id])
        vectorstore.add_documents([Document(page_content=corrected_content, metadata=meta)])
        applied += 1
        print(f"  ✅ Fixed: {fname} page {page}: \"{text}\" -> \"{corrected_text}\"")

    print(f"\nDone. {applied} correction(s) applied, {not_found} skipped (not found).")


if __name__ == "__main__":
    main()