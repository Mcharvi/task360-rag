"""
fix_currency.py

Corrects ONE specific piece of wrong text that got saved into chroma_db
during scanning — without touching the original PDF (the PDF is fine as-is;
this only fixes the saved copy the chatbot actually reads from).

HOW TO USE:

1. Find a line in ocr_currency_review.txt that you've checked against the
   real PDF and confirmed is wrong, e.g.:

     MSME Development Policy 2025.pdf | page 31 | low-confidence currency
     figure: "7300 lakh." (confidence 57)

   ...and you looked at the real PDF page 31 and it actually says
   "Rs.300 lakh" (or "₹300 lakh" — whatever your PDF shows).

2. Run this script with those four pieces of information:

     python fix_currency.py \\
       --file "MSME Development Policy 2025.pdf" \\
       --page 31 \\
       --find "7300 lakh" \\
       --replace "Rs.300 lakh"

   (--page is the page number as printed in the PDF / in the review file
   — this script converts it to the internal 0-indexed page number for you.)

3. It will show you the exact saved text it found, ask "Is this correct?
   (y/n)", and only make the change if you type y.

4. Repeat for every line in ocr_currency_review.txt you've confirmed is
   actually wrong. Skip the ones that turn out to be false alarms.

You do NOT need to run ingest.py again after this — this script directly
fixes the saved copy in chroma_db, which is the only thing the chatbot
reads.
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

CHROMA_DIR = "chroma_db"
DOCS_DIR = "docs"


def main():
    parser = argparse.ArgumentParser(description="Fix one wrong currency figure in chroma_db.")
    parser.add_argument("--file", required=True, help='Exact PDF filename, e.g. "MSME Development Policy 2025.pdf"')
    parser.add_argument("--page", required=True, type=int, help="Page number as printed in the PDF (1-based)")
    parser.add_argument("--find", required=True, help='The wrong text as it currently appears, e.g. "7300 lakh"')
    parser.add_argument("--replace", required=True, help='What it should say instead, e.g. "Rs.300 lakh"')
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)

    source_path = os.path.join(DOCS_DIR, args.file)
    page_index = args.page - 1  # stored 0-indexed, matching ingest.py

    results = vectorstore.get(
        where={"$and": [{"source": source_path}, {"page": page_index}]}
    )

    ids = results.get("ids", [])
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])

    if not ids:
        print(f"No stored chunks found for {args.file}, page {args.page}. "
              f"Double-check the filename and page number match ocr_currency_review.txt exactly.")
        return

    matches = [
        (i, d, m) for i, d, m in zip(ids, docs, metas)
        if args.find in d
    ]

    if not matches:
        print(f"Found {len(ids)} chunk(s) for {args.file} page {args.page}, "
              f"but none contain the exact text \"{args.find}\".")
        print("Here is what IS stored for this page, so you can check the exact wording:\n")
        for d in docs:
            print("---")
            print(d)
        return

    for chunk_id, content, meta in matches:
        print("=" * 60)
        print(f"Found in {args.file}, page {args.page}:\n")
        print(content)
        print("\n" + "-" * 60)
        corrected = content.replace(args.find, args.replace)
        print(f"\nWill change \"{args.find}\" -> \"{args.replace}\"\n")
        print("New text will read:\n")
        print(corrected)
        print()
        answer = input("Apply this fix? (y/n): ").strip().lower()
        if answer != "y":
            print("Skipped.\n")
            continue

        vectorstore.delete(ids=[chunk_id])
        vectorstore.add_documents([Document(page_content=corrected, metadata=meta)])
        print("✅ Fixed and saved.\n")


if __name__ == "__main__":
    main()