from dotenv import load_dotenv
load_dotenv()

import os
import hashlib
import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from concurrent.futures import ThreadPoolExecutor, as_completed

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set.")

DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-26.02.0\Library\bin"
OCR_DPI = 150
OCR_WORKERS = 6

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def is_garbled(text: str) -> bool:
    if not text or len(text.strip()) < 50:
        return True
    garbled_chars = sum(1 for c in text if ord(c) > 1000)
    if garbled_chars / max(len(text), 1) > 0.02:
        return True
    words = text.split()
    real_words = [w for w in words if w.isascii() and len(w) > 2]
    if len(words) > 10 and len(real_words) / len(words) < 0.7:
        return True
    if any(p in text for p in ["ܜ", "LQYHVW", "FDSV", "0DFKLQHU"]):
        return True
    return False


def _ocr_one_page(args):
    page_num, image = args
    return page_num, pytesseract.image_to_string(image, lang="eng")


def load_with_ocr(pdf_path, page_indices=None):
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH, dpi=OCR_DPI)
    wanted = list(range(len(images))) if page_indices is None else page_indices
    total = len(wanted)
    results = {}
    with ThreadPoolExecutor(max_workers=OCR_WORKERS) as executor:
        futures = {executor.submit(_ocr_one_page, (i, images[i])): i for i in wanted}
        done = 0
        for future in as_completed(futures):
            page_num, text = future.result()
            done += 1
            results[page_num] = Document(
                page_content=text,
                metadata={"source": pdf_path, "page": page_num, "loader": "ocr"}
            )
            print(f"    Page {page_num + 1} OCR done ({done}/{total})")
    return results


def load_pdf(pdf_path):
    try:
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
    except Exception as e:
        print(f"  → PyMuPDF failed ({e}), switching to OCR for entire file")
        ocr_map = load_with_ocr(pdf_path)
        return [ocr_map[i] for i in sorted(ocr_map)]

    bad_page_indices = [i for i, d in enumerate(docs) if is_garbled(d.page_content)]
    if not bad_page_indices:
        return docs

    print(f"  → {len(bad_page_indices)} page(s) need OCR: {[i+1 for i in bad_page_indices]}")
    ocr_map = load_with_ocr(pdf_path, page_indices=bad_page_indices)
    for i, doc in ocr_map.items():
        docs[i] = doc
    return docs


def main():
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

    pdf_files = sorted(f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf"))
    print(f"Found {len(pdf_files)} PDFs to process.\n")

    for idx, file in enumerate(pdf_files, start=1):
        path = os.path.join(DOCS_DIR, file)
        print(f"[{idx}/{len(pdf_files)}] Loading: {file}")

        try:
            docs = load_pdf(path)
        except Exception as e:
            print(f"  → FAILED to load {file}: {e}\n")
            continue

        chunks = splitter.split_documents(docs)

        # Check against what's already stored, so re-running after an
        # interruption skips work already saved to disk.
        existing = vectorstore.get(where={"source": path})
        existing_hashes = {
            hashlib.md5(t.encode()).hexdigest() for t in existing.get("documents", [])
        }

        new_chunks = [
            c for c in chunks
            if hashlib.md5(c.page_content.encode()).hexdigest() not in existing_hashes
        ]

        if new_chunks:
            vectorstore.add_documents(new_chunks)
            print(f"  → Added {len(new_chunks)} new chunk(s) "
                  f"(skipped {len(chunks) - len(new_chunks)} already stored)\n")
        else:
            print(f"  → Already up to date, nothing new to add.\n")

    print(f"All done. Total documents in store: {vectorstore._collection.count()}")


if __name__ == "__main__":
    main()