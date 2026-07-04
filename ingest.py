from dotenv import load_dotenv
load_dotenv()

import os
import hashlib
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set.")

# -------------------------
# PATHS
# -------------------------

DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-26.02.0\Library\bin"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# -------------------------
# HELPERS
# -------------------------

def is_garbled(text: str) -> bool:
    """Check if extracted text has too many garbled/unreadable characters."""
    if not text or len(text.strip()) < 50:
        return True

    garbled_chars = sum(1 for c in text if ord(c) > 1000)
    garbled_ratio = garbled_chars / max(len(text), 1)
    if garbled_ratio > 0.02:
        return True

    words = text.split()
    real_words = [w for w in words if w.isascii() and len(w) > 2]
    if len(words) > 10 and len(real_words) / len(words) < 0.7:
        return True

    garbled_patterns = ["ܜ", "LQYHVW", "FDSV", "0DFKLQHU"]
    if any(pattern in text for pattern in garbled_patterns):
        return True

    return False


def load_with_ocr(pdf_path: str) -> list[Document]:
    """Convert PDF pages to images and extract text with OCR."""
    print(f"  → Using OCR for: {os.path.basename(pdf_path)}")
    docs = []
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH, dpi=300)
    for page_num, image in enumerate(images):
        text = pytesseract.image_to_string(image, lang="eng")
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": pdf_path,
                    "page": page_num,
                    "loader": "ocr"
                }
            ))
        print(f"    Page {page_num + 1}/{len(images)} OCR done")
    return docs


def load_pdf(pdf_path: str) -> list[Document]:
    """Try PyMuPDF per page, falling back to OCR only for pages that need it."""
    try:
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
    except Exception as e:
        print(f"  → PyMuPDF failed ({e}), switching to OCR for entire file")
        return load_with_ocr(pdf_path)

    bad_page_indices = [i for i, d in enumerate(docs) if is_garbled(d.page_content)]

    if not bad_page_indices:
        return docs

    print(f"  → {len(bad_page_indices)} page(s) need OCR: {[i+1 for i in bad_page_indices]}")
    ocr_docs = load_with_ocr(pdf_path)

    # Replace only the bad pages with their OCR'd version, keep good pages as-is
    for i in bad_page_indices:
        if i < len(ocr_docs):
            docs[i] = ocr_docs[i]

    return docs


# -------------------------
# LOAD PDFs
# -------------------------

documents = []

for file in sorted(os.listdir(DOCS_DIR)):
    if file.endswith(".pdf"):
        path = os.path.join(DOCS_DIR, file)
        print(f"\nLoading: {file}")
        try:
            docs = load_pdf(path)
            documents.extend(docs)
            print(f"  → {len(docs)} pages loaded")
        except Exception as e:
            print(f"  → FAILED: {e}")

print(f"\nTotal pages loaded: {len(documents)}")

if not documents:
    print("No PDFs found in 'docs/' or all failed to load. Exiting.")
    exit(1)

# -------------------------
# CHUNKING
# -------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")

# -------------------------
# DEDUPLICATE
# -------------------------

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model
)

existing = vectorstore.get()
existing_hashes = set()

for doc_text in existing.get("documents", []):
    h = hashlib.md5(doc_text.encode()).hexdigest()
    existing_hashes.add(h)

new_chunks = []
for chunk in chunks:
    h = hashlib.md5(chunk.page_content.encode()).hexdigest()
    if h not in existing_hashes:
        new_chunks.append(chunk)

print(f"New chunks to add: {len(new_chunks)} "
      f"(skipped {len(chunks) - len(new_chunks)} duplicates)")

if not new_chunks:
    print("Nothing new to ingest. Database is up to date.")
    exit(0)

# -------------------------
# EMBED AND STORE
# -------------------------

print("\nEmbedding and storing chunks...")
vectorstore.add_documents(new_chunks)
print(f"\nDone. Total documents in store: "
      f"{vectorstore._collection.count()}")