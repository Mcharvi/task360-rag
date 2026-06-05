from dotenv import load_dotenv
load_dotenv()

import os
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set.")

DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"

# LOAD PDFs

documents = []

for file in os.listdir(DOCS_DIR):
    if file.endswith(".pdf"):
        path = os.path.join(DOCS_DIR, file)
        loader = PyPDFLoader(path)
        docs = loader.load()
        documents.extend(docs)
        print(f"Loaded: {file} ({len(docs)} pages)")

print(f"\nTotal pages loaded: {len(documents)}")

if not documents:
    print("No PDFs found in /docs. Exiting.")
    exit(1)

# CHUNKING

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=120
)

chunks = splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")

# DEDUPLICATE CHUNKS
# Hash each chunk's text so re-running ingest doesn't add duplicates.

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

# EMBED AND STORE

vectorstore.add_documents(new_chunks)
print(f"\nDone. Total documents in store: "
      f"{vectorstore._collection.count()}")