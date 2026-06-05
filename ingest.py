from openai import OpenAI
import os

from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


# -------------------------
# LOAD PDFS
# -------------------------

documents = []

for file in os.listdir("docs"):

    if file.endswith(".pdf"):

        pdf_path = os.path.join("docs", file)

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        documents.extend(docs)

        print(f"Loaded: {file}")

print(f"\nTotal pages: {len(documents)}")

# -------------------------
# CHUNKING
# -------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=120
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks: {len(chunks)}")

# -------------------------
# OPENAI EMBEDDINGS
# -------------------------

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# -------------------------
# CHROMA
# -------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

print("\nChroma database created successfully!")
