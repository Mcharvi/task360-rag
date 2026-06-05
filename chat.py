from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI

# STARTUP VALIDATION

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set in environment.")

print("API.PY LOADED")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock this down to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# SESSION MEMORY STORE
# Per-session history keyed by session_id.
 

session_histories: dict[str, list[str]] = {}

MAX_HISTORY = 20
SIMILARITY_THRESHOLD = 1.10


# QUERY REWRITING

def rewrite_query(question: str, history: str) -> str:
    if not history:
        return question

    prompt = f"""Given the conversation history and a new question, \
rewrite the question into a fully self-contained standalone question \
that can be understood without any prior context.

Conversation:
{history}

Latest Question:
{question}

Standalone Question:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Query rewrite failed, using original: {e}")
        return question


# REQUEST MODEL

class Question(BaseModel):
    question: str
    session_id: str = "default"  # Client sends a unique session ID


# ASK ENDPOINT

@app.post("/ask")
def ask_question(data: Question):

    # Validate input
    if not data.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    #  Load session history 
    history = session_histories.get(data.session_id, [])
    history_text = "\n".join(history)

    # Rewrite query 
    try:
        rewritten_query = rewrite_query(data.question, history_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query rewriting failed: {e}")

    print(f"\n{'='*50}")
    print(f"SESSION:   {data.session_id}")
    print(f"QUESTION:  {data.question}")
    print(f"REWRITTEN: {rewritten_query}")
    print('='*50)

    # Retrieval 
    try:
        retrieved_docs = vectorstore.max_marginal_relevance_search(
            rewritten_query, k=5, fetch_k=20
        )
        results = vectorstore.similarity_search_with_score(rewritten_query, k=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    #  Relevance check
    if not results:
        return {"answer": "No relevant documents found."}

    # Sort by score ascending (lower = more similar)
    results.sort(key=lambda x: x[1])
    best_score = results[0][1]

    print(f"\nSIMILARITY SCORES: {[round(s, 4) for _, s in results]}")

    if best_score > SIMILARITY_THRESHOLD:
        return {
            "answer": "This question appears unrelated to the uploaded policy documents."
        }

    # Build context 
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    #  Build prompt
    prompt = f"""You are a retrieval-based assistant for CA (Chartered Accountant) \
services and policy documents.

Rules:
1. Answer ONLY using the provided context below.
2. Never use outside knowledge or make assumptions.
3. Never guess or infer facts not explicitly stated.
4. If the answer is not in the context, reply exactly:
   I could not find the answer in the provided documents.
5. Be concise and professional.

Previous Conversation:
{history_text if history_text else "None"}

Context:
{context}

Question:
{data.question}

Answer:"""

    #  LLM response
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    #  Build citations 
    seen = set()
    unique_sources = []

    for doc in retrieved_docs[:3]:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0)
        filename = source.replace("\\", "/").split("/")[-1]
        key = (filename, page)
        if key not in seen:
            seen.add(key)
            unique_sources.append({"file": filename, "page": page + 1})

    print("\nTOP RETRIEVED DOCUMENTS:")
    for doc in retrieved_docs[:3]:
        print(doc.metadata)

    citation_lines = [
        f"[{i}] {s['file']} — Page {s['page']}"
        for i, s in enumerate(unique_sources, start=1)
    ]
    citation_text = "\n".join(citation_lines)

    final_answer = f"{answer}\n\n---\n\n**References**\n{citation_text}"

    # Update session history 
    history.append(f"User: {data.question}")
    history.append(f"Assistant: {answer}")
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    session_histories[data.session_id] = history

    return {"answer": final_answer}


# HEALTH CHECK

@app.get("/health")
def health():
    return {"status": "ok", "docs_in_store": vectorstore._collection.count()}