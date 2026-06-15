from dotenv import load_dotenv
load_dotenv()

import os
import cohere
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI

# -------------------------
# STARTUP VALIDATION
# -------------------------

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set in environment.")
if not os.getenv("COHERE_API_KEY"):
    raise RuntimeError("COHERE_API_KEY is not set in environment.")

print("API.PY LOADED")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI()
co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

session_histories: dict[str, list[str]] = {}

MAX_HISTORY = 20
SIMILARITY_THRESHOLD = 1.50
FALLBACK_PHRASE = "I could not find the answer in the provided documents."
FALLBACK_SCORE_THRESHOLD = 1.30


# -------------------------
# MULTI-QUERY GENERATION
# -------------------------

def generate_queries(question: str, history: str) -> list[str]:
    """Generate 3 different search queries for the same question."""
    prompt = f"""You are an expert in Indian government industrial policy,
specifically MP (Madhya Pradesh) investment and subsidy schemes.

Generate exactly 3 different search queries for retrieving relevant policy
document chunks for the question below.

Query 1: Use the original casual language of the question
Query 2: Use formal policy terminology (expansion, diversification, 
         technological upgradation, EFCI, IPA, transfer of industrial unit etc.)
Query 3: Use a completely different angle or phrasing of the same question

Previous Conversation:
{history if history else "None"}

Question: {question}

Return ONLY the 3 queries, one per line, no numbering, no labels, no explanation:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0
        )
        queries = response.choices[0].message.content.strip().split("\n")
        queries = [q.strip() for q in queries if q.strip()][:3]
        if question not in queries:
            queries.append(question)
        print(f"\nGENERATED QUERIES:")
        for i, q in enumerate(queries, 1):
            print(f"  {i}. {q}")
        return queries
    except Exception as e:
        print(f"Query generation failed, using original: {e}")
        return [question]


# -------------------------
# GPT KNOWLEDGE FALLBACK
# -------------------------

def gpt_fallback(question: str) -> str | None:
    """Fallback to GPT general knowledge for out-of-scope questions."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert on Indian government industrial policy,
specifically Madhya Pradesh investment and subsidy schemes including IPP 2025,
Export Promotion Policy, and Startup Policy. Answer based on your knowledge
of these policies. Be concise and professional. If you are not sure, say so."""
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            max_tokens=800,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Fallback failed: {e}")
        return None


# -------------------------
# REQUEST MODEL
# -------------------------

class Question(BaseModel):
    question: str
    session_id: str = "default"


# -------------------------
# ASK ENDPOINT
# -------------------------

@app.post("/ask")
def ask_question(data: Question):

    if not data.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    history = session_histories.get(data.session_id, [])
    history_text = "\n".join(history)

    print(f"\n{'='*50}")
    print(f"SESSION:   {data.session_id}")
    print(f"QUESTION:  {data.question}")
    print('='*50)

    # -------------------------
    # MULTI-QUERY GENERATION
    # -------------------------

    queries = generate_queries(data.question, history_text)

    # -------------------------
    # MULTI-QUERY RETRIEVAL
    # Search with all queries, merge and deduplicate results
    # -------------------------

    all_candidates = []
    seen_content = set()
    best_score = 999.0

    for query in queries:
        try:
            docs = vectorstore.max_marginal_relevance_search(
                query, k=10, fetch_k=30
            )
            results = vectorstore.similarity_search_with_score(query, k=3)
            results.sort(key=lambda x: x[1])

            if results:
                score = results[0][1]
                best_score = min(best_score, score)
                print(f"\n  Query: '{query[:60]}...'")
                print(f"  Best score: {score:.4f}")

            for doc in docs:
                content_hash = hash(doc.page_content[:200])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_candidates.append(doc)

        except Exception as e:
            print(f"Retrieval failed for query: {e}")

    print(f"\nBEST SCORE ACROSS ALL QUERIES: {best_score:.4f}")
    print(f"TOTAL UNIQUE CANDIDATES: {len(all_candidates)}")

    if not all_candidates:
        return {"answer": "No relevant documents found."}

    if best_score > SIMILARITY_THRESHOLD:
        return {
            "answer": "This question appears unrelated to the uploaded policy documents."
        }

    # Cap candidates for reranker
    candidate_docs = all_candidates[:30]

    # -------------------------
    # RERANKING
    # Rerank all merged candidates with original question
    # -------------------------

    try:
        rerank_response = co.rerank(
            model="rerank-v3.5",
            query=data.question,
            documents=[doc.page_content for doc in candidate_docs],
            top_n=5
        )
        retrieved_docs = [
            candidate_docs[result.index]
            for result in rerank_response.results
        ]
        best_rerank_score = rerank_response.results[0].relevance_score
        print(f"\nRERANK SCORES: {[round(r.relevance_score, 4) for r in rerank_response.results]}")
        print(f"BEST RERANK SCORE: {best_rerank_score:.4f}")
    except Exception as e:
        print(f"Reranking failed, using first 5 candidates: {e}")
        retrieved_docs = candidate_docs[:5]
        best_rerank_score = 0.0

    for i, doc in enumerate(retrieved_docs[:3]):
        print(f"\nCHUNK {i+1}")
        print(doc.page_content[:1500])
        print("-" * 100)

    # -------------------------
    # BUILD CONTEXT
    # -------------------------

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # -------------------------
    # PROMPT
    # -------------------------

    prompt = f"""You are a retrieval-based assistant for CA (Chartered Accountant) \
services and policy documents.

Rules:
1. Answer ONLY using the provided context below.
2. You may explain implications directly supported by the policy text.
3. Do not invent information not present in the context.
4. Do not use outside knowledge.
5. If the context contains relevant policy provisions, answer by applying
   those provisions even if the user's terminology differs from policy
   terminology. Use your knowledge of Indian industrial policy to bridge
   any terminology gaps between casual language and official policy terms.
6. If no relevant information exists at all, reply exactly:
   I could not find the answer in the provided documents.
7. Be concise, professional, and cite policy information accurately.

Previous Conversation:
{history_text if history_text else "None"}

Context:
{context}

Question:
{data.question}

Answer:"""

    # -------------------------
    # LLM RESPONSE
    # -------------------------

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    print("\nGPT RAW ANSWER:")
    print(answer)

    # -------------------------
    # SMART FALLBACK
    # Only triggers when retrieval was also weak
    # -------------------------

    web_used = False

    if FALLBACK_PHRASE in answer and (best_score > FALLBACK_SCORE_THRESHOLD or best_rerank_score < 0.25):

        print(f"\nSmart fallback triggered (score {best_score:.4f} > {FALLBACK_SCORE_THRESHOLD})")
        fallback_answer = gpt_fallback(data.question)
        if fallback_answer:
            answer = fallback_answer
            web_used = True
            print(f"FALLBACK ANSWER: {answer}")
    elif FALLBACK_PHRASE in answer:
        print(f"\nNo fallback (score {best_score:.4f} < {FALLBACK_SCORE_THRESHOLD})")
        print("Documents are relevant — trusting document-based result")

    # -------------------------
    # CITATIONS
    # -------------------------

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

    print("\nTOP RETRIEVED DOCUMENTS AFTER RERANK:")
    for doc in retrieved_docs[:3]:
        print(doc.metadata)

    citation_lines = [
        f"[{i}] {s['file']} — Page {s['page']}"
        for i, s in enumerate(unique_sources, start=1)
    ]
    citation_text = "\n".join(citation_lines)

    if web_used:
        final_answer = f"{answer}\n\n---\n\n⚠️ *This answer is based on general knowledge of MP government policies, not your uploaded documents. Please verify with official sources.*"
    else:
        final_answer = f"{answer}\n\n---\n\n**References**\n{citation_text}"

    # -------------------------
    # UPDATE MEMORY
    # -------------------------

    history.append(f"User: {data.question}")
    history.append(f"Assistant: {answer}")
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    session_histories[data.session_id] = history

    return {"answer": final_answer}


# -------------------------
# HEALTH CHECK
# -------------------------

@app.get("/health")
def health():
    return {"status": "ok", "docs_in_store": vectorstore._collection.count()}