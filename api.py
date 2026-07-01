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

# session_id -> list of "User: ..." / "Assistant: ..." strings
session_histories: dict[str, list[str]] = {}

# session_id -> "en" | "hi" | None (None = not chosen yet, i.e. brand new session)
session_languages: dict[str, str] = {}

# session_id -> the user's very first message (sent before language was
# chosen), so we can answer it automatically once they pick a language,
# instead of making them type it again.
session_pending_question: dict[str, str] = {}

MAX_HISTORY = 20
SIMILARITY_THRESHOLD = 1.50
FALLBACK_PHRASE = "I could not find the answer in the provided documents."
FALLBACK_SCORE_THRESHOLD = 1.30

LANGUAGE_PROMPT = (
    "Hi! Would you like to continue in English or Hindi?\n"
    "नमस्ते! क्या आप अंग्रेज़ी में या हिंदी में बात करना चाहेंगे?"
)


# -------------------------
# LANGUAGE CHOICE DETECTION
# Only ever called on the SECOND message of a session (the reply to
# LANGUAGE_PROMPT). Not used anywhere else — no mid-conversation switching.
# -------------------------

_GREETING_WORDS = {
    "hi", "hello", "hey", "namaste", "namaskar", "hii", "helo", "hola",
    "good morning", "good afternoon", "good evening", "greetings",
}


def _is_just_a_greeting(message: str) -> bool:
    """True only if the message is PURELY a greeting with no other content
    (e.g. 'hi', 'hello there') — so we don't run a real RAG answer on it.
    Anything with extra words/numbers/questions is treated as real content."""
    cleaned = message.strip().lower().strip("!.,? ")
    return cleaned in _GREETING_WORDS


def detect_language_choice(message: str) -> str | None:
    """Return 'en', 'hi', or None if the message doesn't clearly state a choice."""
    prompt = f"""The user was asked to choose between English and Hindi.
Their reply was: "{message}"

Respond with ONLY one word: "en" if they chose English, "hi" if they chose Hindi,
or "unclear" if their reply does not state a language choice at all.
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )
        out = resp.choices[0].message.content.strip().lower()
        if out in ("en", "hi"):
            return out
        return None
    except Exception as e:
        print(f"Language detection failed: {e}")
        return None


# -------------------------
# MULTI-QUERY GENERATION (unchanged from your original)
# -------------------------

def generate_queries(question: str, history: str) -> list[str]:
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
        return queries
    except Exception as e:
        print(f"Query generation failed, using original: {e}")
        return [question]


def gpt_fallback(question: str) -> str | None:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert on Indian government industrial policy,
specifically Madhya Pradesh investment and subsidy schemes. Answer based on your knowledge
of these policies. Be concise and professional. If you are not sure, say so."""
                },
                {"role": "user", "content": question}
            ],
            max_tokens=800,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Fallback failed: {e}")
        return None


# -------------------------
# CORE RAG PIPELINE (your original logic, language added to the prompt)
# -------------------------

def answer_question(question: str, history_text: str, language: str) -> str:
    queries = generate_queries(question, history_text)

    all_candidates = []
    seen_content = set()
    best_score = 999.0

    for query in queries:
        try:
            docs = vectorstore.max_marginal_relevance_search(query, k=10, fetch_k=30)
            results = vectorstore.similarity_search_with_score(query, k=3)
            results.sort(key=lambda x: x[1])

            if results:
                best_score = min(best_score, results[0][1])

            for doc in docs:
                content_hash = hash(doc.page_content[:200])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_candidates.append(doc)
        except Exception as e:
            print(f"Retrieval failed for query: {e}")

    if not all_candidates:
        return "No relevant documents found."

    if best_score > SIMILARITY_THRESHOLD:
        return "This question appears unrelated to the uploaded policy documents."

    candidate_docs = all_candidates[:30]

    try:
        rerank_response = co.rerank(
            model="rerank-v3.5",
            query=question,
            documents=[doc.page_content for doc in candidate_docs],
            top_n=5,
        )
        retrieved_docs = [candidate_docs[r.index] for r in rerank_response.results]
        best_rerank_score = rerank_response.results[0].relevance_score
    except Exception as e:
        print(f"Reranking failed, using first 5 candidates: {e}")
        retrieved_docs = candidate_docs[:5]
        best_rerank_score = 0.0

    context_parts = []
    for doc in retrieved_docs:
        page = doc.metadata.get("page", 0) + 1
        context_parts.append(f"\nPAGE {page}\n\n{doc.page_content}\n")
    context = "\n\n".join(context_parts)

    language_instruction = (
        "Respond ONLY in Hindi (Devanagari script)."
        if language == "hi"
        else "Respond ONLY in English."
    )

    prompt = f"""
You are an expert policy advisor specialising in Madhya Pradesh
industrial investment schemes, including IPP 2025, Export Promotion Policy,
and Energy Policy 2025.

{language_instruction}

Answer the question directly and confidently. State what the policy provides —
do not use phrases like "it seems", "it appears", "according to the policy",
"based on the provided sections", or "the policy mentions". These are implied.
Write as if you have fully internalised the policy and are advising a client.

FORMATTING:
- Use clear structure. If the answer has multiple points, steps, conditions,
  timelines, or categories, present them as a bulleted or numbered list using
  "-" or "1.", "2." etc. — not as one dense paragraph.
- Bold key figures, deadlines, and amounts using **double asterisks**.
- Keep paragraphs short. Use a list whenever there are 3 or more distinct
  points to make.

If the policy is silent on a specific detail but related provisions exist,
give a clear policy-based interpretation and flag it once with a single phrase
like "While not explicitly stated," — then move on.

Only if no relevant provisions exist at all, reply exactly:
I could not find the answer in the provided documents.

Previous Conversation:
{history_text if history_text else "None"}

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    if FALLBACK_PHRASE in answer and (best_score > FALLBACK_SCORE_THRESHOLD or best_rerank_score < 0.25):
        fallback_answer = gpt_fallback(question)
        if fallback_answer:
            answer = fallback_answer

    return answer


# -------------------------
# REQUEST MODEL
# -------------------------

class Question(BaseModel):
    question: str
    session_id: str = "default"


# -------------------------
# ASK ENDPOINT
#
# Rule: the FIRST message of any session (i.e. session_id not yet in
# session_languages at all) ALWAYS gets the language prompt back — no matter
# what the user typed, even "hello" or a real question. Their message is
# simply not processed as a question on this turn.
#
# The user's NEXT message (their reply to the language prompt) is read as
# the language choice. Once language is set, every message after that is
# treated as a normal question — no further language detection happens.
# -------------------------

@app.post("/ask")
def ask_question(data: Question):
    if not data.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = data.session_id
    message = data.question.strip()

    # ---- Turn 1 of a session: always ask language, regardless of input ----
    if session_id not in session_languages:
        # Mark the session as "awaiting language choice" — None is the
        # sentinel for "greeted, but hasn't answered yet" (different from
        # the key being absent entirely, which means "never greeted").
        session_languages[session_id] = None
        # Remember what they typed on turn 1 so we can answer it once
        # they've picked a language, instead of discarding it.
        session_pending_question[session_id] = message
        return {"answer": LANGUAGE_PROMPT}

    language = session_languages.get(session_id)

    # ---- Turn 2: this message IS the language choice ----
    if language is None:
        choice = detect_language_choice(message)
        if choice is None:
            # Didn't understand — ask again, don't guess.
            return {"answer": LANGUAGE_PROMPT}

        session_languages[session_id] = choice

        # Answer the turn-1 message now, in the chosen language —
        # unless it was just a greeting/empty filler with nothing to answer.
        pending = session_pending_question.pop(session_id, None)
        if pending and not _is_just_a_greeting(pending):
            history = session_histories.get(session_id, [])
            history_text = "\n".join(history)
            answer = answer_question(pending, history_text, choice)
            history.append(f"User: {pending}")
            history.append(f"Assistant: {answer}")
            session_histories[session_id] = history[-MAX_HISTORY:]
            return {"answer": answer}

        confirm = (
            "ठीक है, अब हम हिंदी में बात करेंगे। आप क्या जानना चाहते हैं?"
            if choice == "hi"
            else "Great, we'll continue in English. What would you like to know?"
        )
        return {"answer": confirm}

    # ---- Turn 3+: normal question answering ----
    history = session_histories.get(session_id, [])
    history_text = "\n".join(history)

    answer = answer_question(message, history_text, language)

    history.append(f"User: {message}")
    history.append(f"Assistant: {answer}")
    session_histories[session_id] = history[-MAX_HISTORY:]

    return {"answer": answer}


# -------------------------
# RESET ENDPOINT (clear a session during testing, no server restart needed)
# -------------------------

@app.post("/reset")
def reset_session(data: Question):
    session_languages.pop(data.session_id, None)
    session_histories.pop(data.session_id, None)
    session_pending_question.pop(data.session_id, None)
    return {"status": "reset"}


# -------------------------
# HEALTH CHECK
# -------------------------

@app.get("/health")
def health():
    return {"status": "ok", "docs_in_store": vectorstore._collection.count()}