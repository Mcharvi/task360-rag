from dotenv import load_dotenv
load_dotenv()

import os
import re
import json
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

app = FastAPI(title="Task360 Policy Assistant")

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

# -------------------------
# LOAD POLICY METADATA
# -------------------------

with open("policy_metadata.json", encoding="utf-8") as f:
    POLICY_METADATA = json.load(f)

META_BY_FILE = {entry["filename"]: entry for entry in POLICY_METADATA}
ALL_FILENAMES = list(META_BY_FILE.keys())
COMMON_FILES = [e["filename"] for e in POLICY_METADATA if e.get("is_common")]

# -------------------------
# SECTOR -> FILENAME MAP
# (mirrors the dropdown shown on the frontend)
# -------------------------

SECTOR_MAP = {
    "Manufacturing & Industry": ["Industrial Promotion Policy 2025.pdf", "Industrial Policy and Investment Promotion Scheme.pdf"],
    "Export & Trade": ["Export-Promotion-Policy.pdf", "Export Promotion Scheme 2025.pdf"],
    "Healthcare": ["Health Sector Investment Promotion policy.pdf"],
    "Energy & Renewables": ["Energy-Policy-2025.pdf", "Pumped Hydro Storage (PHS) scheme.pdf", "BioFuel Project Scheme for Implementation.pdf"],
    "Electric Vehicles": ["Electric Vehicle Policy 2025.pdf"],
    "IT, Tech & Digital": ["IT,ITes and ESDM Promotion Policy.pdf", "Global Capability Centers Policy.pdf", "Drone Promotion and Usage Policy.pdf"],
    "Tourism & Film": ["Tourism Policy 2025.pdf", "Film Tourism Policy 2025.pdf"],
    "Urban Development & Real Estate": ["Housing Redevelopment Policy 2022.pdf", "Real Estate Policy 2019.pdf", "Madhya Pradesh Integrated Township Policy 2025.pdf", "Land Pooling Policy.pdf"],
    "Logistics & Warehousing": ["Madhya Pradesh Logistics Policy.pdf", "Madhya Pradesh Logistics Scheme 2025.pdf"],
    "Startup & Innovation": ["Start-up-Policy-Inner-page-1.pdf"],
    "Space & Advanced Tech": ["SpaceTech Policy 2026.pdf", "Semiconductor Policy.pdf"],
    "Animation, Gaming & XR": ["AVGC-XR Policy.pdf"],
    "City Gas & Infrastructure": ["City Gas Distribution Network and Expansion Policy 2025.pdf"],
    "Civil Aviation": ["Civil Aviation Policy 2025.pdf"],
    "MSME": ["MSME Development Policy 2025.pdf"],
}

# health sub-questions shown by the frontend onboarding
HEALTH_SUBQUESTIONS = {
    "facility_type": [
        "Multi-Speciality Hospital", "Super-Speciality Hospital", "Medical College",
        "Ayurveda / AYUSH Facility", "Diagnostic Centre", "Medical Device Manufacturing"
    ],
    "district_category": ["A", "B", "C"]
}

FALLBACK_PHRASE = "I could not find the answer in the provided documents."


# -------------------------
# PARSE CONTEXT PREFIX
# Frontend sends:
# [Context: Sector = X, Facility Type = Y, District Category = Z, Language = hindi]\n\n<question>
# -------------------------

def parse_context(raw_question: str):
    ctx_match = re.match(r"\[Context:(.*?)\]\n\n", raw_question, re.DOTALL)

    if not ctx_match:
        return raw_question.strip(), None, False, ""

    ctx = ctx_match.group(1)
    actual_question = raw_question[ctx_match.end():].strip()

    respond_in_hindi = "Language = hindi" in ctx

    sector_match = re.search(r"Sector\s*=\s*([^,\]]+)", ctx)
    sector = sector_match.group(1).strip() if sector_match else None

    context_summary = ctx.strip()

    return actual_question, sector, respond_in_hindi, context_summary


def files_for_sector(sector: str | None) -> list[str]:
    """Files that should always be deep-searched for a given sector context."""
    files = set(COMMON_FILES)
    if sector and sector in SECTOR_MAP:
        files.update(SECTOR_MAP[sector])
    return list(files)


def expand_with_pairs(filenames: list[str]) -> list[str]:
    """Make sure a policy and its paired scheme are always searched together."""
    expanded = set(filenames)
    for fn in list(expanded):
        parent = META_BY_FILE.get(fn, {}).get("parent_policy")
        if parent:
            expanded.add(parent)
        for other_fn, meta in META_BY_FILE.items():
            if meta.get("parent_policy") == fn:
                expanded.add(other_fn)
    return list(expanded)


# -------------------------
# LAYER 1 — LLM FILE SELECTION
# -------------------------

def select_relevant_files(question: str, sector: str | None, history: str) -> list[str]:
    """
    Ask the LLM which policy/scheme files are relevant to the question,
    using only the metadata summaries (cheap, no vector search yet).
    Candidate universe is narrowed to the chosen sector + common policies
    when a sector is known, otherwise all files are considered.
    """
    candidate_files = files_for_sector(sector) if sector else ALL_FILENAMES
    candidate_files = expand_with_pairs(candidate_files)

    catalogue = "\n".join(
        f'- "{fn}" | type={META_BY_FILE[fn].get("type")} | sector={META_BY_FILE[fn].get("sector")} | {META_BY_FILE[fn].get("summary", "")}'
        for fn in candidate_files if fn in META_BY_FILE
    )

    prompt = f"""You are routing a question to the correct Madhya Pradesh government
policy documents. Below is a catalogue of candidate documents (filename | type | sector | summary).

Catalogue:
{catalogue}

Previous conversation:
{history if history else "None"}

Question: {question}

Return ONLY a JSON array of the exact filenames (from the catalogue above) that are
relevant to answering this question. Include a policy AND its paired scheme when both
could contain relevant provisions. If genuinely unsure, include more rather than fewer.
Return valid JSON only, e.g. ["Industrial Promotion Policy 2025.pdf"]"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        selected = json.loads(raw)
        selected = [fn for fn in selected if fn in META_BY_FILE]
        if not selected:
            selected = candidate_files
        print(f"\nLAYER 1 SELECTED FILES: {selected}")
        return selected
    except Exception as e:
        print(f"Layer 1 selection failed, falling back to sector candidates: {e}")
        return candidate_files


# -------------------------
# LAYER 2 — DEEP SEARCH ON SELECTED FILES
# -------------------------

def deep_search(question: str, filenames: list[str]):
    filenames = expand_with_pairs(filenames)
    docs = []
    for fn in filenames:
        try:
            results = vectorstore.max_marginal_relevance_search(
                question, k=12, fetch_k=40,
                filter={"source": {"$contains": fn}}
            )
            docs.extend(results)
        except Exception as e:
            print(f"Deep search failed for {fn}: {e}")
    return docs, filenames


# -------------------------
# LAYER 3 — LIGHT SEARCH ACROSS REMAINING DOCS
# -------------------------

def light_search_remaining(question: str, already_searched: list[str]):
    try:
        results = vectorstore.similarity_search(question, k=3)
    except Exception as e:
        print(f"Layer 3 light search failed: {e}")
        return []
    remaining = []
    for doc in results:
        source = doc.metadata.get("source", "")
        if not any(fn in source for fn in already_searched):
            remaining.append(doc)
    return remaining


# -------------------------
# REQUEST MODEL
# -------------------------

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


# -------------------------
# POST /api/v1/chat
# -------------------------

@app.post("/api/v1/chat")
def chat(data: ChatRequest):
    if not data.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    actual_question, sector, respond_in_hindi, context_summary = parse_context(data.question)

    history = session_histories.get(data.session_id, [])
    history_text = "\n".join(history)

    print(f"\n{'='*50}")
    print(f"SESSION:  {data.session_id}")
    print(f"QUESTION: {actual_question}")
    print(f"SECTOR:   {sector}")
    print(f"HINDI:    {respond_in_hindi}")
    print('='*50)

    # LAYER 1
    selected_files = select_relevant_files(actual_question, sector, history_text)

    # LAYER 2
    layer2_docs, searched_files = deep_search(actual_question, selected_files)

    # FALLBACK — if the filtered deep search returned nothing, search everything
    if not layer2_docs:
        print("\nLayer 2 returned no results — falling back to unfiltered search of all docs")
        try:
            layer2_docs = vectorstore.max_marginal_relevance_search(actual_question, k=12, fetch_k=40)
        except Exception as e:
            print(f"Fallback full search failed: {e}")
            layer2_docs = []
        searched_files = ALL_FILENAMES

    # LAYER 3
    layer3_docs = light_search_remaining(actual_question, searched_files)

    # MERGE + DEDUPE
    all_docs = []
    seen = set()
    for doc in layer2_docs + layer3_docs:
        h = hash(doc.page_content[:200])
        if h not in seen:
            seen.add(h)
            all_docs.append(doc)

    print(f"\nTOTAL MERGED CANDIDATES: {len(all_docs)}")

    if not all_docs:
        msg = "कोई प्रासंगिक दस्तावेज़ नहीं मिला।" if respond_in_hindi else "No relevant documents found."
        return {"answer": msg}

    # RERANK
    try:
        rerank_response = co.rerank(
            model="rerank-v3.5",
            query=actual_question,
            documents=[doc.page_content for doc in all_docs][:100],
            top_n=6
        )
        retrieved_docs = [all_docs[r.index] for r in rerank_response.results]
        print(f"\nRERANK SCORES: {[round(r.relevance_score, 4) for r in rerank_response.results]}")
    except Exception as e:
        print(f"Reranking failed: {e}")
        retrieved_docs = all_docs[:6]

    context_parts = []
    for doc in retrieved_docs:
        page = doc.metadata.get("page", 0) + 1
        source = doc.metadata.get("source", "").replace("\\", "/").split("/")[-1]
        context_parts.append(f"SOURCE: {source} | PAGE {page}\n\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    language_instruction = "\nRespond entirely in Hindi. Use formal Hindi suitable for a government policy context.\n" if respond_in_hindi else ""
    sector_hint = f"\nThis question was asked in the context of: {context_summary}\n" if context_summary else ""

    # --- HALLUCINATION FIX ---
    # Answer ONLY from explicitly stated provisions; any inference must be flagged.
    prompt = f"""You are an expert policy advisor specialising in Madhya Pradesh
government industrial investment policies and schemes.
{sector_hint}{language_instruction}
Answer ONLY from the explicitly stated provisions in the context below.
Do not use phrases like "it seems", "it appears", "according to the policy",
"based on the provided sections" — these are implied.

If the policy is silent on a specific detail, you may offer a reasoned
interpretation based on closely related provisions, but you MUST prefix
that sentence with "While not explicitly stated,". Never present an
inference as if it were an explicitly stated provision.

Use markdown formatting:
- Use **bold** for key terms, amounts, percentages, deadlines, and important conditions
- Use ## headings if the answer has clearly distinct sections
- Use bullet points for lists of conditions or requirements

Only if no relevant provisions exist at all, reply exactly:
{FALLBACK_PHRASE}

Previous Conversation:
{history_text if history_text else "None"}

Context:
{context}

Question:
{actual_question}

Answer:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    print("\nGPT RAW ANSWER:")
    print(answer)

    seen_src = set()
    unique_sources = []
    for doc in retrieved_docs[:4]:
        source = doc.metadata.get("source", "Unknown").replace("\\", "/").split("/")[-1]
        page = doc.metadata.get("page", 0)
        key = (source, page)
        if key not in seen_src:
            seen_src.add(key)
            unique_sources.append({"file": source, "page": page + 1})

    citation_lines = [f"[{i}] {s['file']} — Page {s['page']}" for i, s in enumerate(unique_sources, start=1)]
    citation_text = "\n".join(citation_lines)

    if FALLBACK_PHRASE in answer:
        final_answer = answer
    else:
        final_answer = f"{answer}\n\n---\n\n**References**\n{citation_text}"

    history.append(f"User: {actual_question}")
    history.append(f"Assistant: {answer}")
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    session_histories[data.session_id] = history

    return {"answer": final_answer}


# -------------------------
# GET /api/v1/sectors
# -------------------------

@app.get("/api/v1/sectors")
def get_sectors():
    return {
        "sectors": list(SECTOR_MAP.keys()),
        "health_subquestions": HEALTH_SUBQUESTIONS
    }


# -------------------------
# GET /api/v1/health
# -------------------------

@app.get("/api/v1/health")
def health():
    return {"status": "ok", "docs_in_store": vectorstore._collection.count()}