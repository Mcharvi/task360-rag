from dotenv import load_dotenv
load_dotenv()

import os
import re
import json
import ssl
import cohere
import httpx
from urllib.parse import urlencode
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request



# -------------------------
# STARTUP VALIDATION
# -------------------------

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set in environment.")
if not os.getenv("COHERE_API_KEY"):
    raise RuntimeError("COHERE_API_KEY is not set in environment.")

print("API.PY LOADED")

app = FastAPI(title="Task360 Policy Assistant")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5500,http://127.0.0.1:5500"  # dev defaults
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
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

# -------------------------
# PER-POLICY SUBQUESTIONS
# Asked right after the user picks a sector, before they start chatting.
# Every subquestion is skippable — a skipped answer is simply omitted from
# both the RAG context and the calculator pre-fill (left blank + required
# there instead).
#
# "context_label" is the human-readable key used inside the
# "[Context: ...]" prefix the frontend sends (parsed by parse_context()).
# "calculator_slug" (optional) wires a sector to an entry in
# POLICY_CALCULATORS + a "<slug>-calculator.html" page on the frontend.
# -------------------------

POLICY_SUBQUESTIONS = {
    "Healthcare": {
        "calculator_slug": "health",
        "questions": [
            {
                "id": "facility_type",
                "context_label": "Facility Type",
                "type": "select",
                "label": "What type of healthcare facility are you setting up?",
                "label_hi": "आप किस प्रकार की स्वास्थ्य सुविधा स्थापित कर रहे हैं?",
                # Only the two project types the official calculator actually supports.
                "options": ["Multi-Specialty Hospital", "Super-Specialty Hospital"],
            },
            {
                "id": "district_category",
                "context_label": "District Category",
                "type": "buttons",
                "label": "Which district category is your project in?",
                "label_hi": "आपकी परियोजना किस जिला श्रेणी में है?",
                "options": ["A", "B", "C"],
            },
        ],
    }
}

FALLBACK_PHRASE = "I could not find the answer in the provided documents."


# -------------------------
# PARSE CONTEXT PREFIX
# Frontend sends:
# [Context: Sector = X, Facility Type = Y, District Category = Z, Language = hindi]\n\n<question>
#
# Any "Key = Value" pair other than Sector/Language is treated as a generic
# onboarding field (facility type, district category, etc. for whichever
# sector is active) and passed downstream as `extra_context`. Skipped
# subquestions are never present here, since the frontend leaves them out
# of the prefix entirely.
# -------------------------

def parse_context(raw_question: str):
    ctx_match = re.match(r"\[Context:(.*?)\]\n\n", raw_question, re.DOTALL)

    if not ctx_match:
        return raw_question.strip(), None, {}, False

    ctx = ctx_match.group(1)
    actual_question = raw_question[ctx_match.end():].strip()

    respond_in_hindi = "Language = hindi" in ctx

    sector_match = re.search(r"Sector\s*=\s*([^,\]]+)", ctx)
    sector = sector_match.group(1).strip() if sector_match else None

    extra_context: dict[str, str] = {}
    for field_match in re.finditer(r"([A-Za-z][A-Za-z ]*?)\s*=\s*([^,\]]+)", ctx):
        key = field_match.group(1).strip()
        val = field_match.group(2).strip()
        if key in ("Sector", "Language"):
            continue
        if val:
            extra_context[key] = val

    return actual_question, sector, extra_context, respond_in_hindi


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


def build_onboarding_note(sector: str | None, extra_context: dict[str, str]) -> str:
    """
    Shared "use this onboarding context only if relevant" note, used both
    when picking files (Layer 1) and when writing the final answer.
    Kept deliberately soft (LLM judgement, not a hard filter) — see product
    notes: tightening this to a deterministic rule is a follow-up, not now.
    """
    if not sector:
        return ""

    note = f"""
The user has already selected the "{sector}" sector as onboarding context.
Treat this as a strong prior: prefer documents/information related to this
sector. However, if the question clearly needs provisions from another
sector to be answered completely, include those too.
"""
    if extra_context:
        extra_lines = "\n".join(f"- {k}: {v}" for k, v in extra_context.items())
        note += f"""
The user also provided these onboarding details:
{extra_lines}
Use these ONLY if directly relevant to answering this specific question —
for example, use "Facility Type" if the question is about eligibility or
incentive amounts, but don't mention or lean on it for an unrelated
procedural question. Never force these details into an answer where they
don't belong.
"""
    return note


def build_calculator_link(sector: str | None, extra_context: dict[str, str]) -> str | None:
    """
    If this sector has a configured calculator, build a relative link to its
    "<slug>-calculator.html" page, pre-filled with whatever onboarding
    answers we already have (matched by context_label -> subquestion id).
    Returns None if the sector has no calculator configured.
    """
    cfg = POLICY_SUBQUESTIONS.get(sector) if sector else None
    if not cfg or not cfg.get("calculator_slug"):
        return None

    params = {}
    for q in cfg.get("questions", []):
        label = q.get("context_label", q["id"])
        if extra_context.get(label):
            params[q["id"]] = extra_context[label]

    slug = cfg["calculator_slug"]
    query = urlencode(params)
    return f"{slug}-calculator.html" + (f"?{query}" if query else "")


def build_calculator_note(sector: str | None, extra_context: dict[str, str]) -> str:
    """
    For any sector with a configured calculator, forbid the model from
    computing or estimating a specific rupee figure itself and point it at
    the real calculator instead. This exists because the official upstream
    calculators apply extra rules (caps, thresholds, special packages) that
    aren't fully captured in the policy text, and an LLM guessing a formula
    (e.g. a linear proportion by bed count that appears nowhere in the
    context) has produced confidently wrong numbers in testing.

    The onboarding subquestions (facility type, district category) only
    cover *some* of the calculator's required fields — things like total
    beds or project cost usually only surface later, typed as free text in
    chat (e.g. "my TPC is 30 crore"). The model has already parsed that from
    the conversation, so rather than losing it, this note tells the model
    exactly which extra query-param keys it's allowed to append to the base
    link for any such field it's confident the user actually stated.
    """
    cfg = POLICY_SUBQUESTIONS.get(sector) if sector else None
    if not cfg or not cfg.get("calculator_slug"):
        return ""

    calc_link = build_calculator_link(sector, extra_context)
    calc_cfg = POLICY_CALCULATORS.get(cfg["calculator_slug"])
    if not calc_link or not calc_cfg:
        return ""

    onboarding_field_ids = {q["id"] for q in cfg.get("questions", [])}
    remaining_fields = [f for f in calc_cfg.get("requires", []) if f not in onboarding_field_ids]

    extra_fields_block = ""
    if remaining_fields:
        remaining_desc = ", ".join(f'"{f}"' for f in remaining_fields)
        example_extra = "&".join(f"{f}=<number>" for f in remaining_fields)
        joiner = "&" if "?" in calc_link else "?"
        example_link = f"{calc_link}{joiner}{example_extra}"
        extra_fields_block = f"""
  If the user has ALSO stated any of these remaining fields anywhere in
  this conversation (onboarding details or something typed in chat, e.g.
  "55 beds" or "my TPC is 30 crore") — {remaining_desc} — append them to
  that SAME link as extra query parameters, using exactly those field names
  as keys and plain numeric values only (no ₹ symbol, no commas, no words
  like "crore" or "beds" in the value itself). A fully filled-in link looks
  like: {example_link}
  Only include a field you are confident the user actually stated — never
  guess or invent a value just to fill a field.
"""

    return f"""
CALCULATOR RULE (non-negotiable, overrides anything else in this prompt):
This sector has an official government calculator that computes the exact
incentive/subsidy amount. You must NEVER state, compute, or estimate a
specific rupee/lakh/crore figure yourself for this sector — not even by
scaling a number in the context using a formula the user or you invent
(e.g. proportioning a capped amount by bed count, area, or any other
factor) unless that exact scaling formula is explicitly written in the
context. Estimating produces numbers that can be materially wrong, because
the real calculator applies rules not fully captured in the policy text.
Instead:
- Explain qualitatively what the incentive depends on, using only what is
  explicitly stated in the context (e.g. "it's the lower of a fixed cap or
  a percentage of your total project cost, for facilities up to a certain
  size").
- Point the user to the calculator for the exact, officially computed
  figure, as a markdown link with clear anchor text, e.g.
  [Open the incentive calculator]({calc_link})
{extra_fields_block}- If the calculator still needs an input nobody has given yet in this
  conversation, ask for it as a clarifying question instead of guessing —
  but even once you have every input, still do not do the math yourself;
  direct them to the calculator for the final number.
"""


# -------------------------
# LAYER 1 — LLM FILE SELECTION
# -------------------------

def select_relevant_files(question: str, sector: str | None, extra_context: dict[str, str], history: str) -> tuple[list[str], str]:
    candidate_files = expand_with_pairs(ALL_FILENAMES)

    catalogue = "\n".join(
        f'- "{fn}" | type={META_BY_FILE[fn].get("type")} | sector={META_BY_FILE[fn].get("sector")} | {META_BY_FILE[fn].get("summary", "")}'
        for fn in candidate_files if fn in META_BY_FILE
    )

    sector_note = build_onboarding_note(sector, extra_context)

    prompt = f"""You are routing a question to the correct Madhya Pradesh government
policy documents, and preparing a search query for vector retrieval.

Catalogue (ALL candidate documents):
{catalogue}
{sector_note}
Previous conversation:
{history if history else "None"}

Question: {question}

Return ONLY a JSON object with two fields:
- "files": array of exact filenames from the catalogue relevant to answering
  this question. Include a policy AND its paired scheme when both could
  contain relevant provisions. If genuinely unsure, include more rather than
  fewer.
- "search_query": a rewritten, self-contained, keyword-rich version of the
  question suitable for semantic vector search. Resolve vague references
  ("this", "that", "expansion") using the conversation history. Use policy
  terminology. Do not answer the question — only rewrite it.

Example:
{{"files": ["Industrial Promotion Policy 2025.pdf"], "search_query": "interest subsidy eligibility for new manufacturing unit"}}"""

    fallback_files = files_for_sector(sector) if sector else candidate_files
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        selected = [fn for fn in parsed.get("files", []) if fn in META_BY_FILE]
        search_query = parsed.get("search_query", "").strip() or question
        if not selected:
            selected = fallback_files
        print(f"\nLAYER 1 SELECTED FILES: {selected}")
        print(f"LAYER 1 SEARCH QUERY: {search_query}")
        return selected, search_query
    except Exception as e:
        print(f"Layer 1 selection failed, falling back to sector candidates: {e}")
        return fallback_files, question

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
                filter={"source": f"docs\\{fn}"}
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
        results = vectorstore.similarity_search(question, k=5)
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
@limiter.limit("10/minute")
def chat(request: Request, data: ChatRequest):
    if not data.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    actual_question, sector, extra_context, respond_in_hindi = parse_context(data.question)

    history = session_histories.get(data.session_id, [])
    history_text = "\n".join(history)

    print(f"\n{'='*50}")
    print(f"SESSION:  {data.session_id}")
    print(f"QUESTION: {actual_question}")
    print(f"SECTOR:   {sector}")
    print(f"EXTRA:    {extra_context}")
    print(f"HINDI:    {respond_in_hindi}")
    print('='*50)

    # LAYER 1
    selected_files, search_query = select_relevant_files(actual_question, sector, extra_context, history_text)

# LAYER 2
    layer2_docs, searched_files = deep_search(search_query, selected_files)

# FALLBACK
    if not layer2_docs:
        print("\nLayer 2 returned no results — falling back to unfiltered search of all docs")
        try:
            layer2_docs = vectorstore.max_marginal_relevance_search(search_query, k=12, fetch_k=40)
        except Exception as e:
            print(f"Fallback full search failed: {e}")
            layer2_docs = []
        searched_files = ALL_FILENAMES

# LAYER 3
    layer3_docs = light_search_remaining(search_query, searched_files)

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
        return {"type": "answer", "answer": msg, "suggestions": []}

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

    language_instruction = "\nRespond entirely in Hindi. Use formal Hindi suitable for a government context.\n" if respond_in_hindi else ""

    sector_note = build_onboarding_note(sector, extra_context)
    calculator_note = build_calculator_note(sector, extra_context)

    prompt = f"""You are an investment advisor helping someone evaluate setting up
or expanding a business in Madhya Pradesh, using only the government policy
excerpts provided below as your source of truth.
{sector_note}{calculator_note}{language_instruction}
Grounding rules (non-negotiable):
- Answer ONLY using information explicitly stated in the context below.
- Do not infer, guess, or fill in eligibility, incentives, deadlines, or
  conditions that are not explicitly stated.
- NEVER invent a calculation, scaling method, or proportion that is not
  explicitly written in the context — for example, do not assume an amount
  scales linearly with bed count, area, headcount, or any other factor
  unless the context states that scaling rule in those terms. If the user
  asks you to compute a specific figure and the context doesn't give an
  explicit method for that exact computation, say so plainly instead of
  producing a number.
- If the context does not explicitly answer part of the question, say plainly
  that the policy does not specify this — do not speculate or hedge with
  phrases like "it seems" or "it appears."
- Only if no relevant provisions exist at all for the question, reply with
  exactly this sentence and nothing else: {FALLBACK_PHRASE}

Style guidelines:
- Write like an investment advisor talking to a business owner, not like you
  are summarizing a policy PDF. Use plain business language, not legal or
  policy wording, and don't copy provisions verbatim unless a precise figure,
  percentage, or deadline needs to stay exact.
- Answer the user's actual question directly in the first sentence or two,
  before giving supporting details.
- Only mention incentives, conditions, or schemes directly relevant to what
  the user is asking. If something is technically in the context but
  unrelated (e.g. an apparel training subsidy when the user asked about
  manufacturing), leave it out.
- If there are additional benefits that only apply in specific situations
  (a certain sector, district category, or investment size), put those
  separately under an "Additional benefits (if applicable)" heading rather
  than mixing them into the main answer.
- Use bold for specific amounts, percentages, and deadlines. Use bullet
  points for lists. Only add ## headings when the answer genuinely covers
  multiple distinct topics — don't force structure onto a short answer.
- When it fits naturally, structure the answer as: (1) a direct answer,
  (2) key incentives/conditions, (3) important limitations or eligibility
  conditions, (4) next steps, if the user is asking how to proceed.

Question:
{actual_question}

Previous Conversation:
{history_text if history_text else "None"}

Context:
{context}

Cross-questioning (applies to every sector, use sparingly):
- Most questions should get a direct answer. Only ask a clarifying question
  when the context genuinely contains DIFFERENT answers depending on a
  detail the user hasn't given (e.g. which sector/scheme, project type,
  district category, investment size, new vs. expansion unit) — not just
  because more detail would be "nice to have."
- If the user is asking you to calculate or state a specific figure, and
  that figure genuinely depends on a number they haven't given (e.g. total
  project cost, investment amount, capacity) per the context's own formula
  (such as "lower of X or a percentage of TPC"), you MUST ask for that
  number — do not answer with a caveated guess, and do not silently assume
  a value.
- If the question is too vague to search meaningfully at all (e.g. "tell me
  about incentives" with no sector/topic), that also warrants a clarifying
  question instead of guessing.
- A clarifying question should be ONE focused question, not a list of
  questions.
- Never ask for clarification just to pad the conversation, and never ask
  about something the onboarding context or previous conversation already
  answered.

Output format (non-negotiable):
Respond with ONLY a single valid JSON object — no markdown code fences, no
text outside the JSON — with exactly these fields:
{{
  "type": "answer" or "clarify",
  "message": "the full grounded answer (markdown allowed) if type is
    \\"answer\\", OR a single focused clarifying question if type is
    \\"clarify\\"",
  "suggestions": [ up to 4 short strings. If type is "clarify", these are
    likely answer options the user can tap instead of typing (pull them from
    the context/catalogue where possible, e.g. specific district categories,
    project types, or investment ranges actually mentioned). If type is
    "answer", these are short natural follow-up questions the user might
    reasonably want to ask next, grounded in what is actually in the
    context. Use an empty array if nothing sensible applies. ]
}}

Answer:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    print("\nGPT RAW ANSWER:")
    print(raw)

    # Parse the structured {type, message, suggestions} response. If the
    # model ever produces something that isn't valid JSON, fall back to
    # treating the raw text as a plain "answer" with no suggestions, rather
    # than breaking the response entirely.
    response_type = "answer"
    suggestions: list[str] = []
    answer = raw
    try:
        cleaned = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(cleaned)
        if parsed.get("type") in ("answer", "clarify"):
            response_type = parsed["type"]
        answer = (parsed.get("message") or "").strip() or raw
        suggestions = [
            s.strip() for s in parsed.get("suggestions", [])
            if isinstance(s, str) and s.strip()
        ][:4]
    except Exception as e:
        print(f"Structured answer parse failed, treating raw output as a plain answer: {e}")

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

    if response_type == "clarify":
        # A clarifying question has no document context to cite yet.
        final_answer = answer
    elif FALLBACK_PHRASE in answer:
        final_answer = answer
        suggestions = []
    else:
        final_answer = f"{answer}\n\n---\n\n**References**\n{citation_text}"

    history.append(f"User: {actual_question}")
    history.append(f"Assistant: {answer}")
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    session_histories[data.session_id] = history

    return {"type": response_type, "answer": final_answer, "suggestions": suggestions}


# -------------------------
# GET /api/v1/sectors
# -------------------------

@app.get("/api/v1/sectors")
def get_sectors():
    return {
        "sectors": list(SECTOR_MAP.keys()),
        "policy_subquestions": POLICY_SUBQUESTIONS,
    }


# -------------------------
# GET /api/v1/health
# -------------------------

@app.get("/api/v1/health")
def health():
    return {"status": "ok", "docs_in_store": vectorstore._collection.count()}


# -------------------------
# UPSTREAM SSL CONTEXT
# sws.invest.mp.gov.in's TLS stack requires legacy renegotiation, which
# OpenSSL 3.x refuses by default (raises
# "SSL: UNSAFE_LEGACY_RENEGOTIATION_DISABLED"). This explicitly re-allows
# it for calls to that host only — it does NOT disable certificate
# verification, just the renegotiation restriction.
# -------------------------

def _build_legacy_ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.options |= 0x4  # ssl.OP_LEGACY_SERVER_CONNECT
    return ctx


_UPSTREAM_SSL_CONTEXT = _build_legacy_ssl_context()


# -------------------------
# POLICY CALCULATOR REGISTRY
# One entry per policy that has an official MP Invest calculator.
# "field_map": our subquestion/form field id -> their exact API field name
# "transform": optional per-field conversion before sending upstream
# "requires": which of our fields must be non-empty to call their API
# -------------------------

POLICY_CALCULATORS = {
    "health": {
        "upstream_url": "https://sws.invest.mp.gov.in/api/api/v1/incentives/health-incentive",
        "field_map": {
            "facility_type": "project_type",
            "district_category": "category",
            "total_no_of_beds": "total_no_of_beds",
            "investment_amount_cr": "total_amount_investment",
        },
        "transform": {
            # our field is entered in ₹ Crore; their API expects plain rupees
            "investment_amount_cr": lambda v: str(int(float(v) * 1_00_00_000)),
        },
        "requires": ["facility_type", "district_category", "total_no_of_beds", "investment_amount_cr"],
    }
}


class CalculateRequest(BaseModel):
    values: dict[str, str]


@app.post("/api/v1/calculate/{policy_slug}")
@limiter.limit("15/minute")
def calculate_subsidy(request: Request, policy_slug: str, data: CalculateRequest):
    cfg = POLICY_CALCULATORS.get(policy_slug)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"No calculator configured for '{policy_slug}'.")

    missing = [f for f in cfg["requires"] if not data.values.get(f)]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required field(s): {', '.join(missing)}")

    payload = {}
    for our_id, their_key in cfg["field_map"].items():
        raw_val = data.values.get(our_id, "")
        transform = cfg.get("transform", {}).get(our_id)
        try:
            payload[their_key] = transform(raw_val) if transform else raw_val
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"Invalid value for '{our_id}'.")

    try:
        with httpx.Client(verify=_UPSTREAM_SSL_CONTEXT, timeout=15) as upstream:
            resp = upstream.post(cfg["upstream_url"], json=payload)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Calculator service returned an error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Could not reach calculator service: {e}")