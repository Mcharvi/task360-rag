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
    "http://localhost:5500,http://127.0.0.1:5500" 
    "https://task360-rag.onrender.com" # dev defaults
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
# PARENT SECTIONS (optional)
# Only present if your ingest step ever writes parent_sections.json. Loaded
# defensively so this keeps working unchanged against your current/old
# ingest.py, which doesn't produce this file — PARENT_SECTIONS is just {}
# in that case, and every retrieved chunk falls through to the plain
# "page + raw chunk text" path exactly like before. Nothing else changes
# unless you later opt into an ingest that populates it.
# -------------------------

PARENT_SECTIONS_PATH = "parent_sections.json"
if os.path.exists(PARENT_SECTIONS_PATH):
    with open(PARENT_SECTIONS_PATH, encoding="utf-8") as f:
        PARENT_SECTIONS = json.load(f)
else:
    PARENT_SECTIONS = {}

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
# both the RAG context and the calculator pre-fill.
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
    },
    "Tourism & Film": {
        "calculator_slug": "film",
        "questions": [
            {
                "id": "category",
                "context_label": "Production Category",
                "type": "buttons",
                "label": "What are you shooting?",
                "label_hi": "आप क्या शूट कर रहे हैं?",
                "options": [
                    "Feature Film / Web Series / Documentary / Short Film",
                    "International Film / TV Serial",
                ],
            },
            {
                "id": "film_type",
                "context_label": "Type of Film",
                "type": "select",
                "label": "What type of film is this?",
                "label_hi": "यह किस प्रकार की फिल्म है?",
                "options": [
                    "First Feature Film",
                    "Second Feature Film",
                    "Third Feature Film",
                    "Web Series / OTT Original Show",
                    "Documentary Film",
                    "Short Film",
                ],
            },
            {
                "id": "total_shooting_days",
                "context_label": "Total Shooting Days",
                "type": "number",
                "label": "How many total shooting days is the project?",
                "label_hi": "परियोजना के लिए कुल शूटिंग दिन कितने हैं?",
                "placeholder": "e.g. 100",
            },
            {
                "id": "total_cost_of_film",
                "context_label": "Total Cost of the Film (in Cr)",
                "type": "number",
                "label": "What is the total cost of the film (in ₹ Crore)?",
                "label_hi": "फिल्म की कुल लागत क्या है (₹ करोड़ में)?",
                "placeholder": "e.g. 20",
            },
        ],
    },
    "MSME": {
        "calculator_slug": "msme",
        "questions": [
            {
                "id": "investment_in_plant_machinery",
                "context_label": "Investment in P&M (in Cr)",
                "type": "number",
                "label": "What is your investment in Plant & Machinery (in ₹ Crore)?",
                "label_hi": "प्लांट और मशीनरी में आपका निवेश कितना है (₹ करोड़ में)?",
                "placeholder": "e.g. 500",
            },
            {
                "id": "export_percentage",
                "context_label": "Export Percentage",
                "type": "number",
                "label": "What percentage of your production do you export?",
                "label_hi": "आप अपने उत्पादन का कितना प्रतिशत निर्यात करते हैं?",
                "placeholder": "e.g. 30",
            },
            {
                "id": "industry_type",
                "context_label": "Type of Industry",
                "type": "select",
                "label": "What type of industry is this?",
                "label_hi": "यह किस प्रकार का उद्योग है?",
                "options": [
                    "Pharmaceutical and Medical Device Manufacturing Units",
                    "Food Processing Units",
                    "Apparel Sector",
                    "Textile Units",
                    "Powerloom Sector",
                    "Footware, Furniture, Toys and Related Value Chain Products",
                    "General",
                ],
            },
        ],
    },


    "Manufacturing & Industry": {
        "calculator_slug": "dipip",
        "questions": [
            {
                "id": "investment",
                "context_label": "Project Investment (in Cr)",
                "type": "number",
                "label": "What is your total project investment (in ₹ Crore)?",
                "label_hi": "आपका कुल परियोजना निवेश कितना है (₹ करोड़ में)?",
                "placeholder": "e.g. 500",
            },
        ],
    },
}


FALLBACK_PHRASE = "I could not find the answer in the provided documents."


# -------------------------
# PARSE CONTEXT PREFIX
# Frontend sends:
# [Context: Sector = X | Facility Type = Y | District Category = Z | Language = hindi]\n\n<question>
# -------------------------

def _known_context_keys() -> set[str]:
    keys = {"Sector", "Language"}
    for cfg in POLICY_SUBQUESTIONS.values():
        for q in cfg.get("questions", []):
            keys.add(q.get("context_label", q["id"]))
    return keys


def parse_context(raw_question: str):
    """
    Frontend context prefix looks like:
    [Context: Sector = X, Facility Type = Y, District Category = Z, Language = hindi]

    IMPORTANT: this does NOT rely on a single delimiter (comma or pipe)
    between "key = value" pairs. Some sector names contain a literal comma
    themselves (e.g. "IT, Tech & Digital"), and a plain delimiter split
    can't tell that comma apart from a real pair separator — whichever
    character you pick, some value can contain it. Instead we scan for
    known key names directly (Sector, Language, and every subquestion's
    context_label) and treat the text between one recognized "<key> ="
    and the next as that key's value. This works regardless of whether
    the frontend separates pairs with ",", "|", or nothing at all.
    """
    ctx_match = re.match(r"\[Context:(.*?)\]\n\n", raw_question, re.DOTALL)

    if not ctx_match:
        return raw_question.strip(), None, {}, False

    ctx = ctx_match.group(1)
    actual_question = raw_question[ctx_match.end():].strip()

    known_keys = _known_context_keys()
    # Longest-first so a key that's a prefix of another (unlikely here, but
    # cheap insurance) can't match short and steal part of the real value.
    key_pattern = "|".join(re.escape(k) for k in sorted(known_keys, key=len, reverse=True))
    boundary_re = re.compile(rf"({key_pattern})\s*=\s*")
    matches = list(boundary_re.finditer(ctx))

    respond_in_hindi = False
    sector: str | None = None
    extra_context: dict[str, str] = {}

    for i, m in enumerate(matches):
        key = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(ctx)
        # Trim a trailing pair-separator (comma or pipe) and whitespace,
        # whichever the frontend happens to be using.
        val = ctx[start:end].strip().rstrip("|,").strip()
        if not val:
            continue
        if key == "Language":
            if val.lower() == "hindi":
                respond_in_hindi = True
        elif key == "Sector":
            sector = val
        else:
            extra_context[key] = val

    return actual_question, sector, extra_context, respond_in_hindi


# -------------------------
# Deterministic backstop for the most common "I already told you that"
# clarify-loop complaint: export percentage stated in free text (e.g.
# "30% export") wasn't reliably picked up by LLM judgement alone, and
# could get re-asked on a later turn. This folds an explicit match into
# extra_context before Layer 1 / the answer prompt ever see the question,
# so it's treated exactly like an onboarding-provided value. Deliberately
# narrow — one field, one known failure — not a general NLU extractor.
# -------------------------

_EXPORT_PATTERNS = [
    re.compile(r"(\d{1,3})\s*%\s*(?:of\s+(?:its\s+|their\s+)?(?:total\s+)?(?:sales|production|turnover|output))", re.IGNORECASE),
    re.compile(r"export(?:s|ing)?\s*(?:percentage|percent)?\s*(?:is|of|=|:)\s*(\d{1,3})\s*%", re.IGNORECASE),
    re.compile(r"(\d{1,3})\s*%\s*export", re.IGNORECASE),
    re.compile(r"export\s*percentage\s*is\s*(\d{1,3})\s*%?", re.IGNORECASE),
]


def extract_explicit_fields(question: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for pattern in _EXPORT_PATTERNS:
        m = pattern.search(question)
        if m:
            try:
                val = int(m.group(1))
            except ValueError:
                continue
            if 0 <= val <= 100:
                found["Export Percentage"] = str(val)
            break
    return found


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

    Tightened from the original soft wording: a generic question ("I'm
    setting up a general MSME unit...") was still getting answered as if
    it were about whichever sub-sector happened to be sitting in
    onboarding context from earlier in the session. The model must now
    positively check whether the CURRENT question actually names/implies
    that detail before using it, and generic phrasing ("general", "a new
    unit", "my unit") is an explicit signal to ignore stale context. Also
    tells the model to redirect rather than answer off unrelated docs if
    the question is clearly about a different sector.
    """
    if not sector:
        return ""

    note = f"""
The user has already selected the "{sector}" sector as onboarding context.
Treat this as a strong prior: prefer documents/information related to this
sector. However, if the question clearly needs provisions from another
sector to be answered completely, include those too.

If the question is clearly about a different sector/policy than "{sector}"
(for example a named sector that isn't this one), do not attempt to answer
using unrelated documents. Instead, say this question falls outside the
"{sector}" sector they've selected, and suggest they switch their sector
selection to ask about it.
"""
    if extra_context:
        extra_lines = "\n".join(f"- {k}: {v}" for k, v in extra_context.items())
        note += f"""
The user also provided these onboarding details:
{extra_lines}
Before using any of these in your answer, check whether the CURRENT
question actually names, implies, or asks about that specific detail (e.g.
a named sub-sector/product, a facility type, a district category). Only
use a detail if that check passes.
If the question is phrased generically — for example it says "general",
"a new unit", "my unit", "a business", or otherwise does NOT name a
specific sub-sector/product/category — treat the question as
detail-agnostic for that field, even though the detail exists in this
session's onboarding context. Do NOT silently substitute a stale
onboarding detail into an answer for a question that never mentioned it.
When in doubt, leave the detail out rather than assume it applies.
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
    the real calculator instead — the upstream calculators apply rules
    (caps, thresholds, special packages) not fully captured in the policy
    text, and a guessed formula has produced confidently wrong numbers.
    """
    cfg = POLICY_SUBQUESTIONS.get(sector) if sector else None
    if not cfg or not cfg.get("calculator_slug"):
        return ""

    calc_link = build_calculator_link(sector, extra_context)
    calc_cfg = POLICY_CALCULATORS.get(cfg["calculator_slug"])
    if not calc_link or not calc_cfg:
        return ""

    onboarding_field_ids = {q["id"] for q in cfg.get("questions", [])}
    requires = calc_cfg.get("requires", [])
    remaining_fields = [f for f in requires if f not in onboarding_field_ids] if not callable(requires) else []

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
context. Instead:
- Explain qualitatively what the incentive depends on, using only what is
  explicitly stated in the context.
- Point the user to the calculator for the exact, officially computed
  figure, as a markdown link with clear anchor text, e.g.
  [Open the incentive calculator]({calc_link})
{extra_fields_block}- If the calculator still needs an input nobody has given yet in this
  conversation, ask for it as a clarifying question instead of guessing —
  but even once you have every input, still do not do the math yourself;
  direct them to the calculator for the final number.
"""


# -------------------------
# LAYER 1 — LLM FILE SELECTION + QUERY PLANNING
#
# Returns sub_queries (1-4) instead of one blended search query. A single
# merged query for "Compare A vs B" style questions can retrieve mostly
# whichever entity's language embeds closer, starving the other entity out
# of the top-k. If 2+ distinct entities are named/compared, Layer 1 now
# returns one self-contained query per entity; deep_search runs each
# separately and merges the results. For an ordinary question, sub_queries
# is just a single-item list — no behavior change there.
# -------------------------

def select_relevant_files(question: str, sector: str | None, extra_context: dict[str, str], history: str) -> tuple[list[str], list[str]]:
    if sector:
        candidate_files = expand_with_pairs(files_for_sector(sector))
    else:
        candidate_files = expand_with_pairs(ALL_FILENAMES)

    catalogue = "\n".join(
        f'- "{fn}" | type={META_BY_FILE[fn].get("type")} | sector={META_BY_FILE[fn].get("sector")} | {META_BY_FILE[fn].get("summary", "")}'
        for fn in candidate_files if fn in META_BY_FILE
    )

    sector_note = build_onboarding_note(sector, extra_context)

    prompt = f"""You are routing a question to the correct Madhya Pradesh government
policy documents, and preparing search queries for vector retrieval.

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
  fewer. If the user has an onboarding sector selected, that sector's own
  policy file must always be included even if you also include others.
- "sub_queries": array of 1 to 4 rewritten, self-contained, keyword-rich
  search queries suitable for semantic vector search. Resolve vague
  references ("this", "that", "expansion") using the conversation history.
  Use policy terminology. Do not answer the question — only rewrite it.
  IMPORTANT: if the question names or compares 2 or more distinct entities
  (e.g. "Healthcare vs MSME", "Facility A, B, and C"), return ONE separate,
  self-contained query PER entity (each naming that entity explicitly), so
  each gets its own retrieval instead of being blended into one averaged
  query. Otherwise return a single-item array with one query.

Example (comparison question):
{{"files": ["Health Sector Investment Promotion policy.pdf", "MSME Development Policy 2025.pdf"], "sub_queries": ["Healthcare sector capital subsidy eligibility", "MSME capital subsidy eligibility"]}}

Example (single-topic question):
{{"files": ["Industrial Promotion Policy 2025.pdf"], "sub_queries": ["interest subsidy eligibility for new manufacturing unit"]}}"""

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
        sub_queries = [
            q.strip() for q in parsed.get("sub_queries", [])
            if isinstance(q, str) and q.strip()
        ][:4]
        if not sub_queries:
            sub_queries = [question]
        if not selected:
            selected = fallback_files
        # Belt-and-suspenders: if the user has an onboarding sector, always
        # make sure that sector's own file(s) are in the selected set,
        # regardless of what Layer 1 returned. Cheap, and prevents the
        # model confidently answering from an unrelated policy just
        # because Layer 1 missed the obvious file.
        if sector:
            for fn in files_for_sector(sector):
                if fn not in selected:
                    selected.append(fn)
        print(f"\nLAYER 1 SELECTED FILES: {selected}")
        print(f"LAYER 1 SUB-QUERIES: {sub_queries}")
        return selected, sub_queries
    except Exception as e:
        print(f"Layer 1 selection failed, falling back to sector candidates: {e}")
        return fallback_files, [question]


# -------------------------
# LAYER 2 — DEEP SEARCH ON SELECTED FILES
# Runs each sub-query separately per file and merges — see Layer 1 note.
# -------------------------

def deep_search(sub_queries: list[str], filenames: list[str]):
    filenames = expand_with_pairs(filenames)
    docs = []
    for query in sub_queries:
        for fn in filenames:
            try:
                results = vectorstore.max_marginal_relevance_search(
                    query, k=8, fetch_k=30,
                    filter={"source": f"docs\\{fn}"}
                )
                docs.extend(results)
            except Exception as e:
                print(f"Deep search failed for {fn}: {e}")
    return docs, filenames


# -------------------------
# LAYER 3 — LIGHT SEARCH ACROSS REMAINING DOCS
# Adds a relevance-score floor so Layer 3 doesn't drag in loosely-related
# noise from unrelated policies, and only fires when Layer 2 came up short
# (or no sector is set) — avoids reopening the whole corpus on questions
# that already got a strong sector-scoped answer.
# -------------------------

LAYER3_MIN_DOCS = 4
LAYER3_SCORE_FLOOR = 0.35  # tune against your embedding scale

def light_search_remaining(question: str, already_searched: list[str], score_floor: float = LAYER3_SCORE_FLOOR):
    try:
        results = vectorstore.similarity_search_with_relevance_scores(question, k=8)
    except Exception as e:
        print(f"Layer 3 light search failed: {e}")
        return []
    remaining = []
    for doc, score in results:
        source = doc.metadata.get("source", "")
        if any(fn in source for fn in already_searched):
            continue
        if score < score_floor:
            continue
        remaining.append(doc)
    return remaining[:3]


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

    for key, val in extract_explicit_fields(actual_question).items():
        extra_context.setdefault(key, val)

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
    selected_files, sub_queries = select_relevant_files(actual_question, sector, extra_context, history_text)

    # LAYER 2
    layer2_docs, searched_files = deep_search(sub_queries, selected_files)

    # FALLBACK
    if not layer2_docs:
        print("\nLayer 2 returned no results — falling back to unfiltered search of all docs")
        try:
            layer2_docs = []
            for query in sub_queries:
                layer2_docs.extend(
                    vectorstore.max_marginal_relevance_search(query, k=12, fetch_k=40)
                )
        except Exception as e:
            print(f"Fallback full search failed: {e}")
            layer2_docs = []
        searched_files = ALL_FILENAMES

    # LAYER 3 — only reopen the corpus if sector is unknown or Layer 2 came up short
    if sector and len(layer2_docs) >= LAYER3_MIN_DOCS:
        layer3_docs = []
    else:
        layer3_docs = light_search_remaining(sub_queries[0], searched_files)

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

    # RERANK — wider window when Layer 1 detected a multi-entity comparison,
    # so a 2nd/3rd compared entity doesn't get squeezed out of a fixed top_n.
    rerank_top_n = 8 if len(sub_queries) > 1 else 6
    try:
        rerank_response = co.rerank(
            model="rerank-v3.5",
            query=actual_question,
            documents=[doc.page_content for doc in all_docs][:100],
            top_n=rerank_top_n
        )
        retrieved_docs = [all_docs[r.index] for r in rerank_response.results]
        print(f"\nRERANK SCORES: {[round(r.relevance_score, 4) for r in rerank_response.results]}")
    except Exception as e:
        print(f"Reranking failed: {e}")
        retrieved_docs = all_docs[:rerank_top_n]

    # Context building — parent-aware when available, otherwise identical
    # to the plain per-chunk behavior you already had.
    seen_parents = set()
    context_parts = []
    for doc in retrieved_docs:
        parent_id = doc.metadata.get("parent_id")
        parent = PARENT_SECTIONS.get(parent_id) if parent_id else None
        if parent and parent_id not in seen_parents:
            seen_parents.add(parent_id)
            source = parent["source"].replace("\\", "/").split("/")[-1]
            context_parts.append(f"SOURCE: {source} | PAGE {parent['page'] + 1} | SECTION: {parent['heading']}\n\n{parent['text']}")
        elif not parent:
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
  explicitly written in the context. If the user asks you to compute a
  specific figure and the context doesn't give an explicit method for that
  exact computation, say so plainly instead of producing a number.
- When the context explicitly states a numeric cap, limit, percentage, or
  other figure, copy that figure EXACTLY as written, digit for digit — do
  not re-type it from memory of having read it, round it, or rescale it.
  If the same figure appears more than once and looks inconsistent, prefer
  the version embedded in a complete sentence over one that looks
  truncated, and if you're not confident which is correct, say the context
  gives a cap in this range/section without stating the exact digits,
  rather than picking one.
- If two or more excerpts describe what looks like the same kind of
  incentive but use DIFFERENTLY STRUCTURED mechanisms, do NOT merge them
  into a single answer — identify which SOURCE each mechanism comes from
  and use the one that matches the user's onboarding sector / the policy
  the question is actually about. Only mention a mechanism from a
  different policy if you clearly attribute it by name.
- If the context does not explicitly answer part of the question, say plainly
  that the policy does not specify this — do not speculate or hedge with
  phrases like "it seems" or "it appears."
- Only if no relevant provisions exist at all for the question, reply with
  exactly this sentence and nothing else: {FALLBACK_PHRASE}

Style guidelines:
- Write like an investment advisor talking to a business owner, not like you
  are summarizing a policy PDF. Use plain business language, and don't copy
  provisions verbatim unless a precise figure, percentage, or deadline
  needs to stay exact.
- Answer the user's actual question directly in the first sentence or two,
  before giving supporting details.
- Only mention incentives, conditions, or schemes directly relevant to what
  the user is asking. If something is technically in the context but
  unrelated, leave it out.
- If there are additional benefits that only apply in specific situations,
  put those separately under an "Additional benefits (if applicable)"
  heading — but ONLY if that benefit belongs to the same incentive
  category/topic the user is actually asking about. If it belongs to a
  different category, leave it out of this answer entirely rather than
  surfacing it as a tangent.
- Use bold for specific amounts, percentages, and deadlines. Use bullet
  points for lists. Only add ## headings when the answer genuinely covers
  multiple distinct topics — don't force structure onto a short answer.
- If the question explicitly names 2 or more categories/entities to cover,
  structure the answer with one clearly labeled heading or bolded lead-in
  per named category, in the order the user listed them — do not merge
  them into a single undifferentiated list. If a named category genuinely
  has no information in the context, still give it its own heading and say
  so explicitly, rather than omitting it.
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
  detail the user hasn't given — not just because more detail would be
  "nice to have."
- Before asking for ANY field, check three places, in this exact order, and
  stop checking as soon as you find it: (1) the onboarding context above,
  (2) the Previous Conversation below, (3) the current Question text
  itself — including anything the user stated in free text, in any
  phrasing. If the value appears in ANY of these three places, use it
  directly and do NOT ask for it — even if it was phrased differently than
  you'd expect, and even if this is the first turn.
- If the user is asking you to calculate or state a specific figure, and
  that figure genuinely depends on a number they haven't given anywhere
  (per the three checks above) per the context's own formula, you MUST ask
  for that number — do not answer with a caveated guess.
- If the question is too vague to search meaningfully at all, that also
  warrants a clarifying question instead of guessing.
- A clarifying question should be ONE focused question, not a list.
- Never ask about something the onboarding context, previous conversation,
  or the question itself already answered.

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
    the context/catalogue where possible). If type is "answer", these are
    short natural follow-up questions the user might reasonably want to ask
    next, grounded in what is actually in the context. Use an empty array
    if nothing sensible applies. ]
}}

Answer:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
# OpenSSL 3.x refuses by default. This explicitly re-allows it for calls to
# that host only — it does NOT disable certificate verification.
# -------------------------

def _build_legacy_ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.options |= 0x4  # ssl.OP_LEGACY_SERVER_CONNECT
    return ctx


_UPSTREAM_SSL_CONTEXT = _build_legacy_ssl_context()


# -------------------------
# POLICY CALCULATOR REGISTRY
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
            "investment_amount_cr": lambda v: str(int(float(v) * 1_00_00_000)),
        },
        "requires": ["facility_type", "district_category", "total_no_of_beds", "investment_amount_cr"],
    },
    "film": {
        "upstream_url": "https://sws.invest.mp.gov.in/api/api/v1/incentives/film-subsidy",
        "field_map": {
            "film_type": "film_type",
            "total_shooting_days": "total_shooting_days",
            "shooting_days_in_mp": "shooting_days_in_mp",
            "total_cost_of_film": "total_cost_of_film",
            "expenditure_in_mp": "expenditure_in_mp",
            "additional_grant_type": "additional_grant_type",
            "shooting_permission_fees": "shooting_permission_fees",
        },
        "transform": {
            "total_shooting_days": lambda v: int(float(v)),
            "shooting_days_in_mp": lambda v: int(float(v)),
            "total_cost_of_film": lambda v: float(v),
            "expenditure_in_mp": lambda v: float(v),
            "shooting_permission_fees": lambda v: float(v),
        },
        "requires": [
            "film_type",
            "total_shooting_days",
            "shooting_days_in_mp",
            "total_cost_of_film",
            "expenditure_in_mp",
            "additional_grant_type",
            "shooting_permission_fees",
        ],
    },
    "msme": {
        "upstream_url": "https://sws.invest.mp.gov.in/api/api/v1/incentives/msme-calculator",
        "field_map": {
            "investment_in_plant_machinery": "investment_in_plant_machinery",
            "export_percentage": "export_percentage",
            "industry_type": "industry_type",
            "yearly_production_capacity": "yearly_production_capacity",
            "current_year_production": "current_year_production",
            "current_year_employment": "current_year_employment",
            "fdi_percentage": "fdi_percentage",
            "category_of_entrepreneur": "category_of_entrepreneur",
        },
        "requires": ["investment_in_plant_machinery", "export_percentage", "industry_type"],
    },
    "dipip": {
        "method": "GET",  # <-- this one hits the upstream as a GET with query params
        "upstream_url": "https://sws.invest.mp.gov.in/api/api/v1/incentives/guest-incentive-calculator",
        "field_map": {
            "investment": "investment",
            "sector_name": "sector_name",
            "is_bulk_order_company": "is_bulk_order_company",
            "yop": "yop",
            "total_employer": "total_employer",
            "is_company_exports": "is_company_exports",
            "is_company_fdi": "is_company_fdi",
            "export_per": "export_per",
            "fdi_per": "fdi_per",
            "is_back_area_company": "is_back_area_company",
            "is_cement_company": "is_cement_company",
            "is_sez": "is_sez",
            "is_carbonated_industry": "is_carbonated_industry",
        },
        # Only the fields that are ALWAYS present are required; the conditional
        # ones (is_bulk_order_company, is_carbonated_industry, is_sez,
        # export_per, fdi_per, is_cement_company) get sensible defaults
        # ("false" / "") from the calculator page itself when not applicable,
        # so the backend doesn't need to force them.
        "requires": [
            "investment", "sector_name", "yop", "total_employer",
            "is_company_exports", "is_company_fdi", "is_back_area_company",
        ],
    },
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

    method = cfg.get("method", "POST").upper()
    try:
        with httpx.Client(verify=_UPSTREAM_SSL_CONTEXT, timeout=30) as upstream:
            if method == "GET":
                resp = upstream.get(cfg["upstream_url"], params=payload)
            else:
                resp = upstream.post(cfg["upstream_url"], json=payload)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        print(f"\n!!! CALCULATOR UPSTREAM ERROR ({policy_slug}): {e.response.status_code} — {e.response.text}\n")
        raise HTTPException(status_code=502, detail=f"Calculator service returned an error: {e.response.text}")
    except httpx.RequestError as e:
        print(f"\n!!! CALCULATOR REQUEST FAILED ({policy_slug}): {type(e).__name__} — {e}\n")
        raise HTTPException(status_code=502, detail=f"Could not reach calculator service: {e}")