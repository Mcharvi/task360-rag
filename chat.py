from dotenv import load_dotenv
load_dotenv()

import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI

# -------------------------
# VALIDATION
# -------------------------

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set.")

client = OpenAI()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

SIMILARITY_THRESHOLD = 1.50

print(f"RAG Chatbot Ready! ({vectorstore._collection.count()} chunks loaded)")
print("Type 'exit' to quit.\n")

chat_history = []

# -------------------------
# QUERY REWRITING
# -------------------------

def rewrite_query(question: str, history: str) -> str:
    if not history:
        return question
    prompt = f"""Given the conversation history and a new question, \
rewrite the question into a fully self-contained standalone question.

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

# -------------------------
# CHAT LOOP
# -------------------------

while True:

    query = input("You: ").strip()

    if not query:
        continue

    if query.lower() == "exit":
        print("Goodbye!")
        break

    history_text = "\n".join(chat_history)

    # --- Rewrite query ---
    rewritten = rewrite_query(query, history_text)
    if rewritten != query:
        print(f"[Rewritten: {rewritten}]")

    # --- Retrieval ---
    try:
        retrieved_docs = vectorstore.max_marginal_relevance_search(
            rewritten, k=5, fetch_k=20
        )
        results = vectorstore.similarity_search_with_score(rewritten, k=5)
    except Exception as e:
        print(f"Retrieval error: {e}")
        continue

    if not results:
        print("BOT: No relevant documents found.\n")
        continue

    # --- Relevance check ---
    results.sort(key=lambda x: x[1])
    best_score = results[0][1]

    print(f"[Scores: {[round(s, 4) for _, s in results]}]")

    if best_score > SIMILARITY_THRESHOLD:
        print("BOT: This question appears unrelated to the uploaded policy documents.\n")
        continue

    # --- Build context ---
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # --- Prompt ---
    prompt = f"""You are a retrieval-based assistant for CA services and policy documents.

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
{query}

Answer:"""

    # --- LLM ---
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM error: {e}")
        continue

    # --- Citations ---
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

    citation_lines = [
        f"[{i}] {s['file']} — Page {s['page']}"
        for i, s in enumerate(unique_sources, start=1)
    ]

    print(f"\nBOT: {answer}")
    print("\nReferences:")
    print("\n".join(citation_lines))
    print()

    # --- Memory ---
    chat_history.append(f"User: {query}")
    chat_history.append(f"Assistant: {answer}")
    if len(chat_history) > 20:
        chat_history = chat_history[-20:]


        #this is chat.py