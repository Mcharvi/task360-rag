from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI

# -------------------------
# OPENAI CLIENT
# -------------------------

client = OpenAI()

# -------------------------
# EMBEDDINGS
# -------------------------

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# -------------------------
# LOAD CHROMA DATABASE
# -------------------------

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

print("RAG Chatbot Ready!")
print("Type 'exit' to quit.")

# -------------------------
# MEMORY
# -------------------------

chat_history = []

# -------------------------
# CHAT LOOP
# -------------------------

while True:

    query = input("\nYou: ")

    if query.lower() == "exit":
        print("\nGoodbye!")
        break

    # -------------------------
    # RETRIEVAL
    # -------------------------

    retrieved_docs = vectorstore.max_marginal_relevance_search(
        query,
        k=5,
        fetch_k=20
    )

    print("\nRETRIEVED CHUNKS:\n")

    results = vectorstore.similarity_search_with_score(
        query,
        k=5
)
    for doc, score in results:
        print(score)
    for i, doc in enumerate(retrieved_docs):
        print(f"\nChunk {i+1}")
        print(doc.page_content[:700])
        print("-" * 80)

    # -------------------------
    # CHECK METADATA
    # -------------------------

    print("\nFIRST DOCUMENT METADATA:")
    print(retrieved_docs[0].metadata)

    # -------------------------
    # BUILD CONTEXT
    # -------------------------

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    # -------------------------
    # BUILD SOURCES
    # -------------------------

    sources = []

    for doc in retrieved_docs:

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0)

        filename = source.split("\\")[-1]
        filename = filename.split("/")[-1]

        sources.append(
            f"{filename} (Page {page + 1})"
        )

    history_text = "\n".join(chat_history)

    # -------------------------
    # PROMPT
    # -------------------------

    prompt = f"""
You are a document question-answering assistant.

You MUST answer ONLY from the provided context.

Rules:
1. Use only the context.
2. Use previous conversation if needed.
3. Do not use outside knowledge.
4. Do not guess.
5. If the answer is not present in the context, reply exactly:

I could not find the answer in the provided documents.

Previous Conversation:
{history_text}

Context:
{context}

Question:
{query}

Answer:
"""

    # -------------------------
    # GPT RESPONSE
    # -------------------------

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    answer = response.output_text

    # -------------------------
    # CITATIONS
    # -------------------------

    unique_sources = list(set(sources))

    citation_text = "\n".join(unique_sources)

    final_answer = f"""
{answer}

Sources:
{citation_text}
"""

    # -------------------------
    # DISPLAY ANSWER
    # -------------------------

    print("\nBOT:")
    print(final_answer)

    # -------------------------
    # SAVE MEMORY
    # -------------------------

    chat_history.append(f"User: {query}")
    chat_history.append(f"Assistant: {answer}")

    # Keep only last 10 exchanges

    if len(chat_history) > 20:
        chat_history = chat_history[-20:]