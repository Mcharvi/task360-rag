from dotenv import load_dotenv
load_dotenv()

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI

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

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)
chat_history = []

def rewrite_query(question, history):

    if not history:
        return question

    prompt = f"""
Given the conversation history and latest question,
rewrite the latest question into a standalone question.

Conversation:
{history}

Latest Question:
{question}

Standalone Question:
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    return response.output_text.strip()

class Question(BaseModel):
    question: str


@app.post("/ask")
def ask_question(data: Question):

    history_text = "\n".join(chat_history)

    rewritten_query = rewrite_query(
        data.question,
        history_text
    )

    print("\n" + "="*50)
    print("REQUEST RECEIVED")
    print("QUESTION:", data.question)
    print("REWRITTEN:", rewritten_query)
    print("="*50)

    # RETRIEVAL

    retrieved_docs = vectorstore.max_marginal_relevance_search(
        rewritten_query,
        k=5,
        fetch_k=20
)
    # SIMILARITY SCORES

    results = vectorstore.similarity_search_with_score(
    rewritten_query,
    k=5
)

   

    best_score = results[0][1]

    if best_score > 1.10:
         return {
            "answer": "This question appears unrelated to the uploaded policy documents."
    }

    print("\nSIMILARITY SCORES:", flush=True)

    for doc, score in results:
        print(score, flush=True)
   

    # CONTEXT
    

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    # PROMPT

    prompt = f"""
You are a retrieval-based assistant.

Rules:
1. Use ONLY the provided context.
2. Never use outside knowledge.
3. Never guess.
4. Never infer facts not explicitly present.
5. If the answer is not present in the context, reply exactly:

I could not find the answer in the provided documents.



Previous Conversation:
{history_text}

Context:
{context}

Question:
{data.question}

Answer:
"""

    # GPT RESPONSE

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    answer = response.output_text

    # SOURCES

    sources = []

    for doc in retrieved_docs[:2]:

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0)

        filename = source.split("\\")[-1]
        filename = filename.split("/")[-1]

        print("\nTOP RETRIEVED DOCUMENTS:")

        for doc in retrieved_docs[:2]:
            print(doc.metadata)

        sources.append({
            "file": filename,
            "page": page + 1
})

        unique_sources = []

    for source in sources:
        if source not in unique_sources:
            unique_sources.append(source)

    citation_lines = []

    for i, source in enumerate(unique_sources, start=1):
        citation_lines.append(
            f"[{i}] {source['file']} - Page {source['page']}"
    )

    citation_text = "\n".join(citation_lines)

    final_answer = f"""
    {answer}

    --------------------------------

    References

    {citation_text}
    """

    
    chat_history.append(
        f"User: {data.question}"
    )

    chat_history.append(
        f"Assistant: {answer}"
    )

    if len(chat_history) > 20:
        chat_history[:] = chat_history[-20:]

    return {
        "answer": final_answer
    }