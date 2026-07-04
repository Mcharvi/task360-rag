from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

results = vectorstore.similarity_search(
    "export goal target 2029",
    k=10,
    filter={"source": "docs\\Export-Promotion-Policy.pdf"}
)
for r in results:
    print("page:", r.metadata.get("page"))
    print(r.page_content)
    print("---")