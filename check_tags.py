from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from collections import Counter

vs = Chroma(persist_directory="chroma_db", embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
r = vs.get(where={"source": "docs\\IT,ITes and ESDM Promotion Policy.pdf"}, include=["metadatas"])
print(Counter(m["project_type"] for m in r["metadatas"]))