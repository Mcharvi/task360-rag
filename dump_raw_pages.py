from dotenv import load_dotenv
load_dotenv()

from ingest import load_pdf

PDF_PATH = r"docs\MSME Development Policy 2025.pdf"
docs = load_pdf(PDF_PATH)

# 0-indexed pages 6-9 correspond to printed pages 7-10
for i in [6, 7, 8, 9]:
    print(f"\n{'='*20} PAGE {i+1} (index {i}) {'='*20}")
    print(docs[i].page_content[:1500])