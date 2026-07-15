from dotenv import load_dotenv
load_dotenv()

from ingest import load_pdf, split_into_sections

PDF_PATH = r"docs\MSME Development Policy 2025.pdf"

docs = load_pdf(PDF_PATH)
sections = split_into_sections(docs)

print(f"\n{len(sections)} section(s) detected in {PDF_PATH}\n")
for i, s in enumerate(sections):
    text_preview = s["text"][:60].replace("\n", " ")
    print(f"[{i:>3}] page {s['page_start']+1:>3}  chars={len(s['text']):>5}  "
          f"cat={s['category'] or 'general':10}  heading={s['heading'][:60]!r}")
    print(f"      preview: {text_preview!r}")