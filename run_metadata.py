# run_metadata.py  — run once, generates policy_metadata.json
from dotenv import load_dotenv
load_dotenv()

import os, json
from langchain_community.document_loaders import PyMuPDFLoader
from openai import OpenAI

client = OpenAI()
DOCS_DIR = "docs"

results = []

for file in sorted(os.listdir(DOCS_DIR)):
    if not file.endswith(".pdf"):
        continue
    path = os.path.join(DOCS_DIR, file)
    print(f"Processing: {file}")
    try:
        loader = PyMuPDFLoader(path)
        pages = loader.load()
        # Use first 6 pages — enough for scope/sector info
        sample = "\n\n".join([p.page_content for p in pages[:6]])

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"""
Read this Madhya Pradesh government policy document excerpt and return ONLY a JSON object with these fields:

{{
  "policy_name": "Full official name of the policy",
  "filename": "{file}",
  "common_policy": true/false (true if applicable across all sectors, like IPP),
  "primary_sectors": ["sector1", "sector2"],
  "enterprise_sizes": ["Micro", "Small", "Medium", "Large"] or ["All"],
  "keywords": ["keyword1", "keyword2", ...],
  "summary": "One sentence describing what this policy covers"
}}

Return ONLY the JSON. No explanation, no markdown backticks.

Policy text:
{sample[:6000]}
"""}],
            temperature=0
        )
        data = json.loads(response.choices[0].message.content.strip())
        results.append(data)
        print(f"  ✓ {data['policy_name']}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results.append({"filename": file, "error": str(e)})

with open("policy_metadata.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nDone. {len(results)} policies processed → policy_metadata.json")