from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    input="Hello"
)

print(response.output_text)