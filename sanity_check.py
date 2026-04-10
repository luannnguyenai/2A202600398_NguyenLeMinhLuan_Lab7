from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Lay cau hinh tu .env ma toi da setup cho ban
base_url = os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
model = os.getenv("OPENAI_EMBEDDING_MODEL", "jina-embeddings-v5-text-nano-retrieval")

print(f"--- Sanity Check ---")
print(f"Target: {base_url}")
print(f"Model : {model}")

client = OpenAI(base_url=base_url, api_key="lm-studio")

def get_embedding(text, model=model):
   text = text.replace("\n", " ")
   try:
       # Su dung input=[text] nhu mau ban gui
       resp = client.embeddings.create(input=[text], model=model)
       return resp.data[0].embedding
   except Exception as e:
       return f"Error: {e}"

result = get_embedding("Once upon a time, there was a cat.")
if isinstance(result, list):
    print(f"SUCCESS! Embedding length: {len(result)}")
    print(f"First 5 values: {result[:5]}")
else:
    print(f"FAILURE: {result}")
