import os
import time
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Missing API keys! Please set PINECONE_API_KEY and GEMINI_API_KEY.")


pc = Pinecone(api_key=PINECONE_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

#our_remedies_cloud_DB
index_name = "ayush-remedies"

 
if index_name not in pc.list_indexes().names():
    print(f"Creating Pinecone index: '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=768, 
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(index_name).status['ready']:
        print("Waiting for index to be ready...")
        time.sleep(2)
else:
    print(f"Index '{index_name}' already exists.")