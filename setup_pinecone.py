import os
import time
from pinecone import Pinecone, ServerlessSpec
import google.genai as genai
from google.genai import types
from dotenv import load_dotenv
from remedies_list import remedies

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Missing API keys! Please set PINECONE_API_KEY and GEMINI_API_KEY.")


pc = Pinecone(api_key=PINECONE_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY) #NEW SDK SYNTAX

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




index = pc.Index(index_name)

#generating embeddings and uploading onto the index
print("Converting text to vector embeddings and uploading...")
vectors_to_upload = []

for item in remedies:
    result = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=item["text"],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768
        )
    )
    
    vectors_to_upload.append({
        "id": item["id"],
        "values": result.embeddings[0].values,
        "metadata": {"text": item["text"]}
    })

index.upsert(vectors=vectors_to_upload)
print(f"Successfully uploaded {len(vectors_to_upload)} remedies to Pinecone!")