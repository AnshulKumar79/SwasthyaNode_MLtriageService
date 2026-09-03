import os
import io
import re
import json
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from groq import Groq
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from pinecone import Pinecone
import google.generativeai as genai

app = FastAPI(title="ML: Updated Triage Chatbot")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("ayush-remedies")
genai.configure(api_key=GEMINI_API_KEY)


chat_sessions = {}

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    patient_id: str
    message: str
    language: str

class ChatResponse(BaseModel):
    success: bool
    session_id: str
    reply: str
    zone: Optional[str]
    is_final: bool
    remedy_suggestion: Optional[str]
    follow_up_question: Optional[str]




#checking the panic_condition in the patient's message
PANIC_REGEX = re.compile(
r"\b(help+|bachao|save\s*me|dying|mar\s*raha|emergency|ambulance|choking|unconscious|behosh)\b", 
re.IGNORECASE
)

def is_instant_panic(text: str) -> bool:
    clean = text.lower().strip()
    words = clean.split()
    #Detecting spam repetitions like "help help help"
    if len(words) >= 3 and len(set(words)) <= 2:
        return True
    # Match critical keywords
    return bool(PANIC_REGEX.search(clean))