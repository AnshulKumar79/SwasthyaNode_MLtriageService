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
from prompts import TRIAGE_SYSTEM_PROMPT

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




#chat_end-point
@app.post("/ml/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        session_id = req.session_id or os.urandom(8).hex()
        
        if is_instant_panic(req.message):
            emergency_reply = "EMERGENCY DETECTED: Please stay calm. Contact emergency services or reach the nearest hospital immediately. We are alerting facility personnel."
            chat_sessions[session_id] = [{"role": "system", "content": "Emergency Panic Triggered"}]
            return ChatResponse(
                success=True,
                session_id=session_id,
                reply=emergency_reply,
                zone="red",
                is_final=True,
                remedy_suggestion=None,
                follow_up_question=None
            )

        
        if session_id not in chat_sessions:
            chat_sessions[session_id] = [{"role": "system", "content": TRIAGE_SYSTEM_PROMPT}]
            
        chat_sessions[session_id].append({"role": "user", "content": req.message})

        
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=chat_sessions[session_id],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=250
        )
        
        parsed_llm = json.loads(completion.choices[0].message.content)
        triage_zone = parsed_llm.get("triage_zone", "green").lower()
        reply_text = parsed_llm.get("reply_to_patient", "Could you describe what you are feeling?")
        follow_up = parsed_llm.get("follow_up_question")
        is_critical = parsed_llm.get("is_critical", False) or (triage_zone == "red")

        chat_sessions[session_id].append({"role": "assistant", "content": reply_text})



        #Safe Interruption Engine
        turn_limit_reached = len(chat_sessions[session_id]) >= 9
        is_final = is_critical or turn_limit_reached
        
        final_zone = triage_zone if is_final else None
        remedy = None

        

        return ChatResponse(
            success=True,
            session_id=session_id,
            reply=reply_text,
            zone=final_zone,
            is_final=is_final,
            remedy_suggestion=remedy,
            follow_up_question=follow_up if not is_final else None
        )

    except Exception as e:
        return ChatResponse(
            success=False,
            session_id=req.session_id or "unknown",
            reply="We are experiencing a temporary error. If this is an emergency, please visit the nearest hospital.",
            zone="red",  # Fail-safe: errors in triage default to high alert
            is_final=True,
            remedy_suggestion=None,
            follow_up_question=None
        )