import os
import io
import re
import json
import base64
import textwrap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from groq import Groq
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from pinecone import Pinecone
import google.genai as genai
from google.genai import types
from prompts import TRIAGE_SYSTEM_PROMPT
from dotenv import load_dotenv


app = FastAPI(title="ML: Updated Triage Chatbot")


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("ayush-remedies")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


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

class ReportRequest(BaseModel):
    session_id: str




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
            chat_sessions[session_id] = [
                {"role": "user", "content": req.message},
                {"role": "assistant", "content": emergency_reply},
                {"role": "summary", "zone": "RED", "symptoms": "Critical distress detected", "remedy": "Immediate emergency care required."}
            ]
            return ChatResponse(
                success=True,
                session_id=session_id,
                reply=emergency_reply,
                zone="red",
                is_final=True,
                remedy_suggestion=None,
                follow_up_question=None
            )



        #stopping further chat if the session is already closed
        if session_id in chat_sessions:
            is_already_closed = any(msg.get("role") == "summary" for msg in chat_sessions[session_id])
            if is_already_closed:
                return ChatResponse(
                    success=True,
                    session_id=session_id,
                    reply="Your triage session is already completed. Please download your report.",
                    zone=None, 
                    is_final=True,
                    remedy_suggestion=None,
                    follow_up_question=None
                )
        
        if session_id not in chat_sessions:
            chat_sessions[session_id] = [{"role": "system", "content": TRIAGE_SYSTEM_PROMPT}]
            
        chat_sessions[session_id].append({"role": "user", "content": req.message})

        
        valid_api_messages = [
            msg for msg in chat_sessions[session_id] 
            if msg.get("role") in ["system", "user", "assistant"]
        ]
        completion = groq_client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=valid_api_messages,
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=800
        )



        raw_llm = completion.choices[0].message.content.strip()
        if raw_llm.startswith("```"):
            raw_llm = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_llm, flags=re.IGNORECASE)
        
        parsed_llm = json.loads(raw_llm)
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


        #getting remedies for Green/Orange cases from Pinecone
        if is_final:
            if final_zone == "red":
                follow_up = None
            elif final_zone in ["green", "orange"]:
                symptoms_str = " ".join(parsed_llm.get("extracted_symptoms", [req.message]))
                #Query
                embedding_result = gemini_client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=symptoms_str,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_QUERY",
                        output_dimensionality=768
                    )
                )
                query_vector = embedding_result.embeddings[0].values

                results = pinecone_index.query(vector=query_vector, top_k=1, include_metadata=True)
                if results['matches']:
                    remedy = results['matches'][0]['metadata']['text']

            chat_sessions[session_id].append({
                "role": "summary",
                "zone": final_zone.upper() if final_zone else "RED",
                "symptoms": symptoms_str if 'symptoms_str' in locals() else " ".join(parsed_llm.get("extracted_symptoms", [])),
                "remedy": remedy or "Immediate emergency care required."
            })
        

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
            reply=f"PYTHON ERROR: {str(e)}",
            zone="red",  # Fail-safe: errors in triage default to high alert
            is_final=True,
            remedy_suggestion=None,
            follow_up_question=None
        )





#pdf-report generation endpoint
@app.post("/ml/generate-report")
async def generate_report(req: ReportRequest):
    try:
        if req.session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
            
        session_data = chat_sessions[req.session_id]
        
        history = [msg for msg in session_data if msg.get("role") not in ["system", "summary"]]
        summary = next((msg for msg in session_data if msg.get("role") == "summary"), None)

        pdf_buffer = io.BytesIO()
        pdf = canvas.Canvas(pdf_buffer, pagesize=A4)
        
        #text wrapping
        def draw_wrapped_text(c, text, x, y, max_width=80):
            lines = textwrap.wrap(text, width=max_width)
            for line in lines:
                c.drawString(x, y, line)
                y -= 18
            return y

        #header
        pdf.setFont("Helvetica-Bold", 16)
        y = 800
        pdf.drawString(50, y, f"Health Triage Report (Session: {req.session_id})")
        y -= 30
        

        #clinical summary
        if summary:
            pdf.setFont("Helvetica-Bold", 14)
            pdf.drawString(50, y, "Clinical Summary")
            y -= 20
            
            pdf.setFont("Helvetica", 12)
            y = draw_wrapped_text(pdf, f"Triage Zone: {summary.get('zone', 'UNKNOWN')}", 50, y)
            y = draw_wrapped_text(pdf, f"Extracted Symptoms: {summary.get('symptoms', 'N/A')}", 50, y)
            y = draw_wrapped_text(pdf, f"Recommended Action: {summary.get('remedy', 'N/A')}", 50, y)
            
            # Draw a divider line
            pdf.line(50, y, 550, y)
            y -= 25

        #conversation history
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(50, y, "Consultation Transcript")
        y -= 20
        
        pdf.setFont("Helvetica", 12)
        for msg in history:
            prefix = "Patient: " if msg["role"] == "user" else "AI Assistant: "
            text = prefix + msg["content"]
            
            y = draw_wrapped_text(pdf, text, 50, y, max_width=85)
            y -= 5
            
            if y < 50:
                pdf.showPage()
                pdf.setFont("Helvetica", 12)
                y = 800
                
        pdf.showPage()
        pdf.save()
        
        pdf_base64 = base64.b64encode(pdf_buffer.getvalue()).decode('utf-8')
        
        return {
            "success": True,
            "pdf_base64": pdf_base64,
            "filename": f"triage-report-{req.session_id}.pdf"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}