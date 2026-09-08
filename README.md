# SwasthyaNode_ML: Medical Triage & RAG Microservice

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688)
![Groq](https://img.shields.io/badge/Groq-LPU_Inference-f55036)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-000000)

## Overview
**Swasthya** is a multilingual, AI-driven healthcare triage platform designed for rural India. This repository contains the **Core Triage Microservice**, which acts as the intelligent reasoning engine of the platform. 

It evaluates patient symptoms, safely categorizes medical urgency (Green/Yellow/Red zones), retrieves validated Ayush remedies using a Retrieval-Augmented Generation (RAG) pipeline, and generates clinical summary PDFs—all optimized to run seamlessly under strict memory constraints.

---

## System Architecture 
This microservice is part of a broader distributed architecture designed for low-latency rural connectivity:
**ML Triage Service (This Repo):** Processes the English/Hinglish text via Fast LLM inference, vector searches, and safety checks, returning a structured clinical response.

---

## Key Features
* **High-Speed Clinical Reasoning:** Utilizes the `qwen/qwen3.8-27b` model via Groq's LPU infrastructure to enforce strict JSON schemas and rapid multilingual contextual understanding.
* **RAG-Powered Remedies:** Uses Google Gemini Embeddings (`gemini-embedding-001`) and Pinecone Vector DB to map symptoms to localized Ayush medical remedies without hallucination.
* **Zero-Latency Panic Engine:** A Regex-based interceptor that detects critical emergency keywords (e.g., "heart attack", "chest pain") and instantly bypasses the LLM to trigger a Red Zone alert.
* **Stateless Memory Management:** Tracks patient history via a UUID-based in-memory hash map (`chat_sessions`). Simply passing `session_id: null` instantiates a brand new medical file.
* **In-Memory PDF Generation:** Dynamically builds a formatted Clinical Summary Report using `ReportLab`, returning it as a Base64 string to avoid ephemeral disk-write permissions in cloud environments.

---

## Technology Stack
* **Core Framework:** Python 3 & FastAPI (ASGI asynchronous routing)
* **LLM Inference:** Groq Cloud API
* **Vector Database:** Pinecone Serverless
* **Embeddings:** Google GenAI
* **Document Generation:** ReportLab
* **Cloud Deployment:** Render (Maintained via Google Apps Script Keep-Awake Ping)

---

## API Endpoints

### 1. `POST /ml/chat`
Handles multi-turn conversational triage. Enforces a strict 9-message limit before concluding the session.
* **Payload:** `{"text": "Patient symptoms here", "session_id": "uuid-or-null"}`
* **Response (JSON):** 
  ```json
  {
    "success": true,
    "session_id": "a1b2c3d4",
    "reply": "Are you experiencing any shortness of breath?",
    "zone": "YELLOW",
    "is_final": false,
    "remedy_suggestion": null,
    "follow_up_question": "Are you experiencing any shortness of breath?"
  }

```

### 2. `POST /ml/generate-report`

Generates a downloadable clinical PDF summary once a session reaches `is_final: true`.

* **Payload:** `{"session_id": "a1b2c3d4"}`
* **Response:** A JSON payload containing the PDF encoded as a Base64 string.

---

## Local Setup & Installation

**1. Clone the repository**

```bash
git clone [https://github.com/AnshulKumar79/SwasthyaNode_MLtriageService.git](https://github.com/AnshulKumar79/SwasthyaNode_MLtriageService.git)
cd SwasthyaNode_MLtriageService

```

**2. Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```

**3. Install dependencies**

```bash
pip install -r requirements.txt

```

**4. Configure Environment Variables**
Create a `.env` file in the root directory and add the following keys:

```env
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

```

**5. Run the Server**

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001

```

*Access the interactive Swagger UI documentation at: `http://localhost:8001/docs*`

---

## Deployment Notes

This service is optimized for deployment on **Render's Free Tier (512MB RAM)**.

* It uses a dynamic port binding (`--port $PORT`).
* To prevent Cold Starts (the 15-minute inactivity sleep timer), a lightweight Google Apps Script triggers a `GET` request to the `/docs` endpoint every 10 minutes, preserving API rate limits while keeping the server awake.

---

*Machine Learning & GenAI Architecture built for the Smart India Hackathon by Anshul Kumar.*

```
