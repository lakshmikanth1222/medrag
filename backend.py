from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from dotenv import load_dotenv
import json
import asyncpg
from datetime import datetime
import uuid

# ==========================================
# LOAD ENV
# ==========================================
load_dotenv(dotenv_path=".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
NEON_DATABASE_URI = os.getenv("NEON_DATABASE_URI", "")

print("🔑 OPENROUTER:", "SET" if OPENROUTER_API_KEY else "MISSING")
print("🗄️ DATABASE:", "SET" if NEON_DATABASE_URI else "MISSING")

# ==========================================
# APP INIT
# ==========================================
app = FastAPI(title="Medical AI Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor()

# ==========================================
# MODELS
# ==========================================
class Patient(BaseModel):
    patient_id: str
    abha_id: str
    name: str
    date_of_birth: str
    gender: str
    phone_number: str
    created_at: Optional[str] = None

    @field_validator('patient_id', mode='before')
    def convert_uuid(cls, v):
        return str(v) if isinstance(v, uuid.UUID) else v


class ChatRequest(BaseModel):
    message: str
    patient_id: Optional[str] = None
    session_id: Optional[str] = None


# ==========================================
# DATABASE
# ==========================================
class DatabaseManager:
    def __init__(self):
        self.pool = None

    async def initialize(self):
        try:
            self.pool = await asyncpg.create_pool(NEON_DATABASE_URI)
            print("✅ Database Connected")
            return True
        except Exception as e:
            print("❌ DB ERROR:", e)
            return False

    async def get_all_patients(self):
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM patient")
                return [dict(r) for r in rows]
        except Exception as e:
            print("❌ get_all_patients ERROR:", e)
            return []

    async def get_patient_by_id(self, pid):
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM patient WHERE patient_id=$1", pid
                )
                return dict(row) if row else None
        except Exception as e:
            print("❌ get_patient ERROR:", e)
            return None

    async def search_patient_records(self, embedding, patient_id=None):
        try:
            async with self.pool.acquire() as conn:
                emb = '[' + ','.join(map(str, embedding)) + ']'

                if patient_id:
                    rows = await conn.fetch("""
                        SELECT text, metadata,
                        1 - (embedding <=> $1::vector) as score
                        FROM patient_records
                        WHERE metadata->>'patient_id'=$2
                        LIMIT 5
                    """, emb, patient_id)
                else:
                    rows = await conn.fetch("""
                        SELECT text, metadata,
                        1 - (embedding <=> $1::vector) as score
                        FROM patient_records
                        LIMIT 5
                    """, emb)

                return rows

        except Exception as e:
            print("❌ VECTOR SEARCH ERROR:", e)
            return []

    async def close(self):
        if self.pool:
            await self.pool.close()


# ==========================================
# AI ENGINE
# ==========================================
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from openai import OpenAI


class MedicalAIEngine:
    def __init__(self, db):
        self.db = db
        self.client = None
        self.embed_model = None
        self.initialized = False
        self.lock = asyncio.Lock()

    async def initialize(self):
        async with self.lock:
            if self.initialized:
                return

            if OPENROUTER_API_KEY:
                self.client = OpenAI(
                    api_key=OPENROUTER_API_KEY,
                    base_url="https://openrouter.ai/api/v1"
                )
                print("✅ OpenRouter Connected")
            else:
                print("⚠️ OpenRouter key missing")

            self.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5"
            )

            Settings.embed_model = self.embed_model
            self.initialized = True

    async def generate_embedding(self, text):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            self.embed_model.get_text_embedding,
            text
        )

    async def search(self, message, patient_id=None):
        query_embedding = await self.generate_embedding(message)
        results = await self.db.search_patient_records(query_embedding, patient_id)

        sources = []
        context = []

        for row in results:
            meta = row.get("metadata", {})

            # Safe metadata parsing
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except:
                    meta = {}
            elif not isinstance(meta, dict):
                meta = {}

            sources.append({
                "file_name": meta.get("file_name", "Unknown"),
                "patient_id": meta.get("patient_id", "Unknown"),
                "abha_id": meta.get("abha_id", "Unknown"),
                "text": row.get("text", "")[:300]
            })

            context.append(row.get("text", ""))

        return {
            "sources": sources,
            "context": "\n\n".join(context)
        }

    async def generate(self, message, context, patient_info=None):
        if not self.client:
            return "⚠️ OpenRouter API not configured"

        # Patient context
        patient_context = ""
        if patient_info:
            try:
                dob = datetime.strptime(patient_info.get('date_of_birth', ''), '%Y-%m-%d')
                age = datetime.now().year - dob.year
            except:
                age = "Unknown"

            patient_context = f"""
Patient Information:
- Name: {patient_info.get('name', 'Unknown')}
- Age: {age}
- Gender: {patient_info.get('gender', 'Unknown')}
- ABHA ID: {patient_info.get('abha_id', 'Unknown')}
"""

        # ✅ MEDICAL ASSISTANT SYSTEM PROMPT
        system_prompt = """
You are an expert AI medical assistant.

You will receive patient medical records and a user query.

### RULES:
1. Use ONLY the provided medical records
2. Do NOT assume missing information
3. Do NOT give definitive diagnosis
4. Always recommend consulting a doctor

### TASK:
- Analyze the medical records
- Identify key findings
- Provide possible interpretations
- Suggest next steps

### OUTPUT FORMAT:
- **Findings**
- **Possible Conditions / Interpretation**
- **Suggested Next Steps**
- **Confidence Level**

### TONE:
- Clear and professional
- No unnecessary jargon
- Always include disclaimer:
"This analysis is AI-generated and must be reviewed by a qualified medical professional."
"""

        full_prompt = f"""
{patient_context}

Medical Records:
{context}

Question:
{message}
"""

        loop = asyncio.get_event_loop()

        def call_llm():
            try:
                response = self.client.chat.completions.create(
                    model="nvidia/nemotron-3-super-120b-a12b",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1200
                )
                return response.choices[0].message.content
            except Exception as e:
                print("❌ LLM ERROR:", e)
                return f"⚠️ LLM Error: {str(e)}"

        return await loop.run_in_executor(executor, call_llm)


# ==========================================
# INIT SERVICES
# ==========================================
db = DatabaseManager()
ai = MedicalAIEngine(db)


@app.on_event("startup")
async def startup():
    print("🚀 Starting Backend...")
    await db.initialize()
    await ai.initialize()


@app.on_event("shutdown")
async def shutdown():
    await db.close()


# ==========================================
# ROUTES
# ==========================================
@app.get("/health")
async def health():
    try:
        db_status = "Connected" if db.pool else "Disconnected"
        api_status = "Connected" if OPENROUTER_API_KEY else "Missing"

        patients = await db.get_all_patients() if db.pool else []

        return {
            "status": "healthy",
            "gemini_api": api_status,
            "database": db_status,
            "patients_count": len(patients)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/patients")
async def get_patients():
    return {"patients": await db.get_all_patients()}


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        print("📩 Query:", req.message)

        patient_info = None
        if req.patient_id:
            patient_info = await db.get_patient_by_id(req.patient_id)

        search_results = await ai.search(req.message, req.patient_id)

        if not search_results["sources"]:
            return {"answer": "No relevant medical records found."}

        answer = await ai.generate(
            req.message,
            search_results["context"],
            patient_info
        )

        return {
            "answer": answer,
            "sources": search_results["sources"],
            "session_id": req.session_id or "default"
        }

    except Exception as e:
        print("❌ CHAT ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
