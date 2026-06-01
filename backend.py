
import os, json, asyncio, uuid, logging
from contextlib import asynccontextmanager
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
NEON_DATABASE_URI = os.getenv("NEON_DATABASE_URI")
FRONTEND_URL = os.getenv("FRONTEND_URL","*")

logging.basicConfig(level=logging.INFO)
executor = ThreadPoolExecutor(max_workers=4)

class ChatRequest(BaseModel):
    message:str
    patient_id:str|None=None
    session_id:str|None=None

class Database:
    def __init__(self):
        self.pool=None

    async def connect(self):
        self.pool = await asyncpg.create_pool(
            NEON_DATABASE_URI,min_size=1,max_size=10,command_timeout=60
        )

    async def close(self):
        if self.pool:
            await self.pool.close()

db=Database()

class AIEngine:
    def __init__(self):
        self.client=None
        self.embed_model=None

    async def initialize(self):
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )

    async def embedding(self,text):
        if self.embed_model is None:
            self.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5"
            )
        loop=asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            self.embed_model.get_text_embedding,
            text
        )

ai=AIEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()
    await ai.initialize()
    yield
    await db.close()

app=FastAPI(title="Medical AI Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL] if FRONTEND_URL!="*" else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    try:
        async with db.pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return {"status":"healthy"}
    except Exception as e:
        return {"status":"unhealthy","error":str(e)}

@app.get("/patients")
async def patients():
    async with db.pool.acquire() as conn:
        rows=await conn.fetch("SELECT * FROM patient ORDER BY name")
    return {"patients":[dict(r) for r in rows]}

@app.post("/chat")
async def chat(req:ChatRequest):
    emb = await ai.embedding(req.message)
    emb='['+','.join(map(str,emb))+']'

    async with db.pool.acquire() as conn:
        rows=await conn.fetch("""
        SELECT text,metadata,1-(embedding <=> $1::vector) score
        FROM patient_records
        WHERE metadata->>'patient_id'=$2
        ORDER BY embedding <=> $1::vector
        LIMIT 5
        """,emb,req.patient_id)

    if not rows:
        return {"answer":"No relevant records found."}

    context="\n\n".join([r["text"] for r in rows])

    prompt=f"""
Medical Records:
{context}

Question:
{req.message}
"""

    response=ai.client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b",
        messages=[
            {"role":"system","content":"Use only supplied medical records. Recommend consulting a doctor."},
            {"role":"user","content":prompt}
        ],
        temperature=0.2,
        max_tokens=1000
    )

    return {
        "answer":response.choices[0].message.content,
        "session_id":req.session_id or str(uuid.uuid4())
    }
