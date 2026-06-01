import os
import uuid
import asyncpg
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
NEON_DATABASE_URI = os.getenv("NEON_DATABASE_URI")
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")

if not OPENROUTER_API_KEY:
    raise Exception("OPENROUTER_API_KEY missing")

if not NEON_DATABASE_URI:
    raise Exception("NEON_DATABASE_URI missing")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

pool = None


class ChatRequest(BaseModel):
    message: str
    patient_id: str | None = None
    session_id: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool

    pool = await asyncpg.create_pool(
        NEON_DATABASE_URI,
        min_size=1,
        max_size=10,
        command_timeout=60
    )

    yield

    await pool.close()


app = FastAPI(
    title="Medical AI Assistant",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL] if FRONTEND_URL != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Medical AI Backend Running"}


@app.get("/health")
async def health():
    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")

        return {
            "status": "healthy",
            "database": "connected"
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.get("/patients")
async def patients():

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM patient ORDER BY name"
        )

    return {
        "patients": [dict(r) for r in rows]
    }


async def generate_embedding(query: str):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )

    return response.data[0].embedding


@app.post("/chat")
async def chat(req: ChatRequest):

    try:

        embedding = await generate_embedding(
            req.message
        )

        embedding_str = "[" + ",".join(
            map(str, embedding)
        ) + "]"

        async with pool.acquire() as conn:

            results = await conn.fetch(
                """
                SELECT
                    text,
                    metadata,
                    1 - (embedding <=> $1::vector) AS score
                FROM patient_records
                WHERE metadata->>'patient_id' = $2
                ORDER BY embedding <=> $1::vector
                LIMIT 5
                """,
                embedding_str,
                req.patient_id
            )

        if not results:
            return {
                "answer": "No relevant medical records found."
            }

        context = "\n\n".join(
            [row["text"] for row in results]
        )

        prompt = f"""
Patient Medical Records:

{context}

Question:
{req.message}

Use only the supplied records.
Always recommend consulting a doctor.
"""

        response = client.chat.completions.create(
            model="nvidia/nemotron-3-super-120b-a12b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=1000
        )

        return {
            "answer": response.choices[0].message.content,
            "session_id":
            req.session_id or str(uuid.uuid4())
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
