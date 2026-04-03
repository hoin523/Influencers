import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import select

from config import settings
import database
from models import ContentQueue, ContentStatus, InvalidTransitionError, Persona
from services.pipeline import (
    generate_content_for_persona,
    generate_images_for_persona,
    sync_all_personas,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    personas = sync_all_personas()
    logger.info("Synced %d personas from YAML", len(personas))
    yield


app = FastAPI(title="AI Influencer Factory", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Personas ---


@app.get("/personas")
def list_personas():
    with database.get_session() as session:
        return session.exec(select(Persona)).all()


@app.get("/personas/{persona_id}")
def get_persona(persona_id: int):
    with database.get_session() as session:
        persona = session.get(Persona, persona_id)
        if not persona:
            raise HTTPException(404, "Persona not found")
        return persona


# --- Content Generation ---


@app.post("/personas/{persona_id}/generate")
async def generate_content(persona_id: int, days: int = 7):
    with database.get_session() as session:
        if not session.get(Persona, persona_id):
            raise HTTPException(404, "Persona not found")

    try:
        items = await generate_content_for_persona(persona_id, days)
        return {"generated": len(items), "persona_id": persona_id}
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error("Content generation failed: %s", e)
        raise HTTPException(500, f"Generation failed: {e}")


@app.post("/personas/{persona_id}/generate-images")
async def generate_images(persona_id: int):
    with database.get_session() as session:
        if not session.get(Persona, persona_id):
            raise HTTPException(404, "Persona not found")

    try:
        stats = await generate_images_for_persona(persona_id)
        return stats
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error("Image generation failed: %s", e)
        raise HTTPException(500, f"Image generation failed: {e}")


# --- Content Queue ---


@app.get("/content-queue")
def list_content_queue(
    persona_id: int | None = None,
    status: ContentStatus | None = None,
):
    with database.get_session() as session:
        query = select(ContentQueue)
        if persona_id is not None:
            query = query.where(ContentQueue.persona_id == persona_id)
        if status is not None:
            query = query.where(ContentQueue.status == status)
        query = query.order_by(ContentQueue.post_date)
        return session.exec(query).all()


class StatusUpdate(BaseModel):
    status: ContentStatus


@app.patch("/content-queue/{item_id}")
def update_content_status(item_id: int, body: StatusUpdate):
    with database.get_session() as session:
        item = session.get(ContentQueue, item_id)
        if not item:
            raise HTTPException(404, "Content item not found")
        try:
            item.transition_to(body.status)
            session.commit()
            session.refresh(item)
            return item
        except InvalidTransitionError as e:
            raise HTTPException(400, str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
