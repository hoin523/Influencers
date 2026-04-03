import os

import pytest
from sqlmodel import Session, SQLModel, create_engine

# Use in-memory SQLite for tests
os.environ["DATABASE_URL"] = "sqlite://"

from models import ContentQueue, ContentStatus, Persona


@pytest.fixture
def engine():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def session(engine):
    with Session(engine) as session:
        yield session


@pytest.fixture
def sample_persona(session):
    persona = Persona(
        name="test_mina",
        age=23,
        gender="female",
        niche="fashion",
        personality_traits="trendy, warm",
        speaking_style="casual Korean",
        visual_base_prompt="young korean woman, photorealistic",
        reference_face_images='["assets/test/face.png"]',
        platforms='["instagram"]',
    )
    session.add(persona)
    session.commit()
    session.refresh(persona)
    return persona


@pytest.fixture
def sample_content(session, sample_persona):
    item = ContentQueue(
        persona_id=sample_persona.id,
        post_date="2026-04-10",
        concept="cafe daily",
        caption="morning coffee vibes",
        image_prompt="cozy cafe, latte art",
        hashtags='["cafe", "daily"]',
        status=ContentStatus.PLANNED,
    )
    session.add(item)
    session.commit()
    session.refresh(item)
    return item
