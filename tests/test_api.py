import os
import sys

os.environ["DATABASE_URL"] = "sqlite://"
os.environ["ANTHROPIC_API_KEY"] = "test"

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from models import ContentQueue, ContentStatus, Persona


@pytest.fixture
def client():
    # StaticPool forces a single connection to be reused, so in-memory SQLite works
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(test_engine)

    with Session(test_engine) as session:
        persona = Persona(
            name="api_test",
            age=25,
            gender="female",
            niche="tech",
            personality_traits="smart",
            speaking_style="casual",
            visual_base_prompt="test prompt",
        )
        session.add(persona)
        session.commit()

        item = ContentQueue(
            persona_id=1,
            post_date="2026-04-10",
            concept="test concept",
            caption="test caption",
            image_prompt="test image",
            status=ContentStatus.GENERATED,
        )
        session.add(item)
        session.commit()

    import database
    database.engine = test_engine
    database.get_session = lambda: Session(test_engine)
    database.init_db = lambda: None

    # Force fresh import of main
    if "main" in sys.modules:
        del sys.modules["main"]

    with patch("services.pipeline.sync_all_personas", return_value=[]):
        import main as main_module
        with TestClient(main_module.app) as tc:
            yield tc


class TestPersonaEndpoints:
    def test_list_personas(self, client):
        resp = client.get("/personas")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_get_nonexistent_persona(self, client):
        resp = client.get("/personas/999")
        assert resp.status_code == 404


class TestContentQueueEndpoints:
    def test_list_content_queue(self, client):
        resp = client.get("/content-queue")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_update_status_approve(self, client):
        resp = client.patch("/content-queue/1", json={"status": "approved"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "approved"

    def test_update_status_invalid_transition(self, client):
        client.patch("/content-queue/1", json={"status": "approved"})
        resp = client.patch("/content-queue/1", json={"status": "generating"})
        assert resp.status_code == 400

    def test_update_nonexistent_item(self, client):
        resp = client.patch("/content-queue/999", json={"status": "approved"})
        assert resp.status_code == 404
