import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from sqlmodel import Session, SQLModel, create_engine

from models import ContentQueue, ContentStatus, Persona
from services.llm import ContentCalendar, ContentItem


@pytest.fixture
def pipeline_session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        persona = Persona(
            name="pipeline_test",
            age=23,
            gender="female",
            niche="fashion",
            personality_traits="trendy",
            speaking_style="casual",
            visual_base_prompt="test prompt",
            reference_face_images='["face.png"]',
            platforms='["instagram"]',
        )
        session.add(persona)
        session.commit()
        session.refresh(persona)
        yield session, persona


class TestSyncPersona:
    def test_sync_creates_new(self, tmp_path):
        persona_yaml = tmp_path / "test.yaml"
        persona_yaml.write_text(
            "name: sync_test\n"
            "age: 25\n"
            "gender: male\n"
            "niche: tech\n"
            "personality_traits:\n  - smart\n"
            "speaking_style: casual\n"
            "visual_base_prompt: test\n"
        )

        from services.pipeline import load_persona_yaml

        with patch("services.pipeline.settings") as mock_settings:
            mock_settings.personas_dir = tmp_path
            data = load_persona_yaml("test")
            assert data["name"] == "sync_test"
            assert data["age"] == 25


class TestGenerateContent:
    @pytest.mark.asyncio
    async def test_generates_content_items(self):
        mock_calendar = ContentCalendar(
            items=[
                ContentItem(
                    post_date="2026-04-10",
                    concept="cafe",
                    caption="morning coffee",
                    image_prompt="cozy cafe",
                    hashtags=["cafe", "daily"],
                ),
                ContentItem(
                    post_date="2026-04-11",
                    concept="ootd",
                    caption="today's outfit",
                    image_prompt="street fashion",
                    hashtags=["ootd"],
                ),
            ]
        )

        with (
            patch("services.pipeline.generate_calendar", new_callable=AsyncMock) as mock_gen,
            patch("services.pipeline.load_prompt_template", return_value="template {{ name }}"),
            patch("services.pipeline.get_session") as mock_session_fn,
        ):
            mock_gen.return_value = mock_calendar

            engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
            SQLModel.metadata.create_all(engine)

            with Session(engine) as session:
                persona = Persona(
                    name="gen_test",
                    age=23,
                    gender="female",
                    niche="fashion",
                    personality_traits="trendy",
                    speaking_style="casual",
                    visual_base_prompt="test",
                )
                session.add(persona)
                session.commit()
                session.refresh(persona)
                pid = persona.id

            mock_session_fn.return_value = Session(engine)

            from services.pipeline import generate_content_for_persona

            items = await generate_content_for_persona(pid, days=2)
            assert len(items) == 2
            mock_gen.assert_called_once()
