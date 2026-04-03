"""
Content generation pipeline orchestrator.

PIPELINE FLOW
═══════════════════════════════════════════
  1. Load persona YAML + prompt template
  2. Generate content calendar via LLM
  3. Insert calendar items into content_queue
  4. For each item: generate image via ComfyUI
  5. Update status throughout
"""

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import yaml

from config import settings
from database import get_session
from models import ContentQueue, ContentStatus, Persona
from services.comfyui import ComfyUIError, generate_image
from services.llm import ContentCalendar, LLMError, generate_calendar, render_prompt

logger = logging.getLogger(__name__)

DEFAULT_NEGATIVE_PROMPT = (
    "deformed, blurry, cartoon, anime, 3d render, painting, "
    "low quality, bad anatomy, extra fingers, mutated hands"
)


def load_persona_yaml(persona_name: str) -> dict:
    path = settings.personas_dir / f"{persona_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Persona file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompt_template(template_name: str = "system_prompt.yaml") -> str:
    path = settings.prompts_dir / template_name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("template", "")


def sync_persona_to_db(persona_data: dict) -> Persona:
    """Sync a persona YAML definition to the database. YAML is source of truth."""
    with get_session() as session:
        existing = session.query(Persona).filter(
            Persona.name == persona_data["name"]
        ).first()

        fields = {
            "name": persona_data["name"],
            "age": persona_data["age"],
            "gender": persona_data["gender"],
            "niche": persona_data["niche"],
            "personality_traits": ", ".join(persona_data.get("personality_traits", [])),
            "speaking_style": persona_data.get("speaking_style", ""),
            "visual_base_prompt": persona_data.get("visual_base_prompt", ""),
            "lora_path": persona_data.get("lora_path"),
            "reference_face_images": json.dumps(
                persona_data.get("reference_face_images", [])
            ),
            "platforms": json.dumps(persona_data.get("platforms", [])),
        }

        if existing:
            for key, value in fields.items():
                setattr(existing, key, value)
            session.commit()
            session.refresh(existing)
            return existing
        else:
            persona = Persona(**fields)
            session.add(persona)
            session.commit()
            session.refresh(persona)
            return persona


def sync_all_personas() -> list[Persona]:
    """Sync all persona YAML files to the database."""
    personas = []
    for yaml_file in settings.personas_dir.glob("*.yaml"):
        try:
            data = load_persona_yaml(yaml_file.stem)
            persona = sync_persona_to_db(data)
            personas.append(persona)
            logger.info("Synced persona: %s", persona.name)
        except Exception as e:
            logger.error("Failed to sync persona %s: %s", yaml_file.stem, e)
    return personas


async def generate_content_for_persona(persona_id: int, days: int = 7) -> list[ContentQueue]:
    """Generate a content calendar and insert into content_queue."""
    with get_session() as session:
        persona = session.get(Persona, persona_id)
        if not persona:
            raise ValueError(f"Persona not found: {persona_id}")

        persona_name = persona.name
        persona_vars = {
            "name": persona.name,
            "age": persona.age,
            "gender": persona.gender,
            "niche": persona.niche,
            "personality_traits": persona.personality_traits,
            "speaking_style": persona.speaking_style,
        }

    prompt_template = load_prompt_template()
    calendar = await generate_calendar(persona_vars, prompt_template, days=days)

    items = []
    with get_session() as session:
        for item in calendar.items:
            entry = ContentQueue(
                persona_id=persona_id,
                post_date=item.post_date,
                concept=item.concept,
                caption=item.caption,
                image_prompt=item.image_prompt,
                hashtags=json.dumps(item.hashtags),
                status=ContentStatus.PLANNED,
            )
            session.add(entry)
            items.append(entry)
        session.commit()
        for item in items:
            session.refresh(item)

    logger.info("Generated %d content items for persona %s", len(items), persona_name)
    return items


async def generate_images_for_persona(persona_id: int) -> dict[str, int]:
    """Generate images for all PLANNED content items of a persona."""
    stats = {"success": 0, "failed": 0, "skipped": 0}

    with get_session() as session:
        persona = session.get(Persona, persona_id)
        if not persona:
            raise ValueError(f"Persona not found: {persona_id}")

        persona_name = persona.name
        persona_data = load_persona_yaml(persona_name)
        ref_faces = persona.get_reference_faces()
        ref_face = ref_faces[0] if ref_faces else None

        items = session.query(ContentQueue).filter(
            ContentQueue.persona_id == persona_id,
            ContentQueue.status == ContentStatus.PLANNED,
        ).all()

    workflow_path = settings.workflows_dir / "default.json"
    if not workflow_path.exists():
        raise FileNotFoundError(
            f"Default workflow not found: {workflow_path}. "
            "Export a workflow from ComfyUI GUI and save it as workflows/default.json"
        )

    base_prompt = persona_data.get("visual_base_prompt", "")
    negative_prompt = persona_data.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)

    for item in items:
        with get_session() as session:
            entry = session.get(ContentQueue, item.id)
            if not entry:
                continue

            entry.transition_to(ContentStatus.GENERATING)
            session.commit()

            full_prompt = f"{base_prompt}, {entry.image_prompt}" if base_prompt else entry.image_prompt
            output_path = (
                settings.assets_dir
                / persona_name
                / "generated"
                / f"{entry.post_date}_{entry.id:04d}.png"
            )

            try:
                await generate_image(
                    workflow_path=workflow_path,
                    positive_prompt=full_prompt,
                    negative_prompt=negative_prompt,
                    reference_face_path=ref_face,
                    output_path=output_path,
                )
                entry.image_path = str(output_path)
                entry.transition_to(ContentStatus.GENERATED)
                session.commit()
                stats["success"] += 1
                logger.info("Generated image for content %d", entry.id)
            except ComfyUIError as e:
                entry.status = ContentStatus.ERROR
                entry.error_message = str(e)
                session.commit()
                stats["failed"] += 1
                logger.error("Image generation failed for content %d: %s", entry.id, e)

    return stats
