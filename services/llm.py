"""
LLM content generation service.

DATA FLOW
═════════════════════════════════════
  persona YAML vars ──> Jinja2 render system prompt
       │
       ▼
  LLM API (structured output) ──> 7-day content calendar JSON
       │
       ▼
  Pydantic validation ──> list[ContentItem]
       │
       ▼
  Insert into content_queue (status: planned)
"""

import json
import logging
from typing import Any

import anthropic
import openai
from jinja2 import Template
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import settings

logger = logging.getLogger(__name__)


class ContentItem(BaseModel):
    post_date: str  # YYYY-MM-DD
    concept: str
    caption: str
    image_prompt: str
    hashtags: list[str]


class ContentCalendar(BaseModel):
    items: list[ContentItem]


class LLMError(Exception):
    pass


class LLMParseError(LLMError):
    pass


def render_prompt(template_str: str, variables: dict[str, Any]) -> str:
    return Template(template_str).render(**variables)


def _build_system_prompt(persona_vars: dict[str, Any], prompt_template: str) -> str:
    return render_prompt(prompt_template, persona_vars)


def _strip_markdown_json(raw: str) -> str:
    """Strip markdown code fences if present."""
    text = raw.strip()
    if text.startswith("```"):
        first_newline = text.index("\n")
        text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _parse_calendar(raw: str) -> ContentCalendar:
    try:
        data = json.loads(_strip_markdown_json(raw))
    except json.JSONDecodeError as e:
        raise LLMParseError(f"Invalid JSON from LLM: {e}") from e

    if isinstance(data, list):
        data = {"items": data}

    return ContentCalendar.model_validate(data)


@retry(
    stop=stop_after_attempt(settings.llm_retries),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((anthropic.APIError, openai.APIError)),
    reraise=True,
)
async def _call_anthropic(system_prompt: str, user_prompt: str) -> str:
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    message = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text


@retry(
    stop=stop_after_attempt(settings.llm_retries),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(openai.APIError),
    reraise=True,
)
async def _call_openai(system_prompt: str, user_prompt: str) -> str:
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


async def generate_calendar(
    persona_vars: dict[str, Any],
    prompt_template: str,
    days: int = 7,
) -> ContentCalendar:
    from datetime import date, timedelta

    system_prompt = _build_system_prompt(persona_vars, prompt_template)
    start_date = date.today()
    end_date = start_date + timedelta(days=days - 1)
    user_prompt = (
        f"Generate a {days}-day content calendar as JSON. "
        f"Start from {start_date.isoformat()} to {end_date.isoformat()}. "
        f"Return a JSON object with an 'items' array. Each item must have: "
        f"post_date (YYYY-MM-DD), concept, caption, image_prompt, hashtags (array)."
    )

    providers = (
        [("anthropic", _call_anthropic), ("openai", _call_openai)]
        if settings.default_llm == "anthropic"
        else [("openai", _call_openai), ("anthropic", _call_anthropic)]
    )

    last_error: Exception | None = None
    for provider_name, call_fn in providers:
        for parse_attempt in range(2):
            try:
                raw = await call_fn(system_prompt, user_prompt)
                return _parse_calendar(raw)
            except LLMParseError:
                logger.warning(
                    "Parse attempt %d failed for %s, retrying",
                    parse_attempt + 1,
                    provider_name,
                )
                continue
            except Exception as e:
                logger.warning("Provider %s failed: %s", provider_name, e)
                last_error = e
                break

    raise LLMError(f"All LLM providers failed. Last error: {last_error}")
