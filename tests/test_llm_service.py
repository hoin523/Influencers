import json
from unittest.mock import AsyncMock, patch

import pytest

from services.llm import (
    ContentCalendar,
    LLMError,
    LLMParseError,
    _parse_calendar,
    render_prompt,
)


class TestRenderPrompt:
    def test_basic_substitution(self):
        result = render_prompt("Hello {{ name }}, age {{ age }}", {"name": "Mina", "age": 23})
        assert "Mina" in result
        assert "23" in result

    def test_missing_variable_renders_empty(self):
        result = render_prompt("Hello {{ name }}", {})
        assert "Hello" in result


class TestParseCalendar:
    def test_valid_json_object(self):
        data = {
            "items": [
                {
                    "post_date": "2026-04-10",
                    "concept": "cafe",
                    "caption": "morning vibes",
                    "image_prompt": "cozy cafe",
                    "hashtags": ["cafe", "daily"],
                }
            ]
        }
        result = _parse_calendar(json.dumps(data))
        assert isinstance(result, ContentCalendar)
        assert len(result.items) == 1
        assert result.items[0].concept == "cafe"

    def test_valid_json_array(self):
        """LLM sometimes returns a bare array instead of an object."""
        data = [
            {
                "post_date": "2026-04-10",
                "concept": "cafe",
                "caption": "morning vibes",
                "image_prompt": "cozy cafe",
                "hashtags": ["cafe"],
            }
        ]
        result = _parse_calendar(json.dumps(data))
        assert len(result.items) == 1

    def test_invalid_json_raises(self):
        with pytest.raises(LLMParseError):
            _parse_calendar("not json at all")

    def test_missing_required_field_raises(self):
        data = {"items": [{"post_date": "2026-04-10"}]}
        with pytest.raises(Exception):  # Pydantic ValidationError
            _parse_calendar(json.dumps(data))


class TestGenerateCalendar:
    @pytest.mark.asyncio
    async def test_fallback_to_second_provider(self):
        """If primary provider fails, should try fallback."""
        valid_response = json.dumps({
            "items": [
                {
                    "post_date": "2026-04-10",
                    "concept": "test",
                    "caption": "test caption",
                    "image_prompt": "test prompt",
                    "hashtags": ["test"],
                }
            ]
        })

        with (
            patch("services.llm.settings") as mock_settings,
            patch("services.llm._call_anthropic", new_callable=AsyncMock) as mock_anthropic,
            patch("services.llm._call_openai", new_callable=AsyncMock) as mock_openai,
        ):
            mock_settings.default_llm = "anthropic"
            mock_settings.llm_retries = 1
            mock_anthropic.side_effect = Exception("API down")
            mock_openai.return_value = valid_response

            from services.llm import generate_calendar

            result = await generate_calendar(
                {"name": "Mina", "niche": "fashion"},
                "Generate content for {{ name }}",
            )
            assert len(result.items) == 1
            mock_openai.assert_called_once()

    @pytest.mark.asyncio
    async def test_all_providers_fail_raises(self):
        with (
            patch("services.llm.settings") as mock_settings,
            patch("services.llm._call_anthropic", new_callable=AsyncMock) as mock_anthropic,
            patch("services.llm._call_openai", new_callable=AsyncMock) as mock_openai,
        ):
            mock_settings.default_llm = "anthropic"
            mock_settings.llm_retries = 1
            mock_anthropic.side_effect = Exception("down")
            mock_openai.side_effect = Exception("also down")

            from services.llm import generate_calendar

            with pytest.raises(LLMError):
                await generate_calendar(
                    {"name": "Mina", "niche": "fashion"},
                    "Generate for {{ name }}",
                )
