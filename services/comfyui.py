"""
ComfyUI API integration service.

WORKFLOW JSON SUBSTITUTION + API CALL
══════════════════════════════════════════════
  workflow.json (from GUI export)
       │
       ▼
  substitute prompt text + reference face paths
       │
       ▼
  POST /prompt to ComfyUI ──> queue_id
       │
       ▼
  Poll /history/{queue_id} until done
       │
       ▼
  Download generated image ──> save to assets/{persona}/generated/
"""

import json
import logging
import uuid
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from config import settings

logger = logging.getLogger(__name__)


class ComfyUIError(Exception):
    pass


class ComfyUIConnectionError(ComfyUIError):
    pass


class ComfyUITimeoutError(ComfyUIError):
    pass


async def health_check() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.comfyui_url}/system_stats")
            return resp.status_code == 200
    except httpx.ConnectError:
        return False


def substitute_workflow(
    workflow: dict,
    positive_prompt: str,
    negative_prompt: str = "",
    reference_face_path: str | None = None,
) -> dict:
    """Replace placeholder values in a ComfyUI workflow JSON."""
    workflow_str = json.dumps(workflow)
    workflow_str = workflow_str.replace("{{POSITIVE_PROMPT}}", positive_prompt)
    workflow_str = workflow_str.replace("{{NEGATIVE_PROMPT}}", negative_prompt)
    if reference_face_path:
        workflow_str = workflow_str.replace("{{REFERENCE_FACE}}", reference_face_path)
    return json.loads(workflow_str)


@retry(
    stop=stop_after_attempt(settings.image_gen_retries),
    wait=wait_fixed(5),
    retry=retry_if_exception_type((ComfyUITimeoutError, httpx.HTTPStatusError)),
    reraise=True,
)
async def generate_image(
    workflow_path: Path,
    positive_prompt: str,
    negative_prompt: str,
    reference_face_path: str | None,
    output_path: Path,
) -> Path:
    """Send a workflow to ComfyUI and save the generated image."""
    if not await health_check():
        raise ComfyUIConnectionError(
            f"ComfyUI is not running at {settings.comfyui_url}. "
            "Start ComfyUI with: python main.py --listen"
        )

    with open(workflow_path) as f:
        workflow = json.load(f)

    workflow = substitute_workflow(
        workflow, positive_prompt, negative_prompt, reference_face_path
    )

    client_id = str(uuid.uuid4())
    payload = {"prompt": workflow, "client_id": client_id}

    async with httpx.AsyncClient(timeout=settings.comfyui_timeout) as client:
        # Queue the prompt
        try:
            resp = await client.post(
                f"{settings.comfyui_url}/prompt", json=payload
            )
            resp.raise_for_status()
        except httpx.ConnectError as e:
            raise ComfyUIConnectionError(str(e)) from e

        prompt_id = resp.json()["prompt_id"]

        # Poll for completion
        import asyncio

        for _ in range(settings.comfyui_timeout):
            await asyncio.sleep(1)
            history_resp = await client.get(
                f"{settings.comfyui_url}/history/{prompt_id}"
            )
            history = history_resp.json()
            if prompt_id in history:
                break
        else:
            raise ComfyUITimeoutError(
                f"Image generation timed out after {settings.comfyui_timeout}s"
            )

        # Extract output image
        outputs = history[prompt_id].get("outputs", {})
        image_info = None
        for node_output in outputs.values():
            if "images" in node_output:
                image_info = node_output["images"][0]
                break

        if not image_info:
            raise ComfyUIError("No image in ComfyUI output")

        # Download the image
        filename = image_info["filename"]
        subfolder = image_info.get("subfolder", "")
        img_resp = await client.get(
            f"{settings.comfyui_url}/view",
            params={"filename": filename, "subfolder": subfolder, "type": "output"},
        )
        img_resp.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(img_resp.content)
        logger.info("Image saved to %s", output_path)
        return output_path
