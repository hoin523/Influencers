"""
Database models for the AI Influencer Factory.

STATUS STATE MACHINE (content_queue)
═══════════════════════════════════════════

  planned ──> generating ──> generated ──> approved ──> posted
     ▲            │              │
     │            ▼              │
     │         error ◄───────────┘
     │            │
     │            ▼
     └──── [dashboard retry]

     ▲
     └──── rejected (dashboard → planned)
"""

import enum
import json
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


class ContentStatus(str, enum.Enum):
    PLANNED = "planned"
    GENERATING = "generating"
    GENERATED = "generated"
    APPROVED = "approved"
    POSTED = "posted"
    ERROR = "error"


# Valid state transitions
_VALID_TRANSITIONS: dict[ContentStatus, set[ContentStatus]] = {
    ContentStatus.PLANNED: {ContentStatus.GENERATING},
    ContentStatus.GENERATING: {ContentStatus.GENERATED, ContentStatus.ERROR},
    ContentStatus.GENERATED: {ContentStatus.APPROVED, ContentStatus.PLANNED},  # PLANNED = rejected
    ContentStatus.APPROVED: {ContentStatus.POSTED},
    ContentStatus.POSTED: set(),
    ContentStatus.ERROR: {ContentStatus.PLANNED},  # retry
}


class InvalidTransitionError(ValueError):
    pass


def validate_transition(current: ContentStatus, target: ContentStatus) -> None:
    allowed = _VALID_TRANSITIONS.get(current, set())
    if target not in allowed:
        raise InvalidTransitionError(
            f"Cannot transition from {current.value!r} to {target.value!r}. "
            f"Allowed: {[s.value for s in allowed]}"
        )


class Persona(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    age: int
    gender: str
    niche: str
    personality_traits: str  # comma-separated
    speaking_style: str
    visual_base_prompt: str
    lora_path: Optional[str] = None
    reference_face_images: str = "[]"  # JSON array of file paths
    platforms: str = "[]"  # JSON array of platform names
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def get_reference_faces(self) -> list[str]:
        return json.loads(self.reference_face_images)

    def get_platforms(self) -> list[str]:
        return json.loads(self.platforms)


class ContentQueue(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    persona_id: int = Field(foreign_key="persona.id", index=True)
    post_date: str  # YYYY-MM-DD
    concept: str
    caption: str = ""
    image_prompt: str = ""
    hashtags: str = "[]"  # JSON array
    status: ContentStatus = ContentStatus.PLANNED
    image_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def transition_to(self, target: ContentStatus) -> None:
        validate_transition(self.status, target)
        self.status = target
        self.updated_at = datetime.now(timezone.utc)

    def get_hashtags(self) -> list[str]:
        return json.loads(self.hashtags)


class Post(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    content_queue_id: int = Field(foreign_key="contentqueue.id", index=True)
    platform: str
    posted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    url: Optional[str] = None
    engagement_metrics: Optional[str] = None  # JSON: {likes, comments, shares, saves, reach}
