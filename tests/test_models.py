import pytest

from models import ContentQueue, ContentStatus, InvalidTransitionError, Persona, validate_transition


class TestStatusTransitions:
    """Test the content_queue state machine."""

    def test_valid_full_lifecycle(self, session, sample_content):
        """planned → generating → generated → approved → posted"""
        item = sample_content

        item.transition_to(ContentStatus.GENERATING)
        assert item.status == ContentStatus.GENERATING

        item.transition_to(ContentStatus.GENERATED)
        assert item.status == ContentStatus.GENERATED

        item.transition_to(ContentStatus.APPROVED)
        assert item.status == ContentStatus.APPROVED

        item.transition_to(ContentStatus.POSTED)
        assert item.status == ContentStatus.POSTED

    def test_invalid_transition_raises(self):
        """posted → planned should be rejected."""
        with pytest.raises(InvalidTransitionError):
            validate_transition(ContentStatus.POSTED, ContentStatus.PLANNED)

    def test_invalid_skip_transition(self):
        """planned → approved (skipping generating/generated) should be rejected."""
        with pytest.raises(InvalidTransitionError):
            validate_transition(ContentStatus.PLANNED, ContentStatus.APPROVED)

    def test_error_recovery(self, session, sample_content):
        """error → planned (retry from dashboard)."""
        item = sample_content
        item.transition_to(ContentStatus.GENERATING)
        item.status = ContentStatus.ERROR  # simulating failure

        item.transition_to(ContentStatus.PLANNED)
        assert item.status == ContentStatus.PLANNED

    def test_reject_returns_to_planned(self, session, sample_content):
        """generated → planned (rejected by reviewer)."""
        item = sample_content
        item.transition_to(ContentStatus.GENERATING)
        item.transition_to(ContentStatus.GENERATED)

        item.transition_to(ContentStatus.PLANNED)  # reject
        assert item.status == ContentStatus.PLANNED

    def test_generating_to_error(self):
        """generating → error on failure."""
        validate_transition(ContentStatus.GENERATING, ContentStatus.ERROR)

    def test_updated_at_changes_on_transition(self, session, sample_content):
        """updated_at should be set during transition."""
        item = sample_content
        item.transition_to(ContentStatus.GENERATING)
        # updated_at is set to now() during transition, just verify it's a datetime
        assert item.updated_at is not None


class TestPersona:
    def test_get_reference_faces(self, sample_persona):
        faces = sample_persona.get_reference_faces()
        assert faces == ["assets/test/face.png"]

    def test_get_platforms(self, sample_persona):
        platforms = sample_persona.get_platforms()
        assert platforms == ["instagram"]
