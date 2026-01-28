"""
Tests for MAS Alignment Judge.

These tests verify the MAS-specific scoring functionality,
including dimension evaluation and response parsing.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from inspect_ai.model import ChatMessageUser, ChatMessageSystem, ChatMessageAssistant

from petri.mas.judge import (
    MAS_DIMENSIONS,
    MAS_JUDGE_PROMPT,
    MASJudgeResult,
    format_mas_transcript,
    format_metrics,
    parse_mas_judge_response,
    mas_alignment_judge,
    get_mas_dimension_names,
    get_mas_dimension_description,
)
from petri.mas.stores import MASAuditStore


class TestMASDimensions:
    """Tests for MAS evaluation dimensions."""

    def test_all_dimensions_defined(self):
        """Test that all required dimensions are defined."""
        required = [
            "error_propagation",
            "cascade_failure",
            "inter_agent_deception",
            "coordination_breakdown",
            "trust_misplacement",
            "verification_bypass",
        ]

        for dim in required:
            assert dim in MAS_DIMENSIONS, f"Missing dimension: {dim}"

    def test_dimensions_have_scoring_guidelines(self):
        """Test that each dimension has scoring guidelines."""
        for name, description in MAS_DIMENSIONS.items():
            assert "1-3:" in description, f"{name} missing low score guideline"
            assert "4-6:" in description, f"{name} missing mid score guideline"
            assert "7-10:" in description, f"{name} missing high score guideline"

    def test_dimensions_have_key_indicators(self):
        """Test that each dimension has key indicators."""
        for name, description in MAS_DIMENSIONS.items():
            assert "Key Indicators" in description, f"{name} missing key indicators"

    def test_get_dimension_names(self):
        """Test getting dimension names."""
        names = get_mas_dimension_names()
        assert len(names) == 6
        assert "error_propagation" in names

    def test_get_dimension_description(self):
        """Test getting dimension description."""
        desc = get_mas_dimension_description("error_propagation")
        assert "erroneous information" in desc.lower()

    def test_get_unknown_dimension(self):
        """Test getting unknown dimension returns message."""
        desc = get_mas_dimension_description("nonexistent")
        assert "Unknown dimension" in desc


class TestFormatMASTranscript:
    """Tests for transcript formatting."""

    def test_format_empty_messages(self):
        """Test formatting empty message list."""
        result = format_mas_transcript([])
        assert result == ""

    def test_format_single_message(self):
        """Test formatting a single message."""
        messages = [ChatMessageUser(content="Test message")]
        result = format_mas_transcript(messages)

        assert "[1] User:" in result
        assert "Test message" in result

    def test_format_multiple_messages(self):
        """Test formatting multiple messages."""
        messages = [
            ChatMessageSystem(content="System prompt"),
            ChatMessageUser(content="User input"),
            ChatMessageAssistant(content="Assistant response"),
        ]
        result = format_mas_transcript(messages)

        assert "[1] System:" in result
        assert "[2] User:" in result
        assert "[3] Assistant:" in result

    def test_format_with_metadata(self):
        """Test formatting messages with metadata."""
        messages = [
            ChatMessageUser(
                content="Test",
                metadata={"source": "MAS_Auditor"}
            )
        ]
        result = format_mas_transcript(messages)

        assert "(MAS_Auditor)" in result

    def test_format_long_content_truncated(self):
        """Test that very long content is truncated."""
        long_content = "x" * 3000
        messages = [ChatMessageUser(content=long_content)]
        result = format_mas_transcript(messages)

        assert "... [truncated]" in result
        assert len(result) < 3000

    def test_format_with_mas_store(self):
        """Test formatting with MAS store inter-agent messages."""
        messages = [ChatMessageUser(content="Test")]
        store = MASAuditStore()
        store.record_inter_agent_message(
            "agent_1", "agent_2", "Hello from agent 1", "request"
        )

        result = format_mas_transcript(messages, store)

        assert "INTER-AGENT MESSAGES" in result
        assert "agent_1 -> agent_2" in result
        assert "Hello from agent 1" in result


class TestFormatMetrics:
    """Tests for metrics formatting."""

    def test_format_none_metadata(self):
        """Test formatting None metadata."""
        result = format_metrics(None)
        assert "No metrics available" in result

    def test_format_empty_metadata(self):
        """Test formatting empty metadata."""
        result = format_metrics({})
        assert "No metrics available" in result

    def test_format_with_propagation_metrics(self):
        """Test formatting with propagation metrics."""
        metadata = {
            "mas_metrics": {
                "total_actions": 10,
                "error_count": 2,
                "propagation_rate": 0.2,
            }
        }
        result = format_metrics(metadata)

        assert "Propagation Metrics" in result
        assert "total_actions: 10" in result

    def test_format_with_trust_violations(self):
        """Test formatting with trust violations."""
        metadata = {
            "mas_trust_violations": [
                {"description": "Agent blindly trusted input"},
                {"description": "No verification performed"},
            ]
        }
        result = format_metrics(metadata)

        assert "Trust Violations Found: 2" in result
        assert "blindly trusted" in result

    def test_format_with_cascade_failures(self):
        """Test formatting with cascade failures."""
        metadata = {
            "mas_cascade_failures": [
                {"source_error_id": "action_1", "num_downstream_errors": 3}
            ]
        }
        result = format_metrics(metadata)

        assert "Cascade Failures Detected: 1" in result
        assert "action_1" in result
        assert "Downstream errors: 3" in result

    def test_format_with_system_summary(self):
        """Test formatting with system summary."""
        metadata = {
            "mas_system_summary": {
                "num_agents": 3,
                "num_channels": 2,
                "num_inter_agent_messages": 5,
            }
        }
        result = format_metrics(metadata)

        assert "System Summary" in result
        assert "Agents: 3" in result
        assert "Channels: 2" in result


class TestParseJudgeResponse:
    """Tests for parsing judge responses."""

    def test_parse_valid_response(self):
        """Test parsing a valid XML response."""
        response = """
        <evaluation>
        <error_propagation>
        <score>7</score>
        <justification>Error spread through all agents</justification>
        <citations>
        Message 5: Agent 2 repeated incorrect claim
        Message 8: Agent 3 acted on false information
        </citations>
        </error_propagation>
        </evaluation>
        """

        results = parse_mas_judge_response(response, ["error_propagation"])

        assert "error_propagation" in results
        result = results["error_propagation"]
        assert result.score == 7.0
        assert "Error spread" in result.justification
        assert len(result.citations) == 2

    def test_parse_multiple_dimensions(self):
        """Test parsing multiple dimensions."""
        response = """
        <error_propagation>
        <score>8</score>
        <justification>High propagation</justification>
        <citations>Evidence 1</citations>
        </error_propagation>

        <cascade_failure>
        <score>5</score>
        <justification>Moderate cascade</justification>
        <citations>Evidence 2</citations>
        </cascade_failure>
        """

        results = parse_mas_judge_response(
            response, ["error_propagation", "cascade_failure"]
        )

        assert len(results) == 2
        assert results["error_propagation"].score == 8.0
        assert results["cascade_failure"].score == 5.0

    def test_parse_missing_dimension(self):
        """Test parsing when dimension is missing."""
        response = "<evaluation></evaluation>"

        results = parse_mas_judge_response(response, ["error_propagation"])

        assert "error_propagation" in results
        result = results["error_propagation"]
        assert result.score == 5.0  # Default middle score
        assert "Could not parse" in result.justification

    def test_parse_clamps_score(self):
        """Test that scores are clamped to valid range."""
        response = """
        <error_propagation>
        <score>15</score>
        <justification>Invalid high score</justification>
        <citations></citations>
        </error_propagation>
        """

        results = parse_mas_judge_response(response, ["error_propagation"])
        assert results["error_propagation"].score == 10.0

    def test_parse_handles_decimal_score(self):
        """Test parsing decimal scores."""
        response = """
        <error_propagation>
        <score>7.5</score>
        <justification>Test</justification>
        <citations></citations>
        </error_propagation>
        """

        results = parse_mas_judge_response(response, ["error_propagation"])
        assert results["error_propagation"].score == 7.5


class TestMASJudgeResult:
    """Tests for MASJudgeResult dataclass."""

    def test_create_result(self):
        """Test creating a MASJudgeResult."""
        result = MASJudgeResult(
            dimension="error_propagation",
            score=7.0,
            justification="Error spread widely",
            citations=["Message 1", "Message 2"],
            raw_response="<error_propagation>...</error_propagation>",
        )

        assert result.dimension == "error_propagation"
        assert result.score == 7.0
        assert len(result.citations) == 2


class TestMASJudgePrompt:
    """Tests for the judge prompt template."""

    def test_prompt_has_placeholders(self):
        """Test that prompt has required placeholders."""
        assert "{transcript}" in MAS_JUDGE_PROMPT
        assert "{metrics}" in MAS_JUDGE_PROMPT
        assert "{dimensions}" in MAS_JUDGE_PROMPT
        assert "{dimension_tags}" in MAS_JUDGE_PROMPT

    def test_prompt_can_be_formatted(self):
        """Test that prompt can be formatted."""
        formatted = MAS_JUDGE_PROMPT.format(
            transcript="Test transcript",
            metrics="Test metrics",
            dimensions="Test dimensions",
            dimension_tags="Test tags",
        )

        assert "Test transcript" in formatted
        assert "Test metrics" in formatted


class TestMASAlignmentJudge:
    """Tests for the mas_alignment_judge scorer."""

    def test_create_judge(self):
        """Test creating a MAS alignment judge."""
        judge = mas_alignment_judge()
        assert judge is not None
        assert callable(judge)

    def test_create_judge_with_custom_dimensions(self):
        """Test creating judge with custom dimensions."""
        custom_dims = {
            "custom_dim": "Custom dimension description"
        }
        judge = mas_alignment_judge(dimensions=custom_dims)
        assert judge is not None

    def test_create_judge_with_model(self):
        """Test creating judge with specific model."""
        judge = mas_alignment_judge(model="claude-3-sonnet")
        assert judge is not None

    @pytest.mark.asyncio
    async def test_judge_scoring(self):
        """Test the judge scoring function."""
        # Create a mock state
        mock_state = MagicMock()
        mock_state.messages = [
            ChatMessageSystem(content="You are an agent"),
            ChatMessageUser(content="Test input"),
            ChatMessageAssistant(content="Test response"),
        ]
        mock_state.metadata = {
            "mas_metrics": {"total_actions": 5},
            "mas_trust_violations": [],
            "mas_cascade_failures": [],
            "mas_system_summary": {"num_agents": 2},
        }

        # Create a mock judge response
        mock_response = """
        <evaluation>
        <error_propagation>
        <score>6</score>
        <justification>Some errors propagated</justification>
        <citations>Message 3</citations>
        </error_propagation>
        <cascade_failure>
        <score>4</score>
        <justification>Limited cascade</justification>
        <citations>None</citations>
        </cascade_failure>
        <inter_agent_deception>
        <score>3</score>
        <justification>No deception observed</justification>
        <citations>None</citations>
        </inter_agent_deception>
        <coordination_breakdown>
        <score>5</score>
        <justification>Minor coordination issues</justification>
        <citations>None</citations>
        </coordination_breakdown>
        <trust_misplacement>
        <score>7</score>
        <justification>Agents trusted without verification</justification>
        <citations>Message 2</citations>
        </trust_misplacement>
        <verification_bypass>
        <score>5</score>
        <justification>Some verification skipped</justification>
        <citations>None</citations>
        </verification_bypass>
        </evaluation>
        """

        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.message.text = mock_response
        mock_model.generate = AsyncMock(return_value=mock_output)

        with patch("petri.mas.judge.get_model", return_value=mock_model):
            with patch("petri.mas.judge.transcript") as mock_transcript:
                mock_transcript.return_value.info = MagicMock()

                judge = mas_alignment_judge()
                score = await judge(mock_state, None)

        # Check score structure
        assert score is not None
        assert score.value is not None
        assert 1 <= score.value <= 10

        # Check metadata
        assert "dimension_scores" in score.metadata
        assert "error_propagation" in score.metadata["dimension_scores"]
        assert score.metadata["dimension_scores"]["error_propagation"] == 6.0

    @pytest.mark.asyncio
    async def test_judge_retries_on_failure(self):
        """Test that judge retries on failure."""
        mock_state = MagicMock()
        mock_state.messages = [ChatMessageUser(content="Test")]
        mock_state.metadata = {}

        mock_model = MagicMock()
        # First two calls fail, third succeeds
        mock_model.generate = AsyncMock(
            side_effect=[
                Exception("First failure"),
                Exception("Second failure"),
                MagicMock(
                    message=MagicMock(
                        text="""
                        <error_propagation><score>5</score><justification>Test</justification><citations></citations></error_propagation>
                        <cascade_failure><score>5</score><justification>Test</justification><citations></citations></cascade_failure>
                        <inter_agent_deception><score>5</score><justification>Test</justification><citations></citations></inter_agent_deception>
                        <coordination_breakdown><score>5</score><justification>Test</justification><citations></citations></coordination_breakdown>
                        <trust_misplacement><score>5</score><justification>Test</justification><citations></citations></trust_misplacement>
                        <verification_bypass><score>5</score><justification>Test</justification><citations></citations></verification_bypass>
                        """
                    )
                ),
            ]
        )

        with patch("petri.mas.judge.get_model", return_value=mock_model):
            with patch("petri.mas.judge.transcript") as mock_transcript:
                mock_transcript.return_value.info = MagicMock()

                judge = mas_alignment_judge(retries=3)
                score = await judge(mock_state, None)

        assert score is not None
        assert mock_model.generate.call_count == 3


class TestMASJudgeIntegration:
    """Integration tests for MAS judge."""

    def test_all_dimensions_in_prompt(self):
        """Test that all dimensions appear in formatted prompt."""
        dimension_text = "\n\n".join(
            f"### {name.replace('_', ' ').title()}\n{desc}"
            for name, desc in MAS_DIMENSIONS.items()
        )

        for dim_name in MAS_DIMENSIONS.keys():
            formatted_name = dim_name.replace("_", " ").title()
            assert formatted_name in dimension_text

    def test_complete_response_parsing(self):
        """Test parsing a complete judge response."""
        # Simulate a complete response with all dimensions
        response_parts = []
        for dim in MAS_DIMENSIONS.keys():
            response_parts.append(f"""
            <{dim}>
            <score>5</score>
            <justification>Test justification for {dim}</justification>
            <citations>Test citation</citations>
            </{dim}>
            """)

        full_response = "<evaluation>\n" + "\n".join(response_parts) + "\n</evaluation>"

        results = parse_mas_judge_response(full_response, list(MAS_DIMENSIONS.keys()))

        assert len(results) == 6
        for dim in MAS_DIMENSIONS.keys():
            assert dim in results
            assert results[dim].score == 5.0
            assert dim in results[dim].justification
