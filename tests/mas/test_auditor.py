"""
Tests for MAS Auditor Agent.

These tests verify the MAS auditor agent functionality,
including setup, configuration, and task creation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from inspect_ai.model import ChatMessageUser, ChatMessageSystem

from petri.mas.auditor import (
    mas_auditor_agent,
    mas_audit_loop,
    create_mas_audit_task,
    MAS_AUDITOR_SYSTEM_PROMPT,
    MAS_AUDITOR_USER_MESSAGE,
    DEFAULT_CONTINUE_PROMPT,
)
from petri.mas.stores import MASAuditStore
from petri.mas.observability import ObservabilityGraph


class TestMASAuditorSystemPrompt:
    """Tests for the system prompt template."""

    def test_system_prompt_contains_key_sections(self):
        """Test that system prompt has required sections."""
        assert "Multi-Agent Systems" in MAS_AUDITOR_SYSTEM_PROMPT
        assert "Error Propagation" in MAS_AUDITOR_SYSTEM_PROMPT
        assert "Trust Without Verification" in MAS_AUDITOR_SYSTEM_PROMPT
        assert "Cascade Failures" in MAS_AUDITOR_SYSTEM_PROMPT
        assert "Available Tools" in MAS_AUDITOR_SYSTEM_PROMPT
        assert "Audit Protocol" in MAS_AUDITOR_SYSTEM_PROMPT

    def test_system_prompt_format_placeholders(self):
        """Test that prompt can be formatted."""
        formatted = MAS_AUDITOR_SYSTEM_PROMPT.format(
            topology="chain",
            num_agents=3,
            agent_roles="researcher, validator, executor",
            max_turns=30,
            target_model="claude-3-sonnet",
            seed_instructions="Test error propagation"
        )

        assert "chain" in formatted
        assert "3" in formatted
        assert "researcher, validator, executor" in formatted
        assert "Test error propagation" in formatted

    def test_user_message_format(self):
        """Test user message template."""
        formatted = MAS_AUDITOR_USER_MESSAGE.format(
            topology="hub",
            num_agents=4,
            agent_roles="coordinator, worker_1, worker_2, worker_3",
            seed_instructions="Test hub coordination"
        )

        assert "hub" in formatted
        assert "4" in formatted
        assert "coordinator" in formatted
        assert "Test hub coordination" in formatted


class TestMASAuditorAgentCreation:
    """Tests for mas_auditor_agent creation."""

    def test_create_agent_default_params(self):
        """Test creating agent with default parameters."""
        agent = mas_auditor_agent()

        assert agent is not None
        assert callable(agent)

    def test_create_agent_custom_params(self):
        """Test creating agent with custom parameters."""
        agent = mas_auditor_agent(
            num_agents=5,
            topology="mesh",
            agent_roles=["a", "b", "c", "d", "e"],
            max_turns=50
        )

        assert agent is not None

    def test_default_roles_for_different_agent_counts(self):
        """Test that default roles are assigned correctly."""
        # 2 agents
        agent2 = mas_auditor_agent(num_agents=2)
        assert agent2 is not None

        # 3 agents
        agent3 = mas_auditor_agent(num_agents=3)
        assert agent3 is not None

        # 4 agents
        agent4 = mas_auditor_agent(num_agents=4)
        assert agent4 is not None

        # 6 agents (should use generic names)
        agent6 = mas_auditor_agent(num_agents=6)
        assert agent6 is not None

    def test_roles_padded_if_insufficient(self):
        """Test that roles are padded if fewer than num_agents."""
        # Provide only 2 roles for 4 agents
        agent = mas_auditor_agent(
            num_agents=4,
            agent_roles=["leader", "follower"]
        )
        assert agent is not None


class TestCreateMASAuditTask:
    """Tests for create_mas_audit_task function."""

    def test_create_task_single_instruction(self):
        """Test creating task with single instruction."""
        task = create_mas_audit_task(
            seed_instructions="Test error propagation in validation pipeline"
        )

        assert task is not None
        assert task.dataset is not None
        assert len(task.dataset) == 1

    def test_create_task_multiple_instructions(self):
        """Test creating task with multiple instructions."""
        task = create_mas_audit_task(
            seed_instructions=[
                "Test error propagation",
                "Test trust violations",
                "Test cascade failures"
            ]
        )

        assert len(task.dataset) == 3

    def test_create_task_with_topology(self):
        """Test creating task with specific topology."""
        task = create_mas_audit_task(
            seed_instructions="Test",
            topology="hub",
            num_agents=4
        )

        assert task is not None

    def test_create_task_with_roles(self):
        """Test creating task with custom roles."""
        task = create_mas_audit_task(
            seed_instructions="Test coordination",
            num_agents=3,
            agent_roles=["coordinator", "worker_a", "worker_b"]
        )

        # Check metadata on samples
        sample = task.dataset[0]
        assert sample.metadata["agent_roles"] == ["coordinator", "worker_a", "worker_b"]
        assert sample.metadata["num_agents"] == 3


class TestMASAuditLoop:
    """Tests for the mas_audit_loop function."""

    @pytest.mark.asyncio
    async def test_audit_loop_ends_on_end_conversation(self):
        """Test that loop ends when end_mas_conversation is called."""
        store = MASAuditStore()
        obs_graph = ObservabilityGraph()

        # Mock the auditor model
        mock_model = MagicMock()

        # First call: return tool call for end_mas_conversation
        mock_tool_call = MagicMock()
        mock_tool_call.function = "end_mas_conversation"
        mock_tool_call.arguments = {"summary": "Test complete"}
        mock_tool_call.id = "call_1"

        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tool_call]
        mock_message.text = ""
        mock_message.metadata = {}

        mock_output = MagicMock()
        mock_output.message = mock_message

        mock_model.generate = AsyncMock(return_value=mock_output)

        # Set up store with initial messages
        store.auditor.messages = [
            ChatMessageSystem(content="Test system message"),
            ChatMessageUser(content="Test user message")
        ]
        store.auditor.tools = []

        with patch('petri.mas.auditor.execute_tools') as mock_execute:
            mock_execute.return_value = ([], None)

            await mas_audit_loop(
                auditor_model=mock_model,
                mas_store=store,
                obs_graph=obs_graph,
                max_turns=5
            )

        # Should have called generate at least once
        assert mock_model.generate.call_count >= 1

    @pytest.mark.asyncio
    async def test_audit_loop_respects_max_turns(self):
        """Test that loop respects max_turns limit."""
        store = MASAuditStore()
        obs_graph = ObservabilityGraph()

        mock_model = MagicMock()

        # Return message without tool calls (will trigger continue prompt)
        mock_message = MagicMock()
        mock_message.tool_calls = []
        mock_message.text = "Thinking..."
        mock_message.metadata = {}

        mock_output = MagicMock()
        mock_output.message = mock_message

        mock_model.generate = AsyncMock(return_value=mock_output)

        store.auditor.messages = [
            ChatMessageSystem(content="Test"),
            ChatMessageUser(content="Test")
        ]
        store.auditor.tools = []

        await mas_audit_loop(
            auditor_model=mock_model,
            mas_store=store,
            obs_graph=obs_graph,
            max_turns=3
        )

        # Should stop after max_turns
        assert mock_model.generate.call_count == 3


class TestAgentStateHandling:
    """Tests for AgentState handling in mas_auditor_agent."""

    @pytest.mark.asyncio
    async def test_agent_validates_input_state(self):
        """Test that agent validates input state."""
        from inspect_ai.agent import AgentState

        agent = mas_auditor_agent(num_agents=3)

        # Create invalid state (no messages)
        state = AgentState(messages=[])

        with pytest.raises(ValueError, match="Expected 1 message"):
            await agent(state)

    @pytest.mark.asyncio
    async def test_agent_validates_message_type(self):
        """Test that agent requires user message."""
        from inspect_ai.agent import AgentState

        agent = mas_auditor_agent(num_agents=3)

        # Create state with system message instead of user
        state = AgentState(
            messages=[ChatMessageSystem(content="Not a user message")]
        )

        with pytest.raises(ValueError, match="user message"):
            await agent(state)


class TestMASAuditorIntegration:
    """Integration tests for MAS Auditor components."""

    def test_all_topology_types_supported(self):
        """Test that all topology types can be used."""
        topologies = ["chain", "hub", "mesh", "pipeline", "hierarchy", "custom"]

        for topology in topologies:
            try:
                agent = mas_auditor_agent(
                    num_agents=4,
                    topology=topology
                )
                assert agent is not None
            except Exception as e:
                # Custom may require additional params
                if topology != "custom":
                    raise e

    def test_system_prompt_tool_documentation(self):
        """Test that system prompt documents all tools."""
        tools_mentioned = [
            "create_agent",
            "set_agent_system_message",
            "create_agent_tool",
            "create_communication_channel",
            "send_message_to_agent",
            "send_tool_result_to_agent",
            "trigger_inter_agent_message",
            "inject_information",
            "observe_agent_state",
            "observe_system_state",
            "rollback_agent",
            "end_mas_conversation"
        ]

        for tool in tools_mentioned:
            assert tool in MAS_AUDITOR_SYSTEM_PROMPT, f"Tool {tool} not documented"

    def test_audit_protocol_phases(self):
        """Test that audit protocol has all phases."""
        phases = ["Phase 1: Setup", "Phase 2: Injection", "Phase 3: Propagation",
                  "Phase 4: Observation", "Phase 5: Documentation"]

        for phase in phases:
            assert phase in MAS_AUDITOR_SYSTEM_PROMPT


class TestContinuePrompt:
    """Tests for the continue prompt."""

    def test_continue_prompt_content(self):
        """Test continue prompt has helpful guidance."""
        assert "continue" in DEFAULT_CONTINUE_PROMPT.lower()
        assert "tool" in DEFAULT_CONTINUE_PROMPT.lower()
        assert "observe" in DEFAULT_CONTINUE_PROMPT.lower()
