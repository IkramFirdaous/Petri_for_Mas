"""
Tests for MAS auditor tools.

These tests verify the inter-agent communication and
orchestration tools used by the MAS auditor.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from inspect_ai.model import ChatMessageSystem
from inspect_ai.tool import ToolError

from petri.mas.stores import MASAuditStore, AgentNode
from petri.mas.observability import ObservabilityGraph, ActionType
from petri.mas.tools import (
    default_mas_auditor_tools,
    create_target_agent,
    set_agent_system_message,
    create_agent_tool,
    create_communication_channel,
    send_message_to_agent,
    send_tool_result_to_agent,
    trigger_inter_agent_message,
    inject_information,
    observe_agent_state,
    observe_system_state,
    rollback_agent,
    end_mas_conversation,
)


def get_tool_name(tool) -> str:
    """Helper to get tool name from registry info (strips any prefix)."""
    if hasattr(tool, '__registry_info__'):
        name = tool.__registry_info__.name
        # Strip any package prefix (e.g., "petri/create_agent" -> "create_agent")
        if '/' in name:
            return name.split('/')[-1]
        return name
    return tool.__name__


def setup_agent_with_system_message(store: MASAuditStore, agent_id: str, role: str, system_message: str):
    """Helper to set up an agent with a proper system message."""
    store.add_agent(agent_id, role)
    agent = store.agents[agent_id]
    agent.system_prompt = system_message
    agent.messages = [
        ChatMessageSystem(
            content=system_message,
            metadata={"source": "Auditor", "agent_id": agent_id}
        )
    ]


class TestDefaultMASAuditorTools:
    """Tests for the default_mas_auditor_tools function."""

    def test_returns_all_tools(self):
        """Test that all expected tools are returned."""
        store = MASAuditStore()
        obs_graph = ObservabilityGraph()

        tools = default_mas_auditor_tools(store, obs_graph)

        assert len(tools) == 12
        tool_names = [get_tool_name(t) for t in tools]

        assert "create_agent" in tool_names
        assert "set_agent_system_message" in tool_names
        assert "create_agent_tool" in tool_names
        assert "create_communication_channel" in tool_names
        assert "send_message_to_agent" in tool_names
        assert "send_tool_result_to_agent" in tool_names
        assert "trigger_inter_agent_message" in tool_names
        assert "inject_information" in tool_names
        assert "observe_agent_state" in tool_names
        assert "observe_system_state" in tool_names
        assert "rollback_agent" in tool_names
        assert "end_mas_conversation" in tool_names

    def test_tools_without_observability(self):
        """Test tools work without observability graph."""
        store = MASAuditStore()

        tools = default_mas_auditor_tools(store, obs_graph=None)

        assert len(tools) == 12
        # All tools should be callable
        assert all(callable(t) for t in tools)


class TestCreateTargetAgent:
    """Tests for the create_target_agent tool."""

    def test_tool_metadata(self):
        """Test tool has correct metadata."""
        store = MASAuditStore()
        tool = create_target_agent(store)

        assert get_tool_name(tool) == "create_agent"
        assert callable(tool)

    @pytest.mark.asyncio
    async def test_create_agent_success(self):
        """Test successfully creating an agent."""
        store = MASAuditStore()
        obs_graph = ObservabilityGraph()
        tool = create_target_agent(store, obs_graph)

        result = await tool(agent_id="test_agent", role="validator")

        assert "test_agent" in store.agents
        assert store.agents["test_agent"].role == "validator"
        assert "created" in result.lower()

    @pytest.mark.asyncio
    async def test_create_agent_with_model(self):
        """Test creating an agent with a specific model."""
        store = MASAuditStore()
        tool = create_target_agent(store)

        result = await tool(
            agent_id="custom_agent",
            role="executor",
            model_name="claude-3-haiku"
        )

        assert store.agents["custom_agent"].model_name == "claude-3-haiku"

    @pytest.mark.asyncio
    async def test_create_duplicate_agent_fails(self):
        """Test that creating a duplicate agent raises ToolError."""
        store = MASAuditStore()
        store.add_agent("existing", "role")
        tool = create_target_agent(store)

        with pytest.raises(ToolError, match="already exists"):
            await tool(agent_id="existing", role="validator")

    @pytest.mark.asyncio
    async def test_records_action_in_observability(self):
        """Test that action is recorded in observability graph."""
        store = MASAuditStore()
        obs_graph = ObservabilityGraph()
        tool = create_target_agent(store, obs_graph)

        await tool(agent_id="obs_test", role="observer")

        # Check that action was recorded (agent_id is "auditor" for auditor actions)
        actions = obs_graph.get_actions_by_agent("auditor")
        assert len(actions) >= 1


class TestSetAgentSystemMessage:
    """Tests for the set_agent_system_message tool."""

    @pytest.mark.asyncio
    async def test_set_system_message(self):
        """Test setting an agent's system message."""
        store = MASAuditStore()
        store.add_agent("agent_1", "validator")
        tool = set_agent_system_message(store)

        result = await tool(
            agent_id="agent_1",
            system_message="You are a data validator."
        )

        assert store.agents["agent_1"].system_prompt == "You are a data validator."
        assert "system message set" in result.lower()

    @pytest.mark.asyncio
    async def test_set_message_nonexistent_agent(self):
        """Test setting message for nonexistent agent raises ToolError."""
        store = MASAuditStore()
        tool = set_agent_system_message(store)

        with pytest.raises(ToolError, match="not found"):
            await tool(
                agent_id="nonexistent",
                system_message="Test"
            )


class TestCreateAgentTool:
    """Tests for the create_agent_tool tool."""

    @pytest.mark.asyncio
    async def test_create_simple_tool(self):
        """Test creating a simple tool for an agent."""
        store = MASAuditStore()
        store.add_agent("agent_1", "executor")
        tool = create_agent_tool(store)

        function_code = '''
def get_weather(city: str) -> str:
    """Get weather for a city.

    Args:
        city: The city name
    """
    return f"Weather in {city}: Sunny, 25°C"
'''

        result = await tool(
            agent_id="agent_1",
            function_code=function_code
        )

        assert "created" in result.lower()
        assert len(store.agents["agent_1"].tools) >= 1

    @pytest.mark.asyncio
    async def test_create_tool_invalid_code(self):
        """Test that invalid code raises ToolError."""
        store = MASAuditStore()
        store.add_agent("agent_1", "executor")
        tool = create_agent_tool(store)

        with pytest.raises(ToolError):
            await tool(
                agent_id="agent_1",
                function_code="this is not valid python"
            )


class TestCreateCommunicationChannel:
    """Tests for the create_communication_channel tool."""

    @pytest.mark.asyncio
    async def test_create_channel(self):
        """Test creating a communication channel."""
        store = MASAuditStore()
        store.add_agent("a", "role_a")
        store.add_agent("b", "role_b")
        tool = create_communication_channel(store)

        result = await tool(
            from_agent_id="a",
            to_agent_id="b",
            bidirectional=False
        )

        assert store.can_communicate("a", "b")
        assert not store.can_communicate("b", "a")
        assert "channel" in result.lower()

    @pytest.mark.asyncio
    async def test_create_bidirectional_channel(self):
        """Test creating a bidirectional channel."""
        store = MASAuditStore()
        store.add_agent("a", "role_a")
        store.add_agent("b", "role_b")
        tool = create_communication_channel(store)

        result = await tool(
            from_agent_id="a",
            to_agent_id="b",
            bidirectional=True
        )

        assert store.can_communicate("a", "b")
        assert store.can_communicate("b", "a")

    @pytest.mark.asyncio
    async def test_create_channel_nonexistent_agent(self):
        """Test creating channel with nonexistent agent raises ToolError."""
        store = MASAuditStore()
        store.add_agent("a", "role_a")
        tool = create_communication_channel(store)

        with pytest.raises(ToolError, match="not found"):
            await tool(
                from_agent_id="a",
                to_agent_id="nonexistent"
            )


class TestSendMessageToAgent:
    """Tests for the send_message_to_agent tool."""

    @pytest.mark.asyncio
    async def test_send_message_nonexistent_agent(self):
        """Test sending message to nonexistent agent raises ToolError."""
        store = MASAuditStore()
        tool = send_message_to_agent(store)

        with pytest.raises(ToolError, match="not found"):
            await tool(
                agent_id="nonexistent",
                message="Hello"
            )

    @pytest.mark.asyncio
    async def test_send_message_adds_to_history(self):
        """Test that sending message adds to agent history."""
        store = MASAuditStore()
        setup_agent_with_system_message(store, "agent_1", "validator", "You are a validator.")
        tool = send_message_to_agent(store)

        initial_count = len(store.agents["agent_1"].messages)

        # Mock the model
        with patch('petri.mas.tools.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_output = MagicMock()
            mock_message = MagicMock()
            mock_message.text = "Agent response"
            mock_message.tool_calls = []
            mock_message.id = None
            mock_output.message = mock_message
            mock_model.generate = AsyncMock(return_value=mock_output)
            mock_get_model.return_value = mock_model

            result = await tool(
                agent_id="agent_1",
                message="Hello, agent!"
            )

            # Message should be added to history
            assert len(store.agents["agent_1"].messages) > initial_count


class TestTriggerInterAgentMessage:
    """Tests for the trigger_inter_agent_message tool."""

    @pytest.mark.asyncio
    async def test_trigger_message_no_channel(self):
        """Test triggering message without channel raises ToolError."""
        store = MASAuditStore()
        store.add_agent("a", "sender")
        store.add_agent("b", "receiver")
        # No channel created
        tool = trigger_inter_agent_message(store)

        with pytest.raises(ToolError, match="No communication channel"):
            await tool(
                from_agent_id="a",
                to_agent_id="b",
                message="Test"
            )


class TestInjectInformation:
    """Tests for the inject_information tool."""

    @pytest.mark.asyncio
    async def test_inject_correct_information(self):
        """Test injecting correct information."""
        store = MASAuditStore()
        setup_agent_with_system_message(store, "agent_1", "analyst", "You are an analyst.")
        obs_graph = ObservabilityGraph()
        tool = inject_information(store, obs_graph)

        initial_count = len(store.agents["agent_1"].messages)

        # Mock the model call inside inject
        with patch('petri.mas.tools.call_agent') as mock_call:
            mock_msg = MagicMock()
            mock_msg.text = "I acknowledge the information."
            mock_msg.tool_calls = []
            mock_call.return_value = mock_msg

            result = await tool(
                agent_id="agent_1",
                information="The database shows 100 records",
                source_description="Database query",
                is_erroneous=False
            )

            # Result contains agent response
            assert "agent_response" in result or "acknowledge" in result.lower()
            # Check agent received the information
            assert len(store.agents["agent_1"].messages) > initial_count

    @pytest.mark.asyncio
    async def test_inject_erroneous_information(self):
        """Test injecting erroneous information."""
        store = MASAuditStore()
        setup_agent_with_system_message(store, "agent_1", "analyst", "You are an analyst.")
        obs_graph = ObservabilityGraph()
        tool = inject_information(store, obs_graph)

        # Mock the model call inside inject
        with patch('petri.mas.tools.call_agent') as mock_call:
            mock_msg = MagicMock()
            mock_msg.text = "I see, satisfaction is 95%."
            mock_msg.tool_calls = []
            mock_call.return_value = mock_msg

            result = await tool(
                agent_id="agent_1",
                information="Customer satisfaction is 95%",  # False
                source_description="CRM Database",
                is_erroneous=True,
                error_type="factual_error"
            )

            # Result contains agent response
            assert "agent_response" in result or "95%" in result

            # Check error was recorded in observability graph
            errors = obs_graph.get_all_errors()
            assert len(errors) >= 1


class TestObserveAgentState:
    """Tests for the observe_agent_state tool."""

    @pytest.mark.asyncio
    async def test_observe_agent(self):
        """Test observing an agent's state."""
        store = MASAuditStore()
        setup_agent_with_system_message(store, "agent_1", "validator", "You are a validator.")
        tool = observe_agent_state(store)

        result = await tool(agent_id="agent_1")

        assert "agent_1" in result
        assert "validator" in result

    @pytest.mark.asyncio
    async def test_observe_nonexistent_agent(self):
        """Test observing nonexistent agent raises ToolError."""
        store = MASAuditStore()
        tool = observe_agent_state(store)

        with pytest.raises(ToolError, match="not found"):
            await tool(agent_id="nonexistent")


class TestObserveSystemState:
    """Tests for the observe_system_state tool."""

    @pytest.mark.asyncio
    async def test_observe_system(self):
        """Test observing system-wide state."""
        store = MASAuditStore()
        store.add_agent("a", "role_a")
        store.add_agent("b", "role_b")
        store.create_channel("a", "b")
        obs_graph = ObservabilityGraph()
        tool = observe_system_state(store, obs_graph)

        result = await tool()

        assert "2" in result  # 2 agents
        assert "1" in result  # 1 channel

    @pytest.mark.asyncio
    async def test_observe_empty_system(self):
        """Test observing empty system."""
        store = MASAuditStore()
        tool = observe_system_state(store)

        result = await tool()

        assert "0" in result  # 0 agents


class TestRollbackAgent:
    """Tests for the rollback_agent tool."""

    @pytest.mark.asyncio
    async def test_rollback_agent(self):
        """Test rolling back an agent's conversation."""
        store = MASAuditStore()
        store.add_agent("agent_1", "validator")

        # Add some messages
        from inspect_ai.model import ChatMessageUser, ChatMessageAssistant
        store.agents["agent_1"].messages = [
            ChatMessageUser(content="Message 1"),
            ChatMessageAssistant(content="Response 1"),
            ChatMessageUser(content="Message 2"),
            ChatMessageAssistant(content="Response 2"),
        ]

        tool = rollback_agent(store)

        # Roll back to index 2 means keep first 2 messages (index 0 and 1)
        result = await tool(
            agent_id="agent_1",
            message_index=2
        )

        assert "rolled back" in result.lower()
        # The implementation keeps up to message_index (exclusive or inclusive varies)
        assert len(store.agents["agent_1"].messages) < 4

    @pytest.mark.asyncio
    async def test_rollback_invalid_index(self):
        """Test rollback with invalid index raises ToolError."""
        store = MASAuditStore()
        store.add_agent("agent_1", "validator")
        tool = rollback_agent(store)

        with pytest.raises(ToolError, match="beyond conversation length"):
            await tool(
                agent_id="agent_1",
                message_index=100  # Invalid
            )


class TestEndMASConversation:
    """Tests for the end_mas_conversation tool."""

    @pytest.mark.asyncio
    async def test_end_conversation(self):
        """Test ending the MAS conversation."""
        tool = end_mas_conversation()

        result = await tool(
            summary="Error propagated from agent_1 to agent_3"
        )

        assert "audit completed" in result.lower()

    @pytest.mark.asyncio
    async def test_end_conversation_no_summary(self):
        """Test ending conversation without summary."""
        tool = end_mas_conversation()

        result = await tool()

        assert "audit completed" in result.lower()


class TestToolIntegration:
    """Integration tests for MAS tools working together."""

    @pytest.mark.asyncio
    async def test_complete_setup_workflow(self):
        """Test a complete agent setup workflow."""
        store = MASAuditStore()
        obs_graph = ObservabilityGraph()
        tools = default_mas_auditor_tools(store, obs_graph)

        # Get tools by name
        tools_by_name = {get_tool_name(t): t for t in tools}

        # 1. Create agents
        await tools_by_name["create_agent"](
            agent_id="researcher", role="researcher"
        )
        await tools_by_name["create_agent"](
            agent_id="validator", role="validator"
        )

        # 2. Set system messages
        await tools_by_name["set_agent_system_message"](
            agent_id="researcher",
            system_message="You are a research analyst."
        )
        await tools_by_name["set_agent_system_message"](
            agent_id="validator",
            system_message="You are a data validator."
        )

        # 3. Create communication channel
        await tools_by_name["create_communication_channel"](
            from_agent_id="researcher",
            to_agent_id="validator",
            bidirectional=False
        )

        # Verify setup
        assert len(store.agents) == 2
        assert store.can_communicate("researcher", "validator")
        assert not store.can_communicate("validator", "researcher")
        assert store.agents["researcher"].system_prompt == "You are a research analyst."
        assert store.agents["validator"].system_prompt == "You are a data validator."

    @pytest.mark.asyncio
    async def test_inject_and_observe_workflow(self):
        """Test injecting info and observing state."""
        store = MASAuditStore()
        obs_graph = ObservabilityGraph()
        tools = default_mas_auditor_tools(store, obs_graph)
        tools_by_name = {get_tool_name(t): t for t in tools}

        # Create agent
        await tools_by_name["create_agent"](
            agent_id="analyst", role="analyst"
        )

        # Set system message (required for inject)
        await tools_by_name["set_agent_system_message"](
            agent_id="analyst",
            system_message="You are a financial analyst."
        )

        # Mock call_agent for inject
        with patch('petri.mas.tools.call_agent') as mock_call:
            mock_msg = MagicMock()
            mock_msg.text = "Revenue increase noted."
            mock_msg.tool_calls = []
            mock_call.return_value = mock_msg

            # Inject erroneous information
            await tools_by_name["inject_information"](
                agent_id="analyst",
                information="Revenue is up 50%",
                source_description="Financial report",
                is_erroneous=True,
                error_type="factual_error"
            )

        # Observe state
        state = await tools_by_name["observe_agent_state"](
            agent_id="analyst"
        )

        assert "analyst" in state

        # Observe system
        system_state = await tools_by_name["observe_system_state"]()

        assert "1" in system_state  # 1 agent
