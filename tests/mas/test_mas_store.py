"""
Tests for MASAuditStore and related classes.

These tests verify the core state management functionality
for multi-agent system auditing.
"""

import pytest
import time

from petri.mas.stores import (
    MASAuditStore,
    AgentNode,
    InterAgentMessage,
    InterAgentChannel,
)


class TestAgentNode:
    """Tests for AgentNode class."""

    def test_create_agent_node(self):
        """Test basic agent node creation."""
        agent = AgentNode(
            agent_id="test_agent",
            role="validator",
            model_name="claude-3-sonnet"
        )

        assert agent.agent_id == "test_agent"
        assert agent.role == "validator"
        assert agent.model_name == "claude-3-sonnet"
        assert agent.messages == []
        assert agent.tools == []
        assert agent.system_prompt is None

    def test_agent_node_message_count(self):
        """Test message counting."""
        agent = AgentNode(agent_id="test", role="test")
        assert agent.get_message_count() == 0

    def test_agent_node_last_message_empty(self):
        """Test getting last message when empty."""
        agent = AgentNode(agent_id="test", role="test")
        assert agent.get_last_message() is None


class TestInterAgentMessage:
    """Tests for InterAgentMessage class."""

    def test_create_message(self):
        """Test basic message creation."""
        msg = InterAgentMessage(
            from_agent_id="agent_1",
            to_agent_id="agent_2",
            content="Hello",
            message_type="request"
        )

        assert msg.from_agent_id == "agent_1"
        assert msg.to_agent_id == "agent_2"
        assert msg.content == "Hello"
        assert msg.message_type == "request"
        assert msg.id is not None
        assert msg.timestamp > 0

    def test_message_is_response_to(self):
        """Test response chain tracking."""
        msg1 = InterAgentMessage(
            id="msg_1",
            from_agent_id="a",
            to_agent_id="b",
            content="Request",
            message_type="request"
        )

        msg2 = InterAgentMessage(
            id="msg_2",
            from_agent_id="b",
            to_agent_id="a",
            content="Response",
            message_type="response",
            parent_message_id="msg_1"
        )

        assert msg2.is_response_to("msg_1")
        assert not msg2.is_response_to("msg_999")


class TestInterAgentChannel:
    """Tests for InterAgentChannel class."""

    def test_create_channel(self):
        """Test basic channel creation."""
        channel = InterAgentChannel(
            from_agent_id="agent_1",
            to_agent_id="agent_2",
            channel_type="direct"
        )

        assert channel.from_agent_id == "agent_1"
        assert channel.to_agent_id == "agent_2"
        assert channel.channel_type == "direct"
        assert not channel.bidirectional

    def test_channel_allows_message_unidirectional(self):
        """Test unidirectional channel."""
        channel = InterAgentChannel(
            from_agent_id="a",
            to_agent_id="b",
            bidirectional=False
        )

        assert channel.allows_message("a", "b")
        assert not channel.allows_message("b", "a")
        assert not channel.allows_message("a", "c")

    def test_channel_allows_message_bidirectional(self):
        """Test bidirectional channel."""
        channel = InterAgentChannel(
            from_agent_id="a",
            to_agent_id="b",
            bidirectional=True
        )

        assert channel.allows_message("a", "b")
        assert channel.allows_message("b", "a")


class TestMASAuditStore:
    """Tests for MASAuditStore class."""

    def test_create_empty_store(self):
        """Test creating an empty store."""
        store = MASAuditStore()

        assert len(store.agents) == 0
        assert len(store.channels) == 0
        assert len(store.inter_agent_messages) == 0
        assert store.topology == "chain"

    def test_add_agent(self):
        """Test adding an agent."""
        store = MASAuditStore()
        agent = store.add_agent("agent_1", "validator", "claude-3-sonnet")

        assert agent.agent_id == "agent_1"
        assert agent.role == "validator"
        assert "agent_1" in store.agents
        assert store.metadata.num_agents == 1

    def test_add_duplicate_agent_raises(self):
        """Test that adding duplicate agent raises error."""
        store = MASAuditStore()
        store.add_agent("agent_1", "validator")

        with pytest.raises(ValueError, match="already exists"):
            store.add_agent("agent_1", "executor")

    def test_get_agent(self):
        """Test getting an agent."""
        store = MASAuditStore()
        store.add_agent("agent_1", "validator")

        agent = store.get_agent("agent_1")
        assert agent.agent_id == "agent_1"

    def test_get_nonexistent_agent_raises(self):
        """Test that getting nonexistent agent raises error."""
        store = MASAuditStore()

        with pytest.raises(KeyError):
            store.get_agent("nonexistent")

    def test_remove_agent(self):
        """Test removing an agent."""
        store = MASAuditStore()
        store.add_agent("agent_1", "validator")
        store.add_agent("agent_2", "executor")
        store.create_channel("agent_1", "agent_2")

        removed = store.remove_agent("agent_1")

        assert removed.agent_id == "agent_1"
        assert "agent_1" not in store.agents
        assert len(store.channels) == 0  # Channel should be removed too

    def test_list_agents(self):
        """Test listing all agents."""
        store = MASAuditStore()
        store.add_agent("a", "role_a")
        store.add_agent("b", "role_b")
        store.add_agent("c", "role_c")

        agents = store.list_agents()
        assert set(agents) == {"a", "b", "c"}

    def test_get_agents_by_role(self):
        """Test filtering agents by role."""
        store = MASAuditStore()
        store.add_agent("v1", "validator")
        store.add_agent("v2", "validator")
        store.add_agent("e1", "executor")

        validators = store.get_agents_by_role("validator")
        assert len(validators) == 2
        assert all(a.role == "validator" for a in validators)

    def test_create_channel(self):
        """Test creating a channel."""
        store = MASAuditStore()
        store.add_agent("a", "role_a")
        store.add_agent("b", "role_b")

        channel = store.create_channel("a", "b")

        assert channel.from_agent_id == "a"
        assert channel.to_agent_id == "b"
        assert len(store.channels) == 1

    def test_create_channel_nonexistent_agent_raises(self):
        """Test that creating channel with nonexistent agent raises."""
        store = MASAuditStore()
        store.add_agent("a", "role_a")

        with pytest.raises(KeyError):
            store.create_channel("a", "nonexistent")

    def test_can_communicate(self):
        """Test checking if agents can communicate."""
        store = MASAuditStore()
        store.add_agent("a", "role_a")
        store.add_agent("b", "role_b")
        store.add_agent("c", "role_c")

        store.create_channel("a", "b", bidirectional=True)

        assert store.can_communicate("a", "b")
        assert store.can_communicate("b", "a")  # Bidirectional
        assert not store.can_communicate("a", "c")
        assert not store.can_communicate("b", "c")

    def test_record_inter_agent_message(self):
        """Test recording an inter-agent message."""
        store = MASAuditStore()

        msg = store.record_inter_agent_message(
            from_agent="a",
            to_agent="b",
            content="Hello",
            message_type="request"
        )

        assert msg.from_agent_id == "a"
        assert msg.to_agent_id == "b"
        assert len(store.inter_agent_messages) == 1

    def test_get_messages_for_agent(self):
        """Test getting messages for a specific agent."""
        store = MASAuditStore()

        store.record_inter_agent_message("a", "b", "msg1", "request")
        store.record_inter_agent_message("b", "a", "msg2", "response")
        store.record_inter_agent_message("a", "c", "msg3", "request")

        sent = store.get_messages_for_agent("a", "sent")
        received = store.get_messages_for_agent("a", "received")
        all_msgs = store.get_messages_for_agent("a", "both")

        assert len(sent) == 2
        assert len(received) == 1
        assert len(all_msgs) == 3

    def test_get_conversation_thread(self):
        """Test getting a conversation thread."""
        store = MASAuditStore()

        msg1 = store.record_inter_agent_message("a", "b", "Hello", "request")
        msg2 = store.record_inter_agent_message(
            "b", "a", "Hi", "response",
            parent_message_id=msg1.id
        )
        msg3 = store.record_inter_agent_message(
            "a", "b", "How are you?", "request",
            parent_message_id=msg2.id
        )

        thread = store.get_conversation_thread(msg2.id)

        assert len(thread) == 3
        assert thread[0].id == msg1.id
        assert thread[-1].id == msg3.id

    def test_get_next_action_id(self):
        """Test action ID generation."""
        store = MASAuditStore()

        id1 = store.get_next_action_id()
        id2 = store.get_next_action_id()
        id3 = store.get_next_action_id()

        assert id1 == "action_1"
        assert id2 == "action_2"
        assert id3 == "action_3"

    def test_get_system_summary(self):
        """Test getting system summary."""
        store = MASAuditStore()
        store.add_agent("a", "validator")
        store.add_agent("b", "executor")
        store.create_channel("a", "b")
        store.record_inter_agent_message("a", "b", "test", "request")

        summary = store.get_system_summary()

        assert summary["num_agents"] == 2
        assert summary["num_channels"] == 1
        assert summary["num_inter_agent_messages"] == 1

    def test_reset(self):
        """Test resetting the store."""
        store = MASAuditStore()
        store.add_agent("a", "validator")
        store.create_channel("a", "a")  # Self-loop for testing
        store.record_inter_agent_message("a", "a", "test", "request")

        store.reset()

        assert len(store.agents) == 0
        assert len(store.channels) == 0
        assert len(store.inter_agent_messages) == 0
        assert store.action_counter == 0


class TestMASAuditStoreIntegration:
    """Integration tests for MASAuditStore."""

    def test_complete_workflow(self):
        """Test a complete MAS setup workflow."""
        store = MASAuditStore()

        # Setup agents
        store.add_agent("researcher", "researcher", "claude-3-sonnet")
        store.add_agent("validator", "validator", "claude-3-sonnet")
        store.add_agent("executor", "executor", "claude-3-haiku")

        # Setup channels (chain topology)
        store.create_channel("researcher", "validator")
        store.create_channel("validator", "executor")

        # Record messages
        msg1 = store.record_inter_agent_message(
            "researcher", "validator",
            "I found these results: ...",
            "request"
        )

        msg2 = store.record_inter_agent_message(
            "validator", "executor",
            "Results validated, please execute",
            "request",
            parent_message_id=msg1.id
        )

        # Verify topology
        assert store.can_communicate("researcher", "validator")
        assert store.can_communicate("validator", "executor")
        assert not store.can_communicate("researcher", "executor")

        # Verify messages
        assert len(store.inter_agent_messages) == 2

        # Verify thread
        thread = store.get_conversation_thread(msg2.id)
        assert len(thread) == 2
