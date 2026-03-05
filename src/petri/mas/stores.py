"""
Multi-Agent System (MAS) audit store for Petri.

This module provides state management for auditing multi-agent systems,
extending Petri's single-agent capabilities to handle N concurrent agents
with inter-agent communication tracking.


"""

from __future__ import annotations

import time
import uuid
from typing import Literal, Any

from inspect_ai.model import ChatMessage
from inspect_ai.tool import Tool, ToolDef, ToolSource
from pydantic import BaseModel, ConfigDict, Field

from petri.transcript import Event, TranscriptMetadata
from petri.types import ToolDefinition


# Note: We use BaseModel instead of StoreModel for flexibility.
# When used with Inspect AI, wrap with store_as() as needed.
# StoreModel is a singleton pattern that doesn't work well with unit tests.


class AgentNode(BaseModel):
    """
    State for an individual target agent within the MAS.

    Each agent has its own conversation history, tools, and role in the system.
    This enables tracking of per-agent behavior and decision-making patterns.

    Attributes:
        agent_id: Unique identifier for this agent
        role: The agent's role in the system (e.g., "validator", "executor")
        messages: Conversation history for this agent
        tools: Synthetic tools available to this agent
        system_prompt: The agent's system prompt (if set)
        model_name: Name of the model backing this agent
        created_at: Unix timestamp of agent creation
        metadata: Additional agent-specific metadata

    Example:
        ```python
        agent = AgentNode(
            agent_id="validator_1",
            role="validator",
            model_name="claude-3-sonnet"
        )
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_id: str
    role: str
    messages: list[ChatMessage] = Field(default_factory=list)
    tools: list[ToolDefinition] = Field(default_factory=list)
    system_prompt: str | None = None
    model_name: str | None = None
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_last_message(self) -> ChatMessage | None:
        """Get the most recent message in this agent's conversation."""
        return self.messages[-1] if self.messages else None

    def get_message_count(self) -> int:
        """Get the total number of messages in this agent's conversation."""
        return len(self.messages)


class InterAgentMessage(BaseModel):
    """
    A message exchanged between two target agents.

    This captures all inter-agent communication for observability
    and error propagation analysis.

    Attributes:
        id: Unique message identifier
        from_agent_id: ID of the sending agent
        to_agent_id: ID of the receiving agent
        content: The message content
        message_type: Type of message (request, response, broadcast, error)
        timestamp: Unix timestamp when message was sent
        causality_id: ID of the action that triggered this message
        parent_message_id: For responses, the ID of the message being replied to
        metadata: Additional message metadata (e.g., confidence scores)

    Example:
        ```python
        msg = InterAgentMessage(
            id="msg_001",
            from_agent_id="researcher",
            to_agent_id="validator",
            content="Here are my findings...",
            message_type="request"
        )
        ```
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    from_agent_id: str
    to_agent_id: str
    content: str
    message_type: Literal["request", "response", "broadcast", "error"]
    timestamp: float = Field(default_factory=time.time)
    causality_id: str | None = None
    parent_message_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_response_to(self, message_id: str) -> bool:
        """Check if this message is a response to a specific message."""
        return self.parent_message_id == message_id


class InterAgentChannel(BaseModel):
    """
    A communication channel between two agents.

    Channels define how agents can communicate with each other,
    enabling different topologies (chain, hub, mesh).

    Attributes:
        channel_id: Unique channel identifier
        from_agent_id: ID of the source agent
        to_agent_id: ID of the destination agent
        channel_type: Type of channel (direct, broadcast, queue)
        bidirectional: Whether communication flows both ways
        created_at: Unix timestamp of channel creation
        metadata: Additional channel configuration

    Example:
        ```python
        channel = InterAgentChannel(
            from_agent_id="researcher",
            to_agent_id="validator",
            channel_type="direct",
            bidirectional=True
        )
        ```
    """

    channel_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    from_agent_id: str
    to_agent_id: str
    channel_type: Literal["direct", "broadcast", "queue"] = "direct"
    bidirectional: bool = False
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def allows_message(self, from_id: str, to_id: str) -> bool:
        """Check if this channel allows a message between the given agents."""
        if self.from_agent_id == from_id and self.to_agent_id == to_id:
            return True
        if self.bidirectional and self.from_agent_id == to_id and self.to_agent_id == from_id:
            return True
        return False


class AuditorStore(BaseModel):
    """
    Store for the MAS auditor's conversation state and tools.

    This is similar to Petri's AuditorStore but adapted for MAS auditing.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[ChatMessage] = Field(default_factory=list)
    tools: list[Tool | ToolDef | ToolSource | None] | ToolSource | None = Field(default_factory=list)


class JudgeStore(BaseModel):
    """Store for the MAS judge's evaluation messages."""

    messages: list[ChatMessage] = Field(default_factory=list)


class MASMetadata(BaseModel):
    """
    Metadata about a MAS audit session.

    Extends TranscriptMetadata with MAS-specific information.
    """

    transcript_id: str | None = None
    auditor_model: str | None = None
    target_models: dict[str, str] = Field(default_factory=dict)  # agent_id -> model_name
    topology: str = "chain"
    num_agents: int = 0
    agent_roles: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    seed_instruction: str | None = None
    tags: list[str] = Field(default_factory=list)


class MASAuditStore(BaseModel):
    """
    Central store for managing all state during a multi-agent system audit.

    This store coordinates the conversation state between the auditor and
    multiple target agents, tracks inter-agent communications, and maintains
    metadata for transcript generation and analysis.

    The store is organized into:
    - Auditor store: The MAS auditor's conversation and tools
    - Agent nodes: Individual state for each target agent
    - Channels: Communication channels between agents
    - Inter-agent messages: All messages exchanged between agents
    - Judge store: Evaluation messages from the MAS judge

    Attributes:
        auditor: Store for the auditor's conversation state
        agents: Dictionary of agent_id -> AgentNode
        channels: List of communication channels
        inter_agent_messages: All inter-agent messages
        topology: System topology (chain, hub, mesh, custom)
        topology_config: Additional topology configuration
        seed_instructions: Initial audit instructions
        events: Event history for transcript generation
        metadata: MAS-specific metadata
        judge: Store for judge evaluation
        action_counter: Counter for generating unique action IDs

    Note:
        This class uses BaseModel instead of StoreModel for better test isolation.
        When using with Inspect AI tasks, use store_as() to register it properly.

    Example:
        ```python
        # For standalone use (tests, scripts):
        mas_store = MASAuditStore()

        # For use within Inspect AI tasks:
        # from inspect_ai.util import store_as
        # mas_store = store_as(MASAuditStore, instance="my_mas_audit")

        # Add a new agent
        agent = mas_store.add_agent("validator", "validator", "claude-3-sonnet")

        # Create a communication channel
        mas_store.create_channel("researcher", "validator")

        # Record an inter-agent message
        mas_store.record_inter_agent_message(
            from_agent="researcher",
            to_agent="validator",
            content="Please verify these findings...",
            message_type="request"
        )
        ```
    """

    # Auditor state
    auditor: AuditorStore = Field(default_factory=AuditorStore)
    seed_instructions: str | None = None

    # Target model name — stored at setup so tools can use it directly
    # instead of relying on get_model(role="target") in async tool context
    target_model_name: str | None = None

    # Multi-agent state
    agents: dict[str, AgentNode] = Field(default_factory=dict)
    channels: list[InterAgentChannel] = Field(default_factory=list)
    inter_agent_messages: list[InterAgentMessage] = Field(default_factory=list)

    # Topology configuration
    topology: Literal["chain", "hub", "mesh", "custom"] = "chain"
    topology_config: dict[str, Any] = Field(default_factory=dict)

    # Event tracking
    events: list[Event] = Field(default_factory=list)
    metadata: MASMetadata = Field(default_factory=MASMetadata)

    # Judge state
    judge: JudgeStore = Field(default_factory=JudgeStore)

    # Counters for unique IDs
    action_counter: int = 0
    message_counter: int = 0

    # -------------------------------------------------------------------------
    # Agent Management
    # -------------------------------------------------------------------------

    def add_agent(
        self,
        agent_id: str,
        role: str,
        model_name: str | None = None,
        system_prompt: str | None = None,
        **metadata
    ) -> AgentNode:
        """
        Add a new agent to the multi-agent system.

        Args:
            agent_id: Unique identifier for the agent
            role: The agent's role (e.g., "validator", "executor")
            model_name: Name of the model to use for this agent
            system_prompt: Initial system prompt for the agent
            **metadata: Additional metadata to attach to the agent

        Returns:
            The created AgentNode

        Raises:
            ValueError: If an agent with this ID already exists
        """
        if agent_id in self.agents:
            raise ValueError(f"Agent '{agent_id}' already exists in the MAS")

        agent = AgentNode(
            agent_id=agent_id,
            role=role,
            model_name=model_name,
            system_prompt=system_prompt,
            metadata=metadata
        )
        self.agents[agent_id] = agent

        # Update metadata
        self.metadata.num_agents = len(self.agents)
        if role not in self.metadata.agent_roles:
            self.metadata.agent_roles.append(role)
        if model_name:
            self.metadata.target_models[agent_id] = model_name

        return agent

    def get_agent(self, agent_id: str) -> AgentNode:
        """
        Get an agent by its ID.

        Args:
            agent_id: The agent's unique identifier

        Returns:
            The AgentNode

        Raises:
            KeyError: If the agent doesn't exist
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent '{agent_id}' not found in MAS")
        return self.agents[agent_id]

    def remove_agent(self, agent_id: str) -> AgentNode:
        """
        Remove an agent from the system.

        Also removes any channels involving this agent.

        Args:
            agent_id: The agent's unique identifier

        Returns:
            The removed AgentNode

        Raises:
            KeyError: If the agent doesn't exist
        """
        agent = self.get_agent(agent_id)
        del self.agents[agent_id]

        # Remove associated channels
        self.channels = [
            ch for ch in self.channels
            if ch.from_agent_id != agent_id and ch.to_agent_id != agent_id
        ]

        # Update metadata
        self.metadata.num_agents = len(self.agents)
        if agent_id in self.metadata.target_models:
            del self.metadata.target_models[agent_id]

        return agent

    def list_agents(self) -> list[str]:
        """Get a list of all agent IDs in the system."""
        return list(self.agents.keys())

    def get_agents_by_role(self, role: str) -> list[AgentNode]:
        """Get all agents with a specific role."""
        return [agent for agent in self.agents.values() if agent.role == role]

    # -------------------------------------------------------------------------
    # Channel Management
    # -------------------------------------------------------------------------

    def create_channel(
        self,
        from_agent_id: str,
        to_agent_id: str,
        channel_type: Literal["direct", "broadcast", "queue"] = "direct",
        bidirectional: bool = False,
        **metadata
    ) -> InterAgentChannel:
        """
        Create a communication channel between two agents.

        Args:
            from_agent_id: Source agent ID
            to_agent_id: Destination agent ID
            channel_type: Type of channel
            bidirectional: Whether the channel allows two-way communication
            **metadata: Additional channel configuration

        Returns:
            The created InterAgentChannel

        Raises:
            KeyError: If either agent doesn't exist
        """
        # Validate agents exist
        self.get_agent(from_agent_id)
        self.get_agent(to_agent_id)

        channel = InterAgentChannel(
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            channel_type=channel_type,
            bidirectional=bidirectional,
            metadata=metadata
        )
        self.channels.append(channel)
        return channel

    def get_channels_for_agent(
        self,
        agent_id: str,
        direction: Literal["outgoing", "incoming", "both"] = "both"
    ) -> list[InterAgentChannel]:
        """
        Get all channels connected to an agent.

        Args:
            agent_id: The agent's ID
            direction: Filter by direction

        Returns:
            List of channels
        """
        result = []
        for channel in self.channels:
            is_outgoing = channel.from_agent_id == agent_id
            is_incoming = channel.to_agent_id == agent_id
            is_bidirectional_incoming = (
                channel.bidirectional and channel.from_agent_id != agent_id
            )

            if direction == "outgoing" and (is_outgoing or is_bidirectional_incoming):
                result.append(channel)
            elif direction == "incoming" and (is_incoming or is_bidirectional_incoming):
                result.append(channel)
            elif direction == "both" and (is_outgoing or is_incoming):
                result.append(channel)

        return result

    def can_communicate(self, from_agent_id: str, to_agent_id: str) -> bool:
        """
        Check if one agent can send a message to another.

        Args:
            from_agent_id: Source agent ID
            to_agent_id: Destination agent ID

        Returns:
            True if communication is allowed
        """
        return any(
            ch.allows_message(from_agent_id, to_agent_id)
            for ch in self.channels
        )

    # -------------------------------------------------------------------------
    # Inter-Agent Message Management
    # -------------------------------------------------------------------------

    def record_inter_agent_message(
        self,
        from_agent: str,
        to_agent: str,
        content: str,
        message_type: Literal["request", "response", "broadcast", "error"],
        causality_id: str | None = None,
        parent_message_id: str | None = None,
        **metadata
    ) -> InterAgentMessage:
        """
        Record a message between two agents.

        Args:
            from_agent: Source agent ID
            to_agent: Destination agent ID
            content: Message content
            message_type: Type of message
            causality_id: ID of the action that caused this message
            parent_message_id: For responses, the message being replied to
            **metadata: Additional message metadata

        Returns:
            The recorded InterAgentMessage
        """
        self.message_counter += 1

        message = InterAgentMessage(
            id=f"iam_{self.message_counter}",
            from_agent_id=from_agent,
            to_agent_id=to_agent,
            content=content,
            message_type=message_type,
            causality_id=causality_id,
            parent_message_id=parent_message_id,
            metadata=metadata
        )
        self.inter_agent_messages.append(message)
        return message

    def get_messages_for_agent(
        self,
        agent_id: str,
        direction: Literal["sent", "received", "both"] = "both"
    ) -> list[InterAgentMessage]:
        """
        Get all inter-agent messages involving an agent.

        Args:
            agent_id: The agent's ID
            direction: Filter by direction

        Returns:
            List of messages
        """
        result = []
        for msg in self.inter_agent_messages:
            if direction == "sent" and msg.from_agent_id == agent_id:
                result.append(msg)
            elif direction == "received" and msg.to_agent_id == agent_id:
                result.append(msg)
            elif direction == "both" and (
                msg.from_agent_id == agent_id or msg.to_agent_id == agent_id
            ):
                result.append(msg)
        return result

    def get_conversation_thread(
        self,
        message_id: str
    ) -> list[InterAgentMessage]:
        """
        Get the full conversation thread containing a message.

        Follows parent_message_id links to reconstruct the thread.

        Args:
            message_id: ID of any message in the thread

        Returns:
            List of messages in chronological order
        """
        # Find the message
        target_msg = None
        for msg in self.inter_agent_messages:
            if msg.id == message_id:
                target_msg = msg
                break

        if not target_msg:
            return []

        # Find thread root
        root = target_msg
        while root.parent_message_id:
            for msg in self.inter_agent_messages:
                if msg.id == root.parent_message_id:
                    root = msg
                    break
            else:
                break

        # Collect all messages in thread
        thread = [root]
        queue = [root.id]

        while queue:
            current_id = queue.pop(0)
            for msg in self.inter_agent_messages:
                if msg.parent_message_id == current_id and msg not in thread:
                    thread.append(msg)
                    queue.append(msg.id)

        # Sort by timestamp
        thread.sort(key=lambda m: m.timestamp)
        return thread

    # -------------------------------------------------------------------------
    # Action ID Generation
    # -------------------------------------------------------------------------

    def get_next_action_id(self) -> str:
        """Generate a unique action ID for the observability graph."""
        self.action_counter += 1
        return f"action_{self.action_counter}"

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_combined_transcript(self) -> list[ChatMessage]:
        """
        Get a combined view of all agent conversations.

        Useful for the judge to see the full system behavior.

        Returns:
            List of messages from all agents, sorted by time where possible
        """
        all_messages = []
        for agent in self.agents.values():
            for msg in agent.messages:
                # Add agent ID to metadata
                msg_copy = msg.model_copy()
                if msg_copy.metadata is None:
                    msg_copy.metadata = {}
                msg_copy.metadata["agent_id"] = agent.agent_id
                msg_copy.metadata["agent_role"] = agent.role
                all_messages.append(msg_copy)
        return all_messages

    def get_system_summary(self) -> dict[str, Any]:
        """
        Get a summary of the current MAS state.

        Returns:
            Dictionary with system statistics
        """
        return {
            "num_agents": len(self.agents),
            "agents": [
                {
                    "id": a.agent_id,
                    "role": a.role,
                    "message_count": len(a.messages),
                    "tool_count": len(a.tools),
                }
                for a in self.agents.values()
            ],
            "num_channels": len(self.channels),
            "num_inter_agent_messages": len(self.inter_agent_messages),
            "topology": self.topology,
            "action_count": self.action_counter,
        }

    def reset(self) -> None:
        """Reset the store to its initial state."""
        self.agents.clear()
        self.channels.clear()
        self.inter_agent_messages.clear()
        self.events.clear()
        self.action_counter = 0
        self.message_counter = 0
        self.auditor = AuditorStore()
        self.judge = JudgeStore()
        self.metadata = MASMetadata()
        self.seed_instructions = None
