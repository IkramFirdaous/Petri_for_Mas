"""
Topology definitions for Multi-Agent Systems.

This module provides predefined topology patterns and utilities
for configuring how agents communicate in a MAS.

Supported topologies:
- Chain: Linear sequence (A -> B -> C)
- Hub: Central coordinator (A <-> Hub <-> B, C)
- Mesh: Fully connected (all agents can communicate)
- Pipeline: Specialized chain with validation stages
- Hierarchy: Tree structure with supervisors

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Callable, Any

from petri.mas.stores import MASAuditStore, InterAgentChannel


TopologyType = Literal["chain", "hub", "mesh", "pipeline", "hierarchy", "custom"]


@dataclass
class AgentRole:
    """
    Definition of an agent role in a topology.

    Roles define what an agent does and how it interacts
    with other agents in the system.

    Attributes:
        name: Role identifier (e.g., "researcher", "validator")
        description: Human-readable description
        capabilities: List of capabilities this role has
        can_initiate: Whether this role can initiate communication
        can_receive: Whether this role can receive communication
        trust_level: Default trust level for this role (0-1)
    """

    name: str
    description: str = ""
    capabilities: list[str] = field(default_factory=list)
    can_initiate: bool = True
    can_receive: bool = True
    trust_level: float = 0.5


# Predefined roles for common MAS scenarios
PREDEFINED_ROLES = {
    "researcher": AgentRole(
        name="researcher",
        description="Gathers and synthesizes information from various sources",
        capabilities=["search", "summarize", "analyze"],
        can_initiate=True,
        can_receive=True,
        trust_level=0.6,
    ),
    "validator": AgentRole(
        name="validator",
        description="Validates and fact-checks information from other agents",
        capabilities=["verify", "cross_reference", "flag_errors"],
        can_initiate=True,
        can_receive=True,
        trust_level=0.8,
    ),
    "executor": AgentRole(
        name="executor",
        description="Executes actions based on decisions from other agents",
        capabilities=["execute", "implement", "deploy"],
        can_initiate=False,
        can_receive=True,
        trust_level=0.5,
    ),
    "coordinator": AgentRole(
        name="coordinator",
        description="Coordinates activities between other agents",
        capabilities=["delegate", "prioritize", "monitor"],
        can_initiate=True,
        can_receive=True,
        trust_level=0.7,
    ),
    "oversight": AgentRole(
        name="oversight",
        description="Provides oversight and final approval for actions",
        capabilities=["approve", "reject", "escalate"],
        can_initiate=True,
        can_receive=True,
        trust_level=0.9,
    ),
    "analyst": AgentRole(
        name="analyst",
        description="Analyzes data and provides insights",
        capabilities=["analyze", "model", "predict"],
        can_initiate=True,
        can_receive=True,
        trust_level=0.6,
    ),
    "reporter": AgentRole(
        name="reporter",
        description="Compiles and reports findings to stakeholders",
        capabilities=["summarize", "report", "visualize"],
        can_initiate=True,
        can_receive=True,
        trust_level=0.5,
    ),
}


@dataclass
class TopologyConfig:
    """
    Configuration for a MAS topology.

    Attributes:
        topology_type: Type of topology
        num_agents: Number of agents in the system
        roles: List of roles for each agent
        hub_agent_id: For hub topology, the central agent ID
        hierarchy_levels: For hierarchy topology, levels configuration
        custom_channels: Custom channel definitions
        bidirectional: Whether channels are bidirectional by default
    """

    topology_type: TopologyType
    num_agents: int
    roles: list[str] = field(default_factory=list)
    hub_agent_id: str | None = None
    hierarchy_levels: list[list[str]] = field(default_factory=list)
    custom_channels: list[tuple[str, str]] = field(default_factory=list)
    bidirectional: bool = False


class TopologyBuilder(ABC):
    """
    Abstract base class for topology builders.

    Topology builders create the agent and channel structure
    for a specific topology pattern.
    """

    @abstractmethod
    def build(self, store: MASAuditStore, config: TopologyConfig) -> None:
        """
        Build the topology in the given store.

        Args:
            store: The MASAuditStore to populate
            config: Topology configuration
        """
        pass

    @abstractmethod
    def validate(self, config: TopologyConfig) -> list[str]:
        """
        Validate the topology configuration.

        Args:
            config: Configuration to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        pass


class ChainTopologyBuilder(TopologyBuilder):
    """
    Builds a chain topology: A -> B -> C -> ...

    In a chain topology, each agent can only communicate with
    its immediate neighbors (previous and next in the chain).

    This is useful for:
    - Sequential processing pipelines
    - Validation chains
    - Approval workflows
    """

    def build(self, store: MASAuditStore, config: TopologyConfig) -> None:
        """Build a chain topology."""
        roles = config.roles or [f"agent_{i}" for i in range(config.num_agents)]

        if len(roles) < config.num_agents:
            roles.extend([f"agent_{i}" for i in range(len(roles), config.num_agents)])

        # Create agents
        agent_ids = []
        for i in range(config.num_agents):
            agent_id = f"agent_{i}"
            role = roles[i] if i < len(roles) else f"agent_{i}"
            store.add_agent(agent_id, role)
            agent_ids.append(agent_id)

        # Create chain channels
        for i in range(len(agent_ids) - 1):
            store.create_channel(
                agent_ids[i],
                agent_ids[i + 1],
                channel_type="direct",
                bidirectional=config.bidirectional
            )

        store.topology = "chain"
        store.topology_config = {
            "chain_order": agent_ids,
            "bidirectional": config.bidirectional,
        }

    def validate(self, config: TopologyConfig) -> list[str]:
        """Validate chain topology configuration."""
        errors = []
        if config.num_agents < 2:
            errors.append("Chain topology requires at least 2 agents")
        return errors


class HubTopologyBuilder(TopologyBuilder):
    """
    Builds a hub topology: All agents connect to a central hub.

    In a hub topology, all communication goes through a central
    coordinator agent.

    This is useful for:
    - Centralized coordination
    - Star patterns
    - Supervisor-worker architectures
    """

    def build(self, store: MASAuditStore, config: TopologyConfig) -> None:
        """Build a hub topology."""
        roles = config.roles or ["coordinator"] + [f"worker_{i}" for i in range(config.num_agents - 1)]

        # Create hub agent first
        hub_id = config.hub_agent_id or "hub"
        hub_role = roles[0] if roles else "coordinator"
        store.add_agent(hub_id, hub_role)

        # Create spoke agents
        spoke_ids = []
        for i in range(1, config.num_agents):
            agent_id = f"agent_{i}"
            role = roles[i] if i < len(roles) else f"worker_{i}"
            store.add_agent(agent_id, role)
            spoke_ids.append(agent_id)

            # Connect to hub (bidirectional)
            store.create_channel(
                hub_id,
                agent_id,
                channel_type="direct",
                bidirectional=True
            )

        store.topology = "hub"
        store.topology_config = {
            "hub_id": hub_id,
            "spoke_ids": spoke_ids,
        }

    def validate(self, config: TopologyConfig) -> list[str]:
        """Validate hub topology configuration."""
        errors = []
        if config.num_agents < 2:
            errors.append("Hub topology requires at least 2 agents")
        return errors


class MeshTopologyBuilder(TopologyBuilder):
    """
    Builds a mesh topology: All agents can communicate with all others.

    In a mesh topology, every agent has a direct channel to every
    other agent.

    This is useful for:
    - Fully collaborative systems
    - Consensus protocols
    - Peer-to-peer architectures
    """

    def build(self, store: MASAuditStore, config: TopologyConfig) -> None:
        """Build a mesh topology."""
        roles = config.roles or [f"peer_{i}" for i in range(config.num_agents)]

        # Create all agents
        agent_ids = []
        for i in range(config.num_agents):
            agent_id = f"agent_{i}"
            role = roles[i] if i < len(roles) else f"peer_{i}"
            store.add_agent(agent_id, role)
            agent_ids.append(agent_id)

        # Create all-to-all channels
        for i, from_id in enumerate(agent_ids):
            for j, to_id in enumerate(agent_ids):
                if i < j:  # Avoid duplicates
                    store.create_channel(
                        from_id,
                        to_id,
                        channel_type="direct",
                        bidirectional=True
                    )

        store.topology = "mesh"
        store.topology_config = {
            "agent_ids": agent_ids,
            "num_channels": len(agent_ids) * (len(agent_ids) - 1) // 2,
        }

    def validate(self, config: TopologyConfig) -> list[str]:
        """Validate mesh topology configuration."""
        errors = []
        if config.num_agents < 2:
            errors.append("Mesh topology requires at least 2 agents")
        if config.num_agents > 10:
            errors.append(
                f"Warning: Mesh topology with {config.num_agents} agents "
                f"creates {config.num_agents * (config.num_agents - 1) // 2} channels"
            )
        return errors


class PipelineTopologyBuilder(TopologyBuilder):
    """
    Builds a pipeline topology: A specialized chain with validation stages.

    The pipeline has distinct stages:
    1. Input stage (receives external data)
    2. Processing stages (transform/analyze)
    3. Validation stages (check quality)
    4. Output stage (final output)

    This is useful for:
    - Data processing pipelines
    - Multi-stage validation
    - ETL workflows
    """

    def build(self, store: MASAuditStore, config: TopologyConfig) -> None:
        """Build a pipeline topology."""
        # Default pipeline stages
        default_roles = ["input", "processor", "validator", "output"]
        roles = config.roles or default_roles

        if len(roles) < config.num_agents:
            # Repeat processor/validator pattern
            while len(roles) < config.num_agents:
                roles.insert(-1, "processor" if len(roles) % 2 == 0 else "validator")

        # Create agents
        agent_ids = []
        for i in range(config.num_agents):
            agent_id = f"stage_{i}"
            role = roles[i] if i < len(roles) else f"stage_{i}"
            store.add_agent(agent_id, role)
            agent_ids.append(agent_id)

        # Create pipeline channels (forward only by default)
        for i in range(len(agent_ids) - 1):
            store.create_channel(
                agent_ids[i],
                agent_ids[i + 1],
                channel_type="queue",  # Queue for ordered processing
                bidirectional=False
            )

        # Add feedback channels from validators to processors
        for i, agent_id in enumerate(agent_ids):
            agent = store.get_agent(agent_id)
            if agent.role == "validator" and i > 0:
                # Validator can send feedback to previous processor
                store.create_channel(
                    agent_id,
                    agent_ids[i - 1],
                    channel_type="direct",
                    bidirectional=False
                )

        store.topology = "pipeline"
        store.topology_config = {
            "stages": agent_ids,
            "stage_roles": {aid: store.get_agent(aid).role for aid in agent_ids},
        }

    def validate(self, config: TopologyConfig) -> list[str]:
        """Validate pipeline topology configuration."""
        errors = []
        if config.num_agents < 3:
            errors.append("Pipeline topology requires at least 3 agents")
        return errors


class HierarchyTopologyBuilder(TopologyBuilder):
    """
    Builds a hierarchy topology: Tree structure with supervisors.

    Agents are organized in levels, with each level supervised
    by the level above.

    This is useful for:
    - Organizational structures
    - Command chains
    - Approval hierarchies
    """

    def build(self, store: MASAuditStore, config: TopologyConfig) -> None:
        """Build a hierarchy topology."""
        # Determine hierarchy structure
        if config.hierarchy_levels:
            levels = config.hierarchy_levels
        else:
            # Default: binary tree-ish structure
            levels = self._compute_default_levels(config.num_agents)

        # Create agents level by level
        all_agent_ids = []
        level_agent_ids: list[list[str]] = []

        agent_counter = 0
        for level_idx, level_roles in enumerate(levels):
            level_ids = []
            for role in level_roles:
                agent_id = f"agent_{agent_counter}"
                store.add_agent(agent_id, role)
                level_ids.append(agent_id)
                all_agent_ids.append(agent_id)
                agent_counter += 1
            level_agent_ids.append(level_ids)

        # Create hierarchical channels
        for level_idx in range(len(level_agent_ids) - 1):
            upper_level = level_agent_ids[level_idx]
            lower_level = level_agent_ids[level_idx + 1]

            # Connect each upper agent to lower agents
            # Distribute lower agents among upper agents
            agents_per_supervisor = max(1, len(lower_level) // len(upper_level))

            for i, supervisor_id in enumerate(upper_level):
                start_idx = i * agents_per_supervisor
                end_idx = start_idx + agents_per_supervisor
                if i == len(upper_level) - 1:
                    end_idx = len(lower_level)  # Last supervisor gets remaining

                for subordinate_id in lower_level[start_idx:end_idx]:
                    store.create_channel(
                        supervisor_id,
                        subordinate_id,
                        channel_type="direct",
                        bidirectional=True
                    )

        store.topology = "hierarchy"
        store.topology_config = {
            "levels": level_agent_ids,
            "root": level_agent_ids[0][0] if level_agent_ids and level_agent_ids[0] else None,
        }

    def _compute_default_levels(self, num_agents: int) -> list[list[str]]:
        """Compute default hierarchy levels for a given number of agents."""
        if num_agents <= 2:
            return [["supervisor"], ["worker"] * (num_agents - 1)]

        # Roughly binary tree
        levels = []
        remaining = num_agents
        level_size = 1

        while remaining > 0:
            actual_size = min(level_size, remaining)
            if len(levels) == 0:
                roles = ["executive"] * actual_size
            elif remaining - actual_size <= 0:
                roles = ["worker"] * actual_size
            else:
                roles = ["supervisor"] * actual_size
            levels.append(roles)
            remaining -= actual_size
            level_size *= 2

        return levels

    def validate(self, config: TopologyConfig) -> list[str]:
        """Validate hierarchy topology configuration."""
        errors = []
        if config.num_agents < 2:
            errors.append("Hierarchy topology requires at least 2 agents")
        return errors


class CustomTopologyBuilder(TopologyBuilder):
    """
    Builds a custom topology from explicit channel definitions.

    This allows full control over the topology structure.
    """

    def build(self, store: MASAuditStore, config: TopologyConfig) -> None:
        """Build a custom topology."""
        roles = config.roles or [f"agent_{i}" for i in range(config.num_agents)]

        # Create agents
        agent_ids = []
        for i in range(config.num_agents):
            agent_id = f"agent_{i}"
            role = roles[i] if i < len(roles) else f"agent_{i}"
            store.add_agent(agent_id, role)
            agent_ids.append(agent_id)

        # Create custom channels
        for from_id, to_id in config.custom_channels:
            if from_id in agent_ids and to_id in agent_ids:
                store.create_channel(
                    from_id,
                    to_id,
                    channel_type="direct",
                    bidirectional=config.bidirectional
                )

        store.topology = "custom"
        store.topology_config = {
            "agent_ids": agent_ids,
            "channels": config.custom_channels,
        }

    def validate(self, config: TopologyConfig) -> list[str]:
        """Validate custom topology configuration."""
        errors = []
        if config.num_agents < 1:
            errors.append("Custom topology requires at least 1 agent")

        # Check channel references
        agent_ids = {f"agent_{i}" for i in range(config.num_agents)}
        for from_id, to_id in config.custom_channels:
            if from_id not in agent_ids:
                errors.append(f"Channel references unknown agent: {from_id}")
            if to_id not in agent_ids:
                errors.append(f"Channel references unknown agent: {to_id}")

        return errors


# Registry of topology builders
TOPOLOGY_BUILDERS: dict[TopologyType, type[TopologyBuilder]] = {
    "chain": ChainTopologyBuilder,
    "hub": HubTopologyBuilder,
    "mesh": MeshTopologyBuilder,
    "pipeline": PipelineTopologyBuilder,
    "hierarchy": HierarchyTopologyBuilder,
    "custom": CustomTopologyBuilder,
}


def build_topology(
    store: MASAuditStore,
    topology_type: TopologyType,
    num_agents: int,
    roles: list[str] | None = None,
    **kwargs
) -> None:
    """
    Build a topology in the given MAS store.

    This is the main entry point for creating topologies.

    Args:
        store: The MASAuditStore to populate
        topology_type: Type of topology to build
        num_agents: Number of agents
        roles: Optional list of roles for agents
        **kwargs: Additional configuration for the topology

    Raises:
        ValueError: If the topology type is unknown or configuration is invalid

    Example:
        ```python
        store = MASAuditStore()
        build_topology(
            store,
            topology_type="chain",
            num_agents=3,
            roles=["researcher", "validator", "executor"]
        )
        ```
    """
    if topology_type not in TOPOLOGY_BUILDERS:
        raise ValueError(f"Unknown topology type: {topology_type}")

    config = TopologyConfig(
        topology_type=topology_type,
        num_agents=num_agents,
        roles=roles or [],
        **kwargs
    )

    builder = TOPOLOGY_BUILDERS[topology_type]()

    # Validate configuration
    errors = builder.validate(config)
    if errors:
        raise ValueError(f"Invalid topology configuration: {'; '.join(errors)}")

    # Build the topology
    builder.build(store, config)


def get_topology_description(topology_type: TopologyType) -> str:
    """
    Get a human-readable description of a topology type.

    Args:
        topology_type: The topology type

    Returns:
        Description string
    """
    descriptions = {
        "chain": "Linear sequence where each agent communicates with neighbors (A -> B -> C)",
        "hub": "Star pattern with central coordinator connecting all agents",
        "mesh": "Fully connected network where all agents can communicate directly",
        "pipeline": "Sequential processing with validation stages and feedback loops",
        "hierarchy": "Tree structure with supervisors managing subordinate agents",
        "custom": "User-defined topology with explicit channel definitions",
    }
    return descriptions.get(topology_type, "Unknown topology type")
