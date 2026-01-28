"""
Observability Graph for Multi-Agent System auditing.

This module provides a directed graph structure for tracking causal dependencies
between actions in a multi-agent system, enabling error propagation analysis
and cascade failure detection.

References:
    - Wicaksono et al. (2025) - "Mind the Gap: Action-Graph Observability"
    - Cemri et al. (2025) - "Why Do Multi-Agent LLM Systems Fail?"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Literal

import networkx as nx


class ActionType(Enum):
    """
    Types of actions that can occur in a multi-agent system.

    These categorize the different kinds of events we track
    for observability and error propagation analysis.
    """

    # Agent actions
    AGENT_MESSAGE = "agent_message"  # Message from an agent
    AGENT_DECISION = "agent_decision"  # Decision made by an agent
    AGENT_REASONING = "agent_reasoning"  # Reasoning/thinking by an agent

    # Tool-related actions
    TOOL_CALL = "tool_call"  # Agent calling a tool
    TOOL_RESULT = "tool_result"  # Result returned by a tool

    # Inter-agent communication
    INTER_AGENT_REQUEST = "inter_agent_request"  # Request from one agent to another
    INTER_AGENT_RESPONSE = "inter_agent_response"  # Response between agents
    INTER_AGENT_BROADCAST = "inter_agent_broadcast"  # Broadcast to multiple agents

    # System events
    SYSTEM_INJECT = "system_inject"  # Auditor injecting information
    SYSTEM_ERROR = "system_error"  # System-level error

    # Error markers
    ERROR_INTRODUCED = "error_introduced"  # Initial error injection point
    ERROR_PROPAGATED = "error_propagated"  # Error that propagated from another action


class DependencyType(Enum):
    """
    Types of dependencies between actions.

    These classify the relationship between two actions,
    which is critical for understanding error propagation patterns.
    """

    # Causal dependencies
    CAUSAL = "causal"  # A directly caused B
    CAUSAL_WEAK = "causal_weak"  # A contributed to B but wasn't the only cause

    # Temporal dependencies
    TEMPORAL = "temporal"  # A happened before B (no direct causation)
    SEQUENTIAL = "sequential"  # A must happen before B in protocol

    # Information flow
    INFORMATIONAL = "informational"  # B uses information from A
    DERIVED = "derived"  # B is derived/computed from A

    # Trust relationships
    TRUST = "trust"  # B trusts A's output without verification
    VERIFIED_TRUST = "verified_trust"  # B trusts A after verification

    # Error propagation
    ERROR_PROPAGATION = "error_propagation"  # Error from A propagated to B
    ERROR_AMPLIFICATION = "error_amplification"  # Error from A was amplified in B


@dataclass
class ActionNode:
    """
    A node representing an action in the observability graph.

    Actions are the atomic units of behavior we track in the MAS.
    Each action is associated with an agent and has a type, content,
    and various metadata for analysis.

    Attributes:
        action_id: Unique identifier for this action
        agent_id: ID of the agent that performed this action
        action_type: Category of the action
        content: The action's content (message, tool call, etc.)
        timestamp: Unix timestamp when the action occurred
        tool_name: For tool calls, the name of the tool
        tool_args: For tool calls, the arguments passed
        is_error: Whether this action is/contains an error
        error_type: Classification of the error (if any)
        error_severity: Severity of the error (1-10 scale)
        confidence: Agent's confidence in this action (0-1)
        verified: Whether this action was verified by another agent
        metadata: Additional action-specific data

    Example:
        ```python
        action = ActionNode(
            action_id="action_42",
            agent_id="validator",
            action_type=ActionType.AGENT_DECISION,
            content="Approved the transaction",
            is_error=False,
            confidence=0.95
        )
        ```
    """

    action_id: str
    agent_id: str
    action_type: ActionType
    content: str
    timestamp: float = field(default_factory=time.time)

    # Tool-specific fields
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None

    # Error tracking
    is_error: bool = False
    error_type: str | None = None
    error_severity: int | None = None  # 1-10 scale
    original_error_id: str | None = None  # For propagated errors, the source

    # Quality metrics
    confidence: float | None = None  # 0-1 scale
    verified: bool = False
    verified_by: str | None = None

    # General metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def mark_as_error(
        self,
        error_type: str = "unknown",
        severity: int = 5,
        original_id: str | None = None
    ) -> None:
        """Mark this action as containing an error."""
        self.is_error = True
        self.error_type = error_type
        self.error_severity = severity
        self.original_error_id = original_id

    def mark_as_verified(self, verifier_id: str) -> None:
        """Mark this action as verified by another agent."""
        self.verified = True
        self.verified_by = verifier_id


@dataclass
class DependencyEdge:
    """
    An edge representing a dependency between two actions.

    Dependencies capture the relationships between actions,
    enabling analysis of how information (and errors) flow
    through the multi-agent system.

    Attributes:
        source_id: ID of the source action
        target_id: ID of the target action
        dependency_type: Type of dependency
        weight: Strength of the dependency (0-1)
        description: Human-readable description
        metadata: Additional edge-specific data

    Example:
        ```python
        edge = DependencyEdge(
            source_id="action_1",
            target_id="action_2",
            dependency_type=DependencyType.TRUST,
            weight=0.9,
            description="Validator trusted researcher's data"
        )
        ```
    """

    source_id: str
    target_id: str
    dependency_type: DependencyType
    weight: float = 1.0
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PropagationChain:
    """
    Represents a chain of error propagation through the system.

    A propagation chain tracks how an error travels from its source
    through multiple agents, potentially being amplified or modified.

    Attributes:
        source_action_id: The original error action
        chain: List of action IDs in propagation order
        affected_agents: Set of agents affected by this error
        amplification_factor: How much the error was amplified (1.0 = no change)
        was_detected: Whether the error was eventually detected
        detected_by: Agent ID that detected the error (if any)
    """

    def __init__(self, source_action_id: str):
        self.source_action_id = source_action_id
        self.chain: list[str] = [source_action_id]
        self.affected_agents: set[str] = set()
        self.amplification_factor: float = 1.0
        self.was_detected: bool = False
        self.detected_by: str | None = None
        self.detection_point: str | None = None

    def add_propagation(
        self,
        action_id: str,
        agent_id: str,
        amplification: float = 1.0
    ) -> None:
        """Add a propagation step to the chain."""
        self.chain.append(action_id)
        self.affected_agents.add(agent_id)
        self.amplification_factor *= amplification

    def mark_detected(self, detector_agent: str, detection_action: str) -> None:
        """Mark the error as detected."""
        self.was_detected = True
        self.detected_by = detector_agent
        self.detection_point = detection_action

    @property
    def length(self) -> int:
        """Get the length of the propagation chain."""
        return len(self.chain)

    @property
    def num_affected_agents(self) -> int:
        """Get the number of affected agents."""
        return len(self.affected_agents)


class ObservabilityGraph:
    """
    Directed graph for tracking causal dependencies in multi-agent systems.

    This graph enables:
    - Tracking all actions performed by agents
    - Recording dependencies between actions
    - Analyzing error propagation patterns
    - Detecting cascade failures
    - Identifying trust violations

    The graph uses NetworkX for efficient graph algorithms.

    Attributes:
        graph: The underlying NetworkX DiGraph
        action_nodes: Dictionary of action_id -> ActionNode
        error_sources: Set of action IDs that are original error sources

    Example:
        ```python
        obs = ObservabilityGraph()

        # Add actions
        a1 = obs.add_action("a1", "researcher", ActionType.AGENT_MESSAGE, "Found data")
        a2 = obs.add_action("a2", "validator", ActionType.AGENT_DECISION, "Approved")

        # Add dependency
        obs.add_dependency("a1", "a2", DependencyType.TRUST)

        # Mark an error and analyze propagation
        obs.mark_as_error("a1", "factual_error")
        chains = obs.find_error_propagation_chains("a1")
        ```
    """

    def __init__(self):
        """Initialize an empty observability graph."""
        self.graph = nx.DiGraph()
        self._action_nodes: dict[str, ActionNode] = {}
        self._error_sources: set[str] = set()

    # -------------------------------------------------------------------------
    # Node Management
    # -------------------------------------------------------------------------

    def add_action(
        self,
        action_id: str,
        agent_id: str,
        action_type: ActionType,
        content: str,
        timestamp: float | None = None,
        **kwargs
    ) -> ActionNode:
        """
        Add an action node to the graph.

        Args:
            action_id: Unique identifier for the action
            agent_id: ID of the agent performing the action
            action_type: Type of action
            content: Action content
            timestamp: When the action occurred (defaults to now)
            **kwargs: Additional ActionNode attributes

        Returns:
            The created ActionNode

        Raises:
            ValueError: If an action with this ID already exists
        """
        if action_id in self._action_nodes:
            raise ValueError(f"Action '{action_id}' already exists in graph")

        node = ActionNode(
            action_id=action_id,
            agent_id=agent_id,
            action_type=action_type,
            content=content,
            timestamp=timestamp or time.time(),
            **kwargs
        )

        self._action_nodes[action_id] = node
        self.graph.add_node(
            action_id,
            agent_id=agent_id,
            action_type=action_type.value,
            timestamp=node.timestamp,
            is_error=node.is_error,
        )

        return node

    def get_action(self, action_id: str) -> ActionNode | None:
        """Get an action node by ID."""
        return self._action_nodes.get(action_id)

    def get_actions_by_agent(self, agent_id: str) -> list[ActionNode]:
        """Get all actions performed by a specific agent."""
        return [
            node for node in self._action_nodes.values()
            if node.agent_id == agent_id
        ]

    def get_actions_by_type(self, action_type: ActionType) -> list[ActionNode]:
        """Get all actions of a specific type."""
        return [
            node for node in self._action_nodes.values()
            if node.action_type == action_type
        ]

    # -------------------------------------------------------------------------
    # Edge Management
    # -------------------------------------------------------------------------

    def add_dependency(
        self,
        source_id: str,
        target_id: str,
        dependency_type: DependencyType,
        weight: float = 1.0,
        description: str | None = None,
        **metadata
    ) -> DependencyEdge:
        """
        Add a dependency edge between two actions.

        Args:
            source_id: ID of the source action
            target_id: ID of the target action
            dependency_type: Type of dependency
            weight: Strength of the dependency (0-1)
            description: Human-readable description
            **metadata: Additional edge metadata

        Returns:
            The created DependencyEdge

        Raises:
            KeyError: If either action doesn't exist
        """
        if source_id not in self._action_nodes:
            raise KeyError(f"Source action '{source_id}' not found")
        if target_id not in self._action_nodes:
            raise KeyError(f"Target action '{target_id}' not found")

        edge = DependencyEdge(
            source_id=source_id,
            target_id=target_id,
            dependency_type=dependency_type,
            weight=weight,
            description=description,
            metadata=metadata
        )

        self.graph.add_edge(
            source_id,
            target_id,
            dependency_type=dependency_type.value,
            weight=weight,
            description=description,
        )

        return edge

    def get_dependencies_from(self, action_id: str) -> list[tuple[str, DependencyType]]:
        """Get all actions that depend on the given action."""
        if action_id not in self.graph:
            return []
        return [
            (target, DependencyType(data["dependency_type"]))
            for target, data in self.graph[action_id].items()
        ]

    def get_dependencies_to(self, action_id: str) -> list[tuple[str, DependencyType]]:
        """Get all actions that the given action depends on."""
        if action_id not in self.graph:
            return []
        return [
            (source, DependencyType(self.graph[source][action_id]["dependency_type"]))
            for source in self.graph.predecessors(action_id)
        ]

    # -------------------------------------------------------------------------
    # Error Tracking
    # -------------------------------------------------------------------------

    def mark_as_error(
        self,
        action_id: str,
        error_type: str = "unknown",
        severity: int = 5,
        is_original: bool = True
    ) -> None:
        """
        Mark an action as containing an error.

        Args:
            action_id: ID of the action to mark
            error_type: Classification of the error
            severity: Severity on a 1-10 scale
            is_original: Whether this is the original error source
        """
        if action_id not in self._action_nodes:
            raise KeyError(f"Action '{action_id}' not found")

        node = self._action_nodes[action_id]
        node.mark_as_error(error_type, severity)

        # Update graph attributes
        self.graph.nodes[action_id]["is_error"] = True
        self.graph.nodes[action_id]["error_type"] = error_type
        self.graph.nodes[action_id]["error_severity"] = severity

        if is_original:
            self._error_sources.add(action_id)

    def get_error_sources(self) -> list[ActionNode]:
        """Get all original error source actions."""
        return [self._action_nodes[aid] for aid in self._error_sources]

    def get_all_errors(self) -> list[ActionNode]:
        """Get all actions marked as errors."""
        return [
            node for node in self._action_nodes.values()
            if node.is_error
        ]

    # -------------------------------------------------------------------------
    # Error Propagation Analysis
    # -------------------------------------------------------------------------

    def find_error_propagation_chains(
        self,
        error_action_id: str,
        max_depth: int | None = None
    ) -> list[PropagationChain]:
        """
        Find all propagation chains starting from an error.

        This traces how an error spreads through the system,
        identifying all affected actions and agents.

        Args:
            error_action_id: ID of the original error action
            max_depth: Maximum chain depth to explore

        Returns:
            List of PropagationChain objects
        """
        if error_action_id not in self.graph:
            return []

        chains: list[PropagationChain] = []

        # Get all descendants of the error
        descendants = nx.descendants(self.graph, error_action_id)

        if not descendants:
            # No propagation
            chain = PropagationChain(error_action_id)
            source_node = self._action_nodes[error_action_id]
            chain.affected_agents.add(source_node.agent_id)
            chains.append(chain)
            return chains

        # Find all paths from error to leaf nodes
        leaf_nodes = [
            n for n in descendants
            if self.graph.out_degree(n) == 0
        ]

        for leaf in leaf_nodes:
            for path in nx.all_simple_paths(
                self.graph,
                error_action_id,
                leaf,
                cutoff=max_depth
            ):
                chain = PropagationChain(error_action_id)

                for action_id in path[1:]:  # Skip source
                    node = self._action_nodes.get(action_id)
                    if node:
                        # Calculate amplification based on error severity
                        amp = 1.0
                        if node.is_error and node.error_severity:
                            source = self._action_nodes.get(error_action_id)
                            if source and source.error_severity:
                                amp = node.error_severity / source.error_severity

                        chain.add_propagation(action_id, node.agent_id, amp)

                        # Check if error was detected (verified = error caught)
                        if node.verified and not node.is_error:
                            chain.mark_detected(node.verified_by or node.agent_id, action_id)
                            break

                chains.append(chain)

        return chains

    def calculate_propagation_metrics(self) -> dict[str, float | int]:
        """
        Calculate comprehensive error propagation metrics.

        Returns:
            Dictionary with metrics:
            - error_propagation_rate: % of downstream actions affected
            - avg_chain_length: Average propagation chain length
            - max_chain_length: Maximum chain length
            - affected_agents: Number of unique agents affected
            - detection_rate: % of errors that were eventually detected
            - avg_amplification: Average error amplification factor
        """
        if not self._error_sources:
            return {
                "error_propagation_rate": 0.0,
                "avg_chain_length": 0.0,
                "max_chain_length": 0,
                "affected_agents": 0,
                "detection_rate": 0.0,
                "avg_amplification": 1.0,
                "num_error_sources": 0,
                "total_actions": len(self._action_nodes),
            }

        all_chains: list[PropagationChain] = []
        all_affected_agents: set[str] = set()
        detected_count = 0
        total_amplification = 0.0

        for error_id in self._error_sources:
            chains = self.find_error_propagation_chains(error_id)
            all_chains.extend(chains)

            for chain in chains:
                all_affected_agents.update(chain.affected_agents)
                total_amplification += chain.amplification_factor
                if chain.was_detected:
                    detected_count += 1

        chain_lengths = [chain.length for chain in all_chains] if all_chains else [0]
        num_chains = len(all_chains) if all_chains else 1

        # Calculate propagation rate
        total_downstream = sum(
            len(list(nx.descendants(self.graph, eid)))
            for eid in self._error_sources
        )
        total_possible = len(self._action_nodes) - len(self._error_sources)
        propagation_rate = total_downstream / total_possible if total_possible > 0 else 0.0

        return {
            "error_propagation_rate": propagation_rate,
            "avg_chain_length": sum(chain_lengths) / len(chain_lengths),
            "max_chain_length": max(chain_lengths),
            "affected_agents": len(all_affected_agents),
            "detection_rate": detected_count / num_chains if num_chains > 0 else 0.0,
            "avg_amplification": total_amplification / num_chains if num_chains > 0 else 1.0,
            "num_error_sources": len(self._error_sources),
            "total_actions": len(self._action_nodes),
        }

    # -------------------------------------------------------------------------
    # Trust Analysis
    # -------------------------------------------------------------------------

    def find_trust_violations(self) -> list[dict[str, Any]]:
        """
        Identify cases of misplaced trust leading to error propagation.

        A trust violation occurs when:
        1. Agent B trusts Agent A's output (TRUST dependency)
        2. Agent A's output contains or derives from an error
        3. Agent B doesn't detect the error

        Returns:
            List of violation records with details
        """
        violations = []

        for u, v, data in self.graph.edges(data=True):
            dep_type = data.get("dependency_type", "")

            # Only check trust dependencies
            if dep_type != DependencyType.TRUST.value:
                continue

            source_node = self._action_nodes.get(u)
            target_node = self._action_nodes.get(v)

            if not source_node or not target_node:
                continue

            # Check if source is an error or derives from one
            is_problematic = source_node.is_error

            if not is_problematic:
                # Check ancestors for errors
                ancestors = nx.ancestors(self.graph, u)
                for ancestor_id in ancestors:
                    ancestor = self._action_nodes.get(ancestor_id)
                    if ancestor and ancestor.is_error:
                        is_problematic = True
                        break

            if is_problematic and not target_node.verified:
                violations.append({
                    "trusting_action_id": v,
                    "trusting_agent_id": target_node.agent_id,
                    "trusted_action_id": u,
                    "trusted_agent_id": source_node.agent_id,
                    "error_source": source_node.original_error_id or u,
                    "was_verified": target_node.verified,
                })

        return violations

    def find_unverified_trust_chains(self) -> list[list[str]]:
        """
        Find chains of trust without verification.

        Returns:
            List of action ID chains where trust propagates without verification
        """
        trust_edges = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if d.get("dependency_type") == DependencyType.TRUST.value
        ]

        if not trust_edges:
            return []

        # Build subgraph of trust relationships
        trust_graph = nx.DiGraph(trust_edges)

        chains = []
        for start in trust_graph.nodes():
            if trust_graph.in_degree(start) == 0:  # Root nodes
                for end in trust_graph.nodes():
                    if trust_graph.out_degree(end) == 0:  # Leaf nodes
                        for path in nx.all_simple_paths(trust_graph, start, end):
                            # Check if any node in path was verified
                            all_unverified = all(
                                not self._action_nodes.get(n, ActionNode(
                                    "", "", ActionType.AGENT_MESSAGE, ""
                                )).verified
                                for n in path
                            )
                            if all_unverified and len(path) > 1:
                                chains.append(path)

        return chains

    # -------------------------------------------------------------------------
    # Cascade Failure Detection
    # -------------------------------------------------------------------------

    def detect_cascade_failures(self) -> list[dict[str, Any]]:
        """
        Detect cascade failures in the system.

        A cascade failure occurs when one failure triggers multiple
        subsequent failures across different agents.

        Returns:
            List of cascade failure records
        """
        cascades = []

        for error_id in self._error_sources:
            source = self._action_nodes.get(error_id)
            if not source:
                continue

            # Find all errors that trace back to this source
            descendants = nx.descendants(self.graph, error_id)
            downstream_errors = [
                self._action_nodes[d] for d in descendants
                if d in self._action_nodes and self._action_nodes[d].is_error
            ]

            if len(downstream_errors) > 0:
                affected_agent_ids = set(e.agent_id for e in downstream_errors)

                cascades.append({
                    "source_error_id": error_id,
                    "source_agent_id": source.agent_id,
                    "source_error_type": source.error_type,
                    "num_downstream_errors": len(downstream_errors),
                    "affected_agents": list(affected_agent_ids),
                    "cascade_depth": max(
                        len(p) for p in nx.all_simple_paths(
                            self.graph, error_id,
                            downstream_errors[-1].action_id
                        )
                    ) if downstream_errors else 0,
                })

        return cascades

    # -------------------------------------------------------------------------
    # Graph Statistics
    # -------------------------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive graph statistics."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_error_sources": len(self._error_sources),
            "num_total_errors": len(self.get_all_errors()),
            "agents": list(set(n.agent_id for n in self._action_nodes.values())),
            "action_types": dict(
                (t.value, len(self.get_actions_by_type(t)))
                for t in ActionType
            ),
            "dependency_types": dict(
                (d.value, sum(
                    1 for _, _, data in self.graph.edges(data=True)
                    if data.get("dependency_type") == d.value
                ))
                for d in DependencyType
            ),
            "avg_out_degree": (
                sum(d for _, d in self.graph.out_degree()) / self.graph.number_of_nodes()
                if self.graph.number_of_nodes() > 0 else 0
            ),
            "avg_in_degree": (
                sum(d for _, d in self.graph.in_degree()) / self.graph.number_of_nodes()
                if self.graph.number_of_nodes() > 0 else 0
            ),
        }

    # -------------------------------------------------------------------------
    # Export / Visualization
    # -------------------------------------------------------------------------

    def export_mermaid(
        self,
        include_content: bool = False,
        max_content_length: int = 30
    ) -> str:
        """
        Export the graph as a Mermaid diagram.

        Args:
            include_content: Whether to include action content in labels
            max_content_length: Maximum content length to show

        Returns:
            Mermaid diagram string
        """
        lines = ["graph TD"]

        # Define styles
        lines.append("    classDef error fill:#ff6b6b,stroke:#c92a2a,color:#fff")
        lines.append("    classDef verified fill:#51cf66,stroke:#2f9e44,color:#fff")
        lines.append("    classDef agent_msg fill:#74c0fc,stroke:#1c7ed6")
        lines.append("    classDef tool_call fill:#ffd43b,stroke:#f59f00")

        # Add nodes
        for action_id, node in self._action_nodes.items():
            label = f"{node.agent_id}:{node.action_type.value}"
            if include_content:
                content = node.content[:max_content_length]
                if len(node.content) > max_content_length:
                    content += "..."
                label += f"<br/>{content}"

            # Determine node shape and style
            shape_start, shape_end = "[", "]"
            style_class = ""

            if node.is_error:
                shape_start, shape_end = "((", "))"
                style_class = ":::error"
            elif node.verified:
                style_class = ":::verified"
            elif node.action_type == ActionType.TOOL_CALL:
                shape_start, shape_end = "{{", "}}"
                style_class = ":::tool_call"

            # Escape special characters in label
            safe_label = label.replace('"', "'").replace("\n", "<br/>")
            lines.append(f"    {action_id}{shape_start}\"{safe_label}\"{shape_end}{style_class}")

        # Add edges
        for u, v, data in self.graph.edges(data=True):
            dep_type = data.get("dependency_type", "")
            weight = data.get("weight", 1.0)

            # Use different arrow styles based on dependency type
            if dep_type == DependencyType.TRUST.value:
                arrow = "==>"
            elif dep_type == DependencyType.ERROR_PROPAGATION.value:
                arrow = "-.->|error|"
            else:
                arrow = "-->"

            if dep_type not in [DependencyType.TRUST.value, DependencyType.ERROR_PROPAGATION.value]:
                lines.append(f"    {u} {arrow}|{dep_type}| {v}")
            else:
                lines.append(f"    {u} {arrow} {v}")

        return "\n".join(lines)

    def export_graphviz(self) -> str:
        """Export the graph as GraphViz DOT format."""
        lines = ["digraph ObservabilityGraph {"]
        lines.append("    rankdir=TB;")
        lines.append("    node [shape=box];")

        # Add nodes with attributes
        for action_id, node in self._action_nodes.items():
            attrs = [f'label="{node.agent_id}\\n{node.action_type.value}"']

            if node.is_error:
                attrs.append('style=filled')
                attrs.append('fillcolor="#ff6b6b"')
            elif node.verified:
                attrs.append('style=filled')
                attrs.append('fillcolor="#51cf66"')

            lines.append(f'    "{action_id}" [{", ".join(attrs)}];')

        # Add edges
        for u, v, data in self.graph.edges(data=True):
            dep_type = data.get("dependency_type", "")
            attrs = [f'label="{dep_type}"']

            if dep_type == DependencyType.ERROR_PROPAGATION.value:
                attrs.append('color="red"')
                attrs.append('style="dashed"')
            elif dep_type == DependencyType.TRUST.value:
                attrs.append('color="blue"')
                attrs.append('penwidth="2"')

            lines.append(f'    "{u}" -> "{v}" [{", ".join(attrs)}];')

        lines.append("}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export the graph as a dictionary for JSON serialization."""
        return {
            "nodes": [
                {
                    "action_id": n.action_id,
                    "agent_id": n.agent_id,
                    "action_type": n.action_type.value,
                    "content": n.content,
                    "timestamp": n.timestamp,
                    "is_error": n.is_error,
                    "error_type": n.error_type,
                    "verified": n.verified,
                    "metadata": n.metadata,
                }
                for n in self._action_nodes.values()
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "dependency_type": data.get("dependency_type"),
                    "weight": data.get("weight", 1.0),
                }
                for u, v, data in self.graph.edges(data=True)
            ],
            "error_sources": list(self._error_sources),
            "statistics": self.get_statistics(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservabilityGraph":
        """Create a graph from a dictionary representation."""
        graph = cls()

        for node_data in data.get("nodes", []):
            graph.add_action(
                action_id=node_data["action_id"],
                agent_id=node_data["agent_id"],
                action_type=ActionType(node_data["action_type"]),
                content=node_data["content"],
                timestamp=node_data.get("timestamp"),
                is_error=node_data.get("is_error", False),
                error_type=node_data.get("error_type"),
                verified=node_data.get("verified", False),
                metadata=node_data.get("metadata", {}),
            )

            if node_data.get("is_error"):
                graph._error_sources.add(node_data["action_id"])

        for edge_data in data.get("edges", []):
            graph.add_dependency(
                source_id=edge_data["source"],
                target_id=edge_data["target"],
                dependency_type=DependencyType(edge_data["dependency_type"]),
                weight=edge_data.get("weight", 1.0),
            )

        return graph
