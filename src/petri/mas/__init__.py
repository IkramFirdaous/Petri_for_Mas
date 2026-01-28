"""
Multi-Agent System (MAS) auditing extension for Petri.

This module extends Petri's capabilities to audit multi-agent systems,
enabling detection of error propagation, cascade failures, and
coordination breakdowns between AI agents.

Key Components:
- MASAuditStore: State management for multi-agent audits
- ObservabilityGraph: Causal dependency tracking
- Topologies: Predefined agent communication patterns
- MAS Auditor Agent: Red-teaming agent for MAS testing
- Inter-agent Tools: Tools for orchestrating multi-agent interactions

Example Usage:
    ```python
    from petri.mas import mas_auditor_agent, create_mas_audit_task

    # Create a MAS audit task
    task = create_mas_audit_task(
        seed_instructions="Test error propagation in a 3-agent pipeline",
        num_agents=3,
        topology="chain",
        agent_roles=["researcher", "validator", "executor"]
    )

    # Or use the agent directly
    agent = mas_auditor_agent(
        num_agents=3,
        topology="chain",
        agent_roles=["researcher", "validator", "executor"]
    )
    ```
"""

# Store classes
from petri.mas.stores import (
    MASAuditStore,
    AgentNode,
    InterAgentMessage,
    InterAgentChannel,
    AuditorStore,
    JudgeStore,
    MASMetadata,
)

# Observability classes
from petri.mas.observability import (
    ObservabilityGraph,
    ActionNode,
    DependencyEdge,
    PropagationChain,
    ActionType,
    DependencyType,
)

# Topology utilities
from petri.mas.topologies import (
    build_topology,
    get_topology_description,
    TopologyConfig,
    TopologyBuilder,
    AgentRole,
    PREDEFINED_ROLES,
    TOPOLOGY_BUILDERS,
    TopologyType,
    # Individual builders (for advanced use)
    ChainTopologyBuilder,
    HubTopologyBuilder,
    MeshTopologyBuilder,
    PipelineTopologyBuilder,
    HierarchyTopologyBuilder,
    CustomTopologyBuilder,
)

# Inter-agent tools
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

# MAS Auditor Agent
from petri.mas.auditor import (
    mas_auditor_agent,
    mas_audit_loop,
    create_mas_audit_task,
    MAS_AUDITOR_SYSTEM_PROMPT,
    MAS_AUDITOR_USER_MESSAGE,
)

# MAS Alignment Judge
from petri.mas.judge import (
    mas_alignment_judge,
    MAS_DIMENSIONS,
    MAS_JUDGE_PROMPT,
    MASJudgeResult,
    format_mas_transcript,
    format_metrics,
    parse_mas_judge_response,
    get_mas_dimension_names,
    get_mas_dimension_description,
)

__all__ = [
    # Core store classes
    "MASAuditStore",
    "AgentNode",
    "InterAgentMessage",
    "InterAgentChannel",
    "AuditorStore",
    "JudgeStore",
    "MASMetadata",
    # Observability
    "ObservabilityGraph",
    "ActionNode",
    "DependencyEdge",
    "PropagationChain",
    "ActionType",
    "DependencyType",
    # Topologies
    "build_topology",
    "get_topology_description",
    "TopologyConfig",
    "TopologyBuilder",
    "AgentRole",
    "PREDEFINED_ROLES",
    "TOPOLOGY_BUILDERS",
    "TopologyType",
    "ChainTopologyBuilder",
    "HubTopologyBuilder",
    "MeshTopologyBuilder",
    "PipelineTopologyBuilder",
    "HierarchyTopologyBuilder",
    "CustomTopologyBuilder",
    # Inter-agent tools
    "default_mas_auditor_tools",
    "create_target_agent",
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
    "end_mas_conversation",
    # MAS Auditor Agent
    "mas_auditor_agent",
    "mas_audit_loop",
    "create_mas_audit_task",
    "MAS_AUDITOR_SYSTEM_PROMPT",
    "MAS_AUDITOR_USER_MESSAGE",
    # MAS Alignment Judge
    "mas_alignment_judge",
    "MAS_DIMENSIONS",
    "MAS_JUDGE_PROMPT",
    "MASJudgeResult",
    "format_mas_transcript",
    "format_metrics",
    "parse_mas_judge_response",
    "get_mas_dimension_names",
    "get_mas_dimension_description",
]

__version__ = "0.1.0"
