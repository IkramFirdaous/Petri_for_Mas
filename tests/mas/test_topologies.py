"""
Tests for topology builders and utilities.

These tests verify that different MAS topologies are
correctly constructed.
"""

import pytest

from petri.mas.stores import MASAuditStore
from petri.mas.topologies import (
    build_topology,
    get_topology_description,
    TopologyConfig,
    AgentRole,
    PREDEFINED_ROLES,
    ChainTopologyBuilder,
    HubTopologyBuilder,
    MeshTopologyBuilder,
    PipelineTopologyBuilder,
    HierarchyTopologyBuilder,
    CustomTopologyBuilder,
)


class TestAgentRole:
    """Tests for AgentRole class."""

    def test_create_agent_role(self):
        """Test creating an agent role."""
        role = AgentRole(
            name="custom_role",
            description="A custom role for testing",
            capabilities=["analyze", "report"],
            trust_level=0.75
        )

        assert role.name == "custom_role"
        assert role.trust_level == 0.75
        assert "analyze" in role.capabilities

    def test_predefined_roles(self):
        """Test that predefined roles exist."""
        assert "researcher" in PREDEFINED_ROLES
        assert "validator" in PREDEFINED_ROLES
        assert "executor" in PREDEFINED_ROLES
        assert "coordinator" in PREDEFINED_ROLES
        assert "oversight" in PREDEFINED_ROLES


class TestChainTopology:
    """Tests for chain topology."""

    def test_build_chain_topology(self):
        """Test building a chain topology."""
        store = MASAuditStore()
        build_topology(store, "chain", num_agents=3)

        assert len(store.agents) == 3
        assert len(store.channels) == 2  # A->B, B->C
        assert store.topology == "chain"

    def test_chain_topology_with_roles(self):
        """Test chain topology with custom roles."""
        store = MASAuditStore()
        build_topology(
            store,
            "chain",
            num_agents=3,
            roles=["researcher", "validator", "executor"]
        )

        roles = [store.get_agent(f"agent_{i}").role for i in range(3)]
        assert roles == ["researcher", "validator", "executor"]

    def test_chain_topology_connectivity(self):
        """Test that chain topology has correct connectivity."""
        store = MASAuditStore()
        build_topology(store, "chain", num_agents=4)

        # Each agent should only connect to neighbors
        assert store.can_communicate("agent_0", "agent_1")
        assert store.can_communicate("agent_1", "agent_2")
        assert store.can_communicate("agent_2", "agent_3")

        # Should not be able to skip
        assert not store.can_communicate("agent_0", "agent_2")
        assert not store.can_communicate("agent_0", "agent_3")

    def test_chain_topology_bidirectional(self):
        """Test bidirectional chain topology."""
        store = MASAuditStore()
        build_topology(store, "chain", num_agents=3, bidirectional=True)

        assert store.can_communicate("agent_0", "agent_1")
        assert store.can_communicate("agent_1", "agent_0")

    def test_chain_topology_too_few_agents(self):
        """Test that chain with < 2 agents raises error."""
        store = MASAuditStore()

        with pytest.raises(ValueError, match="at least 2 agents"):
            build_topology(store, "chain", num_agents=1)


class TestHubTopology:
    """Tests for hub topology."""

    def test_build_hub_topology(self):
        """Test building a hub topology."""
        store = MASAuditStore()
        build_topology(store, "hub", num_agents=4)

        assert len(store.agents) == 4
        # Hub connects to all spokes: 3 bidirectional channels
        assert len(store.channels) == 3
        assert store.topology == "hub"

    def test_hub_topology_connectivity(self):
        """Test hub topology connectivity."""
        store = MASAuditStore()
        build_topology(store, "hub", num_agents=4)

        hub_id = store.topology_config["hub_id"]

        # All spokes should connect to hub
        for spoke_id in store.topology_config["spoke_ids"]:
            assert store.can_communicate(hub_id, spoke_id)
            assert store.can_communicate(spoke_id, hub_id)

        # Spokes should not connect directly to each other
        spoke_ids = store.topology_config["spoke_ids"]
        if len(spoke_ids) >= 2:
            assert not store.can_communicate(spoke_ids[0], spoke_ids[1])

    def test_hub_topology_with_custom_hub_id(self):
        """Test hub topology with custom hub agent ID."""
        store = MASAuditStore()
        build_topology(
            store,
            "hub",
            num_agents=3,
            hub_agent_id="coordinator"
        )

        assert "coordinator" in store.agents
        assert store.topology_config["hub_id"] == "coordinator"


class TestMeshTopology:
    """Tests for mesh topology."""

    def test_build_mesh_topology(self):
        """Test building a mesh topology."""
        store = MASAuditStore()
        build_topology(store, "mesh", num_agents=4)

        assert len(store.agents) == 4
        # Full mesh: n*(n-1)/2 = 4*3/2 = 6 channels
        assert len(store.channels) == 6
        assert store.topology == "mesh"

    def test_mesh_topology_full_connectivity(self):
        """Test that mesh has full connectivity."""
        store = MASAuditStore()
        build_topology(store, "mesh", num_agents=4)

        # Every agent should be able to communicate with every other
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert store.can_communicate(f"agent_{i}", f"agent_{j}")


class TestPipelineTopology:
    """Tests for pipeline topology."""

    def test_build_pipeline_topology(self):
        """Test building a pipeline topology."""
        store = MASAuditStore()
        build_topology(store, "pipeline", num_agents=4)

        assert len(store.agents) == 4
        assert store.topology == "pipeline"

        # Check stages are set
        assert "stages" in store.topology_config

    def test_pipeline_topology_roles(self):
        """Test pipeline topology has correct roles."""
        store = MASAuditStore()
        build_topology(
            store,
            "pipeline",
            num_agents=4,
            roles=["input", "processor", "validator", "output"]
        )

        assert store.get_agent("stage_0").role == "input"
        assert store.get_agent("stage_3").role == "output"

    def test_pipeline_topology_too_few_agents(self):
        """Test that pipeline with < 3 agents raises error."""
        store = MASAuditStore()

        with pytest.raises(ValueError, match="at least 3 agents"):
            build_topology(store, "pipeline", num_agents=2)


class TestHierarchyTopology:
    """Tests for hierarchy topology."""

    def test_build_hierarchy_topology(self):
        """Test building a hierarchy topology."""
        store = MASAuditStore()
        build_topology(store, "hierarchy", num_agents=7)

        assert len(store.agents) == 7
        assert store.topology == "hierarchy"
        assert "levels" in store.topology_config

    def test_hierarchy_topology_with_levels(self):
        """Test hierarchy with custom levels."""
        store = MASAuditStore()
        build_topology(
            store,
            "hierarchy",
            num_agents=5,
            hierarchy_levels=[
                ["executive"],
                ["manager_1", "manager_2"],
                ["worker_1", "worker_2"]
            ]
        )

        # Check structure
        levels = store.topology_config["levels"]
        assert len(levels) == 3
        assert len(levels[0]) == 1  # Executive level
        assert len(levels[1]) == 2  # Manager level
        assert len(levels[2]) == 2  # Worker level


class TestCustomTopology:
    """Tests for custom topology."""

    def test_build_custom_topology(self):
        """Test building a custom topology."""
        store = MASAuditStore()
        build_topology(
            store,
            "custom",
            num_agents=3,
            custom_channels=[
                ("agent_0", "agent_2"),  # Skip agent_1
                ("agent_1", "agent_2"),
            ]
        )

        assert len(store.agents) == 3
        assert len(store.channels) == 2
        assert store.topology == "custom"

        # Check custom connectivity
        assert store.can_communicate("agent_0", "agent_2")
        assert store.can_communicate("agent_1", "agent_2")
        assert not store.can_communicate("agent_0", "agent_1")


class TestBuildTopologyFunction:
    """Tests for the build_topology utility function."""

    def test_unknown_topology_raises(self):
        """Test that unknown topology type raises error."""
        store = MASAuditStore()

        with pytest.raises(ValueError, match="Unknown topology"):
            build_topology(store, "unknown_type", num_agents=3)

    def test_get_topology_description(self):
        """Test getting topology descriptions."""
        desc = get_topology_description("chain")
        assert "Linear" in desc or "sequence" in desc

        desc = get_topology_description("hub")
        assert "Star" in desc or "coordinator" in desc

        desc = get_topology_description("mesh")
        assert "Fully connected" in desc or "network" in desc


class TestTopologyIntegration:
    """Integration tests for topologies."""

    def test_topology_with_messages(self):
        """Test that topology works with message recording."""
        store = MASAuditStore()
        build_topology(store, "chain", num_agents=3, roles=["a", "b", "c"])

        # Record messages following the chain
        msg1 = store.record_inter_agent_message(
            "agent_0", "agent_1",
            "Passing data to next stage",
            "request"
        )

        msg2 = store.record_inter_agent_message(
            "agent_1", "agent_2",
            "Validated, forwarding",
            "request",
            parent_message_id=msg1.id
        )

        # Verify thread
        thread = store.get_conversation_thread(msg2.id)
        assert len(thread) == 2

    def test_multiple_topologies_reset(self):
        """Test building multiple topologies with reset."""
        store = MASAuditStore()

        # Build first topology
        build_topology(store, "chain", num_agents=3)
        assert len(store.agents) == 3

        # Reset and build different topology
        store.reset()
        build_topology(store, "mesh", num_agents=4)
        assert len(store.agents) == 4
        assert store.topology == "mesh"
