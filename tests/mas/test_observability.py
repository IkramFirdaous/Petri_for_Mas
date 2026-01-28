"""
Tests for ObservabilityGraph and related classes.

These tests verify the causal dependency tracking and
error propagation analysis functionality.
"""

import pytest
import time

from petri.mas.observability import (
    ObservabilityGraph,
    ActionNode,
    DependencyEdge,
    PropagationChain,
    ActionType,
    DependencyType,
)


class TestActionNode:
    """Tests for ActionNode class."""

    def test_create_action_node(self):
        """Test basic action node creation."""
        node = ActionNode(
            action_id="a1",
            agent_id="agent_1",
            action_type=ActionType.AGENT_MESSAGE,
            content="Hello world"
        )

        assert node.action_id == "a1"
        assert node.agent_id == "agent_1"
        assert node.action_type == ActionType.AGENT_MESSAGE
        assert node.content == "Hello world"
        assert not node.is_error
        assert node.timestamp > 0

    def test_mark_as_error(self):
        """Test marking node as error."""
        node = ActionNode(
            action_id="a1",
            agent_id="agent_1",
            action_type=ActionType.AGENT_MESSAGE,
            content="Bad data"
        )

        node.mark_as_error("factual_error", severity=7)

        assert node.is_error
        assert node.error_type == "factual_error"
        assert node.error_severity == 7

    def test_mark_as_verified(self):
        """Test marking node as verified."""
        node = ActionNode(
            action_id="a1",
            agent_id="agent_1",
            action_type=ActionType.AGENT_DECISION,
            content="Approved"
        )

        node.mark_as_verified("validator_1")

        assert node.verified
        assert node.verified_by == "validator_1"


class TestDependencyEdge:
    """Tests for DependencyEdge class."""

    def test_create_dependency_edge(self):
        """Test basic edge creation."""
        edge = DependencyEdge(
            source_id="a1",
            target_id="a2",
            dependency_type=DependencyType.CAUSAL
        )

        assert edge.source_id == "a1"
        assert edge.target_id == "a2"
        assert edge.dependency_type == DependencyType.CAUSAL
        assert edge.weight == 1.0


class TestPropagationChain:
    """Tests for PropagationChain class."""

    def test_create_chain(self):
        """Test chain creation."""
        chain = PropagationChain("error_1")

        assert chain.source_action_id == "error_1"
        assert chain.chain == ["error_1"]
        assert chain.length == 1
        assert chain.num_affected_agents == 0

    def test_add_propagation(self):
        """Test adding propagation steps."""
        chain = PropagationChain("error_1")
        chain.add_propagation("a2", "agent_1", amplification=1.2)
        chain.add_propagation("a3", "agent_2", amplification=1.5)

        assert chain.length == 3
        assert chain.num_affected_agents == 2
        assert chain.amplification_factor == pytest.approx(1.8)

    def test_mark_detected(self):
        """Test marking error as detected."""
        chain = PropagationChain("error_1")
        chain.add_propagation("a2", "agent_1")
        chain.mark_detected("validator", "a2")

        assert chain.was_detected
        assert chain.detected_by == "validator"
        assert chain.detection_point == "a2"


class TestObservabilityGraph:
    """Tests for ObservabilityGraph class."""

    def test_create_empty_graph(self):
        """Test creating an empty graph."""
        graph = ObservabilityGraph()

        assert graph.graph.number_of_nodes() == 0
        assert graph.graph.number_of_edges() == 0

    def test_add_action(self):
        """Test adding an action."""
        graph = ObservabilityGraph()
        node = graph.add_action(
            "a1",
            "agent_1",
            ActionType.AGENT_MESSAGE,
            "Hello"
        )

        assert node.action_id == "a1"
        assert graph.graph.number_of_nodes() == 1
        assert graph.get_action("a1") == node

    def test_add_duplicate_action_raises(self):
        """Test that adding duplicate action raises error."""
        graph = ObservabilityGraph()
        graph.add_action("a1", "agent_1", ActionType.AGENT_MESSAGE, "Hello")

        with pytest.raises(ValueError, match="already exists"):
            graph.add_action("a1", "agent_2", ActionType.AGENT_MESSAGE, "World")

    def test_add_dependency(self):
        """Test adding a dependency."""
        graph = ObservabilityGraph()
        graph.add_action("a1", "agent_1", ActionType.AGENT_MESSAGE, "Data")
        graph.add_action("a2", "agent_2", ActionType.AGENT_DECISION, "Approved")

        edge = graph.add_dependency("a1", "a2", DependencyType.TRUST)

        assert edge.source_id == "a1"
        assert edge.target_id == "a2"
        assert graph.graph.number_of_edges() == 1

    def test_add_dependency_nonexistent_raises(self):
        """Test that adding dependency with nonexistent node raises."""
        graph = ObservabilityGraph()
        graph.add_action("a1", "agent_1", ActionType.AGENT_MESSAGE, "Hello")

        with pytest.raises(KeyError):
            graph.add_dependency("a1", "nonexistent", DependencyType.CAUSAL)

    def test_get_dependencies_from(self):
        """Test getting outgoing dependencies."""
        graph = ObservabilityGraph()
        graph.add_action("a1", "agent_1", ActionType.AGENT_MESSAGE, "Data")
        graph.add_action("a2", "agent_2", ActionType.AGENT_DECISION, "D1")
        graph.add_action("a3", "agent_3", ActionType.AGENT_DECISION, "D2")

        graph.add_dependency("a1", "a2", DependencyType.INFORMATIONAL)
        graph.add_dependency("a1", "a3", DependencyType.TRUST)

        deps = graph.get_dependencies_from("a1")

        assert len(deps) == 2
        targets = {d[0] for d in deps}
        assert targets == {"a2", "a3"}

    def test_get_dependencies_to(self):
        """Test getting incoming dependencies."""
        graph = ObservabilityGraph()
        graph.add_action("a1", "agent_1", ActionType.AGENT_MESSAGE, "D1")
        graph.add_action("a2", "agent_2", ActionType.AGENT_MESSAGE, "D2")
        graph.add_action("a3", "agent_3", ActionType.AGENT_DECISION, "Result")

        graph.add_dependency("a1", "a3", DependencyType.INFORMATIONAL)
        graph.add_dependency("a2", "a3", DependencyType.INFORMATIONAL)

        deps = graph.get_dependencies_to("a3")

        assert len(deps) == 2
        sources = {d[0] for d in deps}
        assert sources == {"a1", "a2"}

    def test_mark_as_error(self):
        """Test marking action as error."""
        graph = ObservabilityGraph()
        graph.add_action("a1", "agent_1", ActionType.AGENT_MESSAGE, "Bad data")

        graph.mark_as_error("a1", "factual_error", severity=7)

        node = graph.get_action("a1")
        assert node.is_error
        assert node.error_type == "factual_error"
        assert "a1" in graph._error_sources

    def test_get_actions_by_agent(self):
        """Test filtering actions by agent."""
        graph = ObservabilityGraph()
        graph.add_action("a1", "agent_1", ActionType.AGENT_MESSAGE, "M1")
        graph.add_action("a2", "agent_1", ActionType.AGENT_MESSAGE, "M2")
        graph.add_action("a3", "agent_2", ActionType.AGENT_MESSAGE, "M3")

        agent1_actions = graph.get_actions_by_agent("agent_1")

        assert len(agent1_actions) == 2
        assert all(a.agent_id == "agent_1" for a in agent1_actions)

    def test_get_actions_by_type(self):
        """Test filtering actions by type."""
        graph = ObservabilityGraph()
        graph.add_action("a1", "agent_1", ActionType.AGENT_MESSAGE, "M1")
        graph.add_action("a2", "agent_1", ActionType.TOOL_CALL, "T1")
        graph.add_action("a3", "agent_2", ActionType.AGENT_MESSAGE, "M2")

        messages = graph.get_actions_by_type(ActionType.AGENT_MESSAGE)

        assert len(messages) == 2


class TestErrorPropagation:
    """Tests for error propagation analysis."""

    @pytest.fixture
    def simple_chain_graph(self):
        """Create a simple chain graph for testing."""
        graph = ObservabilityGraph()

        # Create chain: a1 -> a2 -> a3 -> a4
        graph.add_action("a1", "researcher", ActionType.AGENT_MESSAGE, "Data")
        graph.add_action("a2", "validator", ActionType.AGENT_DECISION, "OK")
        graph.add_action("a3", "executor", ActionType.AGENT_MESSAGE, "Executing")
        graph.add_action("a4", "executor", ActionType.AGENT_DECISION, "Done")

        graph.add_dependency("a1", "a2", DependencyType.TRUST)
        graph.add_dependency("a2", "a3", DependencyType.CAUSAL)
        graph.add_dependency("a3", "a4", DependencyType.SEQUENTIAL)

        return graph

    def test_find_propagation_chains_no_error(self, simple_chain_graph):
        """Test finding chains when there's no error."""
        chains = simple_chain_graph.find_error_propagation_chains("a1")

        # Should still find chains, but source isn't marked as error
        assert len(chains) > 0

    def test_find_propagation_chains_with_error(self, simple_chain_graph):
        """Test finding chains from an error source."""
        simple_chain_graph.mark_as_error("a1", "factual_error")

        chains = simple_chain_graph.find_error_propagation_chains("a1")

        assert len(chains) > 0
        # Check that chains start from a1
        for chain in chains:
            assert chain.source_action_id == "a1"

    def test_calculate_propagation_metrics_no_errors(self):
        """Test metrics when there are no errors."""
        graph = ObservabilityGraph()
        graph.add_action("a1", "agent_1", ActionType.AGENT_MESSAGE, "OK")

        metrics = graph.calculate_propagation_metrics()

        assert metrics["error_propagation_rate"] == 0.0
        assert metrics["num_error_sources"] == 0

    def test_calculate_propagation_metrics_with_errors(self, simple_chain_graph):
        """Test metrics with errors."""
        simple_chain_graph.mark_as_error("a1", "factual_error")

        metrics = simple_chain_graph.calculate_propagation_metrics()

        assert metrics["error_propagation_rate"] > 0
        assert metrics["num_error_sources"] == 1
        assert metrics["total_actions"] == 4


class TestTrustAnalysis:
    """Tests for trust violation analysis."""

    def test_find_trust_violations(self):
        """Test finding trust violations."""
        graph = ObservabilityGraph()

        graph.add_action("a1", "researcher", ActionType.AGENT_MESSAGE, "Bad data")
        graph.add_action("a2", "validator", ActionType.AGENT_DECISION, "Approved")

        graph.add_dependency("a1", "a2", DependencyType.TRUST)
        graph.mark_as_error("a1", "factual_error")

        violations = graph.find_trust_violations()

        assert len(violations) == 1
        assert violations[0]["trusted_action_id"] == "a1"
        assert violations[0]["trusting_action_id"] == "a2"

    def test_find_trust_violations_verified(self):
        """Test that verified trust isn't a violation."""
        graph = ObservabilityGraph()

        graph.add_action("a1", "researcher", ActionType.AGENT_MESSAGE, "Bad data")
        node2 = graph.add_action("a2", "validator", ActionType.AGENT_DECISION, "Approved")
        node2.mark_as_verified("validator")

        graph.add_dependency("a1", "a2", DependencyType.TRUST)
        graph.mark_as_error("a1", "factual_error")

        violations = graph.find_trust_violations()

        # Should not be a violation because a2 was verified
        assert len(violations) == 0


class TestCascadeFailureDetection:
    """Tests for cascade failure detection."""

    def test_detect_cascade_failures(self):
        """Test detecting cascade failures."""
        graph = ObservabilityGraph()

        # Create a cascade: error at a1 causes errors at a2 and a3
        graph.add_action("a1", "agent_1", ActionType.AGENT_MESSAGE, "Bad")
        graph.add_action("a2", "agent_2", ActionType.AGENT_DECISION, "Based on a1")
        graph.add_action("a3", "agent_3", ActionType.AGENT_DECISION, "Based on a2")

        graph.add_dependency("a1", "a2", DependencyType.CAUSAL)
        graph.add_dependency("a2", "a3", DependencyType.CAUSAL)

        # Only a1 is an original error source; a2 and a3 are propagated
        graph.mark_as_error("a1", "initial_error", is_original=True)
        graph.mark_as_error("a2", "propagated_error", is_original=False)
        graph.mark_as_error("a3", "propagated_error", is_original=False)

        cascades = graph.detect_cascade_failures()

        assert len(cascades) == 1
        assert cascades[0]["source_error_id"] == "a1"
        assert cascades[0]["num_downstream_errors"] == 2


class TestExport:
    """Tests for graph export functionality."""

    def test_export_mermaid(self):
        """Test Mermaid export."""
        graph = ObservabilityGraph()
        graph.add_action("a1", "agent_1", ActionType.AGENT_MESSAGE, "Hello")
        graph.add_action("a2", "agent_2", ActionType.AGENT_DECISION, "OK")
        graph.add_dependency("a1", "a2", DependencyType.CAUSAL)

        mermaid = graph.export_mermaid()

        assert "graph TD" in mermaid
        assert "a1" in mermaid
        assert "a2" in mermaid

    def test_export_graphviz(self):
        """Test GraphViz export."""
        graph = ObservabilityGraph()
        graph.add_action("a1", "agent_1", ActionType.AGENT_MESSAGE, "Hello")
        graph.add_action("a2", "agent_2", ActionType.AGENT_DECISION, "OK")
        graph.add_dependency("a1", "a2", DependencyType.CAUSAL)

        dot = graph.export_graphviz()

        assert "digraph" in dot
        assert "a1" in dot
        assert "a2" in dot

    def test_to_dict_and_from_dict(self):
        """Test JSON serialization round-trip."""
        graph = ObservabilityGraph()
        graph.add_action("a1", "agent_1", ActionType.AGENT_MESSAGE, "Hello")
        graph.add_action("a2", "agent_2", ActionType.AGENT_DECISION, "OK")
        graph.add_dependency("a1", "a2", DependencyType.CAUSAL)
        graph.mark_as_error("a1", "test_error")

        # Export to dict
        data = graph.to_dict()

        # Reimport
        graph2 = ObservabilityGraph.from_dict(data)

        assert graph2.graph.number_of_nodes() == 2
        assert graph2.graph.number_of_edges() == 1
        assert graph2.get_action("a1").is_error


class TestStatistics:
    """Tests for graph statistics."""

    def test_get_statistics(self):
        """Test getting graph statistics."""
        graph = ObservabilityGraph()
        graph.add_action("a1", "agent_1", ActionType.AGENT_MESSAGE, "M1")
        graph.add_action("a2", "agent_1", ActionType.TOOL_CALL, "T1")
        graph.add_action("a3", "agent_2", ActionType.AGENT_MESSAGE, "M2")
        graph.add_dependency("a1", "a2", DependencyType.CAUSAL)
        graph.add_dependency("a2", "a3", DependencyType.TRUST)

        stats = graph.get_statistics()

        assert stats["num_nodes"] == 3
        assert stats["num_edges"] == 2
        assert "agent_1" in stats["agents"]
        assert "agent_2" in stats["agents"]
