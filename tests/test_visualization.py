"""
Tests for graph visualization exports (Epic 8).

Tests GraphViz DOT format and JSON format exports for various graph structures.
"""

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from claude_agent_graph import (
    AgentGraph,
    export_graphviz,
    export_json,
    VisualizationError,
)


@pytest.fixture
async def empty_graph() -> AgentGraph:
    """Create an empty graph for testing."""
    return AgentGraph(name="test_graph")


@pytest.fixture
async def simple_graph() -> AgentGraph:
    """Create a simple graph with a few nodes and edges."""
    graph = AgentGraph(name="test_graph")

    # Add nodes
    await graph.add_node("node_1", "System prompt 1", model="claude-sonnet-4-20250514")
    await graph.add_node("node_2", "System prompt 2", model="claude-sonnet-4-20250514")
    await graph.add_node("node_3", "System prompt 3", model="claude-sonnet-4-20250514")

    # Add edges
    await graph.add_edge("node_1", "node_2", directed=True)
    await graph.add_edge("node_2", "node_3", directed=True)
    await graph.add_edge("node_1", "node_3", directed=True)

    return graph


@pytest.fixture
async def complex_graph() -> AgentGraph:
    """Create a more complex graph with different properties."""
    graph = AgentGraph(name="complex_graph")

    # Add nodes with metadata (passed as keyword args, not as 'metadata' dict)
    await graph.add_node(
        "supervisor", "Supervise tasks", role="supervisor"
    )
    await graph.add_node(
        "worker_1", "Execute tasks", role="worker"
    )
    await graph.add_node(
        "worker_2", "Execute tasks", role="worker"
    )
    await graph.add_node(
        "monitor", "Monitor progress", role="monitor"
    )

    # Add edges with properties
    await graph.add_edge("supervisor", "worker_1", directed=True, priority="high")
    await graph.add_edge("supervisor", "worker_2", directed=True, priority="high")
    await graph.add_edge("worker_1", "monitor", directed=True)
    await graph.add_edge("worker_2", "monitor", directed=True)

    return graph


class TestGraphVizExport:
    """Tests for GraphViz DOT format export."""

    @pytest.mark.asyncio
    async def test_export_graphviz_empty_graph(self, empty_graph: AgentGraph) -> None:
        """Test exporting an empty graph."""
        dot = export_graphviz(empty_graph)

        assert "digraph" in dot
        assert "node" not in dot or "node [shape" in dot  # Node declaration OK
        assert "->" not in dot  # No edges

    @pytest.mark.asyncio
    async def test_export_graphviz_simple_graph(self, simple_graph: AgentGraph) -> None:
        """Test exporting a simple graph with nodes and edges."""
        dot = export_graphviz(simple_graph)

        assert "digraph" in dot
        assert "node_1" in dot
        assert "node_2" in dot
        assert "node_3" in dot
        assert "node_1" in dot and "node_2" in dot  # Edge should exist
        assert "->" in dot  # Has directed edges

    @pytest.mark.asyncio
    async def test_export_graphviz_with_metadata(
        self, simple_graph: AgentGraph
    ) -> None:
        """Test exporting with metadata included."""
        dot = export_graphviz(simple_graph, include_metadata=True)

        assert "digraph" in dot
        assert "node_1" in dot

    @pytest.mark.asyncio
    async def test_export_graphviz_to_file(self, simple_graph: AgentGraph) -> None:
        """Test writing GraphViz output to a file."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "graph.dot"

            dot = export_graphviz(simple_graph, output_path=output_path)

            assert output_path.exists()
            content = output_path.read_text()
            assert content == dot
            assert "digraph" in content

    @pytest.mark.asyncio
    async def test_export_graphviz_syntax_valid(self, simple_graph: AgentGraph) -> None:
        """Test that exported DOT is syntactically valid."""
        dot = export_graphviz(simple_graph)

        # Basic syntax checks
        assert dot.count("{") == dot.count("}")
        assert "digraph" in dot
        assert dot.strip().endswith("}")

    @pytest.mark.asyncio
    async def test_export_graphviz_special_chars_escaped(
        self, empty_graph: AgentGraph
    ) -> None:
        """Test that special characters in node IDs are handled."""
        # Add nodes with special chars in ID
        await empty_graph.add_node("node-with-dashes", "Test prompt")
        await empty_graph.add_node("node_with_underscores", "Test prompt")

        dot = export_graphviz(empty_graph)

        # Should contain the nodes (with proper escaping if needed)
        assert "node-with-dashes" in dot or "node_with_dashes" in dot
        assert "digraph" in dot

    @pytest.mark.asyncio
    async def test_export_graphviz_node_colors(
        self, simple_graph: AgentGraph
    ) -> None:
        """Test node coloring by status."""
        dot = export_graphviz(simple_graph, node_style="status")

        # All nodes should be present with color information
        assert "fillcolor" in dot
        assert "node_1" in dot


class TestJSONExport:
    """Tests for JSON format export."""

    @pytest.mark.asyncio
    async def test_export_json_empty_graph(self, empty_graph: AgentGraph) -> None:
        """Test exporting empty graph as JSON."""
        result = export_json(empty_graph)

        assert isinstance(result, dict)
        assert "nodes" in result
        assert "links" in result
        assert result["nodes"] == []
        assert result["links"] == []

    @pytest.mark.asyncio
    async def test_export_json_node_link_format(
        self, simple_graph: AgentGraph
    ) -> None:
        """Test node-link JSON format (D3.js/Cytoscape compatible)."""
        result = export_json(simple_graph, format_type="node-link")

        assert isinstance(result, dict)
        assert "nodes" in result
        assert "links" in result
        assert "graph_info" in result

        nodes = result["nodes"]
        assert len(nodes) == 3
        assert all("id" in node for node in nodes)
        assert all("status" in node for node in nodes)
        assert all("model" in node for node in nodes)

        links = result["links"]
        assert len(links) == 3
        assert all("source" in link for link in links)
        assert all("target" in link for link in links)
        assert all("directed" in link for link in links)

    @pytest.mark.asyncio
    async def test_export_json_preserves_metadata(
        self, complex_graph: AgentGraph
    ) -> None:
        """Test that metadata is preserved in JSON export."""
        result = export_json(complex_graph, include_metadata=True)

        nodes = result["nodes"]
        supervisor = next((n for n in nodes if n["id"] == "supervisor"), None)
        assert supervisor is not None
        assert "metadata" in supervisor
        assert supervisor["metadata"]["role"] == "supervisor"

    @pytest.mark.asyncio
    async def test_export_json_datetime_encoding(
        self, simple_graph: AgentGraph
    ) -> None:
        """Test that datetime objects are properly encoded."""
        result = export_json(simple_graph, include_metadata=True)

        nodes = result["nodes"]
        if "created_at" in nodes[0]:
            # Should be a string, not a datetime object
            assert isinstance(nodes[0]["created_at"], str)
            # Should be ISO format
            assert "T" in nodes[0]["created_at"]

    @pytest.mark.asyncio
    async def test_export_json_to_file(self, simple_graph: AgentGraph) -> None:
        """Test writing JSON output to a file."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "graph.json"

            result = export_json(simple_graph, output_path=output_path)

            assert output_path.exists()
            assert isinstance(result, str)

            # Verify the file contains valid JSON
            loaded = json.loads(output_path.read_text())
            assert "nodes" in loaded
            assert "links" in loaded

    @pytest.mark.asyncio
    async def test_export_json_parsing(self, simple_graph: AgentGraph) -> None:
        """Test that exported JSON can be parsed and analyzed."""
        result = export_json(simple_graph, format_type="node-link")

        # Parse as JSON to verify validity
        json_str = json.dumps(result)
        parsed = json.loads(json_str)

        assert len(parsed["nodes"]) == 3
        assert len(parsed["links"]) == 3

    @pytest.mark.asyncio
    async def test_export_json_adjacency_format(
        self, simple_graph: AgentGraph
    ) -> None:
        """Test adjacency format JSON export."""
        result = export_json(simple_graph, format_type="adjacency")

        assert isinstance(result, dict)
        assert "nodes" in result
        assert "adjacency" in result
        assert "graph_info" in result

        # Check adjacency structure
        adjacency = result["adjacency"]
        assert "node_1" in adjacency
        assert "node_2" in adjacency
        assert "node_3" in adjacency

    @pytest.mark.asyncio
    async def test_export_json_invalid_format(
        self, simple_graph: AgentGraph
    ) -> None:
        """Test error handling for invalid format."""
        with pytest.raises(VisualizationError):
            export_json(simple_graph, format_type="invalid_format")


class TestVisualizationIntegration:
    """Integration tests for visualization exports."""

    @pytest.mark.asyncio
    async def test_export_both_formats(self, complex_graph: AgentGraph) -> None:
        """Test exporting the same graph in both formats."""
        dot = export_graphviz(complex_graph)
        json_data = export_json(complex_graph)

        # Both should have the same nodes
        dot_nodes = set()
        for node in complex_graph.get_nodes():
            if node.node_id in dot:
                dot_nodes.add(node.node_id)

        json_nodes = {node["id"] for node in json_data["nodes"]}

        assert len(dot_nodes) > 0
        assert len(json_nodes) > 0
        assert dot_nodes == json_nodes

    @pytest.mark.asyncio
    async def test_export_preserves_topology(self, complex_graph: AgentGraph) -> None:
        """Test that exports preserve graph topology."""
        json_data = export_json(complex_graph, format_type="node-link")

        # Count edges
        assert len(json_data["links"]) == complex_graph.edge_count
        assert len(json_data["nodes"]) == complex_graph.node_count

    @pytest.mark.asyncio
    async def test_export_large_graph(self) -> None:
        """Test exporting a larger graph."""
        graph = AgentGraph(name="large_graph", max_nodes=500)

        # Add many nodes
        for i in range(100):
            await graph.add_node(f"node_{i}", f"Prompt {i}")

        # Add some edges
        for i in range(99):
            await graph.add_edge(f"node_{i}", f"node_{i+1}")

        # Both exports should complete without error
        dot = export_graphviz(graph)
        json_data = export_json(graph)

        assert "digraph" in dot
        assert len(json_data["nodes"]) == 100
        assert len(json_data["links"]) == 99

    @pytest.mark.asyncio
    async def test_export_isolated_nodes(self) -> None:
        """Test exporting graph with isolated nodes."""
        graph = AgentGraph(name="isolated_graph")

        # Add isolated nodes
        await graph.add_node("isolated_1", "Prompt 1")
        await graph.add_node("isolated_2", "Prompt 2")

        # Add connected nodes
        await graph.add_node("connected_1", "Prompt 3")
        await graph.add_node("connected_2", "Prompt 4")
        await graph.add_edge("connected_1", "connected_2")

        dot = export_graphviz(graph)
        json_data = export_json(graph)

        # All nodes should be present
        assert "isolated_1" in dot
        assert "isolated_2" in dot
        assert "connected_1" in dot
        assert "connected_2" in dot

        assert len(json_data["nodes"]) == 4
        assert len(json_data["links"]) == 1
