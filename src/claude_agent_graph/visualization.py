"""
Visualization export utilities for agent graphs.

This module provides functions to export graph topology in multiple formats
for visualization with external tools:
- GraphViz (DOT format) for tools like Graphviz, Gephi, yEd
- JSON (node-link format) for tools like D3.js, Cytoscape.js

Epic 8: Monitoring & Observability
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .models import Edge, Node

if TYPE_CHECKING:
    from .graph import AgentGraph

logger = logging.getLogger(__name__)


class VisualizationError(Exception):
    """Raised when visualization export fails."""

    pass


def export_graphviz(
    graph: "AgentGraph",  # type: ignore
    output_path: Path | str | None = None,
    include_metadata: bool = True,
    include_message_counts: bool = False,
    node_style: str = "default",
    edge_style: str = "default",
) -> str:
    """
    Export graph as GraphViz DOT format.

    The DOT format can be used with Graphviz, Gephi, yEd, and other
    visualization tools.

    Args:
        graph: AgentGraph instance to export
        output_path: Optional path to write DOT file. If None, returns string.
        include_metadata: Include node/edge properties in labels
        include_message_counts: Add message count annotations to edges
        node_style: How to style nodes ("default", "status", "utilization")
        edge_style: How to style edges ("default", "messages", "control")

    Returns:
        DOT format string

    Raises:
        VisualizationError: If export fails
    """
    try:
        lines: list[str] = []
        lines.append("digraph {")
        lines.append("    rankdir=LR;")
        lines.append("    node [shape=box, style=filled, fillcolor=lightblue];")

        # Add nodes
        for node in graph.get_nodes():
            node_label = _format_node_label(node, include_metadata)
            node_color = _get_node_color(node, node_style)
            lines.append(f'    "{node.node_id}" [label="{node_label}", fillcolor="{node_color}"];')

        # Add edges
        for edge in graph._edges.values():
            edge_label = _format_edge_label(edge, include_message_counts)
            if edge_label:
                lines.append(f'    "{edge.from_node}" -> "{edge.to_node}" [label="{edge_label}"];')
            else:
                lines.append(f'    "{edge.from_node}" -> "{edge.to_node}";')

        lines.append("}")

        dot_content = "\n".join(lines)

        # Write to file if path provided
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(dot_content)
            logger.info(f"Exported GraphViz format to {output_path}")

        return dot_content

    except Exception as e:
        logger.error(f"Failed to export GraphViz: {e}", exc_info=True)
        raise VisualizationError(f"GraphViz export failed: {e}") from e


def export_json(
    graph: "AgentGraph",  # type: ignore
    output_path: Path | str | None = None,
    format_type: str = "node-link",
    include_metadata: bool = True,
    include_message_counts: bool = False,
) -> dict[str, Any] | str:
    """
    Export graph as JSON (node-link or adjacency format).

    The node-link format is compatible with D3.js and Cytoscape.js visualization
    libraries. The adjacency format provides an alternative structure for custom tools.

    Args:
        graph: AgentGraph instance to export
        output_path: Optional path to write JSON file. If None, returns dict.
        format_type: "node-link" (D3.js/Cytoscape) or "adjacency"
        include_metadata: Include full node/edge properties
        include_message_counts: Include message statistics

    Returns:
        Dictionary with JSON structure (or JSON string if output_path provided)

    Raises:
        VisualizationError: If export fails
    """
    try:
        if format_type == "node-link":
            data = _export_json_node_link(graph, include_metadata, include_message_counts)
        elif format_type == "adjacency":
            data = _export_json_adjacency(graph, include_metadata)
        else:
            raise ValueError(f"Unknown format: {format_type}. Use 'node-link' or 'adjacency'.")

        # Write to file if path provided
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            json_content = json.dumps(data, indent=2, default=str)
            path.write_text(json_content)
            logger.info(f"Exported JSON format to {output_path}")
            return json_content

        return data

    except Exception as e:
        logger.error(f"Failed to export JSON: {e}", exc_info=True)
        raise VisualizationError(f"JSON export failed: {e}") from e


def _export_json_node_link(
    graph: "AgentGraph",  # type: ignore
    include_metadata: bool,
    include_message_counts: bool,
) -> dict[str, Any]:
    """
    Export graph as node-link JSON format (D3.js/Cytoscape compatible).

    Node-link format is a standard for network visualization libraries.
    """
    nodes_list: list[dict[str, Any]] = []
    links_list: list[dict[str, Any]] = []

    # Export nodes
    for node in graph.get_nodes():
        node_dict: dict[str, Any] = {
            "id": node.node_id,
            "status": node.status.value,
            "model": node.model,
        }

        if include_metadata:
            node_dict["metadata"] = node.metadata
            node_dict["created_at"] = node.created_at.isoformat()

        nodes_list.append(node_dict)

    # Export edges
    for edge in graph._edges.values():
        link_dict: dict[str, Any] = {
            "source": edge.from_node,
            "target": edge.to_node,
            "directed": edge.directed,
        }

        if include_metadata:
            link_dict["properties"] = edge.properties
            link_dict["created_at"] = edge.created_at.isoformat()

        if include_message_counts:
            # Message count would be populated here if available
            link_dict["message_count"] = 0

        links_list.append(link_dict)

    return {
        "nodes": nodes_list,
        "links": links_list,
        "graph_info": {
            "name": graph.name,
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "timestamp": datetime.now().isoformat(),
        },
    }


def _export_json_adjacency(
    graph: "AgentGraph",  # type: ignore
    include_metadata: bool,
) -> dict[str, Any]:
    """
    Export graph as adjacency JSON format.

    Alternative format providing adjacency list representation.
    """
    adjacency: dict[str, list[str]] = {}
    nodes_dict: dict[str, dict[str, Any]] = {}

    # Build nodes dictionary
    for node in graph.get_nodes():
        node_entry: dict[str, Any] = {
            "status": node.status.value,
            "model": node.model,
        }

        if include_metadata:
            node_entry["metadata"] = node.metadata
            node_entry["created_at"] = node.created_at.isoformat()

        nodes_dict[node.node_id] = node_entry

    # Build adjacency list
    for node_id in graph._nodes:
        adjacency[node_id] = []

    for edge in graph._edges.values():
        adjacency[edge.from_node].append(edge.to_node)
        if not edge.directed and edge.from_node != edge.to_node:
            adjacency[edge.to_node].append(edge.from_node)

    return {
        "nodes": nodes_dict,
        "adjacency": adjacency,
        "graph_info": {
            "name": graph.name,
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "timestamp": datetime.now().isoformat(),
        },
    }


def _format_node_label(node: Node, include_metadata: bool) -> str:
    """Format node label for DOT output."""
    label = f"{node.node_id}\\n({node.status.value})"

    if include_metadata and node.metadata:
        # Add first metadata item if available
        for key, value in list(node.metadata.items())[:1]:
            label += f"\\n{key}={value}"

    return label


def _get_node_color(node: Node, node_style: str) -> str:
    """Get node color based on style preference."""
    if node_style == "status":
        status_colors = {
            "initializing": "yellow",
            "active": "lightgreen",
            "stopped": "lightgray",
            "error": "lightcoral",
        }
        return status_colors.get(node.status.value, "lightblue")
    elif node_style == "utilization":
        # Placeholder for utilization-based coloring
        return "lightblue"
    else:
        return "lightblue"


def _format_edge_label(edge: Edge, include_message_counts: bool) -> str:
    """Format edge label for DOT output."""
    if include_message_counts:
        # Message count would be populated here if available
        return ""

    # Include edge type or other relevant info
    if edge.properties:
        label_parts = []
        for key, value in list(edge.properties.items())[:1]:
            label_parts.append(f"{key}={value}")
        return ", ".join(label_parts) if label_parts else ""

    return ""
