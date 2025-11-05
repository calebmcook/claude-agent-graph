#!/usr/bin/env python3
"""
Example: DAG Pipeline - Sequential Data Processing

This example demonstrates a Directed Acyclic Graph (DAG) for a data processing
pipeline with stages: Ingest → Transform → Validate → Store
"""

import asyncio
from claude_agent_graph import AgentGraph
from claude_agent_graph.backends import FilesystemBackend
from claude_agent_graph.topology import GraphTopology


async def main():
    """Create and demonstrate a DAG data processing pipeline."""

    async with AgentGraph(
        name="data_pipeline",
        storage_backend=FilesystemBackend(base_dir="./conversations/data_pipeline"),
    ) as graph:
        print("Building data processing pipeline...")

        # Create pipeline stages
        await graph.add_node(
            "ingester",
            "You ingest raw data from various sources (APIs, databases, files).",
            model="claude-sonnet-4-20250514",
        )

        await graph.add_node(
            "transformer",
            "You clean and transform data into standardized formats.",
        )

        await graph.add_node(
            "validator",
            "You validate data quality, check for errors, and ensure constraints are met.",
        )

        await graph.add_node(
            "storage", "You store validated data in the warehouse and update indexes."
        )

        # Create DAG edges (directed, no cycles)
        await graph.add_edge("ingester", "transformer", directed=True)
        await graph.add_edge("transformer", "validator", directed=True)
        await graph.add_edge("validator", "storage", directed=True)

        # Verify topology (linear pipeline is detected as CHAIN, which is a type of DAG)
        topology = graph.get_topology()
        print(f"✓ Graph topology: {topology}")
        assert topology in (GraphTopology.DAG, GraphTopology.CHAIN), f"Expected DAG or CHAIN, got {topology}"

        # Display pipeline info
        print(f"\nPipeline Structure:")
        print(f"  Stages: {graph.node_count}")
        print(f"  Connections: {graph.edge_count}")

        # Show data flow
        print(f"\nData Flow:")
        nodes = ["ingester", "transformer", "validator", "storage"]
        for i, node in enumerate(nodes):
            next_nodes = graph.get_neighbors(node, direction="outgoing")
            if next_nodes:
                print(f"  {node} → {next_nodes[0]}")
            else:
                print(f"  {node} (final stage)")

        # Simulate data batch processing
        print(f"\nProcessing data batch:")

        await graph.send_message(
            "ingester",
            "transformer",
            "New data batch: batch_20251104_001.csv (1000 records)",
        )
        print("  Stage 1: Data ingested")

        await graph.send_message(
            "transformer",
            "validator",
            "Transformed batch_20251104_001: standardized 1000 records, removed 15 duplicates",
        )
        print("  Stage 2: Data transformed")

        await graph.send_message(
            "validator",
            "storage",
            "Validated batch_20251104_001: 985 records passed quality checks, ready for storage",
        )
        print("  Stage 3: Data validated")

        # Get full pipeline conversation
        print(f"\n✓ Pipeline execution complete")
        total_messages = 0
        for from_node in nodes[:-1]:
            to_node = graph.get_neighbors(from_node, direction="outgoing")[0]
            messages = await graph.get_conversation(from_node, to_node)
            total_messages += len(messages)

        print(f"✓ Total messages in pipeline: {total_messages}")


if __name__ == "__main__":
    asyncio.run(main())
