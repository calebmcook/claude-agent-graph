#!/usr/bin/env python3
"""
Example: Collaborative Network - Research Team

This example demonstrates a mesh topology where multiple agents collaborate
on a research project. Shows broadcast messaging and multi-agent coordination.

Key concepts covered:
- Mesh topology (partially connected network)
- Broadcast messaging to multiple agents
- Collaborative workflows
- Undirected edges for peer relationships

Run with:
    python examples/collaborative_network.py

Requirements:
    - ANTHROPIC_API_KEY environment variable set
"""

import asyncio

from claude_agent_graph import AgentGraph
from claude_agent_graph.backends import FilesystemBackend


async def main():
    """Demonstrate a collaborative research network."""

    print("=" * 70)
    print("COLLABORATIVE NETWORK: Research Team")
    print("=" * 70)

    # Create graph with filesystem storage
    async with AgentGraph(
        name="research_team",
        storage_backend=FilesystemBackend(base_dir="./conversations/research_team"),
    ) as graph:
        # ==========================================
        # STEP 1: Build the Research Team
        # ==========================================
        print("\n1. Building research team network...")

        # Add team coordinator (connects to everyone)
        await graph.add_node(
            "coordinator",
            "You are the research coordinator. Facilitate collaboration between team members and synthesize findings.",
            metadata={"role": "lead", "department": "research"},
        )

        # Add specialized researchers
        await graph.add_node(
            "data_analyst",
            "You are a data analyst. Analyze datasets and provide statistical insights.",
            metadata={"role": "analyst", "specialization": "statistics"},
        )

        await graph.add_node(
            "ml_researcher",
            "You are a machine learning researcher. Develop and evaluate ML models.",
            metadata={"role": "researcher", "specialization": "ml"},
        )

        await graph.add_node(
            "report_writer",
            "You are a technical writer. Synthesize research findings into clear reports.",
            metadata={"role": "writer", "specialization": "documentation"},
        )

        # Create mesh topology with coordinator as hub
        # Coordinator connects to everyone (directed edges for coordination)
        await graph.add_edge("coordinator", "data_analyst", directed=True)
        await graph.add_edge("coordinator", "ml_researcher", directed=True)
        await graph.add_edge("coordinator", "report_writer", directed=True)

        # Peer relationships (undirected edges for collaboration)
        await graph.add_edge("data_analyst", "ml_researcher", directed=False)
        await graph.add_edge("ml_researcher", "report_writer", directed=False)

        print(f"✓ Created network with {graph.node_count} nodes, {graph.edge_count} edges")
        print(f"✓ Topology: {graph.get_topology()}")

        # ==========================================
        # STEP 2: Show Network Structure
        # ==========================================
        print("\n2. Network structure:")

        for node in graph.get_nodes():
            neighbors = graph.get_neighbors(node.node_id, direction="both")
            print(f"  {node.node_id}:")
            print(f"    - Reports to: {graph.get_controllers(node.node_id)}")
            print(f"    - Collaborates with: {neighbors}")

        # ==========================================
        # STEP 3: Broadcast Message from Coordinator
        # ==========================================
        print("\n3. Coordinator broadcasts research task to team...")

        messages = await graph.broadcast(
            "coordinator",
            "New project: Analyze customer churn patterns. Data analyst: prepare the dataset. "
            "ML researcher: build predictive models. Writer: prepare for documentation.",
            metadata={"project": "churn_analysis", "priority": "high"},
        )

        print(f"✓ Broadcast sent to {len(messages)} team members")
        for msg in messages:
            print(f"  → {msg.to_node}")

        # ==========================================
        # STEP 4: Peer Collaboration
        # ==========================================
        print("\n4. Team members collaborate...")

        # Data analyst shares findings with ML researcher
        await graph.send_message(
            "data_analyst",
            "ml_researcher",
            "Dataset prepared: 10K records, 15 features. Key correlations found in usage patterns and support tickets.",
            metadata={"dataset_id": "churn_v1"},
        )
        print("  ✓ Data analyst → ML researcher: Dataset insights shared")

        # ML researcher responds with model results
        await graph.send_message(
            "ml_researcher",
            "data_analyst",
            "Model trained: 89% accuracy. Top predictive features are usage frequency and support ticket count.",
            metadata={"model_id": "churn_model_v1"},
        )
        print("  ✓ ML researcher → Data analyst: Model results shared")

        # ML researcher shares with writer
        await graph.send_message(
            "ml_researcher",
            "report_writer",
            "Model performance: 89% accuracy, 0.91 F1 score. Key insights ready for documentation.",
            metadata={"model_id": "churn_model_v1"},
        )
        print("  ✓ ML researcher → Writer: Results ready for documentation")

        # ==========================================
        # STEP 5: Report Back to Coordinator
        # ==========================================
        print("\n5. Report writer sends summary to coordinator...")

        await graph.send_message(
            "report_writer",
            "coordinator",
            "Project completed: Churn prediction model achieves 89% accuracy. "
            "Top factors are usage frequency and support tickets. Full report ready for review.",
            metadata={"report_status": "draft_complete"},
        )
        print("  ✓ Report submitted to coordinator")

        # ==========================================
        # STEP 6: Review Conversations
        # ==========================================
        print("\n6. Conversation summary:")

        # Check coordinator's broadcast
        coord_to_analyst = await graph.get_conversation("coordinator", "data_analyst")
        print(f"  Coordinator → Data Analyst: {len(coord_to_analyst)} messages")

        # Check peer collaboration
        analyst_to_ml = await graph.get_conversation("data_analyst", "ml_researcher")
        ml_to_analyst = await graph.get_conversation("ml_researcher", "data_analyst")
        print(
            f"  Data Analyst ↔ ML Researcher: {len(analyst_to_ml) + len(ml_to_analyst)} messages (bidirectional)"
        )

        # Check final report
        writer_to_coord = await graph.get_conversation("report_writer", "coordinator")
        print(f"  Writer → Coordinator: {len(writer_to_coord)} messages")

        # ==========================================
        # STEP 7: Network Metrics
        # ==========================================
        print("\n7. Network metrics:")
        print(f"  Total nodes: {graph.node_count}")
        print(f"  Total edges: {graph.edge_count}")
        print(f"  Topology: {graph.get_topology()}")

        # Count total messages across all edges
        total_messages = 0
        for edge in graph.get_edges():
            messages = await graph.get_conversation(edge.from_node, edge.to_node)
            total_messages += len(messages)
        print(f"  Total messages exchanged: {total_messages}")

        print("\n" + "=" * 70)
        print("✓ Collaborative network example completed successfully!")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
