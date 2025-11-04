#!/usr/bin/env python3
"""
Example: Tree/Hierarchy - Organizational Structure

This example demonstrates creating a tree-structured graph representing
a corporate organizational hierarchy with CEO, VPs, managers, and workers.
"""

import asyncio
from claude_agent_graph import AgentGraph
from claude_agent_graph.backends import FilesystemBackend
from claude_agent_graph.topology import GraphTopology


async def main():
    """Create and demonstrate a tree hierarchy of agents."""

    # Create graph with filesystem storage
    async with AgentGraph(
        name="org_hierarchy",
        storage_backend=FilesystemBackend(base_dir="./conversations/org_hierarchy"),
    ) as graph:
        print("Building organizational hierarchy...")

        # Top level - CEO
        await graph.add_node(
            "ceo",
            "You are the CEO. Set company vision and coordinate VPs.",
            model="claude-sonnet-4-20250514",
        )

        # VP level
        await graph.add_node(
            "vp_engineering", "You are VP of Engineering. Lead the engineering team."
        )
        await graph.add_node("vp_sales", "You are VP of Sales. Lead the sales team.")

        # Manager level
        await graph.add_node(
            "eng_manager", "You are an Engineering Manager. Manage software engineers."
        )
        await graph.add_node(
            "sales_manager", "You are a Sales Manager. Manage sales representatives."
        )

        # Worker level
        await graph.add_node("engineer_1", "You are a Software Engineer. Write code.")
        await graph.add_node(
            "sales_rep_1", "You are a Sales Representative. Handle customer accounts."
        )

        # Build hierarchy with directed edges (control relationships)
        await graph.add_edge("ceo", "vp_engineering", directed=True)
        await graph.add_edge("ceo", "vp_sales", directed=True)
        await graph.add_edge("vp_engineering", "eng_manager", directed=True)
        await graph.add_edge("vp_sales", "sales_manager", directed=True)
        await graph.add_edge("eng_manager", "engineer_1", directed=True)
        await graph.add_edge("sales_manager", "sales_rep_1", directed=True)

        # Verify topology
        topology = graph.get_topology()
        print(f"✓ Graph topology: {topology}")
        assert topology == GraphTopology.TREE, f"Expected TREE, got {topology}"

        # Display structure
        print(f"\nOrganizational Structure:")
        print(f"  Nodes: {graph.node_count}")
        print(f"  Edges: {graph.edge_count}")

        # Show control relationships
        print(f"\nControl Relationships:")
        for node_id in graph.get_nodes():
            controllers = graph.get_controllers(node_id.node_id)
            subordinates = graph.get_subordinates(node_id.node_id)
            if controllers:
                print(f"  {node_id.node_id} reports to: {', '.join(controllers)}")
            if subordinates:
                print(f"  {node_id.node_id} manages: {', '.join(subordinates)}")

        # Example message flow: CEO -> VP -> Manager -> Worker
        print(f"\nExample message flow:")
        await graph.send_message(
            "ceo",
            "vp_engineering",
            "Please prioritize the new feature roadmap for Q1.",
        )
        print("  CEO → VP Engineering: Task delegation sent")

        await graph.send_message(
            "vp_engineering",
            "eng_manager",
            "The CEO wants us to prioritize Q1 features. Please coordinate with your team.",
        )
        print("  VP Engineering → Engineering Manager: Task delegated")

        await graph.send_message(
            "eng_manager",
            "engineer_1",
            "We need to implement the Q1 features. Can you start with the authentication module?",
        )
        print("  Engineering Manager → Engineer: Specific task assigned")

        # Get conversation history
        messages = await graph.get_conversation("eng_manager", "engineer_1")
        print(f"\n✓ Conversation recorded: {len(messages)} messages")


if __name__ == "__main__":
    asyncio.run(main())
