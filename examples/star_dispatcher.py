#!/usr/bin/env python3
"""
Example: Star (Hub-and-Spoke) - Task Distribution

This example demonstrates a star topology with a central dispatcher
coordinating multiple worker agents for task distribution.
"""

import asyncio

from claude_agent_graph import AgentGraph
from claude_agent_graph.backends import FilesystemBackend
from claude_agent_graph.topology import GraphTopology


async def main():
    """Create and demonstrate a star topology for task distribution."""

    async with AgentGraph(
        name="task_dispatcher",
        storage_backend=FilesystemBackend(base_dir="./conversations/task_dispatcher"),
    ) as graph:
        print("Building task dispatcher system...")

        # Create central dispatcher (hub)
        await graph.add_node(
            "dispatcher",
            "You distribute tasks to available workers based on their load and specialization.",
            model="claude-sonnet-4-20250514",
        )

        # Create worker agents (spokes)
        num_workers = 5
        for i in range(num_workers):
            await graph.add_node(
                f"worker_{i}",
                f"You are worker {i}. Execute assigned tasks efficiently.",
            )

            # Connect workers TO dispatcher (star topology has edges pointing to center)
            # Note: edges FROM center would create a tree topology instead
            await graph.add_edge(f"worker_{i}", "dispatcher", directed=True)

        # Verify topology
        topology = graph.get_topology()
        print(f"✓ Graph topology: {topology}")
        assert topology == GraphTopology.STAR, f"Expected STAR, got {topology}"

        # Display system info
        print("\nSystem Structure:")
        print("  Hub: dispatcher")
        print(f"  Spokes: {num_workers} workers")
        print(f"  Total nodes: {graph.node_count}")

        # Show connections
        workers = graph.get_neighbors("dispatcher", direction="incoming")
        print(f"\nWorkers reporting to dispatcher: {', '.join(workers)}")

        # Simulate task distribution
        print("\nDistributing tasks:")

        tasks = [
            ("worker_0", "Completed: customer orders processing"),
            ("worker_1", "Completed: daily sales report"),
            ("worker_2", "Completed: inventory database update"),
            ("worker_3", "Completed: email notifications sent"),
            ("worker_4", "Completed: data backup"),
        ]

        # Workers report completion to dispatcher
        for worker_id, status in tasks:
            await graph.send_message(worker_id, "dispatcher", status)
            print(f"  {worker_id} → dispatcher: {status}")

        # Check message counts
        print("\nMessage counts:")
        for worker_id in workers:
            messages = await graph.get_conversation(worker_id, "dispatcher")
            print(f"  {worker_id} → dispatcher: {len(messages)} messages")

        total_messages = sum(len(await graph.get_conversation(w, "dispatcher")) for w in workers)
        print(f"\n✓ Total status reports received: {total_messages}")


if __name__ == "__main__":
    asyncio.run(main())
