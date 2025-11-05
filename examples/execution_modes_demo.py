#!/usr/bin/env python3
"""
Example: Execution Modes - Manual, Reactive, and Proactive

This example demonstrates all three execution modes and shows how agents
behave differently under each mode.

Key concepts covered:
- Manual execution mode (step-by-step control)
- Reactive execution mode (message-driven)
- Proactive execution mode (periodic activation)
- Switching between execution modes
- Message queue behavior

Run with:
    python examples/execution_modes_demo.py

Requirements:
    - ANTHROPIC_API_KEY environment variable set
"""

import asyncio
from claude_agent_graph import AgentGraph
from claude_agent_graph.execution import ManualController, ReactiveExecutor


async def main():
    """Demonstrate all three execution modes."""

    print("=" * 70)
    print("EXECUTION MODES DEMO")
    print("=" * 70)

    # Create a simple chain graph for demo
    async with AgentGraph(name="execution_demo") as graph:

        # ==========================================
        # SETUP: Create Simple Chain
        # ==========================================
        print("\n0. Setting up agent chain...")

        await graph.add_node(
            "input_processor",
            "You are an input processor. Extract key information from requests.",
        )

        await graph.add_node(
            "data_analyzer", "You are a data analyzer. Analyze the extracted information."
        )

        await graph.add_node(
            "output_formatter",
            "You are an output formatter. Format analysis results for presentation.",
        )

        # Create chain: input -> analyzer -> output
        await graph.add_edge("input_processor", "data_analyzer", directed=True)
        await graph.add_edge("data_analyzer", "output_formatter", directed=True)

        print(f"✓ Created chain with {graph.node_count} nodes")

        # ==========================================
        # MODE 1: Manual Execution
        # ==========================================
        print("\n" + "=" * 70)
        print("MODE 1: MANUAL EXECUTION (Step-by-Step Control)")
        print("=" * 70)

        print("\n1. Starting manual execution mode...")
        await graph.start(mode="manual")
        print("✓ Manual mode active")

        # Send a message (it gets queued, not processed)
        print("\n2. Sending message to input processor...")
        await graph.send_message(
            "input_processor",
            "data_analyzer",
            "Analyze Q3 sales: revenue $1.2M, costs $800K, profit $400K",
        )
        print("✓ Message queued (not processed yet)")

        # Manually step through execution
        print("\n3. Manually stepping through agents...")

        controller = ManualController(graph)

        # Step 1: Process input processor
        print("  Stepping: input_processor...")
        await controller.step("input_processor")
        print("  ✓ Input processor executed")

        # Step 2: Process data analyzer
        print("  Stepping: data_analyzer...")
        await controller.step("data_analyzer")
        print("  ✓ Data analyzer executed")

        # Step 3: Process output formatter
        print("  Stepping: output_formatter...")
        await controller.step("output_formatter")
        print("  ✓ Output formatter executed")

        print("\n✓ Manual execution complete")

        # Stop manual mode
        await graph.stop_execution()
        print("✓ Manual mode stopped")

        # ==========================================
        # MODE 2: Reactive Execution
        # ==========================================
        print("\n" + "=" * 70)
        print("MODE 2: REACTIVE EXECUTION (Automatic Message Processing)")
        print("=" * 70)

        print("\n1. Starting reactive execution mode...")
        await graph.start(mode="reactive")
        print("✓ Reactive mode active")

        print("\n2. Sending message to input processor...")
        await graph.send_message(
            "input_processor",
            "data_analyzer",
            "Analyze Q4 forecast: expected revenue $1.5M, projected costs $900K",
        )
        print("✓ Message sent - agents will process automatically")

        # Give reactive mode time to process
        print("\n3. Waiting for reactive processing...")
        await asyncio.sleep(2)
        print("✓ Agents processed messages automatically")

        # Send another message to demonstrate concurrent processing
        print("\n4. Sending second message...")
        await graph.send_message(
            "input_processor",
            "data_analyzer",
            "Quick analysis: inventory levels at 75%, reorder threshold at 80%",
        )
        print("✓ Second message sent")

        await asyncio.sleep(2)
        print("✓ Second message processed")

        # Stop reactive mode
        await graph.stop_execution()
        print("\n✓ Reactive mode stopped")

        # ==========================================
        # MODE 3: Proactive Execution (Demo)
        # ==========================================
        print("\n" + "=" * 70)
        print("MODE 3: PROACTIVE EXECUTION (Periodic Activation)")
        print("=" * 70)

        print("\n1. Starting proactive execution mode...")
        print("   (Agents activate every 10 seconds)")

        # Note: We use a very short interval for demo purposes
        # In production, you'd typically use 60+ seconds
        await graph.start(mode="proactive", interval=10.0)
        print("✓ Proactive mode active")

        print("\n2. Waiting for first activation cycle...")
        await asyncio.sleep(12)
        print("✓ First activation cycle complete")

        print("\n3. Agents can now initiate conversations periodically...")
        print("   (Stopping early for demo purposes)")

        # Stop proactive mode
        await graph.stop_execution()
        print("\n✓ Proactive mode stopped")

        # ==========================================
        # COMPARISON SUMMARY
        # ==========================================
        print("\n" + "=" * 70)
        print("EXECUTION MODE COMPARISON")
        print("=" * 70)

        print(
            """
Manual Mode:
  ✓ Full control over execution
  ✓ Step-by-step agent processing
  ✓ Useful for debugging and testing
  ✓ Call step() to process one agent at a time

Reactive Mode:
  ✓ Automatic message-driven processing
  ✓ Agents respond to incoming messages
  ✓ Ideal for message-driven workflows
  ✓ No manual intervention needed

Proactive Mode:
  ✓ Periodic agent activation
  ✓ Agents can initiate conversations
  ✓ Useful for monitoring and scheduled tasks
  ✓ Configurable activation interval
"""
        )

        # ==========================================
        # VIEW CONVERSATION HISTORY
        # ==========================================
        print("\n" + "=" * 70)
        print("CONVERSATION HISTORY")
        print("=" * 70)

        messages = await graph.get_conversation("input_processor", "data_analyzer")
        print(f"\nInput Processor → Data Analyzer: {len(messages)} messages")
        for i, msg in enumerate(messages[:3], 1):  # Show first 3
            preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
            print(f"  {i}. {preview}")

        messages = await graph.get_conversation("data_analyzer", "output_formatter")
        print(f"\nData Analyzer → Output Formatter: {len(messages)} messages")
        for i, msg in enumerate(messages[:3], 1):
            preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
            print(f"  {i}. {preview}")

        print("\n" + "=" * 70)
        print("✓ Execution modes demo completed successfully!")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
