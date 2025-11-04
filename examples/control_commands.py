#!/usr/bin/env python3
"""
Example: Control Commands - Hierarchical Task Delegation

This example demonstrates control relationships and command execution
in a hierarchical agent network.

Key concepts covered:
- Control relationships via directed edges
- execute_command() for authorized commands
- Command authorization enforcement
- Hierarchical task delegation
- System prompt injection for control

Run with:
    python examples/control_commands.py

Requirements:
    - ANTHROPIC_API_KEY environment variable set
"""

import asyncio
from claude_agent_graph import AgentGraph
from claude_agent_graph.exceptions import CommandAuthorizationError
from claude_agent_graph.topology import GraphTopology


async def main():
    """Demonstrate control commands in a hierarchy."""

    print("=" * 70)
    print("CONTROL COMMANDS: Hierarchical Task Delegation")
    print("=" * 70)

    async with AgentGraph(name="command_demo") as graph:

        # ==========================================
        # STEP 1: Build Hierarchy
        # ==========================================
        print("\n1. Building organizational hierarchy...")

        # Add manager level
        await graph.add_node(
            "project_manager",
            "You are a project manager. Delegate tasks to team leads and track progress.",
            metadata={"level": "manager", "department": "engineering"},
        )

        # Add team lead level
        await graph.add_node(
            "backend_lead",
            "You are the backend team lead. Coordinate backend development tasks.",
            metadata={"level": "lead", "specialization": "backend"},
        )

        await graph.add_node(
            "frontend_lead",
            "You are the frontend team lead. Coordinate frontend development tasks.",
            metadata={"level": "lead", "specialization": "frontend"},
        )

        # Add developer level
        await graph.add_node(
            "backend_dev",
            "You are a backend developer. Implement server-side features.",
            metadata={"level": "developer", "specialization": "backend"},
        )

        await graph.add_node(
            "frontend_dev",
            "You are a frontend developer. Implement user interface features.",
            metadata={"level": "developer", "specialization": "frontend"},
        )

        # Create hierarchy with directed edges (establishes control relationships)
        await graph.add_edge("project_manager", "backend_lead", directed=True)
        await graph.add_edge("project_manager", "frontend_lead", directed=True)
        await graph.add_edge("backend_lead", "backend_dev", directed=True)
        await graph.add_edge("frontend_lead", "frontend_dev", directed=True)

        print(f"✓ Created hierarchy with {graph.node_count} nodes")
        print(f"✓ Topology: {graph.get_topology()}")

        # ==========================================
        # STEP 2: Show Control Relationships
        # ==========================================
        print("\n2. Control relationships:")

        for node in graph.get_nodes():
            controllers = graph.get_controllers(node.node_id)
            subordinates = graph.get_subordinates(node.node_id)

            print(f"\n  {node.node_id}:")
            if controllers:
                print(f"    Reports to: {controllers}")
            else:
                print(f"    Reports to: (none - top level)")

            if subordinates:
                print(f"    Manages: {subordinates}")
            else:
                print(f"    Manages: (none - individual contributor)")

            # Check if system prompt was modified
            if node.effective_system_prompt != node.original_system_prompt:
                print(f"    ✓ System prompt includes controller information")

        # ==========================================
        # STEP 3: Manager Issues Commands to Leads
        # ==========================================
        print("\n3. Project manager delegates tasks to team leads...")

        # Command to backend lead
        msg1 = await graph.execute_command(
            controller="project_manager",
            subordinate="backend_lead",
            command="implement_api",
            feature="user_authentication",
            priority="high",
            deadline="2025-11-15",
        )
        print(f"✓ Command sent to backend_lead: {msg1.message_id}")
        print(f"  Command: implement_api")
        print(f"  Parameters: feature=user_authentication, priority=high")

        # Command to frontend lead
        msg2 = await graph.execute_command(
            controller="project_manager",
            subordinate="frontend_lead",
            command="build_ui",
            feature="login_page",
            priority="high",
            deadline="2025-11-15",
        )
        print(f"✓ Command sent to frontend_lead: {msg2.message_id}")
        print(f"  Command: build_ui")
        print(f"  Parameters: feature=login_page, priority=high")

        # ==========================================
        # STEP 4: Leads Issue Commands to Developers
        # ==========================================
        print("\n4. Team leads delegate to developers...")

        # Backend lead to developer
        msg3 = await graph.execute_command(
            controller="backend_lead",
            subordinate="backend_dev",
            command="implement_endpoint",
            endpoint="/auth/login",
            method="POST",
            auth_required=False,
        )
        print(f"✓ Backend lead → Backend dev: implement_endpoint")

        # Frontend lead to developer
        msg4 = await graph.execute_command(
            controller="frontend_lead",
            subordinate="frontend_dev",
            command="create_component",
            component="LoginForm",
            framework="React",
        )
        print(f"✓ Frontend lead → Frontend dev: create_component")

        # ==========================================
        # STEP 5: Demonstrate Authorization Enforcement
        # ==========================================
        print("\n5. Testing command authorization...")

        print("\n  a) Valid command (manager to lead): ", end="")
        try:
            await graph.execute_command(
                controller="project_manager",
                subordinate="backend_lead",
                command="status_update",
            )
            print("✓ ALLOWED")
        except CommandAuthorizationError:
            print("✗ DENIED")

        print("  b) Invalid command (developer to manager): ", end="")
        try:
            await graph.execute_command(
                controller="backend_dev",
                subordinate="project_manager",
                command="unauthorized_command",
            )
            print("✗ ALLOWED (should have been denied!)")
        except CommandAuthorizationError:
            print("✓ DENIED (as expected)")

        print("  c) Invalid command (peer to peer): ", end="")
        try:
            await graph.execute_command(
                controller="backend_lead",
                subordinate="frontend_lead",
                command="cross_team_command",
            )
            print("✗ ALLOWED (should have been denied!)")
        except CommandAuthorizationError:
            print("✓ DENIED (as expected)")

        print("  d) Invalid command (no relationship): ", end="")
        try:
            await graph.execute_command(
                controller="project_manager",
                subordinate="backend_dev",  # Skips hierarchy level
                command="skip_level_command",
            )
            print("✗ ALLOWED (should have been denied!)")
        except CommandAuthorizationError:
            print("✓ DENIED (as expected)")

        # ==========================================
        # STEP 6: View Command History
        # ==========================================
        print("\n6. Command history:")

        # Get messages from manager to backend lead
        messages = await graph.get_conversation("project_manager", "backend_lead")
        print(f"\n  Project Manager → Backend Lead: {len(messages)} commands")
        for msg in messages:
            if msg.metadata.get("type") == "command":
                cmd = msg.metadata.get("command", "unknown")
                params = msg.metadata.get("params", {})
                print(f"    - {cmd}: {params}")

        # Get messages from backend lead to developer
        messages = await graph.get_conversation("backend_lead", "backend_dev")
        print(f"\n  Backend Lead → Backend Dev: {len(messages)} commands")
        for msg in messages:
            if msg.metadata.get("type") == "command":
                cmd = msg.metadata.get("command", "unknown")
                params = msg.metadata.get("params", {})
                print(f"    - {cmd}: {params}")

        # ==========================================
        # STEP 7: Verify System Prompt Injection
        # ==========================================
        print("\n7. System prompt verification:")

        backend_dev_node = graph.get_node("backend_dev")
        print(f"\n  Backend Developer original prompt:")
        print(f"    {backend_dev_node.original_system_prompt[:80]}...")

        print(f"\n  Backend Developer effective prompt (with controller info):")
        print(f"    {backend_dev_node.effective_system_prompt[:150]}...")

        # Check that controller info is present
        if "backend_lead" in backend_dev_node.effective_system_prompt:
            print(f"\n  ✓ Controller information successfully injected into prompt")
        else:
            print(f"\n  ✗ Warning: Controller information not found in prompt")

        print("\n" + "=" * 70)
        print("✓ Control commands example completed successfully!")
        print("=" * 70)

        print(
            """
Key Takeaways:
- Directed edges establish control relationships
- Subordinates' system prompts include controller information
- execute_command() enforces authorization automatically
- Only direct controllers can issue commands
- Commands are logged as special messages with metadata
"""
        )


if __name__ == "__main__":
    asyncio.run(main())
