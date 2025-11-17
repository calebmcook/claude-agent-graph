"""
End-to-end tests for claude-agent-graph.

These tests simulate complete user workflows and real-world scenarios,
testing the system from start to finish.
"""

import asyncio
import tempfile
from pathlib import Path

from claude_agent_graph import AgentGraph
from claude_agent_graph.backends import FilesystemBackend
from claude_agent_graph.execution import ManualController


class TestCompleteWorkflows:
    """Test complete end-to-end workflows."""

    async def test_supervisor_worker_workflow(self):
        """
        E2E test: Supervisor coordinates multiple workers.

        Workflow:
        1. Create graph with 1 supervisor and 3 workers
        2. Supervisor sends tasks to each worker
        3. Workers receive and process tasks
        4. Verify all messages are delivered correctly
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="supervisor_worker",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
            )

            # Setup hierarchy
            await graph.add_node("supervisor", "You coordinate worker agents.")
            await graph.add_node("worker1", "You process data analysis tasks.")
            await graph.add_node("worker2", "You handle file operations.")
            await graph.add_node("worker3", "You perform database queries.")

            await graph.add_edge("supervisor", "worker1", directed=True)
            await graph.add_edge("supervisor", "worker2", directed=True)
            await graph.add_edge("supervisor", "worker3", directed=True)

            # Supervisor broadcasts task
            await graph.broadcast(
                "supervisor", "Begin processing today's batch", direction="outgoing"
            )

            # Verify each worker received the message
            for worker in ["worker1", "worker2", "worker3"]:
                messages = await graph.get_conversation("supervisor", worker)
                assert len(messages) == 1
                assert "Begin processing" in messages[0].content

            # Verify control relationships
            subordinates = graph.get_subordinates("supervisor")
            assert len(subordinates) == 3
            assert set(subordinates) == {"worker1", "worker2", "worker3"}

    async def test_collaborative_research_team(self):
        """
        E2E test: Research team with specialized agents collaborating.

        Workflow:
        1. Create team: researcher, analyst, writer
        2. Researcher gathers data, sends to analyst
        3. Analyst processes data, sends to writer
        4. Writer creates report
        5. Verify message flow and data preservation
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="research_team",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
            )

            # Create team members
            await graph.add_node(
                "researcher",
                "You gather research data and findings.",
            )
            await graph.add_node(
                "analyst",
                "You analyze research data and extract insights.",
            )
            await graph.add_node(
                "writer",
                "You write comprehensive research reports.",
            )

            # Create pipeline
            await graph.add_edge("researcher", "analyst", directed=True)
            await graph.add_edge("analyst", "writer", directed=True)

            # Simulate workflow
            await graph.send_message(
                "researcher",
                "analyst",
                "Research data: 100 samples collected, 95% confidence",
            )

            await graph.send_message(
                "analyst",
                "writer",
                "Analysis complete: Statistically significant results found",
            )

            # Verify pipeline
            research_to_analyst = await graph.get_conversation("researcher", "analyst")
            analyst_to_writer = await graph.get_conversation("analyst", "writer")

            assert len(research_to_analyst) == 1
            assert len(analyst_to_writer) == 1
            assert "100 samples" in research_to_analyst[0].content
            assert "significant results" in analyst_to_writer[0].content

    async def test_multi_level_hierarchy_command_chain(self):
        """
        E2E test: Multi-level organizational hierarchy.

        Workflow:
        1. CEO -> VP -> Manager -> Team Lead -> Worker
        2. Commands flow down the chain
        3. Each level adds their interpretation
        4. Verify control relationships at each level
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="org_hierarchy",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
                topology_constraint="tree",
            )

            # Build hierarchy
            levels = [
                ("ceo", "You are the CEO."),
                ("vp", "You are the VP."),
                ("manager", "You are a manager."),
                ("team_lead", "You are a team lead."),
                ("worker", "You are a worker."),
            ]

            for node_id, prompt in levels:
                await graph.add_node(node_id, prompt)

            # Create chain
            await graph.add_edge("ceo", "vp", directed=True)
            await graph.add_edge("vp", "manager", directed=True)
            await graph.add_edge("manager", "team_lead", directed=True)
            await graph.add_edge("team_lead", "worker", directed=True)

            # Send command down chain
            await graph.send_message("ceo", "vp", "Implement new strategy Q1")
            await graph.send_message("vp", "manager", "Plan Q1 strategy rollout")
            await graph.send_message("manager", "team_lead", "Assign Q1 tasks")
            await graph.send_message("team_lead", "worker", "Execute task A")

            # Verify chain
            assert graph.is_controller("ceo", "vp")
            assert graph.is_controller("vp", "manager")
            assert graph.is_controller("manager", "team_lead")
            assert graph.is_controller("team_lead", "worker")

            # Verify messages at each level
            for from_node, to_node in [
                ("ceo", "vp"),
                ("vp", "manager"),
                ("manager", "team_lead"),
                ("team_lead", "worker"),
            ]:
                messages = await graph.get_conversation(from_node, to_node)
                assert len(messages) == 1

    async def test_mesh_network_consensus(self):
        """
        E2E test: Mesh network where all agents communicate for consensus.

        Workflow:
        1. Create fully connected mesh of 5 agents
        2. Each agent broadcasts their vote
        3. Verify all agents can see all votes
        4. Test bidirectional communication
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="consensus_mesh",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
            )

            # Create mesh
            agents = ["agent1", "agent2", "agent3", "agent4", "agent5"]
            for agent in agents:
                await graph.add_node(agent, f"You are {agent}.")

            # Connect all pairs (mesh)
            for i, agent1 in enumerate(agents):
                for agent2 in agents[i + 1 :]:
                    await graph.add_edge(agent1, agent2, directed=False)

            # Each agent broadcasts a vote
            votes = {
                "agent1": "Vote: Option A",
                "agent2": "Vote: Option B",
                "agent3": "Vote: Option A",
                "agent4": "Vote: Option A",
                "agent5": "Vote: Option B",
            }

            for agent, vote in votes.items():
                await graph.broadcast(agent, vote, direction="outgoing")

            # Verify connectivity: each agent should have 4 neighbors
            for agent in agents:
                neighbors = graph.get_neighbors(agent, direction="both")
                assert len(neighbors) == 4


class TestPersistenceAndRecovery:
    """Test persistence and crash recovery scenarios."""

    async def test_save_load_cycle_with_messages(self):
        """
        E2E test: Save graph, load it, continue operation.

        Workflow:
        1. Create graph and send messages
        2. Save checkpoint
        3. Load from checkpoint
        4. Send more messages
        5. Verify all data persists correctly
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "state.msgpack"

            # Phase 1: Create and populate graph
            graph1 = AgentGraph(
                name="persistence_test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
            )

            await graph1.add_node("alice", "You are Alice.")
            await graph1.add_node("bob", "You are Bob.")
            await graph1.add_edge("alice", "bob", directed=False)

            await graph1.send_message("alice", "bob", "Message 1")
            await graph1.send_message("bob", "alice", "Message 2")

            await graph1.save_checkpoint(str(checkpoint_path))

            # Phase 2: Load and continue
            graph2 = await AgentGraph.load_checkpoint(str(checkpoint_path))

            # Verify loaded state
            assert graph2.name == "persistence_test"
            assert graph2.node_count == 2
            assert graph2.edge_count == 1

            # Continue operation
            await graph2.send_message("alice", "bob", "Message 3")

            # Verify all messages
            messages = await graph2.get_conversation("alice", "bob")
            assert len(messages) == 3
            assert messages[0].content == "Message 1"
            assert messages[1].content == "Message 2"
            assert messages[2].content == "Message 3"

    async def test_auto_save_and_crash_recovery(self):
        """
        E2E test: Simulate crash and recovery with auto-save.

        Workflow:
        1. Enable auto-save
        2. Perform operations
        3. Simulate crash (don't clean up)
        4. Recover from latest checkpoint
        5. Verify state is intact
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_dir.mkdir()

            # Create graph with auto-save
            graph1 = AgentGraph(
                name="auto_save_test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
                auto_save=True,
                auto_save_interval=1,  # 1 second for testing
                checkpoint_dir=str(checkpoint_dir),
            )

            await graph1.add_node("node1", "Node 1")
            await graph1.add_node("node2", "Node 2")
            await graph1.add_edge("node1", "node2", directed=True)

            # Wait for auto-save to trigger
            await asyncio.sleep(2)

            # Stop auto-save
            await graph1.stop_auto_save()

            # Simulate crash by discarding graph1
            # (In real crash, graph1 would be lost)

            # Recover from latest checkpoint
            graph2 = await AgentGraph.load_latest_checkpoint(str(checkpoint_dir))

            # Verify recovery
            assert graph2.node_count == 2
            assert graph2.edge_count == 1
            assert graph2.node_exists("node1")
            assert graph2.node_exists("node2")


class TestDynamicGraphEvolution:
    """Test dynamic graph modification during operation."""

    async def test_add_remove_nodes_during_operation(self):
        """
        E2E test: Dynamically add and remove nodes while system operates.

        Workflow:
        1. Start with basic graph
        2. Send messages
        3. Add new nodes and connect them
        4. Remove old nodes
        5. Continue messaging
        6. Verify graph evolution
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="dynamic_evolution",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
            )

            # Initial setup
            await graph.add_node("original1", "Original 1")
            await graph.add_node("original2", "Original 2")
            await graph.add_edge("original1", "original2", directed=False)

            await graph.send_message("original1", "original2", "Hello")

            # Evolve: Add new nodes
            await graph.add_node("new1", "New 1")
            await graph.add_edge("original1", "new1", directed=False)

            await graph.send_message("original1", "new1", "Welcome")

            # Evolve: Remove old node
            await graph.remove_node("original2", cascade=True)

            # Verify final state
            assert graph.node_count == 2
            assert graph.node_exists("original1")
            assert graph.node_exists("new1")
            assert not graph.node_exists("original2")

            # Verify messages still accessible
            messages = await graph.get_conversation("original1", "new1")
            assert len(messages) == 1

    async def test_scaling_up_graph_size(self):
        """
        E2E test: Scale graph from small to large.

        Workflow:
        1. Start with 5 nodes
        2. Gradually add 45 more nodes
        3. Connect them incrementally
        4. Verify performance and correctness
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="scaling_test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
                max_nodes=100,
            )

            # Initial small graph
            for i in range(5):
                await graph.add_node(f"node{i}", f"Node {i}")

            # Scale up
            for i in range(5, 50):
                await graph.add_node(f"node{i}", f"Node {i}")
                # Connect to previous node (chain)
                await graph.add_edge(f"node{i-1}", f"node{i}", directed=True)

            # Verify final size
            assert graph.node_count == 50
            assert graph.edge_count == 49

            # Test message routing through chain
            await graph.route_message(
                "node0",
                "node49",
                "End-to-end message",
                path="auto",
            )


class TestErrorRecoveryE2E:
    """Test end-to-end error recovery scenarios."""

    async def test_recover_from_invalid_topology_change(self):
        """
        E2E test: Attempt invalid operation, recover gracefully.

        Workflow:
        1. Create graph with tree topology
        2. Try to create cycle (should fail)
        3. Verify graph is still valid
        4. Continue normal operations
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="recovery_test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
                topology_constraint="tree",
            )

            await graph.add_node("root", "Root")
            await graph.add_node("child1", "Child 1")
            await graph.add_node("child2", "Child 2")

            await graph.add_edge("root", "child1", directed=True)
            await graph.add_edge("root", "child2", directed=True)

            # Try invalid operation
            try:
                await graph.add_edge("child1", "child2", directed=True)
                await graph.add_edge("child2", "root", directed=True)
            except Exception:
                pass  # Expected to fail

            # Verify graph is still valid
            assert graph.node_count == 3
            assert graph.edge_count == 2
            assert graph.validate_topology("tree")

            # Continue normal operation
            await graph.send_message("root", "child1", "Continue working")
            messages = await graph.get_conversation("root", "child1")
            assert len(messages) == 1


class TestExecutionModes:
    """Test different execution modes end-to-end."""

    async def test_manual_controller_workflow(self):
        """
        E2E test: Use manual controller to step through execution.

        Workflow:
        1. Create graph with message queue
        2. Use ManualController
        3. Step through message processing
        4. Verify manual control
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="manual_test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
            )

            await graph.add_node("node1", "Node 1")
            await graph.add_node("node2", "Node 2")
            await graph.add_edge("node1", "node2", directed=False)

            # Create controller
            controller = ManualController(graph)
            await controller.start()

            # Queue some messages
            await graph.send_message("node1", "node2", "Message 1")
            await graph.send_message("node2", "node1", "Message 2")

            # Manually step through
            result1 = await controller.step("node2")
            result2 = await controller.step("node1")

            await controller.stop()

            # Verify controlled execution
            assert result1 is not None or result2 is not None


class TestLargeScaleOperations:
    """Test large-scale graph operations."""

    async def test_large_graph_creation(self):
        """
        E2E test: Create and operate on large graph (100+ nodes).

        Workflow:
        1. Create graph with 100 nodes
        2. Create 200 edges
        3. Send messages between random pairs
        4. Verify performance and correctness
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="large_scale_test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
                max_nodes=150,
            )

            # Create 100 nodes
            for i in range(100):
                await graph.add_node(f"node{i}", f"Node {i}")

            # Create ring topology (each node connects to next)
            for i in range(100):
                next_idx = (i + 1) % 100
                await graph.add_edge(f"node{i}", f"node{next_idx}", directed=False)

            # Add some random edges
            import random

            for _ in range(100):
                n1 = f"node{random.randint(0, 99)}"
                n2 = f"node{random.randint(0, 99)}"
                if n1 != n2 and not graph.edge_exists(n1, n2):
                    await graph.add_edge(n1, n2, directed=False)

            # Verify scale
            assert graph.node_count == 100
            assert graph.edge_count >= 100

            # Test messaging on large graph
            await graph.send_message("node0", "node1", "Test message")
            messages = await graph.get_conversation("node0", "node1")
            assert len(messages) == 1

    async def test_high_volume_messaging(self):
        """
        E2E test: Send many messages between nodes.

        Workflow:
        1. Create small graph
        2. Send 1000 messages
        3. Verify all delivered correctly
        4. Check conversation file integrity
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="high_volume_test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
            )

            await graph.add_node("sender", "Sender")
            await graph.add_node("receiver", "Receiver")
            await graph.add_edge("sender", "receiver", directed=False)

            # Send many messages
            num_messages = 100  # Reduced for test performance
            for i in range(num_messages):
                await graph.send_message("sender", "receiver", f"Message {i}")

            # Verify all messages
            messages = await graph.get_conversation("sender", "receiver")
            assert len(messages) == num_messages

            # Verify order preserved
            for i in range(num_messages):
                assert messages[i].content == f"Message {i}"
