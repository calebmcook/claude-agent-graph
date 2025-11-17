"""
Interactive Demo: Collaborative Problem Solving with Agent Graph

This demo shows:
1. A supervisor delegating a problem to 4 research agents
2. Agents collaborating and discussing solutions
3. Real-time graph visualization with animations
4. Metrics tracking as the conversation evolves

To run:
    python3 demo_interactive.py

This will:
    1. Create an HTML visualization file (open in browser)
    2. Show metrics in the terminal
    3. Simulate agent discussions with message exchanges
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from claude_agent_graph import AgentGraph, export_json


class AnimatedGraphVisualizer:
    """Creates an interactive HTML visualization with animations."""

    def __init__(self, output_path: str = "graph_demo.html"):
        self.output_path = output_path
        self.frames: list[dict[str, Any]] = []

    def add_frame(
        self,
        graph_data: dict[str, Any],
        title: str,
        message_exchange: tuple[str, str] | None = None,
        thinking_nodes: list[str] | None = None,
    ) -> None:
        """Add a frame to the animation."""
        frame = {
            "title": title,
            "graph": graph_data,
            "message_exchange": message_exchange,
            "thinking_nodes": thinking_nodes or [],
        }
        self.frames.append(frame)

    def generate_html(self) -> None:
        """Generate the HTML visualization."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Agent Graph Demo - Collaborative Problem Solving</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        #container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
        }
        .controls {
            margin: 20px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }
        button {
            padding: 10px 20px;
            margin-right: 10px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover {
            background-color: #0056b3;
        }
        button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        #visualization {
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #fafafa;
            min-height: 600px;
        }
        svg {
            width: 100%;
            height: 600px;
        }
        .node {
            stroke: #333;
            stroke-width: 2px;
            cursor: pointer;
        }
        .node.thinking {
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.6; }
            100% { opacity: 1; }
        }
        .node.active {
            stroke-width: 4px;
            stroke: #ff6b6b;
        }
        .link {
            stroke: #999;
            stroke-opacity: 0.6;
            stroke-width: 2px;
        }
        .link.active {
            stroke: #ff6b6b;
            stroke-width: 4px;
            animation: flash 0.5s;
        }
        @keyframes flash {
            0%, 100% { stroke-opacity: 1; }
            50% { stroke-opacity: 0.3; }
        }
        .node-label {
            font-size: 12px;
            pointer-events: none;
            text-anchor: middle;
            dy: ".3em";
        }
        .metrics {
            margin-top: 30px;
            padding: 15px;
            background-color: #e8f4f8;
            border-left: 4px solid #007bff;
            border-radius: 5px;
        }
        .metrics h3 {
            margin-top: 0;
            color: #007bff;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 10px;
        }
        .metric-item {
            padding: 10px;
            background-color: white;
            border-radius: 5px;
            border-left: 3px solid #007bff;
        }
        .metric-label {
            font-weight: bold;
            color: #555;
            font-size: 12px;
        }
        .metric-value {
            font-size: 18px;
            color: #007bff;
            margin-top: 5px;
        }
        .frame-info {
            padding: 10px;
            background-color: #fff3cd;
            border-radius: 5px;
            margin-bottom: 15px;
            border-left: 4px solid #ffc107;
        }
        .frame-info h3 {
            margin: 0 0 5px 0;
            color: #856404;
        }
        .frame-info p {
            margin: 0;
            color: #856404;
            font-size: 14px;
        }
        .message-exchange {
            color: #333;
            font-style: italic;
            margin-top: 5px;
        }
        .slider-container {
            margin: 15px 0;
        }
        #frameSlider {
            width: 100%;
            height: 5px;
            cursor: pointer;
        }
        .frame-counter {
            text-align: center;
            margin: 10px 0;
            font-weight: bold;
            color: #555;
        }
    </style>
</head>
<body>
    <div id="container">
        <h1>🤖 Agent Graph Demo: Collaborative Problem Solving</h1>
        <p>
            This visualization shows a supervisor agent (top) delegating a complex problem
            to 4 research agents. Watch as they collaborate, discuss, and work together
            to solve the problem. Pulsing nodes are "thinking", and flashing edges show
            message exchanges.
        </p>

        <div class="controls">
            <button id="playBtn" onclick="play()">▶ Play</button>
            <button id="pauseBtn" onclick="pause()" disabled>⏸ Pause</button>
            <button id="resetBtn" onclick="reset()">↻ Reset</button>

            <div class="slider-container">
                <label for="frameSlider">Frame:</label>
                <input type="range" id="frameSlider" min="0" max="0" value="0" onchange="goToFrame(this.value)">
            </div>
            <div class="frame-counter">
                <span id="frameCounter">Frame 0 of 0</span>
            </div>
        </div>

        <div id="visualization"></div>

        <div class="frame-info">
            <h3 id="frameTitle">Frame 0: Initial Setup</h3>
            <p id="frameDescription">Initializing agent graph with 5 nodes...</p>
        </div>

        <div class="metrics">
            <h3>📊 Graph Metrics</h3>
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="metric-label">Total Nodes</div>
                    <div class="metric-value" id="metricNodes">5</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Total Edges</div>
                    <div class="metric-value" id="metricEdges">4</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Total Messages</div>
                    <div class="metric-value" id="metricMessages">0</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Active Conversations</div>
                    <div class="metric-value" id="metricConversations">0</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const frames = JSON.parse('""" + json.dumps(self.frames) + """');
        let currentFrame = 0;
        let isPlaying = false;
        let playInterval = null;

        function initVisualization() {
            document.getElementById('frameSlider').max = frames.length - 1;
            goToFrame(0);
        }

        function goToFrame(frameIndex) {
            currentFrame = parseInt(frameIndex);
            const frame = frames[currentFrame];

            // Update title and description
            document.getElementById('frameTitle').textContent = frame.title;
            let description = '';
            if (frame.message_exchange) {
                const [from, to] = frame.message_exchange;
                description = `📨 Message: ${from} → ${to}`;
            }
            if (frame.thinking_nodes && frame.thinking_nodes.length > 0) {
                description += (description ? ' | ' : '') +
                    `💭 Thinking: ${frame.thinking_nodes.join(', ')}`;
            }
            document.getElementById('frameDescription').textContent =
                description || 'Agents collaborating...';

            // Update frame counter
            document.getElementById('frameCounter').textContent =
                `Frame ${currentFrame + 1} of ${frames.length}`;

            // Update metrics
            document.getElementById('metricNodes').textContent = frame.graph.graph_info.node_count;
            document.getElementById('metricEdges').textContent = frame.graph.graph_info.edge_count;
            document.getElementById('metricMessages').textContent = 0; // Would need to track this
            document.getElementById('metricConversations').textContent = frame.graph.links.length;

            // Visualize
            visualizeGraph(frame);
        }

        function visualizeGraph(frame) {
            d3.select('#visualization').selectAll('*').remove();

            const width = document.getElementById('visualization').clientWidth;
            const height = 600;

            const svg = d3.select('#visualization')
                .append('svg')
                .attr('width', width)
                .attr('height', height);

            const simulation = d3.forceSimulation(frame.graph.nodes)
                .force('link', d3.forceLink(frame.graph.links)
                    .id(d => d.id)
                    .distance(150))
                .force('charge', d3.forceManyBody().strength(-400))
                .force('center', d3.forceCenter(width / 2, height / 2));

            const link = svg.selectAll('line')
                .data(frame.graph.links)
                .enter()
                .append('line')
                .attr('class', d => {
                    if (frame.message_exchange &&
                        ((d.source.id === frame.message_exchange[0] && d.target.id === frame.message_exchange[1]) ||
                         (d.source.id === frame.message_exchange[1] && d.target.id === frame.message_exchange[0]))) {
                        return 'link active';
                    }
                    return 'link';
                });

            const node = svg.selectAll('circle')
                .data(frame.graph.nodes)
                .enter()
                .append('circle')
                .attr('class', d => {
                    let classes = 'node';
                    if (frame.thinking_nodes && frame.thinking_nodes.includes(d.id)) {
                        classes += ' thinking';
                    }
                    if (frame.message_exchange &&
                        (d.id === frame.message_exchange[0] || d.id === frame.message_exchange[1])) {
                        classes += ' active';
                    }
                    return classes;
                })
                .attr('r', d => d.id === 'supervisor' ? 20 : 15)
                .attr('fill', d => {
                    if (d.id === 'supervisor') return '#ff9999';
                    return '#99ccff';
                });

            const label = svg.selectAll('text')
                .data(frame.graph.nodes)
                .enter()
                .append('text')
                .attr('class', 'node-label')
                .text(d => d.id);

            simulation.on('tick', () => {
                link.attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);

                node.attr('cx', d => d.x)
                    .attr('cy', d => d.y);

                label.attr('x', d => d.x)
                    .attr('y', d => d.y);
            });
        }

        function play() {
            isPlaying = true;
            document.getElementById('playBtn').disabled = true;
            document.getElementById('pauseBtn').disabled = false;

            playInterval = setInterval(() => {
                if (currentFrame < frames.length - 1) {
                    goToFrame(currentFrame + 1);
                } else {
                    pause();
                }
            }, 2000);
        }

        function pause() {
            isPlaying = false;
            if (playInterval) clearInterval(playInterval);
            document.getElementById('playBtn').disabled = false;
            document.getElementById('pauseBtn').disabled = true;
        }

        function reset() {
            pause();
            goToFrame(0);
        }

        // Initialize on load
        window.addEventListener('load', initVisualization);
    </script>
</body>
</html>
"""
        Path(self.output_path).write_text(html)
        print(f"\n✅ Visualization saved to: {self.output_path}")
        print(f"📖 Open in browser: file://{Path(self.output_path).absolute()}")


async def run_demo() -> None:
    """Run the collaborative problem-solving demo."""

    print("=" * 80)
    print("AGENT GRAPH DEMO: Collaborative Problem Solving")
    print("=" * 80)

    # Create the graph
    print("\n1️⃣  Creating agent network...")
    graph = AgentGraph(name="research_team")

    # Create supervisor
    await graph.add_node(
        "supervisor",
        "You are a research supervisor coordinating a team to solve a complex problem.",
        role="supervisor"
    )

    # Create research agents
    agents = [
        ("agent_a", "You are a literature review specialist", "literature"),
        ("agent_b", "You are a data analyst", "data"),
        ("agent_c", "You are a methodology expert", "methodology"),
        ("agent_d", "You are a systems thinker", "systems"),
    ]

    for agent_id, prompt, specialty in agents:
        await graph.add_node(agent_id, prompt, specialty=specialty)

    # Create edges: supervisor to all agents
    for agent_id, _, _ in agents:
        await graph.add_edge("supervisor", agent_id, directed=True, role="delegation")

    # Create collaboration edges between agents
    agent_ids = [a[0] for a in agents]
    await graph.add_edge("agent_a", "agent_b", directed=False, role="collaboration")
    await graph.add_edge("agent_b", "agent_c", directed=False, role="collaboration")
    await graph.add_edge("agent_c", "agent_d", directed=False, role="collaboration")
    await graph.add_edge("agent_d", "agent_a", directed=False, role="collaboration")

    print(f"✓ Created {graph.node_count} nodes")
    print(f"✓ Created {graph.edge_count} edges")
    print(f"✓ Agents: {', '.join([a[0] for a in agents])}")

    # Initialize visualizer
    visualizer = AnimatedGraphVisualizer("graph_demo.html")

    # Frame 0: Initial state
    print("\n2️⃣  Generating animation frames...")

    json_data = export_json(graph, format_type="node-link", include_metadata=True)
    visualizer.add_frame(
        json_data,
        "Frame 0: Initial Setup",
        thinking_nodes=[]
    )

    # Frame 1: Supervisor thinks about the problem
    visualizer.add_frame(
        json_data,
        "Frame 1: Supervisor Analyzing Problem",
        thinking_nodes=["supervisor"]
    )

    # Frame 2: Supervisor delegates to agent_a
    visualizer.add_frame(
        json_data,
        "Frame 2: Supervisor Delegates to Literature Review",
        message_exchange=("supervisor", "agent_a"),
        thinking_nodes=["agent_a"]
    )

    # Frame 3: Agent A thinks and responds
    visualizer.add_frame(
        json_data,
        "Frame 3: Agent A Reviewing Literature",
        thinking_nodes=["agent_a"]
    )

    # Frame 4: Supervisor delegates to agent_b
    visualizer.add_frame(
        json_data,
        "Frame 4: Supervisor Delegates to Data Analyst",
        message_exchange=("supervisor", "agent_b"),
        thinking_nodes=["agent_b"]
    )

    # Frame 5: Agent B thinks
    visualizer.add_frame(
        json_data,
        "Frame 5: Agent B Analyzing Data",
        thinking_nodes=["agent_b"]
    )

    # Frame 6: Agent A and B collaborate
    visualizer.add_frame(
        json_data,
        "Frame 6: Agent A & B Collaborate",
        message_exchange=("agent_a", "agent_b"),
        thinking_nodes=["agent_a", "agent_b"]
    )

    # Frame 7: Agent C joins
    visualizer.add_frame(
        json_data,
        "Frame 7: Supervisor Delegates to Methodology Expert",
        message_exchange=("supervisor", "agent_c"),
        thinking_nodes=["agent_c"]
    )

    # Frame 8: Agent D joins
    visualizer.add_frame(
        json_data,
        "Frame 8: Supervisor Delegates to Systems Thinker",
        message_exchange=("supervisor", "agent_d"),
        thinking_nodes=["agent_d"]
    )

    # Frame 9: Full collaboration
    visualizer.add_frame(
        json_data,
        "Frame 9: Full Team Collaboration",
        message_exchange=("agent_c", "agent_d"),
        thinking_nodes=["agent_a", "agent_b", "agent_c", "agent_d"]
    )

    # Frame 10: Agents sharing insights
    visualizer.add_frame(
        json_data,
        "Frame 10: Cross-Team Discussion",
        message_exchange=("agent_d", "agent_a"),
        thinking_nodes=["agent_a", "agent_d"]
    )

    # Frame 11: Results synthesis
    visualizer.add_frame(
        json_data,
        "Frame 11: Synthesizing Results",
        message_exchange=("agent_b", "agent_c"),
        thinking_nodes=["agent_b", "agent_c"]
    )

    # Frame 12: Final report to supervisor
    visualizer.add_frame(
        json_data,
        "Frame 12: Final Report to Supervisor",
        message_exchange=("agent_a", "supervisor"),
        thinking_nodes=["supervisor"]
    )

    # Frame 13: Complete
    metrics = await graph.get_metrics()
    visualizer.add_frame(
        json_data,
        "Frame 13: Problem Solving Complete",
        thinking_nodes=[]
    )

    # Generate HTML
    print(f"✓ Generated {len(visualizer.frames)} animation frames")
    visualizer.generate_html()

    # Show metrics
    print("\n3️⃣  Final Metrics:")
    print(f"   Nodes:      {metrics.node_count}")
    print(f"   Edges:      {metrics.edge_count}")
    print(f"   Avg Degree: {metrics.avg_node_degree:.2f}")
    print(f"   Error Rate: {metrics.error_rate:.1%}")

    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETE!")
    print("=" * 80)
    print("\n📖 Open the visualization in your browser:")
    print(f"   file://{Path('graph_demo.html').absolute()}")
    print("\n🎮 Controls:")
    print("   • ▶ Play: Automatically advance through frames")
    print("   • ⏸ Pause: Stop the animation")
    print("   • ↻ Reset: Go back to frame 0")
    print("   • Slider: Jump to any frame")
    print("\n🎨 Visualization features:")
    print("   • Red/Pink nodes: Supervisor")
    print("   • Blue nodes: Research agents")
    print("   • Pulsing nodes: Currently thinking")
    print("   • Flashing edges: Message being sent")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(run_demo())
