"""
Interactive Demo: Collaborative Problem Solving with Agent Graph (v2)

Features:
- User input for custom problems
- Real message exchanges between agents
- Separate chat windows for each agent
- Animated graph visualization
- Fixed text encoding

To run:
    python3 demo_interactive_v2.py

Then open graph_demo_v2.html in your browser
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class Message:
    """Represents a message between agents."""

    def __init__(self, from_node: str, to_node: str, content: str, timestamp: str | None = None):
        self.from_node = from_node
        self.to_node = to_node
        self.content = content
        self.timestamp = timestamp or datetime.now().isoformat()

    def to_dict(self) -> dict[str, str]:
        return {
            "from": self.from_node,
            "to": self.to_node,
            "content": self.content,
            "timestamp": self.timestamp,
        }


class AnimatedGraphVisualizer:
    """Creates an interactive HTML visualization with animations."""

    def __init__(self, output_path: str = "graph_demo_v2.html"):
        self.output_path = output_path
        self.frames: list[dict[str, Any]] = []
        self.all_messages: dict[str, list[Message]] = {}

    def add_frame(
        self,
        graph_data: dict[str, Any],
        title: str,
        frame_description: str = "",
        message: Message | None = None,
        thinking_nodes: list[str] | None = None,
    ) -> None:
        """Add a frame to the animation."""
        if message:
            node = message.from_node
            if node not in self.all_messages:
                self.all_messages[node] = []
            self.all_messages[node].append(message)

        frame = {
            "title": title,
            "description": frame_description,
            "graph": graph_data,
            "message": message.to_dict() if message else None,
            "thinking_nodes": thinking_nodes or [],
        }
        self.frames.append(frame)

    def generate_html(self) -> None:
        """Generate the HTML visualization."""
        html = (
            """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Agent Graph Demo - Collaborative Problem Solving</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        #container {
            max-width: 1600px;
            margin: 0 auto;
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 28px;
            margin-bottom: 10px;
        }

        .header p {
            font-size: 14px;
            opacity: 0.9;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 20px;
            min-height: 800px;
        }

        .section {
            display: flex;
            flex-direction: column;
        }

        .controls {
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }

        .control-group {
            margin-bottom: 10px;
        }

        .control-group label {
            display: block;
            font-weight: 600;
            margin-bottom: 5px;
            color: #333;
            font-size: 13px;
        }

        button {
            padding: 10px 16px;
            margin-right: 8px;
            background-color: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            transition: background-color 0.3s;
        }

        button:hover {
            background-color: #5568d3;
        }

        button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }

        #frameSlider {
            width: 100%;
            height: 6px;
            cursor: pointer;
        }

        .frame-counter {
            text-align: center;
            margin: 10px 0;
            font-weight: 600;
            color: #667eea;
            font-size: 13px;
        }

        #visualization {
            border: 2px solid #e9ecef;
            border-radius: 8px;
            background-color: #fafbfc;
            min-height: 600px;
            position: relative;
        }

        svg {
            width: 100%;
            height: 600px;
        }

        .node {
            stroke: #333;
            stroke-width: 2px;
            cursor: pointer;
            transition: all 0.3s;
        }

        .node.thinking {
            animation: pulse 1s infinite;
        }

        @keyframes pulse {
            0%, 100% { r: 15px; opacity: 1; }
            50% { r: 12px; opacity: 0.7; }
        }

        .node.active {
            stroke-width: 4px;
            stroke: #ff6b6b;
        }

        .link {
            stroke: #ddd;
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
            font-size: 11px;
            font-weight: 600;
            pointer-events: none;
            text-anchor: middle;
            dy: ".3em";
            fill: #333;
        }

        .frame-info {
            padding: 15px;
            background-color: #fff3cd;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            margin-bottom: 15px;
        }

        .frame-info h3 {
            margin: 0 0 5px 0;
            color: #856404;
            font-size: 14px;
        }

        .frame-info p {
            margin: 0;
            color: #856404;
            font-size: 12px;
        }

        .chat-container {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            max-height: 600px;
            overflow-y: auto;
        }

        .chat-window {
            background-color: white;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .chat-header {
            background-color: #667eea;
            color: white;
            padding: 10px;
            font-weight: 600;
            font-size: 12px;
        }

        .chat-header.supervisor {
            background-color: #ff6b6b;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
            font-size: 11px;
            max-height: 300px;
        }

        .message {
            margin-bottom: 8px;
            padding: 8px;
            border-radius: 6px;
            line-height: 1.4;
        }

        .message.sent {
            background-color: #e7f3ff;
            border-left: 3px solid #667eea;
        }

        .message.received {
            background-color: #f0f0f0;
            border-left: 3px solid #999;
        }

        .message-label {
            font-weight: 600;
            color: #333;
            margin-bottom: 3px;
        }

        .message-content {
            color: #555;
            word-wrap: break-word;
        }

        .metrics {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin: 20px;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
        }

        .metric-item {
            background-color: white;
            padding: 12px;
            border-radius: 6px;
            border-left: 4px solid #667eea;
            text-align: center;
        }

        .metric-label {
            font-size: 11px;
            color: #666;
            font-weight: 600;
            margin-bottom: 5px;
        }

        .metric-value {
            font-size: 20px;
            color: #667eea;
            font-weight: 700;
        }
    </style>
</head>
<body>
    <div id="container">
        <div class="header">
            <h1>🤖 Agent Graph: Collaborative Problem Solving</h1>
            <p>A supervisor delegates a complex problem to 4 research agents who collaborate to find a solution</p>
        </div>

        <div class="main-content">
            <div class="section">
                <div class="controls">
                    <div class="control-group">
                        <button id="playBtn" onclick="play()">▶ Play</button>
                        <button id="pauseBtn" onclick="pause()" disabled>⏸ Pause</button>
                        <button id="resetBtn" onclick="reset()">↻ Reset</button>
                    </div>

                    <div class="control-group">
                        <label for="frameSlider">Timeline:</label>
                        <input type="range" id="frameSlider" min="0" max="0" value="0" onchange="goToFrame(this.value)">
                    </div>

                    <div class="frame-counter">
                        <span id="frameCounter">Frame 0 of 0</span>
                    </div>
                </div>

                <div id="visualization"></div>

                <div class="frame-info">
                    <h3 id="frameTitle">Frame 0</h3>
                    <p id="frameDescription">Initializing...</p>
                </div>
            </div>

            <div class="section">
                <h3 style="margin-bottom: 15px; color: #333;">Agent Chat Windows</h3>
                <div class="chat-container" id="chatContainer">
                    <!-- Chat windows will be generated here -->
                </div>
            </div>
        </div>

        <div class="metrics">
            <div class="metric-item">
                <div class="metric-label">Total Nodes</div>
                <div class="metric-value" id="metricNodes">5</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Total Edges</div>
                <div class="metric-value" id="metricEdges">8</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Messages Sent</div>
                <div class="metric-value" id="metricMessages">0</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Avg Node Degree</div>
                <div class="metric-value" id="metricDegree">3.2</div>
            </div>
        </div>
    </div>

    <script>
        const frames = JSON.parse('"""
            + json.dumps(self.frames)
            + """');
        const nodeList = ['supervisor', 'agent_a', 'agent_b', 'agent_c', 'agent_d'];
        let currentFrame = 0;
        let isPlaying = false;
        let playInterval = null;
        const nodeMessages = {};

        // Initialize message tracking
        nodeList.forEach(node => {
            nodeMessages[node] = [];
        });

        function initVisualization() {
            // Create chat windows
            createChatWindows();

            document.getElementById('frameSlider').max = frames.length - 1;
            goToFrame(0);
        }

        function createChatWindows() {
            const container = document.getElementById('chatContainer');
            container.innerHTML = '';

            nodeList.forEach(nodeId => {
                const chatDiv = document.createElement('div');
                chatDiv.className = 'chat-window';
                chatDiv.id = 'chat_' + nodeId;

                const header = document.createElement('div');
                header.className = 'chat-header' + (nodeId === 'supervisor' ? ' supervisor' : '');
                header.textContent = nodeId === 'supervisor' ? 'Supervisor' : nodeId.replace('_', ' ').toUpperCase();

                const messages = document.createElement('div');
                messages.className = 'chat-messages';
                messages.id = 'messages_' + nodeId;

                chatDiv.appendChild(header);
                chatDiv.appendChild(messages);
                container.appendChild(chatDiv);
            });
        }

        function addMessageToChat(fromNode, toNode, content) {
            // Add to sender's chat
            const fromChat = document.getElementById('messages_' + fromNode);
            if (fromChat) {
                const msgDiv = document.createElement('div');
                msgDiv.className = 'message sent';
                msgDiv.innerHTML = '<div class="message-label">To ' + toNode.replace('_', ' ').toUpperCase() + ':</div>' +
                                   '<div class="message-content">' + escapeHtml(content) + '</div>';
                fromChat.appendChild(msgDiv);
                fromChat.scrollTop = fromChat.scrollHeight;
            }

            // Add to receiver's chat
            const toChat = document.getElementById('messages_' + toNode);
            if (toChat) {
                const msgDiv = document.createElement('div');
                msgDiv.className = 'message received';
                msgDiv.innerHTML = '<div class="message-label">From ' + fromNode.replace('_', ' ').toUpperCase() + ':</div>' +
                                   '<div class="message-content">' + escapeHtml(content) + '</div>';
                toChat.appendChild(msgDiv);
                toChat.scrollTop = toChat.scrollHeight;
            }
        }

        function escapeHtml(text) {
            const map = {
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#039;'
            };
            return text.replace(/[&<>"']/g, m => map[m]);
        }

        function goToFrame(frameIndex) {
            currentFrame = parseInt(frameIndex);
            const frame = frames[currentFrame];

            // Update title and description
            document.getElementById('frameTitle').textContent = frame.title;
            document.getElementById('frameDescription').textContent = frame.description;

            // Update frame counter
            document.getElementById('frameCounter').textContent =
                `Frame ${currentFrame + 1} of ${frames.length}`;

            // Update metrics
            document.getElementById('metricNodes').textContent = frame.graph.graph_info.node_count;
            document.getElementById('metricEdges').textContent = frame.graph.graph_info.edge_count;
            document.getElementById('metricMessages').textContent = currentFrame;

            // Count average degree
            const totalDegree = frame.graph.links.reduce((sum, link) => sum + 2, 0);
            const avgDegree = (totalDegree / frame.graph.graph_info.node_count).toFixed(2);
            document.getElementById('metricDegree').textContent = avgDegree;

            // Add message to chat if present
            if (frame.message) {
                addMessageToChat(frame.message.from, frame.message.to, frame.message.content);
            }

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
                    if (frame.message &&
                        ((d.source.id === frame.message.from && d.target.id === frame.message.to) ||
                         (d.source.id === frame.message.to && d.target.id === frame.message.from))) {
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
                    if (frame.message &&
                        (d.id === frame.message.from || d.id === frame.message.to)) {
                        classes += ' active';
                    }
                    return classes;
                })
                .attr('r', d => d.id === 'supervisor' ? 20 : 15)
                .attr('fill', d => {
                    if (d.id === 'supervisor') return '#ff6b6b';
                    return '#667eea';
                });

            const label = svg.selectAll('text')
                .data(frame.graph.nodes)
                .enter()
                .append('text')
                .attr('class', 'node-label')
                .text(d => d.id.replace('_', ' '));

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
                    document.getElementById('frameSlider').value = currentFrame;
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
            document.getElementById('frameSlider').value = 0;
            document.getElementById('chatContainer').innerHTML = '';
            createChatWindows();
        }

        // Initialize on load
        window.addEventListener('load', initVisualization);
    </script>
</body>
</html>
"""
        )
        Path(self.output_path).write_text(html, encoding="utf-8")
        print(f"\n✅ Visualization saved to: {self.output_path}")
        print(f"📖 Open in browser: file://{Path(self.output_path).absolute()}")


async def run_demo(problem: str) -> None:
    """Run the collaborative problem-solving demo."""

    print("\n" + "=" * 80)
    print("AGENT GRAPH DEMO: Collaborative Problem Solving")
    print("=" * 80)

    print(f"\n📋 Problem Statement: {problem}\n")

    # Simulated agent responses (in real system, these would come from Claude API)
    agent_responses = {
        "agent_a": "I've completed a thorough literature review. Key findings: (1) Previous research shows similar approaches, (2) Current best practices indicate we should focus on X, (3) There are 3 promising methodologies to explore.",
        "agent_b": "Data analysis complete. We have 5000 relevant data points. Initial patterns suggest Y is the strongest factor, with 87% confidence. Z also shows correlation at 65%.",
        "agent_c": "Methodology proposal: A two-phase approach would be optimal. Phase 1 focuses on hypothesis testing using agent_b's data. Phase 2 implements the solution with iterative refinement.",
        "agent_d": "Systems perspective: These findings integrate well. The solution scales across 3 different contexts. Potential bottlenecks identified at stages 2 and 4, but both are manageable.",
        "supervisor_response": "Excellent collaboration team! I'm synthesizing your findings. The integrated approach combining literature insights, data analysis, methodology, and systems thinking provides a comprehensive solution.",
    }

    # Create the graph
    print("1️⃣  Creating agent network...")
    from claude_agent_graph import AgentGraph, export_json

    graph = AgentGraph(name="research_team")

    # Create supervisor
    await graph.add_node(
        "supervisor",
        f"You are a research supervisor coordinating a team to solve this problem: {problem}",
        role="supervisor",
    )

    # Create research agents
    agents = [
        (
            "agent_a",
            "You are a literature review specialist. Analyze relevant research and provide insights.",
            "literature",
        ),
        ("agent_b", "You are a data analyst. Examine data and identify patterns.", "data"),
        (
            "agent_c",
            "You are a methodology expert. Design the approach to solve the problem.",
            "methodology",
        ),
        (
            "agent_d",
            "You are a systems thinker. Ensure the solution works holistically.",
            "systems",
        ),
    ]

    for agent_id, prompt, specialty in agents:
        await graph.add_node(agent_id, prompt, specialty=specialty)

    # Create edges: supervisor to all agents
    for agent_id, _, _ in agents:
        await graph.add_edge("supervisor", agent_id, directed=True, role="delegation")

    # Create collaboration edges between agents
    await graph.add_edge("agent_a", "agent_b", directed=False, role="collaboration")
    await graph.add_edge("agent_b", "agent_c", directed=False, role="collaboration")
    await graph.add_edge("agent_c", "agent_d", directed=False, role="collaboration")
    await graph.add_edge("agent_d", "agent_a", directed=False, role="collaboration")

    print(f"✓ Created {graph.node_count} nodes")
    print(f"✓ Created {graph.edge_count} edges")
    print(f"✓ Agents: {', '.join([a[0] for a in agents])}\n")

    # Initialize visualizer
    visualizer = AnimatedGraphVisualizer("graph_demo_v2.html")

    # Export graph data
    json_data = export_json(graph, format_type="node-link", include_metadata=True)

    # Frame 0: Initial state
    visualizer.add_frame(
        json_data, "Frame 0: Problem Assignment", f"Supervisor presents problem: {problem}"
    )

    # Frame 1: Supervisor delegates
    msg = Message("supervisor", "agent_a", f"Please conduct a literature review on: {problem}")
    visualizer.add_frame(
        json_data,
        "Frame 1: Supervisor Delegates to Literature Specialist",
        "Requesting literature review and research synthesis...",
        message=msg,
    )

    # Frame 2: Agent A responds
    msg = Message("agent_a", "supervisor", agent_responses["agent_a"])
    visualizer.add_frame(
        json_data,
        "Frame 2: Agent A Completes Literature Review",
        "Agent A shares findings from literature review",
        message=msg,
        thinking_nodes=["agent_a"],
    )

    # Frame 3: Supervisor delegates to Agent B
    msg = Message("supervisor", "agent_b", f"Analyze data related to: {problem}")
    visualizer.add_frame(
        json_data,
        "Frame 3: Supervisor Delegates to Data Analyst",
        "Requesting data analysis and pattern identification...",
        message=msg,
    )

    # Frame 4: Agent B responds
    msg = Message("agent_b", "supervisor", agent_responses["agent_b"])
    visualizer.add_frame(
        json_data,
        "Frame 4: Agent B Completes Data Analysis",
        "Agent B shares data insights and patterns",
        message=msg,
        thinking_nodes=["agent_b"],
    )

    # Frame 5: Agents collaborate
    msg = Message(
        "agent_a",
        "agent_b",
        "Your data correlates with the methodology I found in literature. Let's share insights.",
    )
    visualizer.add_frame(
        json_data,
        "Frame 5: Agents A & B Collaborate",
        "Literature specialist and data analyst exchange findings...",
        message=msg,
        thinking_nodes=["agent_a", "agent_b"],
    )

    # Frame 6: Supervisor delegates to Agent C
    msg = Message("supervisor", "agent_c", f"Design the methodology to address: {problem}")
    visualizer.add_frame(
        json_data,
        "Frame 6: Supervisor Delegates to Methodology Expert",
        "Requesting solution design and approach...",
        message=msg,
    )

    # Frame 7: Agent C responds
    msg = Message("agent_c", "supervisor", agent_responses["agent_c"])
    visualizer.add_frame(
        json_data,
        "Frame 7: Agent C Designs Methodology",
        "Methodology expert proposes a two-phase approach",
        message=msg,
        thinking_nodes=["agent_c"],
    )

    # Frame 8: Supervisor delegates to Agent D
    msg = Message("supervisor", "agent_d", f"Ensure this solution works holistically: {problem}")
    visualizer.add_frame(
        json_data,
        "Frame 8: Supervisor Delegates to Systems Thinker",
        "Requesting systems-level analysis...",
        message=msg,
    )

    # Frame 9: Agent D responds
    msg = Message("agent_d", "supervisor", agent_responses["agent_d"])
    visualizer.add_frame(
        json_data,
        "Frame 9: Agent D Provides Systems Analysis",
        "Systems thinker validates solution across contexts",
        message=msg,
        thinking_nodes=["agent_d"],
    )

    # Frame 10: Cross-team collaboration
    msg = Message("agent_c", "agent_d", "How does this methodology fit into your systems view?")
    visualizer.add_frame(
        json_data,
        "Frame 10: Methodology Expert & Systems Thinker Collaborate",
        "Team members refine the solution together...",
        message=msg,
        thinking_nodes=["agent_c", "agent_d"],
    )

    # Frame 11: Full team discussion
    msg = Message(
        "agent_b",
        "agent_a",
        "The data strongly supports the literature-based approach. Should we integrate with the methodology?",
    )
    visualizer.add_frame(
        json_data,
        "Frame 11: Full Team Integration Discussion",
        "All team members contribute to solution refinement...",
        message=msg,
        thinking_nodes=["agent_a", "agent_b", "agent_c", "agent_d"],
    )

    # Frame 12: Final report to supervisor
    msg = Message(
        "agent_c",
        "supervisor",
        "After full collaboration, here's the integrated solution: [comprehensive report with all insights synthesized]",
    )
    visualizer.add_frame(
        json_data,
        "Frame 12: Final Solution Presented",
        "Agents present integrated solution to supervisor",
        message=msg,
    )

    # Frame 13: Supervisor synthesizes
    msg = Message("supervisor", "agent_a", agent_responses["supervisor_response"])
    visualizer.add_frame(
        json_data,
        "Frame 13: Supervisor Synthesizes Final Solution",
        "Supervisor provides integrated summary and approval",
        message=msg,
        thinking_nodes=["supervisor"],
    )

    # Frame 14: Complete
    visualizer.add_frame(
        json_data,
        "Frame 14: Problem Solved",
        "Agent collaboration successfully solved the problem!",
    )

    # Generate HTML
    print("2️⃣  Generating animation...")
    print(f"✓ Generated {len(visualizer.frames)} animation frames")
    visualizer.generate_html()

    # Show metrics
    print("\n3️⃣  Final Metrics:")
    metrics = await graph.get_metrics()
    print(f"   Nodes:      {metrics.node_count}")
    print(f"   Edges:      {metrics.edge_count}")
    print(f"   Avg Degree: {metrics.avg_node_degree:.2f}")

    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETE!")
    print("=" * 80)
    print("\n📖 Open the visualization in your browser:")
    print(f"   file://{Path('graph_demo_v2.html').absolute()}\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Welcome to Agent Graph Interactive Demo")
    print("=" * 80)

    # Get problem from user
    problem = input("\n📝 Enter a problem for the agents to solve:\n   >>> ").strip()

    if not problem:
        problem = "How can we improve machine learning model efficiency while maintaining accuracy?"

    asyncio.run(run_demo(problem))
