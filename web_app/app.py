"""
Agent Graph Web Application

A Flask web app for collaborative problem-solving with AI agents.
Uses claude-agent-sdk for all agent interactions.

Features:
- User input for custom problems
- Dynamic agent creation
- Real-time claude-agent-sdk integration
- Live visualization updates

To run:
    python3 app.py

Then open http://localhost:5000 in your browser

Requires:
    - ANTHROPIC_API_KEY environment variable set
    - claude-agent-sdk installed
"""

import asyncio
import json
import logging
import os
from datetime import datetime

from claude_agent_graph import AgentGraph
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

# Configure logging with detailed format
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__)),
    static_folder=os.path.join(os.path.dirname(__file__)),
)
CORS(app)

# Global state
current_graph = None
current_messages = {}
agent_sessions = {}  # Store agent sessions


class AgentCollaborationManager:
    """Manages agent collaboration using claude-agent-sdk."""

    def __init__(self):
        self.graph = None
        self.messages = {}
        self.agent_sessions = {}
        self.conversation_history = []

    async def initialize_graph(self, problem: str) -> dict:
        """Initialize the graph with supervisor."""
        logger.info("Initializing new graph for problem-solving session")
        self.graph = AgentGraph(name="collaborative_problem_solving")
        self.messages = {}
        self.agent_sessions = {}  # Reset agent sessions for new graph

        # Create supervisor with claude-agent-sdk
        await self.graph.add_node(
            "supervisor",
            f"""You are the lead problem solver coordinating a team to solve this problem:

{problem}

Your role is to:
1. Analyze the problem
2. Decide what types of specialist agents you need
3. Create those agents dynamically (you can propose: researcher, analyst, engineer, strategist, etc.)
4. Delegate tasks to them
5. Synthesize their findings into a solution

Be specific about what agents you create and why.""",
            model="claude-haiku-4-5-20251001",
        )

        self.messages["supervisor"] = []
        self.conversation_history = problem

        return self._get_graph_state()

    async def supervisor_thinks(self, problem: str) -> dict:
        """Get supervisor's initial analysis and agent recommendations."""
        logger.info("Supervisor analyzing problem...")

        # Create supervisor session
        supervisor_options = ClaudeAgentOptions(
            system_prompt="""You are an expert at analyzing complex problems and planning team collaboration.
When given a problem, analyze it and recommend what types of specialist agents would be needed to solve it.""",
            model="claude-haiku-4-5-20251001",
            max_turns=1,
        )

        supervisor_session = ClaudeSDKClient(supervisor_options)

        # Get supervisor analysis
        analysis_prompt = f"""Problem to solve: {problem}

Analyze this problem and respond in this JSON format:
{{
    "analysis": "Your understanding of the problem",
    "agents_needed": [
        {{"name": "agent_a", "role": "specialist_role", "description": "What this agent will do"}},
        ...
    ],
    "approach": "Your overall strategy"
}}

Only respond with the JSON, no other text."""

        # Use the SDK properly
        async with supervisor_session:
            await supervisor_session.query(analysis_prompt)
            response_text = ""
            message_count = 0
            async for message in supervisor_session.receive_messages():
                message_count += 1
                logger.debug(f"Received message {message_count}: {type(message).__name__}")

                # Check if this is an AssistantMessage (the actual response)
                if type(message).__name__ == "AssistantMessage":
                    if hasattr(message, "content") and message.content:
                        content = message.content
                        if isinstance(content, str):
                            response_text += content
                        elif isinstance(content, list):
                            # Extract text from content blocks
                            for block in content:
                                if hasattr(block, "text"):
                                    response_text += block.text
                    # Once we have the assistant response, stop waiting for more messages
                    logger.debug("Got AssistantMessage, breaking from message loop")
                    break

        logger.info(f"Supervisor response: {response_text}")

        # Parse JSON response
        data = None
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re

            # Try to extract JSON from markdown code blocks first
            code_block_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL
            )
            if code_block_match:
                try:
                    data = json.loads(code_block_match.group(1))
                except json.JSONDecodeError:
                    # Fall back to general JSON extraction
                    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                    if json_match:
                        try:
                            data = json.loads(json_match.group())
                        except json.JSONDecodeError:
                            pass  # Will use defaults below
            else:
                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass  # Will use defaults below

        # Last resort - use defaults if parsing failed
        if data is None:
            data = {
                "analysis": response_text,
                "agents_needed": [
                    {"name": "agent_a", "role": "researcher", "description": "Conduct research"},
                    {"name": "agent_b", "role": "analyst", "description": "Analyze data"},
                    {"name": "agent_c", "role": "engineer", "description": "Design solution"},
                ],
                "approach": "Systematic problem solving",
            }

        # Create agents based on supervisor's recommendation
        agents_created = []
        logger.info(
            f"Creating agents from supervisor recommendation. Graph exists: {self.graph is not None}"
        )

        for i, agent_spec in enumerate(data.get("agents_needed", [])[:4]):  # Max 4 agents
            agent_name = agent_spec.get("name", f"agent_{chr(97+i)}")
            agent_role = agent_spec.get("role", "specialist")
            agent_desc = agent_spec.get("description", "Help solve the problem")

            logger.info(f"Adding agent '{agent_name}' to graph")
            # Create agent in graph
            await self.graph.add_node(
                agent_name,
                f"""You are a {agent_role} on a problem-solving team.
Your task: {agent_desc}
Work collaboratively with other team members to solve the assigned problem.""",
                model="claude-haiku-4-5-20251001",
                role=agent_role,
            )

            self.messages[agent_name] = []
            agents_created.append({"id": agent_name, "role": agent_role})

            # Create agent session
            agent_options = ClaudeAgentOptions(
                system_prompt=f"""You are a specialist agent with role: {agent_role}
{agent_desc}
Work collaboratively with other team members to solve the assigned problem.""",
                model="claude-haiku-4-5-20251001",
                max_turns=1,
            )
            self.agent_sessions[agent_name] = ClaudeSDKClient(agent_options)

            # Create edges: bidirectional between supervisor and agents
            await self.graph.add_edge(agent_name, "supervisor", directed=False)

        # Create collaboration edges between agents
        agent_ids = [a["id"] for a in agents_created]
        for i in range(len(agent_ids) - 1):
            await self.graph.add_edge(agent_ids[i], agent_ids[i + 1], directed=False)

        return {
            "type": "supervisor_analysis",
            "analysis": data.get("analysis", ""),
            "agents": agents_created,
            "graph": self._get_graph_state(),
        }

    async def delegate_to_agents(self, problem: str) -> dict:
        """Have supervisor delegate tasks to each agent."""
        logger.info("Supervisor delegating tasks...")

        if not self.graph:
            logger.warning("Graph is not initialized for delegation")
            return {"delegations": [], "graph": {"nodes": [], "links": []}}

        all_nodes = self.graph.get_nodes()
        logger.info(f"Total nodes in graph: {len(all_nodes)}")
        for node in all_nodes:
            logger.info(f"  - Node: {node.node_id}")

        agent_nodes = [node.node_id for node in all_nodes if node.node_id != "supervisor"]
        logger.info(f"Found {len(agent_nodes)} agents in graph: {agent_nodes}")

        agent_list = ", ".join(agent_nodes)

        # Create supervisor session for delegation
        supervisor_options = ClaudeAgentOptions(
            system_prompt=f"""You are a team coordinator. Your team members are: {agent_list}
Your job is to assign them specific, actionable tasks to solve problems.""",
            model="claude-haiku-4-5-20251001",
            max_turns=1,
        )
        supervisor_session = ClaudeSDKClient(supervisor_options)

        delegation_prompt = f"""You are a team coordinator. You have these team members:
{', '.join(f'{node.node_id}' for node in all_nodes if node.node_id != 'supervisor')}

Problem statement (may be incomplete): {problem}

Your job: Even if the problem statement is incomplete or unclear, create specific, actionable tasks for EACH team member based on their expertise that would help solve this problem. Do NOT ask for clarification - make reasonable assumptions and delegate tasks.

Respond ONLY with this JSON format, nothing else:
{{
    "delegations": [
        {{"agent": "agent_1_name", "task": "specific actionable task"}},
        {{"agent": "agent_2_name", "task": "specific actionable task"}},
        {{"agent": "agent_3_name", "task": "specific actionable task"}}
    ]
}}"""

        # Use the SDK properly
        async with supervisor_session:
            await supervisor_session.query(delegation_prompt)
            response_text = ""
            message_count = 0
            async for message in supervisor_session.receive_messages():
                message_count += 1
                logger.debug(f"Delegation message {message_count}: {type(message).__name__}")

                # Check if this is an AssistantMessage (the actual response)
                if type(message).__name__ == "AssistantMessage":
                    if hasattr(message, "content") and message.content:
                        content = message.content
                        if isinstance(content, str):
                            response_text += content
                        elif isinstance(content, list):
                            # Extract text from content blocks
                            for block in content:
                                if hasattr(block, "text"):
                                    response_text += block.text
                    # Once we have the assistant response, stop waiting for more messages
                    logger.debug("Got AssistantMessage, breaking from delegation loop")
                    break
        logger.info(f"Delegation response: {response_text}")

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            import re

            # Try to extract JSON from markdown code blocks first
            code_block_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL
            )
            if code_block_match:
                try:
                    data = json.loads(code_block_match.group(1))
                except json.JSONDecodeError:
                    # Fall back to general JSON extraction
                    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                    data = json.loads(json_match.group()) if json_match else {"delegations": []}
            else:
                # Fall back to general JSON extraction
                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                data = json.loads(json_match.group()) if json_match else {"delegations": []}

        # Extract agent responses
        results = {"delegations": [], "graph": self._get_graph_state()}

        for delegation in data.get("delegations", [])[:3]:
            agent_id = delegation.get("agent", "").lower()  # Keep underscores, just lowercase
            # Find matching agent
            matching_agent = None
            for node in self.graph.get_nodes():
                # Direct match or node starts with agent_id
                if node.node_id.lower() == agent_id or node.node_id.lower().startswith(agent_id):
                    matching_agent = node.node_id
                    break

            if matching_agent:
                task = delegation.get("task", "")
                results["delegations"].append({"agent": matching_agent, "task": task})

        return results

    async def get_agent_response(self, agent_id: str, task: str) -> str:
        """Get response from a specific agent using its session."""
        logger.info(f"Getting response from {agent_id} for task: {task}")

        # Get or create agent session
        if agent_id not in self.agent_sessions:
            agent_options = ClaudeAgentOptions(
                system_prompt=f"""You are a specialist agent with expertise in your domain.
You have been assigned to work on collaborative problem-solving tasks.
Your name is {agent_id}. Focus on being helpful, specific, and actionable in your responses.""",
                model="claude-haiku-4-5-20251001",
                max_turns=1,
            )
            self.agent_sessions[agent_id] = ClaudeSDKClient(agent_options)

        session = self.agent_sessions[agent_id]

        prompt = f"""You are assigned this task: {task}

Provide a focused, concise response addressing this task. Be specific and actionable."""

        # Use the SDK properly
        async with session:
            await session.query(prompt)
            agent_response = ""
            message_count = 0
            async for message in session.receive_messages():
                message_count += 1
                logger.debug(f"Agent {agent_id} message {message_count}: {type(message).__name__}")

                # Check if this is an AssistantMessage (the actual response)
                if type(message).__name__ == "AssistantMessage":
                    if hasattr(message, "content") and message.content:
                        content = message.content
                        if isinstance(content, str):
                            agent_response += content
                        elif isinstance(content, list):
                            # Extract text from content blocks
                            for block in content:
                                if hasattr(block, "text"):
                                    agent_response += block.text
                    # Once we have the assistant response, stop waiting for more messages
                    logger.debug(f"Got AssistantMessage from {agent_id}, breaking")
                    break

        self.messages[agent_id].append(agent_response)
        return agent_response

    async def chat_with_supervisor(self, user_message: str) -> str:
        """Have a conversation with the supervisor."""
        logger.info(f"User message to supervisor: {user_message[:100]}...")

        # Create or reuse supervisor session
        if "supervisor_chat" not in self.agent_sessions:
            supervisor_options = ClaudeAgentOptions(
                system_prompt="""You are an expert supervisor coordinating a team of AI agents to solve complex problems.
You can ask clarifying questions, provide guidance, and help refine the problem-solving approach.
Be conversational, helpful, and ask follow-up questions when needed to better understand the user's needs.""",
                model="claude-haiku-4-5-20251001",
                max_turns=1,
            )
            self.agent_sessions["supervisor_chat"] = ClaudeSDKClient(supervisor_options)

        session = self.agent_sessions["supervisor_chat"]

        # Use the SDK properly
        async with session:
            await session.query(user_message)
            response_text = ""
            message_count = 0
            async for message in session.receive_messages():
                message_count += 1
                logger.debug(f"Supervisor chat message {message_count}: {type(message).__name__}")

                # Check if this is an AssistantMessage (the actual response)
                if type(message).__name__ == "AssistantMessage":
                    if hasattr(message, "content") and message.content:
                        content = message.content
                        if isinstance(content, str):
                            response_text += content
                        elif isinstance(content, list):
                            # Extract text from content blocks
                            for block in content:
                                if hasattr(block, "text"):
                                    response_text += block.text
                    # Once we have the assistant response, stop waiting for more messages
                    logger.debug("Got AssistantMessage from supervisor, breaking")
                    break

        if "supervisor" not in self.messages:
            self.messages["supervisor"] = []
        self.messages["supervisor"].append({"role": "assistant", "content": response_text})
        logger.info(f"Supervisor response: {response_text[:100]}...")
        return response_text

    def _get_graph_state(self) -> dict:
        """Get current graph state as JSON."""
        if not self.graph:
            return {"nodes": [], "links": []}

        try:
            nodes = [
                {
                    "id": node.node_id,
                    "status": node.status.value,
                    "model": node.model,
                    "metadata": node.metadata,
                }
                for node in self.graph.get_nodes()
            ]

            links = [
                {"source": edge.from_node, "target": edge.to_node, "directed": edge.directed}
                for edge in self.graph._edges.values()
            ]

            return {"nodes": nodes, "links": links}
        except Exception as e:
            logger.error(f"Error getting graph state: {e}")
            return {"nodes": [], "links": []}


# Initialize manager
manager = AgentCollaborationManager()


@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/api/initialize", methods=["POST"])
def initialize():
    """Initialize the graph with a problem statement."""
    data = request.json
    problem = data.get("problem", "")

    if not problem:
        return jsonify({"error": "Problem statement required"}), 400

    try:
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(manager.initialize_graph(problem))
        loop.close()

        return jsonify({"success": True, "graph": result})
    except Exception as e:
        logger.error(f"Error initializing: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/supervisor-think", methods=["POST"])
def supervisor_think():
    """Get supervisor's analysis and recommendations."""
    data = request.json
    problem = data.get("problem", "")

    logger.info(f"supervisor_think endpoint called with problem: {problem[:50]}...")

    try:
        if not problem:
            logger.warning("Empty problem statement provided")
            return jsonify({"error": "Problem statement is required"}), 400

        logger.debug("Creating new event loop for supervisor thinking")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            logger.debug("Calling manager.supervisor_thinks()")
            result = loop.run_until_complete(manager.supervisor_thinks(problem))
            logger.info("Supervisor thinking completed successfully")
            return jsonify(result)
        except Exception as async_error:
            logger.error(f"Async error in supervisor thinking: {async_error}", exc_info=True)
            raise
        finally:
            loop.close()
            logger.debug("Event loop closed")

    except Exception as e:
        logger.error(f"Error in supervisor thinking: {e}", exc_info=True)
        error_response = {"error": str(e), "error_type": type(e).__name__, "details": repr(e)}
        return jsonify(error_response), 500


@app.route("/api/delegate", methods=["POST"])
def delegate():
    """Have supervisor delegate tasks."""
    data = request.json
    problem = data.get("problem", "")

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(manager.delegate_to_agents(problem))
        loop.close()

        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in delegation: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/agent-response", methods=["POST"])
def agent_response():
    """Get response from an agent."""
    data = request.json
    agent_id = data.get("agent")
    task = data.get("task")

    if not agent_id or not task:
        return jsonify({"error": "Agent and task required"}), 400

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(manager.get_agent_response(agent_id, task))
        loop.close()

        return jsonify(
            {"agent": agent_id, "response": response, "timestamp": datetime.now().isoformat()}
        )
    except Exception as e:
        logger.error(f"Error getting agent response: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/supervisor-chat", methods=["POST"])
def supervisor_chat():
    """Chat with the supervisor."""
    data = request.json
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    logger.info(f"Supervisor chat request: {user_message[:50]}...")

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(manager.chat_with_supervisor(user_message))
        loop.close()

        return jsonify(
            {
                "role": "supervisor",
                "message": user_message,
                "response": response,
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        logger.error(f"Error in supervisor chat: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph-state", methods=["GET"])
def graph_state():
    """Get current graph state."""
    return jsonify(manager._get_graph_state())


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Agent Graph Web Application (Using claude-agent-sdk)")
    print("=" * 80)
    print("\nStarting Flask server...")
    print("Open your browser to: http://localhost:5001")
    print("\nFeatures:")
    print("  ✓ Dynamic agent creation")
    print("  ✓ Real claude-agent-sdk integration")
    print("  ✓ Live graph visualization")
    print("  ✓ Real-time collaboration")
    print("\n" + "=" * 80 + "\n")

    app.run(debug=True, port=5001)
