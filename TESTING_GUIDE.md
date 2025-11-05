# Testing Guide for Agent Graph Web Application

## Quick Start

The Flask web application is now fully functional for testing the agent graph framework.

### Running the Application

```bash
# Start the Flask server
python3 app.py

# Then open your browser to: http://localhost:5001
```

### Testing with the Automated Workflow Script

```bash
# Run the complete workflow test
python3 test_workflow.py
```

This script tests all key endpoints:
1. **Initialize** - Creates a new graph with a supervisor agent
2. **Supervisor Analysis** - Gets Claude to analyze the problem and recommend agent roles
3. **Delegate Tasks** - Gets Claude to assign specific tasks to each agent
4. **Get Agent Responses** - Fetches responses from individual agents

## What Was Fixed

### Issue 1: JSON Parsing from Markdown-Wrapped Responses
**Problem**: Claude's API was returning JSON wrapped in markdown code fences (```json...```), which caused JSON parsing to fail.

**Solution**: Added explicit markdown code block detection and extraction:
```python
# Try to extract JSON from markdown code blocks first
code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
```

### Issue 2: Agent Name Matching Failure
**Problem**: The delegation logic was removing underscores from agent names:
- Agent name from Claude: `"domain_expert_agent"`
- Code was converting to: `"domainexpertagent"` (missing underscores)
- This failed to match actual node IDs like `"domain_expert_agent"`

**Solution**: Keep underscores and only lowercase for case-insensitive matching:
```python
agent_id = delegation.get("agent", "").lower()  # Keep underscores
# Then match: node.node_id.lower() == agent_id or node.node_id.lower().startswith(agent_id)
```

### Issue 3: Delegation Prompt Causing Clarification Requests
**Problem**: When given an incomplete problem statement, Claude would ask for clarification instead of delegating tasks.

**Solution**: Made the delegation prompt more directive:
- Explicitly told Claude: "Do NOT ask for clarification - make reasonable assumptions"
- Provided clear examples of expected output format
- Told Claude to assign tasks based on agent expertise even with incomplete info

## API Endpoints

### POST /api/initialize
Initialize a new graph with a supervisor for a problem.

**Request**:
```json
{"problem": "Your problem statement here"}
```

**Response**:
```json
{
  "success": true,
  "graph": {
    "nodes": [...],
    "links": [...]
  }
}
```

### POST /api/supervisor-think
Get the supervisor's analysis and recommended agent roles.

**Request**:
```json
{"problem": "Your problem statement here"}
```

**Response**:
```json
{
  "type": "supervisor_analysis",
  "analysis": "Analysis text",
  "agents": [
    {"id": "agent_1", "role": "role_name"},
    ...
  ],
  "graph": {...}
}
```

### POST /api/delegate
Get task delegations from supervisor to agents.

**Request**:
```json
{"problem": "Your problem statement here"}
```

**Response**:
```json
{
  "delegations": [
    {
      "agent": "agent_1",
      "task": "Specific task description"
    },
    ...
  ],
  "graph": {...}
}
```

### POST /api/agent-response
Get a response from a specific agent for a given task.

**Request**:
```json
{
  "agent": "agent_id",
  "task": "Task description"
}
```

**Response**:
```json
{
  "agent": "agent_id",
  "response": "Agent's response text",
  "timestamp": "2025-11-04T17:04:02.827520"
}
```

### GET /api/graph-state
Get the current state of the graph.

**Response**:
```json
{
  "nodes": [...],
  "links": [...]
}
```

## Browser Interface

The web interface (`index.html`) includes:
- **Problem Input**: Text area to enter your problem statement
- **Interactive Buttons**: Step through the workflow manually or let it run automatically
- **Graph Visualization**: D3.js visualization showing agent nodes and connections
- **Response Display**: Shows agent responses and analysis

### Key Features
- ✓ Real-time graph visualization updates
- ✓ Complete workflow from problem → agents → responses
- ✓ Proper error handling and user feedback
- ✓ Support for incomplete problem statements
- ✓ Multiple agent role generation based on problem type

## Known Limitations

1. **Incomplete Problem Statements**: The current test uses "Get the vote of 3 different agents on what the pri..." which is intentionally incomplete. The system handles this by making reasonable assumptions about the task.

2. **Single Graph Instance**: The current implementation maintains a single graph that resets with each new problem initialization. This is suitable for demo/testing scenarios.

3. **Model Selection**: Currently uses `claude-haiku-4-5-20251001` for all agents. Can be changed in the code or made configurable.

## Testing Scenarios

### Scenario 1: Basic Workflow (Recommended)
```bash
python3 test_workflow.py
```
Tests the complete happy path.

### Scenario 2: Browser Testing
1. Open http://localhost:5001 in browser
2. Enter a problem statement
3. Click "Initialize"
4. Click "Supervisor Thinks"
5. Click "Delegate"
6. Click on individual agents to get their responses

### Scenario 3: cURL Testing
```bash
# Initialize
curl -X POST http://localhost:5001/api/initialize \
  -H "Content-Type: application/json" \
  -d '{"problem": "What is the best approach to system design?"}'

# Supervisor thinks
curl -X POST http://localhost:5001/api/supervisor-think \
  -H "Content-Type: application/json" \
  -d '{"problem": "What is the best approach to system design?"}'
```

## Debugging

### Check Flask Logs
Flask debug mode logs all requests and responses to console. Look for:
- `[INFO] __main__: Supervisor response:`
- `[INFO] __main__: Delegation response:`
- `[DEBUG]` messages showing message types received from Claude SDK

### Test Individual Endpoints
Use the `test_workflow.py` script which provides clear output of each step.

### Browser Developer Tools
Open Developer Tools (F12 in most browsers) and check:
- **Console**: JavaScript errors or logs
- **Network**: Check API request/response payloads
- **Elements**: Inspect the graph visualization

## Performance Notes

- Each API call makes 1-2 requests to Claude (supervisor-think and/or agent responses)
- Expected latency: 2-5 seconds per endpoint call
- Graph visualization updates are instant (client-side)
- No external dependencies beyond Flask and the claude-agent-sdk

## Next Steps

1. ✅ JSON parsing working
2. ✅ Agent delegation working
3. ✅ Test workflow passing
4. Next: Integrate full agent responses into frontend UI
5. Next: Add loading indicators and error messages
6. Next: Create more sophisticated test scenarios

## Support

For issues or questions:
1. Check the Flask console output for error messages
2. Run `test_workflow.py` to verify the backend is working
3. Check browser console for frontend errors
4. Review the code in `app.py` for detailed logging
