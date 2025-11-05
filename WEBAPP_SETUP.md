# Agent Graph Web Application Setup

A real-time web app for collaborative problem-solving with AI agents.

## Features

✅ **Dynamic Agent Creation** - The supervisor decides what agents to create based on the problem
✅ **Real Claude API Integration** - Uses claude-3-5-sonnet for all agent responses
✅ **Live Visualization** - Interactive D3.js graph showing agents and connections
✅ **Real-time Chat** - See actual messages from agents as they're generated
✅ **Responsive Design** - Works on desktop and tablet

## Quick Start

### Prerequisites

- Python 3.8+
- ANTHROPIC_API_KEY environment variable set

### Installation

1. **Install dependencies:**
```bash
pip3 install flask flask-cors
```

2. **Set your API key:**
```bash
export ANTHROPIC_API_KEY="sk-..."
```

### Running the App

```bash
cd /Users/calebmcook/Documents/dev/repos/claude-agent-graph
python3 app.py
```

You should see:
```
================================================================================
Agent Graph Web Application
================================================================================

Starting Flask server...
Open your browser to: http://localhost:5000
...
```

### Using the App

1. **Open browser:** http://localhost:5000
2. **Enter a problem:** e.g., "How can we improve product quality?"
3. **Watch the magic:**
   - Supervisor analyzes the problem
   - Supervisor dynamically creates specialist agents
   - Supervisor delegates tasks
   - Agents respond in real-time
   - Graph visualization updates live

### Example Problems to Try

- "How can we reduce operational costs?"
- "What's the best way to launch a new product?"
- "How can we improve customer experience?"
- "What strategies would help us enter a new market?"
- "How should we organize our engineering team?"

## How It Works

1. **User enters problem** → Sent to supervisor agent
2. **Supervisor analyzes** → Claude API determines what agents are needed
3. **Agents created** → New nodes appear in graph visualization
4. **Tasks delegated** → Supervisor tells each agent what to do
5. **Agents respond** → Real responses from Claude API appear in chat
6. **Messages flow** → Chat windows update in real-time

## API Endpoints

- `POST /api/initialize` - Create supervisor and initialize graph
- `POST /api/supervisor-think` - Get supervisor analysis & agent recommendations
- `POST /api/delegate` - Have supervisor delegate tasks to agents
- `POST /api/agent-response` - Get a specific agent's response
- `GET /api/graph-state` - Get current graph state (nodes/edges)

## Architecture

```
┌─────────────────┐
│  Browser (HTML) │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│   Flask App     │ (app.py)
└────────┬────────┘
         │
         ├─→ AgentGraph (creates agents)
         ├─→ Claude API (gets responses)
         └─→ JSON responses back to browser
```

## Files

- `app.py` - Flask backend with Claude API integration
- `index.html` - Frontend with D3.js visualization
- `WEBAPP_SETUP.md` - This file

## Troubleshooting

**"Module not found: flask"**
```bash
pip3 install flask flask-cors
```

**"ANTHROPIC_API_KEY not set"**
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

**"Connection refused on localhost:5000"**
- Make sure app.py is running
- Check if port 5000 is available

**Slow responses**
- Claude API calls take 2-5 seconds per response
- This is normal - agents are thinking!

## Next Steps

- Add WebSocket support for faster real-time updates
- Add persistent conversation history
- Add ability to save/load problem-solving sessions
- Add performance metrics and analytics
- Deploy to cloud (Heroku, AWS, etc.)

## Support

For issues or questions, check:
1. Make sure ANTHROPIC_API_KEY is set
2. Check Flask is running (no errors in terminal)
3. Check browser console for JavaScript errors
4. Restart Flask app and try again
