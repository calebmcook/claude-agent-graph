# Flask Web App for Agent Graph

This directory contains the Flask web application and demo files for the Agent Graph framework. These are for **local development only** and are not published to GitHub Pages.

## Files

- **app.py** - Main Flask application with REST API endpoints
- **index.html** - Web UI for the Flask app
- **demo_interactive.py** - Interactive demo script
- **demo_interactive_v2.py** - Enhanced interactive demo
- **demo_simple.html** - Simple HTML demo
- **graph_demo.html** - Graph visualization demo
- **graph_demo_v2.html** - Enhanced graph visualization demo

## Running Locally

```bash
# From repo root
python3 web_app/app.py
```

Then open http://localhost:5001 in your browser.

## Note

These files are kept for local development but excluded from GitHub Pages publication. The official documentation site is built from the `docs/` directory and published via the `docs.yml` GitHub Actions workflow to gh-pages branch.
