#!/bin/bash
# Start Jupyter Lab in background
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' &

# Start code tunnel in background
code tunnel --accept-server-license-terms &

# Start Ollama serve in background
ollama serve &

# Keep container running
tail -f /dev/null