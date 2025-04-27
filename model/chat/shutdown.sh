#!/bin/bash

# Step 1: Kill Flask process
if [ -f flask.pid ]; then
  FLASK_PID=$(cat flask.pid)
  echo "Stopping Flask app (PID: $FLASK_PID)..."
  kill "$FLASK_PID"
  rm flask.pid
else
  echo "PID file not found. Is the server running?"
fi

# Step 2: Deactivate virtual environment
deactivate 2>/dev/null

# Step 3: Clean up
echo "Cleaning up..."
rm -rf __pycache__ static/*
> chat_history.json
rm -rf venv

echo "Done! Server shut down and cleaned up."
