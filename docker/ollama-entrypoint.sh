#!/bin/bash
set -e

echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama to be ready..."
for i in {1..30}; do
    if ollama list >/dev/null 2>&1; then
        echo "Ollama is ready"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

if ! ollama list >/dev/null 2>&1; then
    echo "Failed to start Ollama server"
    exit 1
fi

echo "Pulling models..."
ollama pull llama3.2:3b
ollama pull qwen2.5:7b

echo "Models ready. Ollama server running."
wait $OLLAMA_PID
