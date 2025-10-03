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

echo "Pulling local router model..."
ollama pull phi3:3.8b-mini-128k-instruct-q4_K_M

echo "Router model ready. Cloud models (gpt-oss, qwen3-coder) will be accessed via Ollama cloud."
ollama pull gpt-oss:20b-cloud
ollama pull gpt-oss:120b-cloud
ollama pull qwen3-coder:480b-cloud
wait $OLLAMA_PID
