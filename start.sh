#!/bin/bash
export OLLAMA_HOST=127.0.0.1:11434

ollama serve &

echo "Waiting for Ollama..."
until curl -s http://127.0.0.1:11434/api/tags >/dev/null; do
    sleep 1
done

echo "Ollama ready, starting app..."
exec python run_app.py
