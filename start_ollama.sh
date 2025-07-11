#!/bin/bash

# Start the ollama server in the background
echo "Starting ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for the server to start
sleep 5

# Create the ollama model with the adapter
echo "Creating ollama model with adapter..."
ollama create miltronic-adapter -f Modelfile.adapter

# Run the model
echo "Running the miltronic-adapter model..."
ollama run miltronic-adapter

# Stop the ollama server
echo "Stopping ollama server..."
kill $OLLAMA_PID