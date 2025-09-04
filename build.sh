#!/bin/bash

# Exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Initialize the database (safe for an empty database on first deploy)
flask init-db

# Download the model file into Render's persistent data directory when available
MODEL_DEST="forgery_model.h5"
if [ -d "/mnt/data" ]; then
  MODEL_DEST="/mnt/data/forgery_model.h5"
fi

echo "Downloading model to: $MODEL_DEST"
wget -O "$MODEL_DEST" "https://drive.google.com/uc?export=download&id=1Zl8LLFLnVDyzo_qO1b-ZKX3rcPUCjS5q" || true

if [ ! -f "$MODEL_DEST" ]; then
  echo "Model file not found. Download failed or wget couldn't fetch it. Check build logs." 
else
  echo "Model downloaded successfully to $MODEL_DEST"
fi
