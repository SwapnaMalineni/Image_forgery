#!/bin/bash

# Exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Initialize the database (safe for an empty database on first deploy)
flask init-db

# Download the model file
wget -O forgery_model.h5 "https://drive.google.com/uc?export=download&id=1Zl8LLFLnVDyzo_qO1b-ZKX3rcPUCjS5q"

if [ ! -f forgery_model.h5 ]; then
  echo "Model file not found. Download failed."
fi
