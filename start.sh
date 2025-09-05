#!/usr/bin/env bash

# Small start script that sets TF logging and runs gunicorn with one worker.
# Make this your Render start command or set Render to use this script.

export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""
export FLASK_APP=app.py

# Ensure required directories exist
mkdir -p /mnt/data/uploads || true
mkdir -p reports || true

# Initialize the database (idempotent)
flask init-db || true

exec gunicorn -w 1 --bind 0.0.0.0:$PORT app:app
