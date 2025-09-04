#!/usr/bin/env bash

# Small start script that sets TF logging and runs gunicorn with one worker.
# Make this your Render start command or set Render to use this script.

export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""

exec gunicorn -w 1 --bind 0.0.0.0:$PORT app:app
