#!/usr/bin/env bash

# Small start script that sets TF logging and runs gunicorn with one worker.
# Make this your Render start command or set Render to use this script.

export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""
export FLASK_APP=app.py

# Limit TensorFlow and BLAS thread usage to reduce memory/CPU
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Ensure required directories exist
mkdir -p /mnt/data/uploads || true
mkdir -p reports || true

# Initialize the database (idempotent)
flask init-db || true

# Recycle workers and use a tmp dir to reduce memory pressure
exec gunicorn -w 1 --threads 1 --max-requests 50 --max-requests-jitter 20 --worker-tmp-dir /tmp --timeout 120 --bind 0.0.0.0:$PORT app:app
