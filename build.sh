#!/bin/bash

# Exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Download the model file into Render's persistent data directory when available
MODEL_DEST="forgery_model.h5"
if [ -d "/mnt/data" ]; then
  MODEL_DEST="/mnt/data/forgery_model.h5"
fi

echo "Downloading model to: $MODEL_DEST"
# Use gdown for reliable Google Drive downloads (handles large-file confirmation)
pip install --upgrade gdown || true
# Modern gdown usage without deprecated --id flag
gdown 1Zl8LLFLnVDyzo_qO1b-ZKX3rcPUCjS5q -O "$MODEL_DEST" || true

# Quick HDF5 signature sanity check (Keras .h5 should start with HDF5 signature)
# Export MODEL_DEST so the Python snippet can read it from the environment.
export MODEL_DEST="$MODEL_DEST"
python - <<PY
import sys
import os
path = os.environ.get('MODEL_DEST') or os.path.expandvars('$MODEL_DEST')
if not path or not os.path.exists(path):
  print('Model download failed: file not found at', path)
  sys.exit(1)

# Print file size and SHA-256 so build logs include a simple integrity snapshot
import hashlib
size = os.path.getsize(path)
print('Downloaded file size (bytes):', size)
sha = hashlib.sha256()
with open(path, 'rb') as fh:
  for chunk in iter(lambda: fh.read(8192), b''):
    sha.update(chunk)
print('Downloaded file SHA256:', sha.hexdigest())

with open(path, 'rb') as f:
  header = f.read(8)
  # HDF5 files start with 0x89 H D F 0x0d 0x0a 0x1a 0x0a
  if header != b'\x89HDF\r\n\x1a\n':
    print('Downloaded file does not look like a valid HDF5 (.h5) file. Header:', header)
    sys.exit(2)
print('Model downloaded and validated at', path)
PY

if [ $? -ne 0 ]; then
  echo "Model validation failed. Check that gdown downloaded the correct file id and that the file is a valid .h5 model."
else
  echo "Model downloaded successfully to $MODEL_DEST"
fi

# Ensure the start script is executable so you can set it as the Render start command
if [ -f start.sh ]; then
  chmod +x start.sh || true
  echo "Made start.sh executable"
fi
