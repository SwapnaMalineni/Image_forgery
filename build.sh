#!/bin/bash

# Exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Run database migrations
flask db upgrade

# Download the model file
# The forgery_model.h5 file is too large to be included in the repository.
# You should host it on a cloud storage service and provide a direct download link here.
# For example, using wget:
# wget -O forgery_model.h5 YOUR_DIRECT_DOWNLOAD_LINK
#
# If you use Google Drive, you can use a tool like gdown:
# pip install gdown
# gdown --id YOUR_FILE_ID

if [ ! -f forgery_model.h5 ]; then
  echo "Model file not found. Please download it and place it in the root directory or update the build.sh script to download it."
fi
