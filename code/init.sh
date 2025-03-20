#!/bin/bash

# Script to set up spaCy model with error handling
# This script creates a weights directory, downloads the spaCy model if needed,
# and verifies the installation was successful.

set -e  # Exit immediately if a command exits with non-zero status

# Define colors for better output readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function for error handling
handle_error() {
    echo -e "${RED}ERROR: $1${NC}"
    exit 1
}

# Function to display status messages
info() {
    echo -e "${GREEN}INFO:${NC} $1"
}

warning() {
    echo -e "${YELLOW}WARNING:${NC} $1"
}

# Set weights directory (convert to absolute path)
WEIGHTS_DIR="$(realpath "./weights")"

# Create weights directory if it doesn't exist
info "Setting up weights directory at ${WEIGHTS_DIR}..."
mkdir -p "${WEIGHTS_DIR}" || handle_error "Failed to create weights directory"

# Check if spaCy is installed
if ! command -v python3 &> /dev/null; then
    handle_error "Python3 is not installed or not in PATH"
fi

if ! python3 -c "import spacy" &> /dev/null; then
    warning "spaCy is not installed. Attempting to install..."
    python3 -m pip install spacy || handle_error "Failed to install spaCy"
fi

# Check if model already exists in the weights directory
MODEL_PATH="${WEIGHTS_DIR}/en_core_web_md"
if [ ! -d "${MODEL_PATH}" ]; then
    info "Downloading spaCy model to ${WEIGHTS_DIR}..."
    
    # Create a temporary Python script to handle the download
    TMP_SCRIPT=$(mktemp)
    cat > "${TMP_SCRIPT}" << EOF
import os
import spacy.cli
import shutil
from pathlib import Path

# Set target directory
target_dir = Path("${MODEL_PATH}")

# Download to a temporary location first
spacy.cli.download("en_core_web_md")

# Find where spaCy downloaded the model
import en_core_web_md
model_path = Path(en_core_web_md.__file__).parent

# Create the target directory
target_dir.mkdir(parents=True, exist_ok=True)

# Copy all files from the downloaded model to our target directory
for item in model_path.iterdir():
    if item.is_file():
        shutil.copy(item, target_dir)
    elif item.is_dir():
        shutil.copytree(item, target_dir / item.name, dirs_exist_ok=True)

print(f"Model successfully installed to {target_dir}")
EOF

    # Execute the Python script
    python3 "${TMP_SCRIPT}" || handle_error "Failed to download and setup spaCy model"
    rm "${TMP_SCRIPT}"
else
    info "spaCy model already exists in ${WEIGHTS_DIR}"
fi

# Create a .spacy-data-dir file in the model directory to help spaCy locate it
echo "${WEIGHTS_DIR}" > "${WEIGHTS_DIR}/.spacy-data-dir"

# Create a simple Python script to test loading from the custom directory
TEST_SCRIPT=$(mktemp)
cat > "${TEST_SCRIPT}" << EOF
import os
import sys

# Set environment variable to our weights directory
os.environ["SPACY_DATA_DIR"] = "${WEIGHTS_DIR}"

try:
    import spacy
    print(f"Attempting to load model from {os.environ['SPACY_DATA_DIR']}...")
    nlp = spacy.load("en_core_web_md")
    
    # Test the model with a simple sentence
    doc = nlp("This is a test sentence to verify the model works correctly.")
    print("Model loaded and working correctly!")
    sys.exit(0)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)
EOF

# Verify model loads correctly from our directory
info "Verifying model installation..."
if python3 "${TEST_SCRIPT}"; then
    info "spaCy model loaded successfully from ${WEIGHTS_DIR}"
else
    handle_error "Failed to load spaCy model from the specified directory"
fi
rm "${TEST_SCRIPT}"

# Export environment variable for the current session
export SPACY_DATA_DIR="${WEIGHTS_DIR}"
info "SPACY_DATA_DIR environment variable set to ${WEIGHTS_DIR}"

info "Setup complete! To use this model in other scripts, make sure to set:"
info "export SPACY_DATA_DIR=${WEIGHTS_DIR}"