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

# Set weights directory
WEIGHTS_DIR="./weights"

# Create weights directory if it doesn't exist
info "Setting up weights directory at ${WEIGHTS_DIR}..."
mkdir -p "${WEIGHTS_DIR}" || handle_error "Failed to create weights directory"

# Export environment variable
export SPACY_DATA_DIR="${WEIGHTS_DIR}"
info "SPACY_DATA_DIR environment variable set to ${WEIGHTS_DIR}"

# Check if spaCy is installed
if ! command -v python3 &> /dev/null; then
    handle_error "Python3 is not installed or not in PATH"
fi

if ! python3 -c "import spacy" &> /dev/null; then
    warning "spaCy is not installed. Attempting to install..."
    python3 -m pip install spacy || handle_error "Failed to install spaCy"
fi

# Check if model already exists
if [ ! -d "${WEIGHTS_DIR}/en_core_web_md" ]; then
    info "Downloading spaCy model to ${WEIGHTS_DIR}..."
    python3 -m spacy download en_core_web_md || handle_error "Failed to download spaCy model"
else
    info "spaCy model already exists in ${WEIGHTS_DIR}"
fi

# Verify model loads correctly
info "Verifying model installation..."
if python3 -c "import spacy; nlp = spacy.load('en_core_web_md'); print('Model loaded successfully')" &> /dev/null; then
    info "spaCy model loaded successfully"
else
    handle_error "Failed to load spaCy model. Installation may be corrupted"
fi

info "Setup complete! You can now use the spaCy model with SPACY_DATA_DIR=${WEIGHTS_DIR}"