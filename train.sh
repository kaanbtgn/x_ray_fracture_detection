#!/usr/bin/env bash
# ------------------------------------------------------------
# Enhanced training launcher with error handling
# ------------------------------------------------------------
set -e

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found"
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv_tf"
DATA_DIR="$PROJECT_DIR/dataset"

# Create and activate virtual environment
if [[ ! -d "$VENV_DIR" ]]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

    source "$VENV_DIR/bin/activate"

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r "$PROJECT_DIR/requirements.txt"

# Create models directory if it doesn't exist
mkdir -p models

# Start training
echo "🚀 Starting training..."
python3 train_eval_tf.py \
    --data-dir "$DATA_DIR" \
    --epochs 20

echo "[+] Training completed. See training.log for details."