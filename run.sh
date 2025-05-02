#!/bin/bash

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv_tf" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv_tf
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv_tf/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models static templates

# Check if model exists
if [ ! -f "models/best_model.keras" ]; then
    echo "⚠️ Warning: No trained model found at models/best_model.keras"
    echo "   Please run ./train.sh first to train the model."
fi

# Start the server
echo "🚀 Starting FastAPI server..."
python app.py
