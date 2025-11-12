#!/bin/bash
# Setup script for DR.A-EYE project
# This script installs all dependencies needed to run the project

set -e

echo "=========================================="
echo "DR.A-EYE Project Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Python $PYTHON_VERSION found"

# Check Node.js version
echo ""
echo "Checking Node.js version..."
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

NODE_VERSION=$(node --version)
echo "✓ Node.js $NODE_VERSION found"

# Create virtual environment if it doesn't exist
echo ""
echo "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt -q
pip install -r api/requirements.txt -q
echo "✓ Python dependencies installed"

# Install frontend dependencies
echo ""
echo "Installing frontend dependencies..."
cd frontend
if [ ! -d "node_modules" ]; then
    npm install
    echo "✓ Frontend dependencies installed"
else
    npm install
    echo "✓ Frontend dependencies updated"
fi
cd ..

# Verify installations
echo ""
echo "Verifying installations..."
python -c "import torch, torchvision, flask, flask_cors, pandas, numpy, matplotlib, seaborn, sklearn, jupyter, PIL; print('✓ All Python packages verified')" 2>&1

echo ""
echo "=========================================="
echo "Setup Complete! ✓"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start training:"
echo "  python run_training.py"
echo ""
echo "To run the API:"
echo "  cd api && python app.py"
echo ""
echo "To run the frontend:"
echo "  cd frontend && npm start"
echo ""

