#!/bin/bash
# Script to start API and Frontend servers

echo "=========================================="
echo "Starting DR.A-EYE Servers"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if model exists
if [ ! -f "model/best_resnet18.pth" ]; then
    echo "❌ Trained model not found. Please train the model first."
    exit 1
fi

echo "Starting Flask API server..."
cd api
python app.py &
API_PID=$!
cd ..

echo "Waiting for API to start..."
sleep 5

# Check if API is running
if curl -s http://localhost:5000/health > /dev/null; then
    echo "✓ API is running on http://localhost:5000"
else
    echo "⚠ API may not be fully started yet"
fi

echo ""
echo "Starting React Frontend..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

echo ""
echo "=========================================="
echo "Servers Starting..."
echo "=========================================="
echo ""
echo "API: http://localhost:5000"
echo "Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""
echo "Server PIDs:"
echo "  API: $API_PID"
echo "  Frontend: $FRONTEND_PID"
echo ""

# Wait for user interrupt
trap "kill $API_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait

