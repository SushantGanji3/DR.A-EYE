# Quick Start Guide

## Current Status

✅ **Frontend:** Running on http://localhost:3000  
✅ **API:** Running on http://localhost:5000  
✅ **Model:** Trained and loaded

## Access the Application

Open your web browser and navigate to:

```
http://localhost:3000
```

## If You See "This site can't be reached"

### Option 1: Check if servers are running

```bash
# Check frontend
curl http://localhost:3000

# Check API
curl http://localhost:5000/health
```

### Option 2: Restart the frontend

```bash
cd frontend
npm start
```

### Option 3: Check for port conflicts

```bash
# See what's using port 3000
lsof -ti:3000

# Kill any process on port 3000 if needed
kill -9 $(lsof -ti:3000)
```

## Using the Application

1. **Open Browser:** Go to http://localhost:3000
2. **Upload Image:** Click "Select Image" or drag and drop a retinal scan
3. **Analyze:** Click "Analyze Image"
4. **View Results:** See the prediction with confidence scores

## Troubleshooting

### Frontend won't start
```bash
cd frontend
rm -rf node_modules/.cache
npm start
```

### API not responding
```bash
source venv/bin/activate
cd api
python app.py
```

### Port already in use
```bash
# Find and kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Or use a different port
PORT=3001 npm start
```

## Network Access

The frontend is also accessible on your local network at:
- http://10.182.91.16:3000 (or your machine's IP)

This allows access from other devices on the same network.

