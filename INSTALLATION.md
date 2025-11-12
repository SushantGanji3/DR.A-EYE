# Installation Guide

This guide will help you install all dependencies needed to run the DR.A-EYE project.

## Quick Setup

Run the automated setup script:

```bash
./setup.sh
```

Or manually follow the steps below.

## Prerequisites

- **Python 3.9+** (Python 3.14.0 recommended)
- **Node.js 18+** (for frontend)
- **Git** (for cloning the repository)

## Manual Installation

### 1. Python Dependencies

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r api/requirements.txt
```

### 2. Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### 3. Verify Installation

Run the verification script to check if everything is installed correctly:

```bash
source venv/bin/activate
python verify_setup.py
```

You should see all checks passing:
- ✓ Python Packages
- ✓ Model Modules
- ✓ Node.js Dependencies
- ✓ Dataset

## What Gets Installed

### Python Packages
- **PyTorch 2.9.1** - Deep learning framework
- **torchvision 0.24.1** - Computer vision utilities
- **Flask 3.0.0** - Web API framework
- **Jupyter** - Interactive notebooks
- **Pandas, NumPy** - Data manipulation
- **Matplotlib, Seaborn** - Visualization
- **scikit-learn** - Machine learning utilities
- And more...

### Frontend Packages
- **React 19.2.0** - UI framework
- **Tailwind CSS** - Styling
- **react-scripts** - Build tools
- And more...

## Troubleshooting

### Virtual Environment Issues

If you encounter issues with the virtual environment:

```bash
# Remove old virtual environment
rm -rf venv

# Create new one
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Node Modules Issues

If frontend dependencies are missing:

```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
cd ..
```

### Import Errors

If you get import errors, make sure:
1. Virtual environment is activated
2. All packages are installed: `pip install -r requirements.txt`
3. You're in the project root directory

## Next Steps

After installation, you can:

1. **Train the model:**
   ```bash
   source venv/bin/activate
   python run_training.py
   ```

2. **Run the API:**
   ```bash
   source venv/bin/activate
   cd api
   python app.py
   ```

3. **Run the frontend:**
   ```bash
   cd frontend
   npm start
   ```

4. **Use Jupyter Notebooks:**
   ```bash
   source venv/bin/activate
   jupyter notebook
   ```

## System Requirements

- **RAM:** Minimum 8GB (16GB recommended for training)
- **Storage:** ~5GB for dependencies + dataset
- **CPU:** Any modern CPU (GPU optional but recommended for faster training)

## Support

If you encounter any issues during installation, please:
1. Run `python verify_setup.py` to identify missing components
2. Check the error messages carefully
3. Ensure all prerequisites are installed
4. Try the troubleshooting steps above

