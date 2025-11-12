#!/usr/bin/env python
"""
Verification script to check if all dependencies are installed correctly
"""
import sys
import subprocess
from pathlib import Path

def check_python_packages():
    """Check if all required Python packages are installed"""
    print("Checking Python packages...")
    required_packages = [
        'torch', 'torchvision', 'numpy', 'pandas', 'Pillow',
        'matplotlib', 'seaborn', 'sklearn', 'jupyter', 'flask',
        'flask_cors', 'requests', 'tqdm'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            elif package == 'flask_cors':
                __import__('flask_cors')
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    return len(missing) == 0, missing

def check_model_modules():
    """Check if model modules can be imported"""
    print("\nChecking model modules...")
    try:
        sys.path.insert(0, str(Path(__file__).parent / 'model'))
        from model import DiabeticRetinopathyModel
        from dataset import get_dataloaders
        print("  ✓ Model module")
        print("  ✓ Dataset module")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def check_node_modules():
    """Check if Node.js dependencies are installed"""
    print("\nChecking Node.js dependencies...")
    frontend_path = Path(__file__).parent / 'frontend'
    node_modules = frontend_path / 'node_modules'
    
    if node_modules.exists():
        print("  ✓ node_modules directory exists")
        
        # Check for key packages
        key_packages = ['react', 'react-dom', 'react-scripts']
        all_present = True
        for pkg in key_packages:
            pkg_path = node_modules / pkg
            if pkg_path.exists():
                print(f"  ✓ {pkg}")
            else:
                print(f"  ✗ {pkg} - MISSING")
                all_present = False
        
        # Check Tailwind
        tailwind = node_modules / 'tailwindcss'
        if tailwind.exists():
            print("  ✓ tailwindcss")
        else:
            print("  ✗ tailwindcss - MISSING")
            all_present = False
        
        return all_present
    else:
        print("  ✗ node_modules directory not found")
        return False

def check_data_directory():
    """Check if dataset directory exists"""
    print("\nChecking data directory...")
    data_path = Path(__file__).parent / 'data' / 'raw' / 'DiabeticRetinopathyDataset'
    
    if data_path.exists():
        train_csv = data_path / 'train.csv'
        images_dir = data_path / 'gaussian_filtered_images' / 'gaussian_filtered_images'
        
        if train_csv.exists():
            print("  ✓ train.csv found")
        else:
            print("  ✗ train.csv not found")
            return False
        
        if images_dir.exists():
            class_dirs = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
            all_present = True
            for class_dir in class_dirs:
                if (images_dir / class_dir).exists():
                    count = len(list((images_dir / class_dir).glob('*.png')))
                    print(f"  ✓ {class_dir}: {count} images")
                else:
                    print(f"  ✗ {class_dir} directory not found")
                    all_present = False
            return all_present
        else:
            print("  ✗ Images directory not found")
            return False
    else:
        print("  ✗ Dataset directory not found")
        return False

def main():
    print("=" * 60)
    print("DR.A-EYE Setup Verification")
    print("=" * 60)
    print()
    
    all_checks = []
    
    # Check Python packages
    python_ok, missing = check_python_packages()
    all_checks.append(("Python Packages", python_ok))
    
    # Check model modules
    model_ok = check_model_modules()
    all_checks.append(("Model Modules", model_ok))
    
    # Check Node.js dependencies
    node_ok = check_node_modules()
    all_checks.append(("Node.js Dependencies", node_ok))
    
    # Check data directory
    data_ok = check_data_directory()
    all_checks.append(("Dataset", data_ok))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for check_name, passed in all_checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:30s} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All checks passed! Project is ready to use.")
        return 0
    else:
        print("\n✗ Some checks failed. Please install missing dependencies.")
        print("\nTo install all dependencies, run:")
        print("  ./setup.sh")
        return 1

if __name__ == '__main__':
    sys.exit(main())

