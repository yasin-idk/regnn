#!/usr/bin/env python3
"""
ML Regression Dashboard Launcher
Automatically sets up environment and runs the application on any device.
"""

import os
import sys
import subprocess
import platform
import time
from pathlib import Path

def print_banner():
    """Print application banner"""
    print("=" * 70)
    print("🤖 ML REGRESSION DASHBOARD - AUTO LAUNCHER")
    print("=" * 70)
    print("📊 TensorFlow Nightly Edition with Model Persistence")
    print("🚀 Automatic dependency installation & environment setup")
    print("💻 Compatible with Windows, Mac, Linux")
    print("=" * 70)
    print()

def get_python_executable():
    """Get the correct python executable"""
    return sys.executable

def check_pip():
    """Ensure pip is available and up to date"""
    print("🔧 Checking pip installation...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True, text=True)
        print("✅ pip is available")
        
        # Upgrade pip
        print("⬆️  Upgrading pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True, text=True)
        print("✅ pip upgraded successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip is not available")
        return False
    except Exception as e:
        print(f"⚠️  pip check warning: {e}")
        return True  # Continue anyway

def install_package(package_name, package_spec=None):
    """Install a single package with error handling"""
    spec = package_spec or package_name
    try:
        print(f"📦 Installing {package_name}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", spec], 
                              check=True, capture_output=True, text=True)
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e}")
        print(f"   Error output: {e.stderr}")
        return False

def install_requirements():
    """Install all required packages"""
    print("📚 Installing required packages...")
    
    # Define packages with their specifications
    packages = {
        "streamlit": "streamlit>=1.28.0",
        "pandas": "pandas>=1.5.0",
        "numpy": "numpy>=1.21.0",
        "matplotlib": "matplotlib>=3.5.0",
        "seaborn": "seaborn>=0.11.0",
        "scikit-learn": "scikit-learn>=1.1.0",
        "tensorflow": "tf-nightly",  # Use nightly build for latest features
        "scipy": "scipy>=1.9.0",
        "openpyxl": "openpyxl>=3.0.0",  # For Excel file reading
        "joblib": "joblib>=1.1.0",
        "xlrd": "xlrd>=2.0.0",  # Additional Excel support
    }
    
    failed_packages = []
    
    for package_name, package_spec in packages.items():
        if not install_package(package_name, package_spec):
            failed_packages.append(package_name)
            
            # Try alternative installation for tensorflow
            if package_name == "tensorflow":
                print("🔄 Trying regular TensorFlow installation...")
                if install_package("tensorflow-fallback", "tensorflow>=2.10.0"):
                    print("✅ Regular TensorFlow installed as fallback")
                    failed_packages.remove(package_name)
    
    if failed_packages:
        print(f"\n⚠️  Some packages failed to install: {', '.join(failed_packages)}")
        print("The application may still work with reduced functionality.")
        print("You can install them manually later if needed.")
    else:
        print("\n🎉 All packages installed successfully!")
    
    return len(failed_packages) == 0

def check_gpu_support():
    """Check for GPU support and provide information"""
    print("\n🔍 Checking GPU support...")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ Found {len(gpus)} GPU(s) available:")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("ℹ️  No GPU detected - will use CPU (this is fine for most datasets)")
    except Exception as e:
        print(f"⚠️  Could not check GPU support: {e}")

def create_directories():
    """Create necessary directories"""
    print("📁 Creating necessary directories...")
    directories = ["saved_models", "temp", "logs"]
    
    for directory in directories:
        try:
            Path(directory).mkdir(exist_ok=True)
            print(f"✅ Directory '{directory}' ready")
        except Exception as e:
            print(f"⚠️  Could not create directory '{directory}': {e}")

def check_main_file():
    """Check if the main application file exists"""
    main_file = "ml.py"
    if not os.path.exists(main_file):
        print(f"❌ Main application file '{main_file}' not found!")
        print("Please ensure 'ml.py' is in the same directory as this launcher.")
        return False
    print(f"✅ Main application file '{main_file}' found")
    return True

def start_streamlit():
    """Start the Streamlit application"""
    print("\n🚀 Starting ML Regression Dashboard...")
    print("📝 Note: The application will open in your default web browser")
    print("🌐 Default URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C in this window to stop the application")
    print("-" * 50)
    
    try:
        # Try different methods to start streamlit
        commands_to_try = [
            [sys.executable, "-m", "streamlit", "run", "ml.py", "--browser.gatherUsageStats", "false"],
            [sys.executable, "-m", "streamlit", "run", "ml.py"],
            ["streamlit", "run", "ml.py"],
        ]
        
        for i, cmd in enumerate(commands_to_try):
            try:
                print(f"🔄 Attempt {i+1}: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
                break
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                if i == len(commands_to_try) - 1:
                    raise e
                print(f"   Failed, trying next method...")
                
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to start Streamlit: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Try running: pip install streamlit --upgrade")
        print("2. Check if port 8501 is available")
        print("3. Try running manually: streamlit run ml.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

def main():
    """Main launcher function"""
    print_banner()
    
    # System information
    print(f"💻 System: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"📂 Working Directory: {os.getcwd()}")
    print()
    
    # Check main file
    if not check_main_file():
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Check and setup pip
    if not check_pip():
        print("❌ pip is required but not available")
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Install requirements
    print("\n" + "="*50)
    print("📦 DEPENDENCY INSTALLATION")
    print("="*50)
    
    install_success = install_requirements()
    
    # Create directories
    print("\n" + "="*50)
    print("📁 ENVIRONMENT SETUP")
    print("="*50)
    create_directories()
    
    # Check GPU support
    check_gpu_support()
    
    # Final status
    print("\n" + "="*50)
    print("🎯 LAUNCH STATUS")
    print("="*50)
    
    if install_success:
        print("✅ All dependencies installed successfully")
    else:
        print("⚠️  Some dependencies failed - continuing anyway")
    
    print("✅ Environment setup complete")
    print("✅ Ready to launch application")
    
    # Wait a moment for user to read
    print("\n⏳ Starting in 3 seconds...")
    time.sleep(3)
    
    # Start the application
    start_streamlit()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Launcher interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error in launcher: {e}")
        print("\n🔧 Please report this issue with the error details above")
        input("Press Enter to exit...")
        sys.exit(1) 