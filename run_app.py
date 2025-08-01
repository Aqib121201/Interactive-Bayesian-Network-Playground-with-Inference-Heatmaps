#!/usr/bin/env python3
"""
Application launcher for the Bayesian Network Playground.
"""

import sys
import subprocess
import os
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly',
        'networkx'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True


def check_pgmpy():
    """Check if pgmpy is available for inference."""
    try:
        import pgmpy
        return True
    except ImportError:
        print("⚠️  Warning: pgmpy not found. Inference functionality will be limited.")
        print("💡 Install pgmpy with: pip install pgmpy")
        return False


def main():
    """Main launcher function."""
    print("🧠 Bayesian Network Playground")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check pgmpy (optional but recommended)
    check_pgmpy()
    
    # Get app path
    app_path = Path(__file__).parent / "app" / "app.py"
    
    if not app_path.exists():
        print(f"❌ Error: App file not found at {app_path}")
        sys.exit(1)
    
    print("✅ Dependencies check passed!")
    print("🚀 Starting Streamlit application...")
    print("📱 Open your browser to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 40)
    
    try:
        # Run Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 