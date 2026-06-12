import os
import sys
import subprocess
import time
import signal

def check_dependencies():
    print("Checking dependencies...")
    try:
        import fastapi
        import uvicorn
        import streamlit
        import torch
        import torchvision
        import PIL
        import numpy
        import sklearn
        print("All dependencies are satisfied!")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e.name}")
        print("Attempting to install dependencies from requirements.txt...")
        
        # Determine paths to requirements
        backend_req = os.path.join("backend", "requirements.txt")
        frontend_req = os.path.join("frontend", "requirements.txt")
        
        try:
            # Install backend requirements
            print(f"Installing backend requirements from {backend_req}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", backend_req])
            
            # Install frontend requirements
            print(f"Installing frontend requirements from {frontend_req}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", frontend_req])
            
            print("Successfully installed all dependencies!")
            return True
        except Exception as install_err:
            print(f"Failed to install dependencies: {install_err}")
            return False

def main():
    # Change directory to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Verify/Install dependencies
    if not check_dependencies():
        print("Error: Could not resolve dependencies. Exiting.")
        sys.exit(1)
        
    print("\nStarting Neural Studio Server Suite...")
    
    # 1. Start FastAPI Backend
    # Run uvicorn inside the backend directory
    backend_dir = os.path.join(script_dir, "backend")
    print(f"Launching FastAPI Backend on http://127.0.0.1:8000 (cwd: {backend_dir})")
    
    # Note: On Windows, use shell=True or python.exe to run commands safely
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=backend_dir
    )
    
    # Give the backend a brief moment to bind to the port
    time.sleep(2)
    
    # 2. Start Streamlit Frontend
    frontend_dir = os.path.join(script_dir, "frontend")
    print(f"Launching Streamlit UI on http://127.0.0.1:8501 (cwd: {frontend_dir})")
    
    frontend_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"],
        cwd=frontend_dir
    )
    
    print("\nNeural Studio is running!")
    print("Press Ctrl+C to terminate both servers.")
    
    try:
        # Keep orchestrator running and monitor subprocesses
        while True:
            # Check if backend crashed
            backend_code = backend_proc.poll()
            if backend_code is not None:
                print(f"\nBackend process terminated with code {backend_code}")
                break
                
            # Check if frontend crashed
            frontend_code = frontend_proc.poll()
            if frontend_code is not None:
                print(f"\nFrontend process terminated with code {frontend_code}")
                break
                
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Shutting down servers...")
        
    finally:
        # Clean termination of subprocesses
        print("Terminating Backend...")
        backend_proc.terminate()
        try:
            backend_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            print("Force-killing Backend...")
            backend_proc.kill()
            
        print("Terminating Frontend...")
        frontend_proc.terminate()
        try:
            frontend_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            print("Force-killing Frontend...")
            frontend_proc.kill()
            
    print("All processes shut down successfully.")

if __name__ == "__main__":
    main()
