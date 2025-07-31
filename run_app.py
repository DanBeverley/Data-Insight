#!/usr/bin/env python3
"""
DataInsight AI Application Launcher

Launch the FastAPI application with proper configuration and auto-open browser.
"""

import sys
import time
import threading
import webbrowser
import uvicorn
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def open_browser():
    """Open browser after a short delay to ensure server is ready."""
    time.sleep(2)  # Wait for server to start
    webbrowser.open("http://localhost:8000")
    print("🚀 DataInsight AI is now running at http://localhost:8000")

if __name__ == "__main__":
    print("🤖 Starting DataInsight AI...")
    print("⚡ Preparing to launch your browser...")
    
    # Start browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start the FastAPI server
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["src", "static"],
        log_level="info"
    )