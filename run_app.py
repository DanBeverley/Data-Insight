import time
import threading
import webbrowser
import uvicorn
import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Use PORT env var for Railway/cloud deployments, fallback to 8000 for local
PORT = int(os.getenv("PORT", 8000))


def open_browser():
    """Open browser after a short delay to ensure server is ready."""
    time.sleep(2)
    webbrowser.open(f"http://localhost:{PORT}")
    print(f"🚀 DataInsight AI is now running at http://localhost:{PORT}")


if __name__ == "__main__":
    # Only open browser for local development (not Railway)
    if os.getenv("RAILWAY_ENVIRONMENT") is None:
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
    uvicorn.run("src.api:app", host="0.0.0.0", port=PORT, reload=False, loop="asyncio")
