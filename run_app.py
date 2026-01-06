import time
import threading
import webbrowser
import uvicorn
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent / "src"))


def open_browser():
    """Open browser after a short delay to ensure server is ready."""
    time.sleep(2)
    webbrowser.open("http://localhost:8000")
    print("🚀 DataInsight AI is now running at http://localhost:8000")


if __name__ == "__main__":
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False, loop="asyncio")
