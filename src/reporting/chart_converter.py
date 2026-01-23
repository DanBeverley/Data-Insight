import asyncio
import base64
from pathlib import Path
from typing import Optional, Dict
import hashlib
import logging

logger = logging.getLogger(__name__)

_browser_instance = None
_browser_lock = asyncio.Lock()

CACHE_DIR = Path("static/chart_cache")


async def get_browser():
    global _browser_instance
    async with _browser_lock:
        if _browser_instance is None:
            try:
                from playwright.async_api import async_playwright

                playwright = await async_playwright().start()
                _browser_instance = await playwright.chromium.launch(headless=True)
            except Exception as e:
                logger.error(f"Failed to launch browser: {e}")
                return None
        return _browser_instance


async def html_to_png(html_path: Path, width: int = 800, height: int = 600) -> Optional[bytes]:
    browser = await get_browser()
    if not browser:
        return None

    try:
        page = await browser.new_page(viewport={"width": width, "height": height})
        await page.goto(f"file://{html_path.absolute()}", wait_until="networkidle")
        await asyncio.sleep(0.5)
        screenshot = await page.screenshot(type="png", full_page=True)
        await page.close()
        return screenshot
    except Exception as e:
        logger.error(f"Screenshot failed for {html_path}: {e}")
        return None


def get_cache_path(html_path: Path) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hash_key = hashlib.md5(str(html_path).encode()).hexdigest()[:12]
    return CACHE_DIR / f"{html_path.stem}_{hash_key}.png"


async def convert_chart_to_png(html_path: Path, use_cache: bool = True) -> Optional[bytes]:
    if not html_path.exists():
        return None

    cache_path = get_cache_path(html_path)

    if use_cache and cache_path.exists():
        if cache_path.stat().st_mtime >= html_path.stat().st_mtime:
            return cache_path.read_bytes()

    png_bytes = await html_to_png(html_path)

    if png_bytes and use_cache:
        cache_path.write_bytes(png_bytes)

    return png_bytes


def convert_chart_to_png_sync(html_path: Path, use_cache: bool = True) -> Optional[bytes]:
    import nest_asyncio

    nest_asyncio.apply()

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(convert_chart_to_png(html_path, use_cache))


def png_to_base64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("utf-8")


async def convert_charts_batch(html_paths: list[Path], use_cache: bool = True) -> Dict[Path, Optional[bytes]]:
    results = {}
    for path in html_paths:
        results[path] = await convert_chart_to_png(path, use_cache)
    return results


async def html_to_pdf(html_content: str) -> Optional[bytes]:
    browser = await get_browser()
    if not browser:
        return None

    try:
        page = await browser.new_page()
        await page.set_content(html_content, wait_until="networkidle")
        pdf_bytes = await page.pdf(
            format="A4",
            margin={"top": "20mm", "bottom": "20mm", "left": "15mm", "right": "15mm"},
            print_background=True,
        )
        await page.close()
        return pdf_bytes
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        return None


def html_to_pdf_sync(html_content: str) -> Optional[bytes]:
    import nest_asyncio

    nest_asyncio.apply()

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(html_to_pdf(html_content))


async def cleanup_browser():
    global _browser_instance
    async with _browser_lock:
        if _browser_instance:
            await _browser_instance.close()
            _browser_instance = None
