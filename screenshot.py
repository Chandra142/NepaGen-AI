import asyncio
from playwright.async_api import async_playwright
import time

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        print("Navigating to app...")
        await page.goto("http://localhost:8501")
        print("Waiting for load...")
        time.sleep(3)
        print("Taking screenshot...")
        await page.screenshot(path="screenshot.png")
        print("Done.")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
