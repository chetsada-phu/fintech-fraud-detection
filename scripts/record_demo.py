"""Record the real local analyst flow as a short browser video."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

from playwright.sync_api import sync_playwright

DEFAULT_URL = "http://127.0.0.1:8768/analyst-demo"
DEFAULT_OUTPUT = Path("docs/demo/fintech-fraud-demo.webm")


def build_parser() -> argparse.ArgumentParser:
    """Create the recording command parser."""
    parser = argparse.ArgumentParser(description="Record the local analyst demo.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Record one real request and save the browser video."""
    arguments = build_parser().parse_args(argv)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fraud-demo-video-") as video_directory:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1280, "height": 720},
                record_video_dir=video_directory,
                record_video_size={"width": 1280, "height": 720},
            )
            page = context.new_page()
            page.goto(arguments.url, wait_until="networkidle")
            page.wait_for_timeout(1_500)
            amount = page.locator('input[name="tx_amount"]')
            amount.click()
            amount.press("ControlOrMeta+A")
            amount.type("300.00", delay=90)
            page.wait_for_timeout(700)
            page.locator("#score-button").click()
            page.locator("#request-status").filter(has_text="scored").wait_for()
            page.wait_for_timeout(1_500)
            page.locator("#result-ticket").scroll_into_view_if_needed()
            page.wait_for_timeout(3_000)
            video = page.video
            context.close()
            browser.close()
            if video is None:
                raise RuntimeError("Playwright did not create a video")
            shutil.copyfile(video.path(), arguments.output)
    print(f"recorded_demo={arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
