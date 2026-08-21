"""Packaged HTML for the local artifact-backed analyst demo."""

from importlib.resources import files


def load_analyst_demo_html() -> str:
    """Load the analyst page from the installed serving package."""
    return (
        files("fraud_detection.serving")
        .joinpath("static/analyst_demo.html")
        .read_text(encoding="utf-8")
    )
