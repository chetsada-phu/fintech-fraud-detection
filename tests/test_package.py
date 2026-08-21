"""Smoke tests for the project package."""

import fraud_detection


def test_package_is_importable() -> None:
    """Confirm that the configured source tree exposes the project package."""
    assert fraud_detection.__name__ == "fraud_detection"
