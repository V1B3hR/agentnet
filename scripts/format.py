#!/usr/bin/env python3
"""Development utility: Code formatting with black and isort.

DEPRECATED: This script is deprecated and will be removed in a future release.
Please use the unified CLI instead:
    python -m cli.main format
    or: python cli/main.py format
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run code formatting tools."""
    print("⚠️  DEPRECATION WARNING: This script is deprecated.")
    print("    Please use: python -m cli.main format")
    print()

    root_dir = Path(__file__).parent.parent

    print("🎨 Formatting code with black and isort...")

    # Run black
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "black",
                "--line-length",
                "88",
                str(root_dir / "agentnet"),
                str(root_dir / "tests"),
                str(root_dir / "scripts"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print("✅ Black formatting completed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Black formatting failed: {e.stderr}")
        return 1

    # Run isort
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "isort",
                "--profile",
                "black",
                str(root_dir / "agentnet"),
                str(root_dir / "tests"),
                str(root_dir / "scripts"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print("✅ isort import sorting completed")
    except subprocess.CalledProcessError as e:
        print(f"❌ isort failed: {e.stderr}")
        return 1

    print("🎉 Code formatting complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
