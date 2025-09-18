#!/usr/bin/env python3
"""Development utility: Code linting with flake8."""

import subprocess
import sys
from pathlib import Path


def main():
    """Run code linting tools."""
    root_dir = Path(__file__).parent.parent

    print("🔍 Linting code with flake8...")

    # Run flake8
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "flake8",
                "--max-line-length",
                "88",
                "--extend-ignore",
                "E203,W503",  # Ignore conflicts with black
                str(root_dir / "agentnet"),
                str(root_dir / "tests"),
                str(root_dir / "scripts"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print("✅ flake8 linting completed - no issues found")
    except subprocess.CalledProcessError as e:
        print(f"❌ flake8 found issues:")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return 1

    print("🎉 Code linting complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
