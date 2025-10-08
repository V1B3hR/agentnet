"""Code formatting utilities for AgentNet."""

import subprocess
import sys
from pathlib import Path
from typing import Optional


def format_code(root_dir: Optional[Path] = None) -> int:
    """Run code formatting tools.
    
    Args:
        root_dir: Root directory of the project. If None, auto-detects.
        
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    if root_dir is None:
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
