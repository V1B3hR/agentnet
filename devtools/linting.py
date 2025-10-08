"""Code linting utilities for AgentNet."""

import subprocess
import sys
from pathlib import Path
from typing import Optional


def lint_code(root_dir: Optional[Path] = None) -> int:
    """Run code linting tools.
    
    Args:
        root_dir: Root directory of the project. If None, auto-detects.
        
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    if root_dir is None:
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
        print("❌ flake8 found issues:")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return 1

    print("🎉 Code linting complete!")
    return 0
