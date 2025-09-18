#!/usr/bin/env python3
"""Development utility: Test harness for AgentNet."""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """Run test harness."""
    root_dir = Path(__file__).parent.parent

    print("🧪 Running AgentNet test harness...")

    # Set PYTHONPATH to include the project root
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root_dir)

    # Run pytest on tests directory
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(root_dir / "tests"),
                "-v",
                "--tb=short",
            ],
            check=True,
            env=env,
        )
        print("✅ pytest completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ pytest failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print("⚠️  pytest not available, running tests individually...")

        # Fallback: run individual test files
        test_files = list(root_dir.glob("tests/test_*.py"))
        failed_tests = []

        for test_file in test_files:
            print(f"\n📋 Running {test_file.name}...")
            try:
                result = subprocess.run(
                    [sys.executable, str(test_file)],
                    check=True,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                print(f"✅ {test_file.name} passed")
            except subprocess.CalledProcessError as e:
                print(f"❌ {test_file.name} failed")
                if e.stdout:
                    print("STDOUT:", e.stdout)
                if e.stderr:
                    print("STDERR:", e.stderr)
                failed_tests.append(test_file.name)

        if failed_tests:
            print(f"\n❌ {len(failed_tests)} test(s) failed: {failed_tests}")
            return 1
        else:
            print(f"\n✅ All {len(test_files)} test(s) passed!")

    print("🎉 Test harness complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
