"""Tests for devtools modules."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from devtools.formatting import format_code
from devtools.linting import lint_code
from devtools.testing import run_tests


class TestLinting:
    """Tests for linting module."""

    def test_lint_code_returns_int(self):
        """Test that lint_code returns an integer exit code."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = lint_code(Path(__file__).parent.parent.parent)
            assert isinstance(result, int)

    def test_lint_code_success(self):
        """Test successful linting."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = lint_code(Path(__file__).parent.parent.parent)
            assert result == 0

    def test_lint_code_failure(self):
        """Test linting with errors."""
        with patch("subprocess.run") as mock_run:
            from subprocess import CalledProcessError

            mock_run.side_effect = CalledProcessError(1, "flake8", stdout="errors")
            result = lint_code(Path(__file__).parent.parent.parent)
            assert result == 1


class TestFormatting:
    """Tests for formatting module."""

    def test_format_code_returns_int(self):
        """Test that format_code returns an integer exit code."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = format_code(Path(__file__).parent.parent.parent)
            assert isinstance(result, int)

    def test_format_code_success(self):
        """Test successful formatting."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = format_code(Path(__file__).parent.parent.parent)
            assert result == 0

    def test_format_code_black_failure(self):
        """Test formatting when black fails."""
        with patch("subprocess.run") as mock_run:
            from subprocess import CalledProcessError

            mock_run.side_effect = CalledProcessError(1, "black", stderr="error")
            result = format_code(Path(__file__).parent.parent.parent)
            assert result == 1


class TestTesting:
    """Tests for testing module."""

    def test_run_tests_returns_int(self):
        """Test that run_tests returns an integer exit code."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = run_tests(Path(__file__).parent.parent.parent)
            assert isinstance(result, int)

    def test_run_tests_success(self):
        """Test successful test execution."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = run_tests(Path(__file__).parent.parent.parent)
            assert result == 0

    def test_run_tests_failure(self):
        """Test test execution with failures."""
        with patch("subprocess.run") as mock_run:
            from subprocess import CalledProcessError

            mock_run.side_effect = CalledProcessError(1, "pytest")
            result = run_tests(Path(__file__).parent.parent.parent)
            assert result == 1
