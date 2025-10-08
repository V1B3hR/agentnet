"""Tests for CLI main module."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Import CLI module dynamically
import importlib.util

cli_main_path = parent_dir / "cli" / "main.py"
spec = importlib.util.spec_from_file_location("cli_main", cli_main_path)
cli_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli_main)
cli = cli_main.cli


class TestCLI:
    """Tests for CLI commands."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()

    def test_cli_help(self):
        """Test CLI help output."""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "AgentNet CLI" in result.output
        assert "lint" in result.output
        assert "format" in result.output
        assert "test" in result.output

    def test_cli_version(self):
        """Test CLI version output."""
        result = self.runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.5.0" in result.output

    def test_lint_command(self):
        """Test lint command."""
        with patch("devtools.linting.lint_code", return_value=0):
            result = self.runner.invoke(cli, ["lint"])
            assert result.exit_code == 0

    def test_format_command(self):
        """Test format command."""
        with patch("devtools.formatting.format_code", return_value=0):
            result = self.runner.invoke(cli, ["format"])
            assert result.exit_code == 0

    def test_test_command(self):
        """Test test command."""
        with patch("devtools.testing.run_tests", return_value=0):
            result = self.runner.invoke(cli, ["test"])
            assert result.exit_code == 0

    def test_validate_security_command(self):
        """Test validate-security command."""
        with patch("devtools.validation.validate_security", return_value=0):
            result = self.runner.invoke(cli, ["validate-security"])
            assert result.exit_code == 0

    def test_validate_roadmap_command(self):
        """Test validate-roadmap command."""
        with patch("devtools.validation.validate_roadmap", return_value=0):
            result = self.runner.invoke(cli, ["validate-roadmap"])
            assert result.exit_code == 0

    def test_train_debate_model_command(self):
        """Test train-debate-model command."""
        with patch("devtools.training.train_debate_model", return_value=0):
            result = self.runner.invoke(cli, ["train-debate-model"])
            assert result.exit_code == 0

    def test_train_debate_model_with_datasets_dir(self):
        """Test train-debate-model command with datasets-dir option."""
        with patch("devtools.training.train_debate_model", return_value=0) as mock:
            result = self.runner.invoke(
                cli, ["train-debate-model", "--datasets-dir", "custom"]
            )
            assert result.exit_code == 0
            mock.assert_called_once_with("custom")

    def test_command_failure(self):
        """Test command returns non-zero on failure."""
        with patch("devtools.linting.lint_code", return_value=1):
            result = self.runner.invoke(cli, ["lint"])
            assert result.exit_code == 1
