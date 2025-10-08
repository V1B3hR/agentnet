#!/usr/bin/env python3
"""AgentNet CLI - Unified development tools interface.

This CLI provides a single entrypoint for all development utilities:
- lint: Run code linting with flake8
- format: Run code formatting with black and isort
- test: Run test harness with pytest
- validate-security: Validate security and isolation features
- validate-roadmap: Verify roadmap issues are resolved
- train-debate-model: Train debate model using Kaggle datasets

Usage:
    agentnet-cli lint
    agentnet-cli format
    agentnet-cli test
    agentnet-cli validate-security
    agentnet-cli validate-roadmap
    agentnet-cli train-debate-model [--datasets-dir DATASETS]
"""

import sys
from pathlib import Path

import click

# Add parent directory to path for devtools imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from devtools.formatting import format_code
from devtools.linting import lint_code
from devtools.testing import run_tests
from devtools.training import train_debate_model
from devtools.validation import validate_roadmap, validate_security


@click.group()
@click.version_option(version="0.5.0", prog_name="agentnet-cli")
def cli():
    """AgentNet CLI - Unified development tools interface."""
    pass


@cli.command()
def lint():
    """Run code linting with flake8."""
    sys.exit(lint_code())


@cli.command()
def format():
    """Run code formatting with black and isort."""
    sys.exit(format_code())


@cli.command()
def test():
    """Run test harness with pytest."""
    sys.exit(run_tests())


@cli.command("validate-security")
def validate_security_cmd():
    """Validate security and isolation features."""
    sys.exit(validate_security())


@cli.command("validate-roadmap")
def validate_roadmap_cmd():
    """Verify roadmap issues are resolved."""
    sys.exit(validate_roadmap())


@cli.command("train-debate-model")
@click.option(
    "--datasets-dir",
    default="datasets",
    help="Directory containing debate datasets",
    show_default=True,
)
def train_debate_model_cmd(datasets_dir):
    """Train debate model using Kaggle datasets."""
    sys.exit(train_debate_model(datasets_dir))


if __name__ == "__main__":
    cli()
