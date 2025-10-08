# AgentNet CLI Documentation

## Overview

The AgentNet CLI provides a unified interface for all development utilities, replacing the legacy scripts in the `scripts/` directory. This simplifies maintenance, onboarding, and automation.

## Installation

The CLI requires the following dependencies:

```bash
pip install click pytest black isort flake8
```

Or install all development dependencies:

```bash
pip install -e ".[dev]"  # If configured in setup.py
# OR
make install-dev
```

## Usage

### Basic Commands

#### Lint Code

Run flake8 linting on the codebase:

```bash
python -m cli lint
```

#### Format Code

Format code with black and isort:

```bash
python -m cli format
```

#### Run Tests

Execute the test harness with pytest:

```bash
python -m cli test
```

#### Validate Security Features

Run security and isolation validation tests:

```bash
python -m cli validate-security
```

#### Verify Roadmap Issues

Check that roadmap issues have been resolved:

```bash
python -m cli validate-roadmap
```

#### Train Debate Model

Train the debate model using Kaggle datasets:

```bash
python -m cli train-debate-model [--datasets-dir DATASETS]
```

Options:
- `--datasets-dir`: Directory containing debate datasets (default: `datasets`)

### Help

Get help for the CLI:

```bash
python -m cli --help
```

Get help for a specific command:

```bash
python -m cli lint --help
```

### Version

Check the CLI version:

```bash
python -m cli --version
```

## Makefile Integration

The CLI is integrated with the Makefile for convenience:

```bash
# New CLI-based targets
make cli-lint       # Run linting
make cli-format     # Run formatting
make cli-test       # Run tests
make check          # Run format, lint, and minimal tests
make quick-check    # Quick check with format and P0 tests

# Legacy targets (deprecated, show warnings)
make lint           # Deprecated, use cli-lint
make format         # Deprecated, use cli-format
make test           # Deprecated, use cli-test
```

## Architecture

The CLI is built using the following structure:

```
cli/
├── __init__.py      # CLI package initialization
├── __main__.py      # Entry point for 'python -m cli'
└── main.py          # CLI commands and logic

devtools/
├── __init__.py      # Devtools package
├── linting.py       # Linting utilities
├── formatting.py    # Formatting utilities
├── testing.py       # Testing utilities
├── validation.py    # Validation utilities
└── training.py      # Training utilities
```

### Design Principles

1. **Single Responsibility**: Each devtools module handles one concern
2. **Reusability**: Logic is extracted from scripts into reusable functions
3. **Testability**: All utilities are unit-testable
4. **Backward Compatibility**: Legacy scripts remain functional with deprecation warnings
5. **Discoverability**: All commands are documented and have help text

## Migration from Legacy Scripts

### Old Way (Deprecated)

```bash
python scripts/lint.py
python scripts/format.py
python scripts/test.py
python scripts/validate_security.py
python scripts/validate_roadmap_fixes.py
python scripts/train_debate_model.py
```

### New Way (Recommended)

```bash
python -m cli lint
python -m cli format
python -m cli test
python -m cli validate-security
python -m cli validate-roadmap
python -m cli train-debate-model
```

### Deprecation Timeline

- **Current**: Legacy scripts show deprecation warnings
- **Next minor release**: Legacy scripts remain but with louder warnings
- **Future major release**: Legacy scripts will be removed

## Testing

The CLI has comprehensive test coverage:

```bash
# Run CLI tests
python -m pytest tests/cli/ -v

# Run devtools tests
python -m pytest tests/devtools/ -v

# Run all tests
python -m pytest tests/ -v
```

## Development

### Adding a New Command

1. Create a new function in the appropriate devtools module
2. Add a new command to `cli/main.py`
3. Update this documentation
4. Add tests in `tests/cli/` and `tests/devtools/`

Example:

```python
# In devtools/myutility.py
def my_utility_function():
    """Do something useful."""
    print("Running utility...")
    return 0

# In cli/main.py
@cli.command("my-utility")
def my_utility_cmd():
    """Run my utility."""
    sys.exit(my_utility_function())
```

### Code Style

- Use Click for CLI framework
- Follow PEP 8 style guidelines
- Add type hints where possible
- Include docstrings for all functions
- Keep functions small and focused

## Troubleshooting

### Import Errors

If you get import errors, ensure you're running from the repository root:

```bash
cd /path/to/agentnet
python -m cli lint
```

### Missing Dependencies

If commands fail due to missing dependencies:

```bash
make install-dev
```

### Permission Errors

If you get permission errors when running scripts directly:

```bash
chmod +x cli/main.py
```

## Future Enhancements

Planned improvements:

- [ ] Add `--verbose` flag for detailed output
- [ ] Add `--quiet` flag for minimal output
- [ ] Configuration file support (`.agentnet.yaml`)
- [ ] Parallel test execution
- [ ] Watch mode for continuous linting/testing
- [ ] Integration with pre-commit hooks
- [ ] Shell completion (bash, zsh, fish)

## Contributing

When contributing to the CLI:

1. Follow the existing code structure
2. Add tests for new commands
3. Update this documentation
4. Ensure backward compatibility
5. Add deprecation warnings for removed features

## License

Same as the parent AgentNet project.
