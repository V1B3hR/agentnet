# Installation

Get AgentNet up and running on your system.

## Requirements

- Python 3.8 or higher
- pip (Python package installer)

## Installation Options

### Option 1: Install from PyPI (Recommended)

```bash
# Install core AgentNet
pip install agentnet

# Or install with all optional dependencies
pip install agentnet[full]
```

### Option 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/V1B3hR/agentnet.git
cd agentnet

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[full]"
```

### Option 3: Development Installation

```bash
# Clone and set up development environment
git clone https://github.com/V1B3hR/agentnet.git
cd agentnet

# Install development dependencies
pip install -e ".[dev]"

# Run tests to verify installation
python -m pytest tests/test_p0_implementation.py -v
```

## Dependency Groups

AgentNet uses optional dependency groups for different feature sets:

### Core Dependencies (Always Installed)

- `pydantic>=2.0.0` - Data validation and settings management
- `pyyaml>=6.0` - Configuration file support
- `typing-extensions>=4.0.0` - Enhanced typing support

### Full Dependencies (`pip install agentnet[full]`)

Additional dependencies for advanced features:

- `networkx>=3.0` - Graph-based memory and DAG execution
- `numpy>=1.21.0` - Numerical computations
- `faiss-cpu>=1.7.0` - Vector similarity search
- `openai>=1.0.0` - OpenAI API integration
- `chromadb>=0.4.0` - Vector database integration

### Development Dependencies (`pip install agentnet[dev]`)

Tools for development and testing:

- `pytest>=7.0.0` - Testing framework
- `pytest-asyncio>=0.21.0` - Async testing support
- `black>=22.0.0` - Code formatting
- `isort>=5.10.0` - Import sorting
- `flake8>=4.0.0` - Code linting
- `mypy>=1.0.0` - Type checking
- `coverage>=6.0.0` - Code coverage

## Verification

Verify your installation by running:

```python
import agentnet

# Check version
print(f"AgentNet version: {agentnet.__version__}")

# Check phase availability
print(f"Phase status: {agentnet.__phase_status__}")

# Create a simple agent
from agentnet import AgentNet, ExampleEngine

agent = AgentNet(
    name="TestAgent",
    style={"logic": 0.8, "creativity": 0.6},
    engine=ExampleEngine()
)

result = agent.generate_reasoning_tree("Hello, AgentNet!")
print("✅ AgentNet is working correctly!")
```

Expected output:
```
AgentNet version: 0.5.0
Phase status: {'P0': True, 'P1': True, 'P2': True, ...}
✅ AgentNet is working correctly!
```

## Common Issues

### Import Errors

If you encounter import errors:

1. **Missing optional dependencies**: Install the full package with `pip install agentnet[full]`
2. **Python version**: Ensure you're using Python 3.8 or higher
3. **Virtual environment**: Consider using a virtual environment to avoid conflicts

### Performance Issues

For better performance with vector operations:

```bash
# Install optimized FAISS (if available for your platform)
pip install faiss-gpu  # For CUDA-enabled systems
```

### Development Setup Issues

If development setup fails:

```bash
# Update pip and setuptools
pip install --upgrade pip setuptools

# Clean installation
pip uninstall agentnet
pip install -e ".[dev]"
```

## Docker Installation

For containerized deployment:

```dockerfile
FROM python:3.11-slim

# Install AgentNet
RUN pip install agentnet[full]

# Copy your application
COPY . /app
WORKDIR /app

# Run your AgentNet application
CMD ["python", "your_agent_app.py"]
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Build your first agent
- [Basic Concepts](concepts.md) - Understand AgentNet fundamentals
- [Core Features](../guide/core-features.md) - Explore key capabilities