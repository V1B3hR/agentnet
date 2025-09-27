# CI/CD Pipeline Implementation

This document describes the complete CI/CD pipeline implementation for AgentNet, including automated testing, security scanning, deployment, and risk monitoring.

## Pipeline Overview

The AgentNet CI/CD pipeline consists of three main workflows:

1. **CI Pipeline** (`.github/workflows/ci.yml`) - Continuous Integration
2. **Deploy Pipeline** (`.github/workflows/deploy.yml`) - Continuous Deployment  
3. **Risk Monitoring** (`.github/workflows/risk-monitoring.yml`) - Continuous Risk Assessment

## CI Pipeline Features

### Automated Testing
- **Unit Tests**: Full test suite across Python 3.8-3.12
- **Integration Tests**: Component integration validation
- **Contract Tests**: Provider adapter validation
- **Smoke Tests**: Basic functionality validation

### Code Quality
- **Linting**: Ruff for code style and error detection
- **Type Checking**: MyPy for static type analysis
- **Formatting**: Ruff formatter for consistent code style

### Security Scanning
- **Static Analysis**: Bandit for security vulnerability detection
- **Dependency Scanning**: Trivy for container and dependency vulnerabilities
- **SARIF Integration**: Results uploaded to GitHub Security tab

### Coverage Requirements
- **Minimum Coverage**: 85% test coverage required
- **Coverage Reports**: HTML and XML reports generated
- **Codecov Integration**: Automatic coverage tracking

## Deploy Pipeline Features

### Docker Build
- **Multi-stage Build**: Optimized production images
- **Security**: Non-root user, minimal attack surface
- **Metadata**: Proper container labeling and versioning

### Environment Management
- **Staging Deployment**: Automatic deployment on main branch
- **Production Deployment**: Manual approval required
- **Smoke Testing**: Post-deployment validation

### Cost and Risk Integration
- **Cost Analysis**: Automated cost impact assessment
- **Risk Assessment**: Pre-deployment risk validation
- **Monitoring Setup**: Post-deployment monitoring configuration

## Risk Monitoring Pipeline

### Continuous Risk Assessment
- **Scheduled Monitoring**: Every 15 minutes during business hours
- **Emergency Mode**: On-demand high-frequency monitoring
- **Multi-tenant Support**: Tenant-specific risk monitoring

### Alert Generation
- **Critical Risk Alerts**: Automatic GitHub issue creation
- **Cost Anomaly Detection**: Spend velocity and spike alerts
- **Mitigation Tracking**: Automated response verification

### Integration Points
- **Cost Tracking**: Real-time cost and usage monitoring
- **Risk Registry**: Centralized risk event management
- **Mitigation Engine**: Automated response execution

## Configuration

### Environment Variables
```yaml
AGENTNET_DATA_DIR: /app/data
AGENTNET_CONFIG_DIR: /app/configs
PYTHONPATH: /app
```

### GitHub Secrets (Required)
```yaml
GITHUB_TOKEN: # Automatic (GitHub Actions)
CODECOV_TOKEN: # Optional (for Codecov integration)
```

### Pipeline Triggers

#### CI Pipeline
```yaml
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  workflow_dispatch:
```

#### Deploy Pipeline
```yaml
on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]
  workflow_dispatch:
    inputs:
      environment: [staging, prod]
```

#### Risk Monitoring
```yaml
on:
  schedule:
    - cron: '*/15 8-20 * * 1-5'  # Business hours
  workflow_dispatch:
    inputs:
      emergency_mode: boolean
      tenant_id: string
```

## Usage Examples

### Running CI Pipeline
The CI pipeline runs automatically on all pushes and PRs. Manual execution:

```bash
# Trigger via GitHub API
curl -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/V1B3hR/agentnet/actions/workflows/ci.yml/dispatches \
  -d '{"ref":"main"}'
```

### Deployment
```bash
# Staging deployment (automatic on main branch push)
git push origin main

# Production deployment (manual approval required)
# 1. Create release tag
git tag v1.0.0
git push origin v1.0.0

# 2. Approve deployment in GitHub Actions UI
```

### Risk Monitoring
```bash
# Emergency risk assessment
curl -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/V1B3hR/agentnet/actions/workflows/risk-monitoring.yml/dispatches \
  -d '{"ref":"main","inputs":{"emergency_mode":"true"}}'
```

## Local Development

### Prerequisites
```bash
# Install development dependencies
make install-dev

# Install all dependencies (optional)
make install-full
```

### Running Tests Locally
```bash
# Quick validation
make quick-check

# Full test suite
make test

# Specific test categories
make test-p0  # Phase 0 tests only
PYTHONPATH=. python -m pytest tests/ -m integration  # Integration tests
PYTHONPATH=. python -m pytest tests/ -m unit  # Unit tests only
```

### Code Quality Checks
```bash
# Format code
make format

# Lint code
make lint

# Type checking (if mypy available)
mypy agentnet --ignore-missing-imports
```

### Docker Build
```bash
# Build image locally
docker build -t agentnet:local .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/configs:/app/configs \
  agentnet:local
```

## Monitoring and Observability

### Pipeline Metrics
- **Build Success Rate**: Tracked via GitHub Actions
- **Test Coverage**: Reported to Codecov
- **Deployment Frequency**: Tracked via deployment artifacts
- **Lead Time**: Commit to production timing

### Cost Monitoring
- **Real-time Tracking**: Inference cost recording
- **Predictive Analytics**: Monthly cost forecasting
- **Alert Thresholds**: Configurable spend limits
- **Executive Reporting**: Automated cost summaries

### Risk Monitoring
- **Event Tracking**: All risk events logged and persisted
- **Mitigation Success**: Automated response effectiveness
- **Resolution Rates**: Risk resolution performance
- **Escalation Tracking**: Critical risk escalation paths

## Troubleshooting

### Common Issues

#### Build Failures
```bash
# Check Python version compatibility
python --version  # Should be 3.8+

# Verify dependencies
pip list | grep -E "(pydantic|pyyaml|typing-extensions)"

# Check import issues
PYTHONPATH=. python -c "import agentnet; print('OK')"
```

#### Test Failures
```bash
# Run with verbose output
PYTHONPATH=. python -m pytest tests/ -v --tb=long

# Run specific failing test
PYTHONPATH=. python -m pytest tests/test_specific.py::TestClass::test_method -v

# Check test data cleanup
ls -la /tmp/test_*  # Should be empty after tests
```

#### Docker Issues
```bash
# Check Docker build
docker build --no-cache -t agentnet:debug .

# Debug container
docker run -it --entrypoint /bin/bash agentnet:debug

# Check container logs
docker logs <container_id>
```

### Pipeline Debug
```bash
# Enable debug logging in workflows
# Add to workflow file:
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true
```

## Security Considerations

### Code Security
- **Static Analysis**: Bandit scans for security vulnerabilities
- **Dependency Scanning**: Trivy checks for known CVEs
- **Secret Scanning**: Automated detection of exposed secrets

### Container Security
- **Non-root User**: Application runs as unprivileged user
- **Minimal Base**: Slim Python image reduces attack surface
- **Read-only Filesystem**: Immutable container configuration

### Data Security
- **Encryption at Rest**: Cost and risk data encrypted
- **Access Controls**: Tenant isolation enforced
- **Audit Logging**: All access and changes logged

## Performance Optimization

### Build Optimization
- **Caching**: pip dependencies cached between runs
- **Parallel Execution**: Tests run in parallel where possible
- **Resource Limits**: Appropriate resource allocation

### Runtime Optimization
- **Multi-stage Build**: Minimal production image
- **Health Checks**: Container health monitoring
- **Resource Monitoring**: Memory and CPU usage tracking

## Compliance and Governance

### Audit Trail
- **All Changes Tracked**: Git history provides complete audit trail
- **Deployment Records**: Artifact storage maintains deployment history
- **Risk Events**: All risk events logged with timestamps

### Policy Enforcement
- **Required Reviews**: Pull requests require approval
- **Branch Protection**: Main branch protected from direct pushes
- **Status Checks**: All CI checks must pass before merge

### Documentation
- **Pipeline Documentation**: This document and inline comments
- **Runbook**: Operational procedures documented
- **Incident Response**: Risk escalation procedures defined