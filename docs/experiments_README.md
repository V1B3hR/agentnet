# AgentNet Experimental Harness

This directory contains a comprehensive experimental harness for AgentNet, providing 12 different experiment domains to evaluate and analyze various aspects of the multi-agent cognitive system.

## Overview

The experimental harness covers these key areas:
- **Agent Performance**: Single and multi-agent reasoning capabilities
- **Style Influence**: How different style configurations affect output quality
- **Convergence Dynamics**: Multi-agent dialogue convergence patterns
- **Monitor Effectiveness**: Policy and safety monitor performance under stress
- **Resilience**: System behavior under fault conditions
- **Performance**: Async vs sync execution benchmarking
- **Quality Assurance**: Automated quality gates for CI/CD

## Directory Structure

```
experiments/
├── README.md                    # This file
├── config/                      # Configuration files
│   ├── metrics_schema.json      # JSON schema for JSONL metrics
│   ├── monitors_baseline.yaml   # Standard monitor configuration
│   ├── monitors_high_sensitivity.yaml  # Stress test monitor config
│   └── rules_extended.yaml      # Extended policy rules
├── scripts/                     # Experiment scripts
│   ├── runner.py               # Unified experiment dispatcher
│   ├── run_smoke.py            # Smoke/sanity tests
│   ├── run_style_grid.py       # Style influence exploration
│   ├── run_convergence.py      # Multi-agent convergence dynamics
│   ├── run_monitor_stress.py   # Monitor stress testing
│   └── run_quality_gates.py    # Quality gate validation
├── utils/                       # Shared utilities
│   ├── analytics.py            # Metrics computation and analysis
│   └── monitors_custom.py      # Custom monitor implementations
└── data/                        # Generated experiment data
    ├── raw_sessions/           # Raw session outputs (gitignored)
    └── analytics/              # Processed analytics (gitignored)
```

## Quick Start

### Running Individual Experiments

Use the unified runner to execute any experiment:

```bash
# List all available experiments
python -m experiments.scripts.runner --list

# Run smoke tests
python -m experiments.scripts.runner --exp smoke

# Run style exploration
python -m experiments.scripts.runner --exp style

# Run convergence dynamics
python -m experiments.scripts.runner --exp convergence

# Run monitor stress tests
python -m experiments.scripts.runner --exp monitor_stress

# Run quality gates validation
python -m experiments.scripts.runner --exp quality_gates
```

### Running Scripts Directly

You can also run experiment scripts directly:

```bash
# Smoke tests
python experiments/scripts/run_smoke.py

# Style exploration
python experiments/scripts/run_style_grid.py

# Quality gates with custom parameters
python experiments/scripts/run_quality_gates.py --hours-back 48 --verbose
```

## Experiment Descriptions

### 1. Quick Smoke / Sanity Tests (`smoke`)

**Purpose**: Validate basic functionality and fault handling mechanisms.

**What it tests**:
- Single-agent reasoning without monitors
- Single-agent reasoning with monitors enabled
- Forced rule violations and CognitiveFault handling
- Resource budget overruns
- Monitor violation detection

**Output**: 
- `experiments/data/raw_sessions/smoke_tests_*/`
- `smoke_test_results.json` - Detailed test results
- `smoke_tests_metrics.jsonl` - Standardized metrics

**Success criteria**: All tests pass, expected violations are triggered appropriately.

### 2. Style Influence Exploration (`style`)

**Purpose**: Understand how different style weight combinations affect agent output quality and characteristics.

**What it tests**:
- Grid search across logic/creativity/analytical dimensions
- Confidence score changes due to style influence
- Style insight generation patterns
- Content characteristics across style combinations

**Parameters**:
- Style dimensions: `logic`, `creativity`, `analytical` 
- Grid steps: 3 (default) - creates 3³ = 27 combinations
- Trials per combination: 2 (default)
- Multiple reasoning tasks per trial

**Output**:
- `experiments/data/raw_sessions/style_exploration_*/`
- `style_exploration_results.json` - Complete results with trial data
- `style_exploration_analysis.json` - Best combinations and dimension effects
- `style_exploration_metrics.jsonl` - Per-trial metrics

**Key insights**: Identifies optimal style combinations for different task types.

### 3. Convergence Dynamics (Multi-Agent) (`convergence`)

**Purpose**: Analyze how different multi-agent configurations affect dialogue convergence.

**What it tests**:
- Agent diversity effects on convergence
- Overlap threshold sensitivity
- Convergence window size impact
- Participation balance across agents
- Topic influence on convergence patterns

**Parameters**:
- Diversity profiles: `high_diversity`, `medium_diversity`, `low_diversity`
- Agent counts: 2, 3, 4
- Overlap thresholds: 0.3, 0.5, 0.7
- Window sizes: 2, 3, 4
- Multiple topics per configuration

**Output**:
- `experiments/data/raw_sessions/convergence_dynamics_*/`
- `convergence_results.json` - All trial results
- `convergence_analysis.json` - Parameter effect analysis
- `convergence_metrics.jsonl` - Per-session metrics

**Key insights**: Optimal convergence parameters for different scenarios.

### 4. Monitoring Stress & Failure Mode Mapping (`monitor_stress`)

**Purpose**: Test monitor effectiveness under stress conditions and identify failure modes.

**What it tests**:
- Multiple simultaneous monitor violations
- High-sensitivity monitor configurations
- Expected vs actual violation detection
- CognitiveFault triggering under stress
- Monitor performance and coverage

**Test scenarios**:
- **Keyword stacking**: Content with multiple keyword violations
- **Pattern violations**: Regex pattern matches
- **Resource overruns**: Very low resource budgets
- **Repetition triggers**: Similar content to test custom monitors
- **Compound violations**: Multiple violation types simultaneously

**Output**:
- `experiments/data/raw_sessions/monitor_stress_*/`
- `monitor_stress_results.json` - Scenario results and violation details
- `monitor_stress_analysis.json` - Monitor effectiveness analysis
- `monitor_stress_metrics.jsonl` - Per-scenario metrics

**Success criteria**: Expected violations are triggered, no false positives/negatives.

### 5. Validation / Quality Gates (`quality_gates`)

**Purpose**: Automated quality validation for CI/CD pipelines.

**What it validates**:
- Convergence rate bounds (30-100%)
- Lexical diversity minimum (>10%)
- No severe violations in production runs
- Confidence score ranges (0.2-1.2)
- Success rate thresholds (>50%)
- Runtime limits (<5 minutes per session)
- Token count limits (<10,000 per session)

**Usage**:
```bash
# Validate last 24 hours of experiments
python -m experiments.scripts.runner --exp quality_gates

# Custom time window and experiment types
python experiments/scripts/run_quality_gates.py --hours-back 48 --experiment-types smoke style

# With custom thresholds
python experiments/scripts/run_quality_gates.py --thresholds-file custom_thresholds.json
```

**Output**:
- Console output with pass/fail status
- Exit code 0 (pass) or 1 (fail) for CI integration
- Optional JSON output file with detailed results

**Default thresholds** (can be customized):
- `convergence_rate_min`: 0.3
- `lexical_diversity_min`: 0.1  
- `severe_violations_max`: 0
- `confidence_score_min`: 0.2
- `success_rate_min`: 0.5
- `runtime_max`: 300.0 seconds
- `violation_rate_max`: 0.5

## Planned Experiments (Not Yet Implemented)

### 6. Fault Injection & Resilience (`fault_injection`)
- Engine wrapper with probabilistic exceptions
- Delayed runtime simulation
- Monitor exception simulation
- Graceful degradation testing

### 7. Async vs Sync Performance Benchmark (`async_benchmark`)
- Sequential vs parallel dialogue timing
- Speedup measurements across agent counts
- Throughput analysis
- Resource utilization patterns

### 8. Analytics Index Generator (`analytics`)
- Scan persisted session directories
- Compute diversity indices
- Repetition metric analysis
- Trend identification across experiments

### 9. Extensibility Experiments (`extensibility`)
- Custom repetition monitor demonstration
- Semantic similarity monitor (with/without sentence-transformers)
- Plugin system validation
- Monitor registration at runtime

### 10. Policy/Safety Modeling Extensions (`policy_tiers`)
- Multi-tier monitoring (pre_style + post_style)
- Policy escalation patterns
- Safety override mechanisms
- Compliance audit trails

## Metrics Schema

All experiments output standardized JSONL metrics following the schema defined in `config/metrics_schema.json`. Key fields include:

- `timestamp`: ISO 8601 experiment timestamp
- `experiment_type`: Type of experiment
- `session_id`: Unique session identifier
- `agent_count`: Number of agents involved
- `metrics`: Core performance metrics
  - `runtime_seconds`: Total execution time
  - `confidence_score`: Average confidence
  - `token_count`: Total tokens processed
  - `violation_count`: Monitor violations triggered
  - `convergence_rate`: Multi-agent convergence success rate
  - `lexical_diversity`: Vocabulary richness
- `parameters`: Experiment-specific configuration
- `outcomes`: Results and success indicators

## Analytics Utilities

The `experiments.utils.analytics` module provides:

- **Lexical diversity computation**: Unique tokens / total tokens
- **Repetition scoring**: Sliding window similarity analysis
- **Jaccard similarity**: Token-based content similarity
- **Session metrics extraction**: Convert raw sessions to standardized metrics
- **JSONL I/O utilities**: Read/write metrics in standard format
- **Diversity indices**: Shannon entropy across multiple sessions

## Custom Monitors

The `experiments.utils.monitors_custom` module includes:

- **Repetition monitor**: Detects repetitive content using Jaccard similarity
- **Semantic similarity monitor**: Optional sentence-transformers integration with fallback
- **Monitor configuration loader**: YAML config file support
- **Factory functions**: Create custom monitors from specifications

## Configuration Files

### Monitor Configurations

- **`monitors_baseline.yaml`**: Standard production monitors
  - Policy rule checking (severe)
  - Keyword filtering (minor)
  - Resource monitoring (30% tolerance)

- **`monitors_high_sensitivity.yaml`**: Stress testing configuration  
  - Extended keyword lists
  - Strict resource limits (10% tolerance)
  - Regex pattern matching
  - Custom repetition detection

### Rules Configuration

- **`rules_extended.yaml`**: Extended policy rules
  - Self-harm prevention
  - Hate speech detection
  - Manipulation detection
  - Confidence thresholds
  - Token limits

## Integration with CI/CD

The quality gates experiment is designed for CI integration:

```yaml
# Example GitHub Actions step
- name: Run AgentNet Quality Gates
  run: |
    python -m experiments.scripts.runner --exp quality_gates
  env:
    PYTHONPATH: .
```

Exit codes:
- `0`: All quality gates passed
- `1`: Some quality gates failed
- `130`: Interrupted by user

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're running from the repository root and PYTHONPATH includes the current directory.

2. **Missing YAML support**: Install PyYAML if you get YAML-related errors:
   ```bash
   pip install PyYAML
   ```

3. **No metrics found**: Quality gates need existing experiment data. Run other experiments first to generate metrics.

4. **Permission errors**: Ensure write access to `experiments/data/` directory.

### Debug Tips

- Use `--verbose` flag on quality gates for detailed output
- Check `experiments/data/raw_sessions/` for experiment outputs
- Review JSONL files for metrics format issues
- Enable DEBUG logging in AgentNet for detailed execution traces

## Contributing

When adding new experiments:

1. Create script in `experiments/scripts/run_<name>.py`
2. Follow the pattern of existing scripts (main function, metrics output)
3. Add entry to `EXPERIMENTS` registry in `runner.py`
4. Update this README with experiment description
5. Add any new configuration files to `experiments/config/`
6. Ensure metrics follow the standard schema

## License

Same as parent AgentNet project.