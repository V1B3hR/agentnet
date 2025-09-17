# AgentNet

> A governed multi‑agent reasoning platform: policy‑aware LLM orchestration, dialogue strategies (debate / brainstorm / consensus), task graph execution, memory layers, monitoring, and extensible tool & provider adapters.

---

## Table of Contents
- [Why AgentNet?](#why-agentnet)
- [Core Features](#core-features)
- [Architecture Overview](#architecture-overview)
- [Concepts](#concepts)
- [Directory Structure](#directory-structure)
- [Quick Start](#quick-start)
- [Agents & Styles](#agents--styles)
- [Policies & Monitors](#policies--monitors)
- [Multi‑Agent Dialogue](#multi-agent-dialogue)
- [Task Graph (DAG) Execution](#task-graph-dag-execution)
- [Memory System](#memory-system)
- [Tooling & Actions](#tooling--actions)
- [Provider Adapters](#provider-adapters)
- [Observability & Governance](#observability--governance)
- [REST / SDK Roadmap](#rest--sdk-roadmap)
- [Evaluation Harness](#evaluation-harness)
- [Security & Compliance](#security--compliance)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Quick Glossary](#quick-glossary)
- [Status Badges](#status-badges)
- [Need Help?](#need-help)

---

## Why AgentNet?

Modern AI applications demand:
- Collaboration between heterogeneous agents (reasoners, planners, synthesizers).
- Governance: safety, policy compliance, cost & resource budgeting.
- Persistent audit trails of reasoning.
- Extensibility: integrate new LLM providers, tools, memory backends, and custom guardrails quickly.

AgentNet turns raw model calls into structured, monitored, reproducible cognitive processes.

---

## Core Features

| Domain | Capabilities |
|--------|--------------|
| Multi‑Agent Orchestration | Debate, brainstorm, consensus, adaptive topic evolution, convergence detection |
| Policy & Safety | Keyword, regex, custom, resource monitors (extensible for semantic & classifier rules) |
| Cognitive Fault Handling | Structured exceptions (`CognitiveFault`) with severity + violation bundles |
| Task Graph Execution | Planner → DAG → scheduled sub‑tasks with dependency resolution |
| Memory | Short-term transcript window, episodic persistence, semantic (vector) retrieval (pluggable) |
| Style Modulation | Logic / creativity / analytical weight influences confidence & meta‑insights |
| Async + Parallel | Parallel round execution in async dialogues |
| Persistence | Knowledge graph nodes + sessions + violations (JSON now; DB planned) |
| Extensibility | Pluggable monitors, provider adapters, tools, style transformers |
| Evaluation | Scenario harness (planned) for regression scoring & quality signals |
| Cost & Resource | Runtime budgets, token accounting (pricing table integration planned) |
| Observability | Structured logs, monitor traces, runtime metrics (Prometheus integration planned) |

---

## Architecture Overview

```
+------------ API / CLI / SDK (planned) -------------+
                 |
                 v
          Orchestration Layer
   (Session Manager | Dialogue Engine | DAG Planner)
                 |
      +----------+-----------+
      |                      |
 Agent Runtime(s)      Task / Tool Runner
 (style, monitors)      (sandbox / APIs)
      |                      |
      +----------+-----------+
                 |
           Provider Adapters
     (OpenAI | Anthropic | Local/vLLM ...)
                 |
         Policy & Monitor Pipeline
       (rules, resource, semantic*)
                 |
           Persistence & Memory
     (JSON → Postgres + Vector Store*)
                 |
           Observability (metrics, logs, traces*)
```

(*) = upcoming enhancements

---

## Concepts

| Term | Description |
|------|-------------|
| Agent | A reasoning unit with style, engine, monitors, memory access |
| Monitor | A guard evaluating outputs (policy, keyword, regex, resource, custom) |
| Cognitive Fault | Structured failure with severity & violation bundle |
| Session | Multi-agent dialogue or task execution context |
| Style | Weighted attributes shaping meta‑insights & confidence scaling |
| DAG | Directed acyclic graph of decomposed tasks with dependencies |
| Memory Episode | Persisted contextual snippet retrievable by semantic similarity |
| Convergence | Heuristic detection of topic stabilization |

---

## Directory Structure

Proposed / evolving layout:

```
agentnet/
  core/
    agent.py              # DuetMindAgent core
    monitors/             # Monitor implementations
    policy/               # Rule loading & policy engine
    memory/               # Memory abstractions
    provider/             # LLM adapters (openai, local, ...)
    orchestration/        # Sessions, dialogue strategies, DAG
    tools/                # Tool registry & execution
    cost/                 # Pricing & cost recording
  api/                    # REST layer (future)
  cli/                    # Command-line utilities
  evaluations/            # Scenario definitions
  configs/                # Agent & policy configs
  sessions/               # Persisted session logs (gitignored)
  tests/
    unit/
    integration/
    performance/
  docs/
    architecture.md
    policies.md
    roadmap.md
```

---

## Quick Start

Current status: local experimentation with file-based persistence. DB + full APIs are on the roadmap.

### Prerequisites
- Python 3.10+ recommended
- macOS, Linux, or Windows

### 1) Install

```bash
git clone https://github.com/V1B3hR/agentnet.git
cd agentnet
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2) Run Demo

```bash
python agentnet/core/agent_demo.py --demo both --rounds 3
```

Outputs include:
- Single-agent reasoning tree with monitor trace
- Multi-agent session (brainstorm or debate)
- Persisted session JSON under `./sessions/`

### 3) Minimal Usage

```python
from agentnet.core.agent import DuetMindAgent
from agentnet.core.engine import ExampleEngine

engine = ExampleEngine()
agent = DuetMindAgent(
    name="Athena",
    style={"logic": 0.9, "creativity": 0.4, "analytical": 0.8, "resource_budget": 0.05},
    engine=engine
)

res = agent.generate_reasoning_tree("Evaluate trade-offs in edge caching")
print(res["result"]["content"])
```

---

## Agents & Styles

Styles influence:
- Confidence scaling (analytical weight)
- Added meta‑insights (logic/creativity triggers)
- Potential future: dynamic temperature, tool selection bias

Example style config (YAML):

```yaml
name: Athena
style:
  logic: 0.9
  creativity: 0.4
  analytical: 0.8
  resource_budget: 0.05
monitors:
  file: monitors.yaml
dialogue_config:
  max_rounds: 25
  convergence_window: 4
```

---

## Policies & Monitors

Supported (current):
- keyword
- regex
- custom (Python function)
- resource (runtime > budget)
- rcd_policy (rule bundle file)

Planned:
- semantic_similarity
- classifier (toxicity / PII)
- numeric_threshold
- streaming partial monitors

Example policy rules (YAML):

```yaml
rules:
  - name: no_self_harm
    severity: severe
    type: keyword
    keywords: ["self-harm", "suicide"]

  - name: applied_ethics
    severity: minor
    type: custom
    func: applied_ethics_check
```

Monitors config:

```yaml
monitors:
  - name: policy_rcd
    type: rcd_policy
    severity: severe
    params:
      rules_file: rules.yaml
  - name: keyword_guard
    type: keyword
    severity: severe
    params:
      keywords: ["manipulation","hate","racist"]
  - name: resource_guard
    type: resource
    severity: minor
    params:
      budget_key: resource_budget
      tolerance: 0.2
```

---

## Multi‑Agent Dialogue

Modes:
- `brainstorm`: idea expansion
- `debate`: argument + rebuttal
- `consensus`: alignment & synthesis
- `general`: neutral reasoning

Features:
- Topic evolution heuristics (confidence-driven)
- Convergence detection (lexical Jaccard overlap)
- Rolling & final synthesis prompts
- Async variant (`parallel_round=True`)

---

## Task Graph (DAG) Execution

Planned capability:
- Planner agent creates graph: nodes `{id, prompt, agent, deps}`
- Scheduler executes ready nodes
- Retry + fallback chain
- Aggregated final synthesis node

JSON node example:

```json
{
  "id": "risk_analysis",
  "prompt": "Analyze failure cascades",
  "agent": "Athena",
  "deps": ["plan_root"]
}
```

---

## Memory System

| Layer | Status | Notes |
|-------|--------|-------|
| Short-term | Implemented | Sliding window + truncation strategy |
| Episodic | Planned | Persist selected turns w/ tags |
| Semantic | Planned | Vector retrieval (pgvector / Qdrant) |
| Summarization Compression | Planned | Rolling synthetic memory nodes |

Retrieval pipeline (planned):
1. Collect short-term tail
2. Add semantic top-k (cosine threshold)
3. Optional episodic tag matches
4. Summarize if token > budget

---

## Tooling & Actions

Roadmap:
- Tool registry with JSON schema
- Rate limiting + auth scoping
- Sandbox execution for code tools
- Cached deterministic tool responses

Planned tool contract:

```json
{
  "name": "web_search",
  "schema": {
    "type": "object",
    "properties": { "query": { "type": "string" } },
    "required": ["query"]
  },
  "rate_limit_per_min": 30,
  "auth": "api_key_ref:SERPAPI_KEY"
}
```

---

## Provider Adapters

Adapter goals:
- Normalized interface (`infer`, `stream`, `cost`)
- Token + cost accounting
- Fallback chain (primary → backup → local)

Initial targets:
- OpenAI
- Local engine (ExampleEngine stub)
- Future: Anthropic, Azure, vLLM, HuggingFace Inference

---

## Observability & Governance

Current:
- Structured reasoning tree JSON
- Violation objects (rule name, severity, rationale)
- Runtime measurement per turn

Planned:
- Prometheus metrics: `inference_latency_ms`, `violations_total`
- OpenTelemetry traces: spans per turn & monitor pipeline
- Cost aggregation dashboards

---

## REST / SDK Roadmap

| Endpoint | Purpose | Status |
|----------|---------|--------|
| POST /agents | Create agent version | Planned |
| GET /agents/{id} | Retrieve config | Planned |
| POST /sessions | Start session | Planned |
| GET /sessions/{id} | Inspect state | Planned |
| POST /sessions/{id}/advance | Manual next round | Planned |
| POST /tasks/plan | Generate DAG | Planned |
| POST /tasks/execute | Execute DAG | Planned |
| POST /policies | Upload policy bundle | Planned |
| GET /violations | Query violations | Planned |

Python SDK will wrap these calls with typed dataclasses.

---

## Evaluation Harness

Scenario YAML (planned):

```yaml
suite: resilience_core
scenarios:
  - name: edge_recovery_brainstorm
    mode: brainstorm
    agents: ["Athena","Apollo"]
    topic: "Recover from edge partition"
    criteria:
      - type: keyword_presence
        must_include: ["redundancy","failover"]
      - type: semantic_score
        ref: docs/resilience_ref.md
        min_score: 0.78
```

Outputs:
- `coverage_score`
- `novelty_score` (embedding divergence)
- `violation_count`
- `coherence` (LLM grader)

---

## Security & Compliance

Planned stages and mitigations:

| Concern | Mitigation |
|---------|------------|
| Prompt injection | Input sanitization + monitor suites |
| PII leakage | Regex + semantic PII classifiers |
| Secret exposure | Redaction + output scanning |
| Tenant isolation | Row level security (RLS) |
| Tool abuse | Schema validation + sandbox VM |
| Model fallback trust | Signed adapter registry |

---

## Roadmap

| Phase | Focus | Highlights |
|-------|-------|-----------|
| P0 | Stabilize core | Refactor agent, monitors v1, session persistence |
| P1 | Multi-agent polish | Async parallel rounds, improved convergence, basic API |
| P2 | Memory & Tools | Vector store integration, tool registry |
| P3 | DAG & Eval | Task graph planner, evaluation harness |
| P4 | Governance++ | Semantic/classifier monitors, cost engine, RBAC |
| P5 | Observability | Metrics, traces, spend dashboards |
| P6 | Enterprise Hardening | Export controls, audit workflow, plugin SDK |

---

## Contributing

1. Fork & branch naming: `feat/`, `fix/`, `chore/`.
2. Run lint & tests:
   ```bash
   ruff check .
   mypy agentnet
   pytest -q
   ```
3. Add or update docs for new behaviors.
4. Ensure demo still runs:
   ```bash
   python agentnet/core/agent_demo.py --demo sync
   ```
5. Open a PR with a summary and design notes if architecture-affecting.

Planned labels: `good-first-issue`, `monitor`, `provider-adapter`, `memory`, `tooling`.

---

## License

GPL-v3.0

---

## Quick Glossary

| Term | Meaning |
|------|---------|
| Style Insights | Meta phrases indicating internal weighting effect |
| Knowledge Graph Node | Stored reasoning result + metadata |
| Violation | Structured record of a failed policy/monitor check |
| Convergence | Heuristic content overlap threshold reached |

---

## Status Badges

Add badges once services are wired:

```
[![Tests](https://img.shields.io/badge/tests-pending-lightgray)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-pending-lightgray)]()
```

---

## Need Help?

Please open a GitHub Issue with:
- Reproduction steps
- Agent config (redacted if needed)
- Monitor bundle (if relevant)
- Session excerpt

> AgentNet evolves from a foundation of structured, monitorable multi‑agent reasoning. Your contributions can push it toward a robust ecosystem for governed AI collaboration.

Happy building!
