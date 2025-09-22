# AgentNet

Policy-governed multi-agent LLM framework for dialogue, debate, tool-use, memory, and observability.  
Designed for safe, inspectable, and extensible cognitive workflows.

> Status: Early development (scaffolding + initial abstractions). Expect rapid iteration and breaking changes.

---

## Why AgentNet?

Modern LLM systems need:
- Multiple cooperating or adversarial agents (analyst / critic / arbiter / tool-runner).
- Enforceable policies (redaction, action gating, role-based tool access).
- Transparent reasoning (event streams, audit trails, reproducible sessions).
- Memory persistence & retrieval (short-term conversation, semantic vector store, structured / graph memory).
- Configurable orchestration modes (round-robin, debate, critique-revise, arbitration, async, multi-modal).
- Extensibility (new tools, policies, evaluators, memory backends).

AgentNet aims to offer these as composable, inspectable primitives rather than monolith patterns.

---

## New Modes & Features (2025 Update)

- **Modes:**  
  - *Debate*: Analyst vs Critic agent, scoring, arbitration.
  - *Critique-Revise*: Agent self-improvement loop.
  - *Round-Robin*: Sequential, multi-agent dialogue.
  - *Async*: Parallel tool invocation & agent turns.
  - *Arbitration*: Score-based decision resolution.
  - *Multi-modal*: Text-first, extensibility for other modalities.
- **Features:**
  - *Policy engine*: Redaction, blocking, rewriting, deferred actions, human gating.
  - *Memory adapters*: Short-term buffer, vector store, graph/structured memory, caching, LRU/salience retention.
  - *Tools*: API calls, calculation, file access, external system integration.
  - *Evaluators*: Heuristic truthiness, risk tagging, complexity, consensus, revision triggers.
  - *Observability*: Event bus, trace export, OpenTelemetry, audit bundles.
  - *Persistence*: Session checkpoint, resume, replay.
  - *Dashboard*: Live event stream, violation panel, metrics (planned).

---

## The 25 AI Fundamental Law

AgentNet is inspired by the emerging "25 AI Fundamental Law"—a set of best practices for building safe, transparent, and adaptive AI agent systems.  
**Summary of the Law:**

1. Policy-first: All reasoning and actions must pass through explicit policy gates.
2. Memory isolation: Distinct layers for ephemeral, semantic, and structural knowledge.
3. Full observability: Every non-trivial action emits a traceable event.
4. Deterministic input/output surfaces.
5. Explicit orchestration: All agent sequences are logged and reproducible.
6. Role-based tool access.
7. Human-in-the-loop escalation for high-risk actions.
8. Redaction by default for sensitive data.
9. Consensus & arbitration for critical decisions.
10. Adaptive performance feedback.
11. Multi-modal extensibility with provenance tracking.
12. Reward modeling integration for learning loops.
13. Transparent policy violation reporting.
14. Structured audit export (events, config, memory).
15. Pluggable memory retention strategies.
16. Metrics-first: Latency, cost, error rate tracked.
17. Plugin registry for extensibility (tools, policies, evaluators).
18. Safe default toolsets and allowlists.
19. Explicit dependency passing—minimal hidden global state.
20. Persistent session state for replay and analysis.
21. Layered evaluator protocols: critique, scoring, revision.
22. Live monitoring (dashboard, telemetry).
23. Multi-lingual safety policy translation (planned).
24. Streaming partial-output collaboration (planned).
25. Rapid iteration with versioned change tracking.

See [docs/25_AI_fundamental_law.md](docs/25_AI_fundamental_law.md) for full explanation and rationale.

---

## Core Concepts

| Component      | Purpose                                                                |
|----------------|------------------------------------------------------------------------|
| `Agent`        | Encapsulates model interface, role, allowed tools, and policy hooks.   |
| `Orchestrator` | Manages turn-taking, debate modes, arbitration, or async flows.        |
| `Policy`       | Rule engine evaluating actions/messages (allow, block, transform, log).|
| `Tool`         | External capability (API call, computation, retrieval).                |
| `Memory`       | Pluggable adapters (short-term buffer, vector store, structured / graph).|
| `Evaluator`    | Self or cross-agent critique, scoring, revision triggers.              |
| `Monitor`      | Event capture: traces, metrics, violations, tokens, decisions.         |
| `Session`      | Persistent container of configuration + evolving state snapshot.       |

---

## High-Level Architecture

```
User Prompt
   |
Orchestrator  <--- config (mode: debate | critique | linear | async)
   |
   +---> Agents[] ----> Policies ----> Tools
   |          |             |            |
   |          |             |            +--> External Systems / APIs
   |          |             |
   |          |             +--> Memory (vector, structured, graph, ephemeral)
   |          |
   |          +--> Evaluators (critique, scoring, arbitration)
   |
   +---> Event Bus ---> Monitors (logging, telemetry, audit export)
```

---

## Installation (Early Dev)

```bash
git clone https://github.com/V1B3hR/agentnet
cd agentnet
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Optional / Planned Extras (some may not yet be active):
```bash
# Vector store + embeddings
pip install ".[vector]"

# Graph / structured memory (requires networkx)
pip install ".[graph]"

# Monitoring / tracing extras
pip install ".[obs]"

# Everything (when defined)
pip install ".[all]"
```

---

## Minimal (Planned) Usage Example

```python
from agentnet import Agent, Orchestrator, load_policy

policy = load_policy("default")  # placeholder
analyst = Agent(name="Analyst", model="openai:gpt-4o", tools=["search"], policy=policy)
critic  = Agent(name="Critic",  model="openai:gpt-4o", tools=[],         policy=policy)

orch = Orchestrator(
    agents=[analyst, critic],
    mode="debate",
    max_turns=6,
    evaluators=["consistency", "risk_check"]
)

session = orch.run(prompt="Assess architectural risks in the proposed data pipeline.")
print(session.final.output)
```

---

## Design Principles

1. Policy-First: Safety & governance are not add-ons.
2. Composability: Each cognitive layer is swappable and minimal.
3. Observability Everywhere: Every non-trivial action emits an event.
4. Deterministic Surfaces: Inputs and transforms logged for replay where feasible.
5. Layered Memory: Separate ephemeral, semantic, and structural knowledge.
6. Extensible Interfaces: Clear protocols for model, tool, memory, evaluator, monitor.
7. Minimal Hidden State: Session structures explicit and exportable.

---

## Clean Roadmap

Legend:  
[•] Planned | [WIP] In progress | [✓] Complete | [Δ] Partial / provisional | [R] Research / exploratory

### Phase 0 – Foundation (Bootstrapping)
- [✓] Repository scaffolding & license
- [Δ] Core package layout (`agent`, `policy`, `orchestrator`, `memory`, `events`)
- [WIP] Typed event schema + base monitor
- [•] Basic CLI entry point
- [•] Config loader (YAML/py) with validation

### Phase 1 – MVP Orchestration & Policy
- [WIP] Single + multi-agent synchronous turn engine
- [•] Round-robin & termination conditions (max turns, consensus, policy stop)
- [•] Policy rule engine (matchers: regex, role, tool, classification stub)
- [•] Tool protocol + sample tools (search stub, calc, file read)
- [•] Memory v1: short-term conversation buffer
- [•] Vector memory adapter (FAISS/Chroma placeholder)
- [•] Event bus with console/file sinks

### Phase 2 – Debate, Critique, Evaluation
- [•] Debate mode (analyst vs critic)
- [•] Critique-revise loop (agent self-revision)
- [•] Arbitration strategies (score weighting, majority vote)
- [•] Evaluators: truthiness (heuristic), complexity, risk tags
- [•] Policy actions: block, redact, rewrite, require-approval (stub)
- [•] Structured memory tagging & retrieval filters
- [•] Policy violation reporting & counters

### Phase 3 – Persistence, Scaling, Graph Memory
- [Δ] Graph/relational memory (requires `networkx`)
- [•] Session checkpoint + resume
- [•] Async orchestration (parallel tool calls / futures)
- [•] Embedding cache & response caching
- [•] Memory retention policies (LRU, semantic salience)
- [•] Multi-modal placeholder (text-first; extensibility hooks)

### Phase 4 – Observability & Ops
- [•] Token accounting + cost model abstraction
- [•] OpenTelemetry / OTLP export adapter
- [•] Rich audit bundle export (JSONL events + memory snapshot + policy config)
- [•] Web dashboard (live event stream, violation panel)
- [•] Metrics: latency, tool error rate, policy hit distribution

### Phase 5 – Hardening & Ecosystem
- [•] Full test matrix (unit + multi-agent integration)
- [•] Performance harness (turn latency, token utilization)
- [•] Documentation portal (mkdocs) + example notebooks
- [•] PyPI packaging & versioning policy (semver w/ 0.x rapid changes)
- [•] Plugin registry (entry points for tools/policies/evaluators)

### Phase 6 – Advanced / Exploratory (R)
- [R] Meta-controller agent (dynamic agent graph reconfiguration)
- [R] Human-in-loop gating (approval & escalation flow)
- [R] Reward modeling integration / offline evaluation loops
- [R] Adaptive orchestration via performance feedback
- [R] Multi-lingual safety policy translation
- [R] Streaming partial-output collaboration

---

## Policy Model (Planned Shape)

```yaml
policies:
  - name: redact_secrets
    rules:
      - if:
          any:
            - regex:
                target: message.content
                pattern: "(api[_-]?key|secret|password)"
        actions: [redact, log]

  - name: restrict_high_risk_tools
    rules:
      - if:
          all:
            - tool.name: "system_shell"
            - agent.role != "Supervisor"
        actions: [block, flag]
```

Actions (initial set): `allow`, `block`, `redact`, `transform`, `log`, `defer`, `require_human`, `rewrite`.

---

## Event Model (Indicative)

| Event                   | When                                   |
|-------------------------|----------------------------------------|
| `turn.start` / `turn.end`     | Orchestrator loop boundaries      |
| `agent.request`               | Agent forms model prompt          |
| `model.response`              | Raw model output                  |
| `policy.violation`            | Rule triggered                    |
| `tool.invoke` / `tool.result` | External capability usage         |
| `memory.store` / `memory.retrieve` | Memory operations           |
| `evaluator.score`             | Critique or scoring module output |
| `session.checkpoint`          | Persistence snapshot written      |

---

## Development Workflow (Suggested)

```bash
# Lint & type check
ruff check .
mypy agentnet

# Run tests
pytest -q

# Regenerate event schema (planned)
python scripts/gen_schema.py
```

---

## Contributing

1. Open a lightweight design note (if feature is non-trivial).
2. Prefer small PRs aligned with an open phase milestone.
3. Include tests for new adapters / policies / evaluators.
4. Ensure new events update schema + docs if stabilized.
5. Avoid silent global state; pass dependencies explicitly.

---

## Security & Safety Notes

- Policies are not a guarantee of compliance—treat as defense-in-depth.
- Memory redaction must be validated independently for sensitive deployments.
- External tools should implement explicit allowlists & argument validation.

---

## License

GPL-3.0 — see [LICENSE](LICENSE).

---

## Disclaimer

Experimental software. Interfaces may change. Do not deploy in production environments handling sensitive data without rigorous review.

---

## Quick Status Snapshot (Live Edit Area)

| Area            | State             |
|-----------------|-------------------|
| Core orchestration | WIP             |
| Policy engine      | Scaffold planned|
| Vector memory      | Pending adapter |
| Graph memory       | Partial (`networkx`) |
| Event bus          | Basic in progress|
| Debate mode        | Planned         |
| Evaluators         | Stub design     |
| Dashboard          | Not started     |

---

Suggestions / critiques welcome—open a discussion or issue with concise rationale and desired outcome.
