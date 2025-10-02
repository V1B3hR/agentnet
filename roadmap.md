# AgentNet To-Do List (Actionable Backlog)

Source-derived from: README.md and roadmap.md (commit 60f4da8)  
Scope: Only unimplemented, partial, planned, or explicitly outstanding work.  
Priority scale:   P1 (High) < P2 (Medium) < P3 (Strategic / Long-Term)

---

## 🟠 P1 – High Priority (Stability, Safety, Core Feature Completion)

### 1. Mirror Agents with Noise Injection ("Dusty Mirror")
- Purpose: Robustness & variance testing.
- Implementation Plan:
  - `MirrorAgent` wrapper: clone base agent; inject perturbations (temperature jitter, prompt paraphrasing, token dropout / masking).
  - Divergence analysis module: semantic similarity (e.g., cosine on embeddings), factual variance heuristics, output structure delta.
  - Optional adversarial noise profile (extreme temperature + paraphrase).
- Acceptance:
  - Example: `examples/mirror_agents_demo.py`
  - Test: `tests/test_mirror_agents.py` asserting diversity score > configured threshold while maintaining semantic intent bounds.

### 2. Policy Engine Enhancements
- Add:
  - Rule priority / weighting & deterministic conflict resolution.
  - Conflict explanation object (structured: triggering rules, resolution path, suppressed rules).
  - Hot reload for policies without full process restart (file + registry watcher).
- Notes:
  - Baseline engine exists: `agentnet/core/policy/engine.py`
  - No current priority arbitration or reload hooks.
- Acceptance:
  - Tests: conflicting policy scenario yields stable chosen rule.
  - Reload test: modifying a rule file updates active policy set in-process.
  - Explanation object present in violation events.

### 3. Complete Phase 9 Deep Learning Remaining Work
Core scaffolding is present (registry, trainer stubs, embeddings, neural reasoning placeholders). Remaining:
- Model artifact version diffing & rollback utility.
- Distributed training orchestration templates (multi-node launcher).
- Embedding batch optimizer (adaptive batch size, precision switching).
- Fine-tuning evaluation harness integration (auto-run benchmarks on job completion).
- GPU resource scheduler (fair-share across concurrent sessions).
- Full PyTorch trainer (beyond stubs: optimizer lifecycle, gradients, mixed precision, checkpoints).
- Actual embedding generation using `sentence-transformers` (replace stubs).
- Neural reasoning module concrete strategies (chain-of-thought ranking, retrieval-conditioned reasoning).
- Acceptance:
  - End-to-end example: `examples/phase9_training_pipeline_demo.py` trains, evaluates, versions, rolls back.
  - Tests cover: version diff, rollback, distributed launch dry-run, scheduler fairness.

---

## 🟡 P2 – Medium Priority (Expansion & Robustness)

### 9. Edge/Latency Optimization Pass
- Implement adaptive batching for provider calls.
- Token usage prediction caching layer.
- Async I/O instrumentation (trace spans per await cluster).
- Acceptance: Benchmark shows >10% latency or throughput improvement documented in `docs/performance/latency_report.md`.

### 10. Session Persistence & Replay Hardening
- Cryptographic integrity hash chain for session events.
- Partial replay (`session.replay(range)`) with deterministic reproduction.
- Acceptance: Integrity hash validated; replay unit tests pass for arbitrary slice windows.

### 11. Advanced Risk Analytics
- Rolling consolidated risk score trend.
- Incident export bundle (JSONL: violation + context window).
- Acceptance: `risk/report/generate` endpoint returns structured JSON with schema doc.

### 12. Tool Lifecycle Governance Extensions
- Tool capability JWT signing (optional).
- Rate limiting per tool or category.
- Parameter schema linting (validation + suggestion).
- Acceptance: Security doc updated; tests for signature + rate limit enforcement.

### 13. Cost Optimization Strategies
- Dynamic provider selection: latency × cost × quality scoring.
- Acceptance: Scenario test picks cheaper provider (<2% quality delta) with documented metrics.

### 14. Memory Backends – Graph / Structured Finalization
- Node aging / salience-based eviction.
- Cross-layer linking (episodic → semantic → graph edges).
- Acceptance: `examples/memory_graph_enrichment.py` shows enrichment; tests assert aging removes low-salience nodes.

### 15. Chaos & Resilience Testing
- Fault injection harness: random tool failure, provider timeout, memory backend disconnect.
- Recovery metrics collection (mean recovery time, failure containment).
- Acceptance: Report generated with resilience KPIs.

---

## 🟢 P3 – Strategic / Growth / Ecosystem

### 16. Community Edition Packaging
- Optional dependency slimming via extras.
- Acceptance: `pip install agentnet[community]` size threshold documented.

### 17. Enterprise SaaS Path
- Authn/OIDC integration placeholder.
- Billing / usage metering hooks.
- Acceptance: Architecture doc + stub modules compiled without runtime errors.

### 18. Partner Ecosystem Program
- Plugin verification pipeline (signature + metadata validation).
- Acceptance: `docs/ecosystem/plugins.md` with lifecycle + verification steps.

### 19. Training & Certification Materials
- Curriculum outline + sample lab exercise.
- Acceptance: `docs/education/certification_outline.md`.

### 20. Research Partnerships Framework
- Reproducible benchmark harness export.
- Dataset anonymization utilities.
- Acceptance: `agentnet/research/benchmark_runner.py` demo + doc.

### 21. Global Expansion: Localization & Compliance
- Multi-region config templates (EU/US separation).
- Data residency flag in session config.
- Language-specific evaluation heuristics.
- Acceptance: `docs/compliance/data_residency.md`.

### 22. Local Model & Regional Provider Support
- Pluggable registry (e.g., Ollama, Mistral-hosted, OpenRouter).
- Acceptance: Integration tests with ≥2 local providers.

### 23. Cultural Adaptation Layer (UI / Examples)
- Localized examples under `/examples/i18n/`.
- Acceptance: At least 2 non-English language examples.

### 24. Quantum-Ready Stubs (Exploratory)
- Abstraction layer for quantum job submission + simulated adapter.
- Acceptance: Demo script with abstract "quantum assist" step.

---

## ⚙️ Maintenance / Ongoing Improvements

| Task | Description | Cadence | Acceptance |
|------|-------------|---------|-----------|
| Dependency Audit | Automated weekly vulnerability scan | Weekly | Report artifact in Actions |
| Coverage Trend | Track coverage diff vs main | Each PR | Delta comment posted |
| Performance Baseline Refresh | Re-run performance benchmark suite | Monthly | Updated JSON snapshot |
| Roadmap Sync | Reconcile README vs roadmap vs implementation | Quarterly | Changelog entry |
| Security Review | Threat model & policy tuning | Quarterly | Updated `docs/security/threat_model.md` |

---

## 🔍 Traceability Matrix (Task ↔ Source Reference)

| Task ID | Source Snippet / Indicator | Status |
|---------|----------------------------|--------|
| P1-1 (formerly 4) | README marker: mirror agents concept not yet implemented | 🔴 Not Implemented |
| P1-2 (formerly 8) | Policy conflict resolution + hot reload absent | 🔴 Not Implemented |
| P1-3 (formerly 7) | Phase 9 scaffolding present; depth features missing | 🟡 Partially Complete |
| P2-9–15 | Derived from scaling / robustness roadmap sections | 🔴 Not Implemented |
| P3-16–24 | Growth initiatives & future objectives | 🔴 Not Implemented |

### Recently Completed (moved to ✅ section)

| Task ID | Implementation | Status |
|---------|----------------|--------|
| P1-2 (old) | NFR Test Coverage: `tests/test_nfr_comprehensive.py`, `docs/testing/nfr_testing.md` | ✅ Complete |
| P1-3 (old) | Dashboard: `agentnet/observability/dashboard.py`, demo HTML | ✅ Complete |
| P1-5 (old) | Multi-Lingual Safety: `agentnet/core/multilingual_safety.py` (12 languages) | ✅ Complete |
| P1-6 (old) | Streaming: `agentnet/streaming/` module with collaboration | ✅ Complete |

---

## ✅ Completed (Do Not Re-add Unless Regression)

- Core architecture, orchestration, memory layers
- Policy engine baseline & governance
- Evaluation harness
- Risk register runtime integration
- Observability metrics & tracing
- Cost tracking integration
- Tool system w/ governance
- Provider adapters (OpenAI, Anthropic, Azure, etc.)
- DAG planner w/ networkx
- Deep learning scaffolding (registry, training pipeline, fine-tuning, embeddings – baseline)
- NFR Test Coverage:
  - Comprehensive suite (`tests/test_nfr_comprehensive.py`) with reliability, scalability, security cases
  - Documentation: `docs/testing/nfr_testing.md`
- Live Dashboard (Observability UI):
  - Implementation: `agentnet/observability/dashboard.py`
  - Features: cost aggregation, performance metrics, violation tracking
  - Demo artifact (HTML) included
- Multi-Lingual Safety Policy Translation:
  - Implementation: `agentnet/core/multilingual_safety.py`
  - 12 languages supported: EN, ES, FR, DE, IT, PT, RU, ZH, JA, KO, AR, HI
  - Pattern + keyword mapping & fallback translation logic
- Streaming Partial-Output Collaboration:
  - Module: `agentnet/streaming/`
  - Supports token/chunk streaming, mid-stream intervention, incremental memory buffering
  - Associated tests validate intervention control

---

## 📌 Next Immediate Steps (Execution Order)

1. Implement Mirror Agents with Noise Injection (P1 Task 1)
2. Policy Engine Enhancements: priority resolution, conflict explanation, hot reload (P1 Task 2)
3. Complete Phase 9 Deep Learning remaining work (P1 Task 3)
4. Edge/Latency Optimization Pass (P2 Task 9)
5. Advanced Risk Analytics (P2 Task 11)

---

## 🧪 Suggested Branching Strategy

- feature/mirror-agents
- feature/policy-enhancements
- feature/phase9-completion
- feature/edge-optimization
- feature/risk-analytics

Each merges via PR with checklist referencing Task IDs.

---

## ✅ Definition of Done (Global)

- Code + tests + docs updated
- CI green across matrix
- Coverage no regression >2% vs baseline
- Security scan clean (no High/Critical)
- Changelog entry appended (`CHANGELOG.md`)
- Linked issue(s) closed automatically via PR description

---

Let this file evolve; prune delivered tasks and append new planned work with explicit classification.
