# AgentNet To-Do List (Actionable Backlog)

Source-derived from: README.md and roadmap.md (commit 60f4da8)  
Scope: Only unimplemented, partial, planned, or explicitly outstanding work.  
Priority scale:   P1 (High) < P2 (Medium) < P3 (Strategic / Long-Term)


---

## 🟠 P1 – High Priority (Stability, Safety, Core Feature Completion)

### 1. Mirror Agents with Noise Injection ("Dusty Mirror")
- Purpose: Robustness & variance testing.
- Implementation:
  - MirrorAgent wrapper: clones base agent, injects perturbations (sampling temperature jitter, prompt paraphrasing, token dropout).
  - Compare divergence scores (semantic similarity, factual variance).
- Acceptance: Example in `examples/mirror_agents_demo.py`; test asserting diversity > threshold.

### 2. Complete Phase 9 Deep Learning Remaining Work
Core scaffolding is complete with registry, trainer, embeddings, and neural reasoning modules. The following enhancements remain:
- Model artifact version diffing & rollback
- Distributed training orchestration templates (multi-node)
- Embedding batch optimizer (adaptive batch size, precision)
- Fine-tuning evaluation harness integration (auto-benchmark on finish)
- GPU resource scheduler (multi-session fairness)
- Full PyTorch trainer implementation (beyond stubs)
- Actual embedding generation with sentence-transformers (beyond stubs)
- Neural reasoning implementations (beyond stubs)
- Acceptance: All features fully functional; `examples/phase9_training_pipeline_demo.py` demonstrates end-to-end training.

### 3. Policy Engine Enhancements
- Add: rule priority resolution, conflict explanation object, hot reload without restart.
- Note: Basic policy engine exists (`agentnet/core/policy/engine.py`) but these specific enhancements are not yet implemented.
- Acceptance: Tests for conflicting policy rules produce deterministic resolution.

---

## 🟡 P2 – Medium Priority (Expansion & Robustness)

### 9. Edge/Latency Optimization Pass
- Implement adaptive batching for provider calls
- Token usage prediction caching
- Async I/O instrumentation (trace spans per await cluster)
- Acceptance: Performance delta >10% improvement documented.

### 10. Session Persistence & Replay Hardening
- Add cryptographic integrity hash (session event chain)
- Partial replay (range of turns)
- Acceptance: `session.replay(range)` works; hash verified.

### 11. Advanced Risk Analytics
- Consolidated risk score over time (trend)
- Incident export bundle (JSONL of violation + context window)
- Acceptance: `risk/report/generate` endpoint returns structured JSON.

### 12. Tool Lifecycle Governance Extensions
- Add: tool capability JWT signing (optional), rate limiting, parameter schema linting.
- Acceptance: Security doc updated.

### 13. Cost Optimization Strategies
- Implement dynamic provider selection (latency × cost × quality score)
- Acceptance: Example scenario chooses cheaper provider with negligible quality delta (<2% drop).

### 14. Memory Backends – Graph / Structured Finalization
- networkx features integrated but enrich with:
  - Node aging / salience eviction
  - Cross-layer linking (episodic -> semantic -> graph)
- Acceptance: `examples/memory_graph_enrichment.py`.

### 15. Chaos & Resilience Testing
- Fault injection harness: random tool failure, provider timeout, memory backend disconnect.
- Acceptance: Mean recovery time metrics recorded; doc added.

---

## 🟢 P3 – Strategic / Growth / Ecosystem

### 16. Community Edition Packaging
- Slim optional dependencies, separate extras tag.
- Acceptance: `pip install agentnet[community]` < minimal size threshold.

### 17. Enterprise SaaS Path
- Authn/OIDC integration placeholder
- Billing event hooks
- Acceptance: Architecture doc + stub modules.

### 18. Partner Ecosystem Program
- Plugin verification pipeline (signature, metadata validation)
- Acceptance: `docs/ecosystem/plugins.md`.

### 19. Training & Certification Materials
- Structured curriculum outline + sample lab.
- Acceptance: `docs/education/certification_outline.md`.

### 20. Research Partnerships Framework
- Reproducible benchmark harness export
- Dataset anonymization utilities
- Acceptance: `agentnet/research/benchmark_runner.py`.

### 21. Global Expansion: Localization & Compliance
- Multi-region config templates (EU/US)
- Data residency flag in session config
- Language-specific evaluation heuristics
- Acceptance: `docs/compliance/data_residency.md`.

### 22. Local Model & Regional Provider Support
- Add pluggable registry (Ollama, Mistral-hosted, open router)
- Acceptance: integration tests for at least 2 local providers.

### 23. Cultural Adaptation Layer (UI / Examples)
- Localized examples in /examples/i18n/
- Acceptance: at least 2 languages.

### 24. Quantum-Ready Stubs (If Not Already Real)
- Abstraction layer for quantum job submission
- Simulated adapter + placeholder solver
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
| P1-1 (formerly 4) | README: Inline marker `!!!Create "mirror agents"...` | 🔴 Not Implemented |
| P1-2 (formerly 8) | Policy conflict resolution not explicitly claimed | 🔴 Not Implemented |
| P1-3 (formerly 7) | README Phase 9 "scaffolding complete" (implies remaining implementation depth) | 🟡 Partially Complete |
| P2-9–15 | Inference from partial / robustness / scaling ambitions & typical maturity path | 🔴 Not Implemented |
| P3-16–24 | README "Growth Initiatives", "Global Expansion", Phase 10 research, roadmap future objectives | 🔴 Not Implemented |

### Recently Completed (moved to ✅ section):
| Task ID | Implementation | Status |
|---------|----------------|--------|
| P1-2 (old) | NFR Test Coverage: `tests/test_nfr_comprehensive.py`, `docs/testing/nfr_testing.md` | ✅ Complete |
| P1-3 (old) | Dashboard: `agentnet/observability/dashboard.py`, demo HTML | ✅ Complete |
| P1-5 (old) | Multi-Lingual Safety: `agentnet/core/multilingual_safety.py` (12 languages) | ✅ Complete |
| P1-6 (old) | Streaming: `agentnet/streaming/` module with full collaboration | ✅ Complete |

---

## ✅ Completed (Do Not Re-add Unless Regression)
(From roadmap & README; excluded from active list)
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
- **NFR Test Coverage**: Comprehensive test suite (`tests/test_nfr_comprehensive.py`) with 10 passing tests covering reliability, scalability, and security requirements. Documentation in `docs/testing/nfr_testing.md`.
- **Live Dashboard (Observability UI)**: Dashboard implementation in `agentnet/observability/dashboard.py` with cost aggregation, performance metrics visualization, and violation tracking. Standalone demo in `demo_output/p5_standalone_dashboard.html`.
- **Multi-Lingual Safety Policy Translation**: Full implementation in `agentnet/core/multilingual_safety.py` supporting 12 languages (EN, ES, FR, DE, IT, PT, RU, ZH, JA, KO, AR, HI) with pattern/keyword/description sets, language detection, and cultural adaptation.
- **Streaming Partial-Output Collaboration**: Complete streaming module (`agentnet/streaming/`) with `StreamingCollaborator`, partial JSON parsing, collaboration handlers, and intervention capabilities. Tests in `tests/test_p6_streaming.py`.

---

## 📌 Next Immediate Steps (Execution Order)
1. Implement Mirror Agents with Noise Injection (P1 Task 1)
2. Policy Engine Enhancements: priority resolution, conflict explanation, hot reload (P1 Task 2)
3. Complete Phase 9 Deep Learning remaining work: full trainer implementations, embedding generation, neural reasoning (P1 Task 3)
4. Edge/Latency Optimization Pass (P2 Task 9)
5. Advanced Risk Analytics (P2 Task 11)

---

## 🧪 Suggested Branching Strategy
- feature/mirror-agents
- feature/policy-enhancements
- feature/phase9-completion
- feature/edge-optimization
- feature/risk-analytics

Each merges through PR with checklist referencing Task IDs.

---

## ✅ Definition of Done (Global)
- Code + tests + docs updated
- CI green across matrix
- Coverage no regression >2% vs baseline
- Security scan clean (no High/Critical)
- Changelog entry appended (CHANGELOG.md)
- Linked issue(s) closed automatically via PR description

---

Let this file evolve; prune delivered tasks and append new planned work with explicit classification.
