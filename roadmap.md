# AgentNet To-Do List (Actionable Backlog)

Source-derived from: README.md and roadmap.md (commit 60f4da8)  
Scope: Only unimplemented, partial, planned, or explicitly outstanding work.  
Priority scale:   P1 (High) < P2 (Medium) < P3 (Strategic / Long-Term)


---

## 🟠 P1 – High Priority (Stability, Safety, Core Feature Completion)

### 2. Strengthen Non-Functional Requirement (NFR) Test Coverage
- Gaps: Performance thresholds, concurrency stress, failure injection, memory footprint, latency variance.
- Tasks:
  - Add `tests/nfr/` suite: load test (orchestrator concurrency), memory retention stress, tool invocation saturation, policy violation burst.
  - Add performance baselines (store JSON snapshot & compare drift)
  - Introduce chaos tests (simulated provider timeout, tool failure)
- Acceptance: CI job `nfr` passes; documented thresholds in `docs/nfr/metrics.md`.

### 3. Implement Live Dashboard (Observability UI)
- Current: "Dashboard (planned)" placeholder.
- Features:
  - Real-time event stream (WebSocket or SSE)
  - Policy violations panel (filter by agent/session/severity)
  - Metrics tiles (latency, token usage, cost, violation rate)
  - Session replay selector
- Backend: Reuse event bus; add `/ws/events` endpoint.
- Frontend: Lightweight (FastAPI + HTMX or small React/Vite)
- Acceptance: Launch `docker-compose up` -> dashboard accessible; doc in `docs/observability/dashboard.md`.

### 4. Mirror Agents with Noise Injection ("Dusty Mirror")
- Purpose: Robustness & variance testing.
- Implementation:
  - MirrorAgent wrapper: clones base agent, injects perturbations (sampling temperature jitter, prompt paraphrasing, token dropout).
  - Compare divergence scores (semantic similarity, factual variance).
- Acceptance: Example in `examples/mirror_agents_demo.py`; test asserting diversity > threshold.

### 5. Multi-Lingual Safety Policy Translation (Planned)
- Extend safety engine with i18n mapping.
- Features:
  - Language packs: pattern + keyword + description sets
  - Confidence-backed translation fallback
  - False-positive logging + adaptive refinement
- Acceptance: Add `agentnet/policy/language/` with at least EN, ES; tests include detection in two languages.

### 6. Streaming Partial-Output Collaboration
- Current: Marked "planned".
- Requirements:
  - Streaming token-level or chunk events
  - Mid-stream evaluator intervention (halt / revise)
  - Incremental memory buffering (low-latency)
- Acceptance: `StreamingCollaborator` fully implemented with intervention test.

### 7. Complete Phase 9 Deep Learning Remaining Work
Although scaffolding exists, the following likely remain:
- Model artifact version diffing & rollback
- Distributed training orchestration templates (multi-node)
- Embedding batch optimizer (adaptive batch size, precision)
- Fine-tuning evaluation harness integration (auto-benchmark on finish)
- GPU resource scheduler (multi-session fairness)
- Acceptance: `docs/PHASE9_DEEP_LEARNING_PLAN.md` updated with "Delivered" stamps + demo script `examples/phase9_training_pipeline_demo.py`.

### 8. Policy Engine Enhancements (If Not Yet Present)
- Add: rule priority resolution, conflict explanation object, hot reload without restart.
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

| Task ID | Source Snippet / Indicator |
|---------|----------------------------|

| 2 | Roadmap: NFR partially working; coverage gaps |
| 3 | README: "Dashboard (planned)" |
| 4 | README: Inline marker `!!!Create "mirror agents"...` |
| 5 | README: Fundamental Law #23 (planned) |
| 6 | Fundamental Law #24 (planned) + streaming collaboration section (planned enhancements) |
| 7 | README Phase 9 "scaffolding complete" (implies remaining implementation depth) |
| 8 | Potential extension beyond current delivered features (policy conflict resolution not explicitly claimed) |
| 9–15 | Inference from partial / robustness / scaling ambitions & typical maturity path (README performance & scalability emphasis) |
| 16–24 | README "Growth Initiatives", "Global Expansion", Phase 10 research, roadmap future objectives |

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

---

## 📌 Next Immediate Steps (Execution Order)
1. P0-1 Sprint: Implement CI/CD (Task 1)
2. Add NFR test suite & coverage gating (Task 2)
3. Implement Mirror Agents + Dashboard (Tasks 4 & 3 parallel if team-split)
4. Streaming partial-output + multi-lingual policy (Tasks 6 & 5)
5. Phase 9 deep learning completion tasks (Task 7)

---

## 🧪 Suggested Branching Strategy
- feature/ci-cd
- feature/nfr-suite
- feature/dashboard
- feature/mirror-agents
- feature/streaming-collab
- feature/multilingual-policy
- feature/phase9-completion

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
