# AgentNet Roadmap & Architecture

## 1. Product Vision

Build a modular, policy‑governed, multi‑agent reasoning platform that:

- Orchestrates heterogeneous LLMs and custom reasoning engines.
- Supports autonomous and collaborative agents (planning, debate, consensus, delegation).
- Enforces safety, policy, resource, and cost constraints in real time.
- Provides memory (short, episodic, long‑term), tool use, adaptive strategies, and persistent auditability.
- Scales to enterprise multi-tenancy with observability, governance, and extensibility via plugins.

**Primary Value:**
1. Faster experimentation with multi-agent strategies.
2. Safe deployment (policy + runtime monitors).
3. Data-driven iteration (evaluation harness + telemetry).

## 2. Core Use Cases

| Category | Use Case | Description |
|----------|----------|-------------|
| Single Agent | Task reasoning | An agent executes a prompt with adaptive style & policies. |
| Multi-Agent | Debate / brainstorming / consensus | Configurable round orchestration with convergence detection. |
| Workflow Automation | Task graph execution | Planner agent decomposes task into DAG of subtasks assigned to specialists. |
| Tool Augmentation | Retrieval & actions | Agents call tools (search, DB query, code executor, API) under policy. |
| Memory-Enriched Dialog | Knowledge retention | Agent retrieves episodic + semantic memory to ground responses. |
| Governance | Policy compliance | Keyword/regex/custom semantic filters + resource budgets + redaction. |
| Observability | Audit & replay | Persisted reasoning trees + violation logs + cost accounting. |
| Evaluation | Regression scoring | Benchmark tasks to compare model/agent versions. |

## 3. Functional Requirements

- FR1: Create / update / run agents via REST & SDK.
- FR2: Multi-agent session creation: specify participants, mode (debate/brainstorm/consensus/workflow), max rounds, convergence policy.
- FR3: Agents must support sync + async inference and provider fallback (OpenAI, Anthropic, local).
- FR4: Policy monitors: per-turn + final-output evaluation; configurable via uploaded JSON/YAML.
- FR5: Tool interface: register tool with schema, rate limit, allow agent invocation via tool selection policy.
- FR6: Memory system:
  - Short-term: last N turns (bounded).
  - Episodic: persisted transcripts segment indexed by metadata.
  - Semantic: vector store (FAISS / Qdrant / Milvus / PGVector).
- FR7: Task Graph Execution:
  - Planner generates DAG with nodes {task_id, prompt, dependencies, assigned_agent}.
  - Scheduler executes nodes when dependencies complete, aggregates outputs.
- FR8: Session persistence: JSON + optional Postgres schema + object storage (for large artifacts).
- FR9: Observability: metrics (latency, tokens, violations), tracing (OpenTelemetry), structured logs.
- FR10: Cost tracking per provider + per tenant.
- FR11: Versioning: agent configs & policy bundles immutable with version tags.
- FR12: Evaluation harness trigger via API: run scenario set, store metrics & outcome diffs.
- FR13: RBAC: roles (admin, operator, auditor, tenant_user).
- FR14: Redaction pipeline for export (strip sensitive fields).
- FR15: CLI + Python SDK + Web dashboard.

## 4. Non-Functional Requirements

| Attribute | Target |
|-----------|--------|
| Latency (single inference) | P50 < 1.2s (model dependent) |
| Horizontal Scalability | Stateless inference pods behind queue |
| Availability | 99.5% initial target |
| Audit Retention | 90 days hot, then archive |
| Security | JWT + OIDC; encrypted secrets |
| Data Privacy | Tenant isolation (schema or RLS) |
| Throughput | >100 concurrent sessions / node (model dependent) |
| Extensibility | New monitor/tool plugin < 30 min |
| Fault Handling | Fallback chain (primary → backup → local → stub) |

## 5. High-Level Architecture (Textual)

```
+---------------- Web / SDK / CLI ----------------+
                  | REST / GraphQL
                  v
+---------- API Gateway / Auth Layer -------------+
| Rate limiting | JWT/OIDC | Version routing      |
+-------------------+-----------------------------+
                    v
         +--------- Orchestrator Service ---------+
         | Session Manager | Dialogue Engine      |
         | Convergence Check | Turn Scheduler     |
         | DAG Planner | Tool Router              |
         +------------+------+--------------------+
                      |      \
                      v       v
            +---------------+  +--------------+
            | Agent Runtime |  | Tool Runner  |
            | Style/Policy  |  | Sandbox Exec |
            | Memory Fetch  |  +--------------+
            +-------+-------+
                    v
      +----------- Inference Layer -------------+
      | Provider Adapters (OpenAI, Anthropic,   |
      | Local) Retry / Fallback / Streaming     |
      +----------------+------------------------+
                       v
     +----- Observability & Governance ---------+
     | Policy Monitor Bus | Violations | Costs  |
     | Metrics | Tracing | Logs | Alerts        |
     +----------------+-------------------------+
                      v
        +----------- Persistence Layer ---------+
        | Postgres | Vector Store | Object Store|
        +---------------------------------------+
```

## 6. Component Specifications

| Component | Responsibilities | Tech |
|-----------|------------------|------|
| API Gateway | AuthN/Z, rate limit, version routing | FastAPI or Envoy + FastAPI |
| Orchestrator | Sessions, multi-agent control, scheduling | Python |
| Agent Runtime | Wraps DuetMindAgent, hooks | Python |
| Provider Adapters | Normalize calls, streaming, cost | Adapter classes |
| Policy Engine | Rules: regex, semantic, classifier | Python (+ ONNX optional) |
| Monitor Manager | Real-time eval pipeline | Async chain |
| Memory Service | Short/episodic/semantic tiers | Postgres + Vector DB |
| Tool Service | Registration, sandboxed exec, cache | Python + Firecracker/gVisor |
| DAG Planner | Generate/validate DAG | networkx |
| Cost Engine | Token accounting, anomaly detection | Worker |
| Evaluation Harness | Batch scenario runs | Worker queue |
| Telemetry Stack | Metrics/traces/logs | Prometheus, OTel, Loki |
| Dashboard | UI for agents/sessions/policies | Next.js/React |
| Event Bus (opt) | TURN_COMPLETED, VIOLATION_RAISED | Kafka/NATS/Redis |

## 7. Data Model (Initial Schema)

```sql
CREATE TABLE agent (
  id UUID PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  version TEXT NOT NULL,
  style JSONB NOT NULL,
  config JSONB,
  monitors JSONB,
  created_at TIMESTAMPTZ DEFAULT now(),
  archived BOOLEAN DEFAULT FALSE
);

CREATE TABLE session (
  id UUID PRIMARY KEY,
  topic_start TEXT,
  topic_final TEXT,
  mode TEXT,
  strategy TEXT,
  converged BOOLEAN,
  rounds_executed INT,
  participants TEXT[],
  metadata JSONB,
  started_at TIMESTAMPTZ DEFAULT now(),
  ended_at TIMESTAMPTZ
);

CREATE TABLE turn (
  id UUID PRIMARY KEY,
  session_id UUID REFERENCES session(id),
  round INT,
  agent_name TEXT,
  prompt TEXT,
  content TEXT,
  confidence DOUBLE PRECISION,
  raw JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX turn_session_round_idx ON turn(session_id, round);

CREATE TABLE violation (
  id UUID PRIMARY KEY,
  session_id UUID REFERENCES session(id),
  turn_id UUID REFERENCES turn(id),
  monitor_name TEXT,
  severity TEXT,
  violation_type TEXT,
  rationale TEXT,
  meta JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE task_graph (
  id UUID PRIMARY KEY,
  session_id UUID REFERENCES session(id),
  graph_json JSONB,
  status TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE cost_event (
  id UUID PRIMARY KEY,
  session_id UUID,
  agent_name TEXT,
  provider TEXT,
  model TEXT,
  tokens_input INT,
  tokens_output INT,
  cost_usd NUMERIC(12,6),
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE memory_episode (
  id UUID PRIMARY KEY,
  agent_name TEXT,
  embedding VECTOR(1536),
  content TEXT,
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);
```

## 8. Memory Architecture

| Layer | Purpose | Implementation |
|-------|---------|----------------|
| Short-term | Last K turns | In-memory ring buffer |
| Episodic | Persisted chunks | memory_episode table |
| Semantic | Vector similarity | pgvector / Qdrant |
| Structured Facts (future) | Entity graph | Neo4j / typed tables |

**Retrieval Flow:**
1. Pre-inference hook gathers:
   - Short-term tail
   - Top N semantic matches (cosine threshold)
   - Episodic matches by tags  
2. Summarize if token budget exceeded.

## 9. Message / Turn Schema (JSON Contract)

**Implementation Status:** ✅ **COMPLETED** (Step 3 - Message Schema Integration)  
Full pydantic-based schema with AgentNet integration via `to_turn_message()` method. See `agentnet/schemas/` and `docs/STEP3_IMPLEMENTATION_SUMMARY.md`.

```json
{
  "task_id": "uuid",
  "agent": "Athena",
  "input": {
    "prompt": "Analyze X",
    "context": {
      "short_term": ["..."],
      "semantic_refs": [{"id": "e1", "score": 0.83}],
      "episodic_refs": []
    }
  },
  "output": {
    "content": "Reasoned answer...",
    "confidence": 0.87,
    "style_insights": ["Applying rigorous logical validation"],
    "tokens": {"input": 324, "output": 512}
  },
  "monitors": [
    { "name": "keyword_guard", "status": "pass", "elapsed_ms": 2.1 }
  ],
  "cost": {
    "provider": "openai",
    "model": "gpt-4o",
    "usd": 0.01234
  },
  "timing": {
    "started": 1736981000.123,
    "completed": 1736981001.001,
    "latency_ms": 878
  },
  "version": "agent:Athena@1.0.0"
}
```

## 10. Representative API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | /agents | Create agent version |
| GET | /agents/{name} | Fetch agent config |
| POST | /sessions | Start multi-agent session |
| POST | /sessions/{id}/advance | Manual advance |
| GET | /sessions/{id} | Session state summary |
| GET | /sessions/{id}/turns | Paginated turns |
| POST | /tasks/plan | Generate task DAG |
| POST | /tasks/execute | Execute DAG |
| POST | /policies | Upload policy bundle |
| GET | /violations | Query violations |
| POST | /eval/run | Trigger evaluation suite |
| GET | /cost/summary | Cost aggregates |
| POST | /tools | Register tool |
| POST | /tools/invoke | Test tool invocation |
| GET | /metrics/health | Liveness/readiness |

**Create Session Example:**
```json
{
  "topic": "Designing resilient edge network",
  "mode": "brainstorm",
  "strategy": "round_robin",
  "agents": ["Athena@1.0.0", "Apollo@1.0.0"],
  "max_rounds": 6,
  "convergence": true,
  "metadata": {"project": "edgeX"}
}
```

## 11. Multi-Agent Orchestration Logic

Pseudo-round flow:
1. Load session config.
2. For each agent (strategy order):
   - Prepare prompt (mode directive + role + last R summary + diff context).
   - Fetch memory context.
   - Invoke `agent.generate_reasoning_tree()`.
   - Run monitors (pre/post).
   - Persist turn + cost + violations.
3. Update convergence (Jaccard or semantic embedding threshold).
4. If DAG mode: schedule ready nodes.
5. Finalize if converged or max rounds reached.
6. Persist session closure.

## 12. Task Graph Execution

Example:
```json
{
  "nodes": [
    {"id": "root", "prompt": "Plan high availability design", "agent": "Planner", "deps": []},
    {"id": "analysis", "prompt": "Analyze failure modes", "agent": "Athena", "deps": ["root"]},
    {"id": "mitigations", "prompt": "Propose mitigations", "agent": "Apollo", "deps": ["analysis"]},
    {"id": "summary", "prompt": "Integrate plan & mitigations", "agent": "Synthesizer", "deps": ["mitigations"]}
  ]
}
```

Scheduler:
- Maintain `ready_set`.
- Dispatch to queue.
- Worker executes, retries K times, fallback agent on persistent failure.

## 13. LLM Provider Adapter Contract

**Implementation Status:** ✅ **COMPLETED**  
Complete provider implementations for OpenAI, Anthropic, Azure OpenAI, and HuggingFace. All adapters include cost tracking, token counting, streaming support, and built-in retry logic.

```python
class ProviderAdapter:
    def infer(self, model: str, prompt: str, **opts) -> "ProviderResult": ...
    def stream(self, model: str, prompt: str, **opts) -> "Iterable[Chunk]": ...
    def cost(self, tokens_in: int, tokens_out: int, model: str) -> float: ...
```

**Available Providers:**
- ✅ OpenAI (GPT-4, GPT-4o, GPT-3.5-turbo)
- ✅ Anthropic (Claude 3 Opus, Sonnet, Haiku)
- ✅ Azure OpenAI (Enterprise deployments)
- ✅ HuggingFace (Open-source models, local inference)
- ✅ ExampleEngine (Testing and development)

**Fallback Chain Example:** `gpt-4o → gpt-4o-mini → local → synthetic stub`

## 14. Tool System

**Implementation Status:** ✅ **COMPLETED**  
Complete tool system with governance, approval workflows, risk assessment, and audit logging. See `agentnet/tools/` and `docs/security_governance_workflows.md`.

Registration:
```json
{
  "name": "web_search",
  "input_schema": {
    "type": "object",
    "properties": { "query": { "type": "string" } },
    "required": ["query"]
  },
  "rate_limit_per_min": 30,
  "exec_mode": "external_api",
  "auth": { "type": "api_key_ref", "key_id": "serpapi_key" },
  "allowed_agents": ["Athena", "*"]
}
```

**Governance Features:**
- ✅ Human-in-the-loop approval workflows with timeout management
- ✅ Risk assessment (low, medium, high, critical) with custom assessors
- ✅ Policy-based tool restrictions with custom validators
- ✅ Audit logging and compliance tracking
- ✅ Configurable approval policies per tool pattern

Flow: Select → Risk Assessment → Approval (if required) → Validate (rate/auth) → Execute → Audit → Cache → Append context.

## 15. Policy & Governance Extensions

**Implementation Status:** ✅ **COMPLETED**  
Complete policy system with advanced rule types including semantic similarity, LLM classifiers, and numerical thresholds. See `agentnet/core/policy/`.

**Implemented Rule Types:**
- ✅ `regex` - Pattern matching for keywords and phrases
- ✅ `semantic_similarity` - Embedding-based similarity detection using sentence-transformers
- ✅ `llm_classifier` - LLM-based content classification for toxicity, violations, etc.
- ✅ `numerical_threshold` - Resource limits, token counts, cost tracking
- ✅ `role` - Role-based access control
- ✅ `custom` - User-defined validation functions

Example:
```yaml
rules:
  - name: no_pii
    type: regex
    severity: severe
    pattern: "(\\b\\d{3}-\\d{2}-\\d{4}\\b)"
  - name: toxicity_screen
    type: llm_classifier
    severity: major
    params: { model: "moderation-small", threshold: 0.78 }
  - name: semantic_block_list
    type: semantic_similarity
    severity: severe
    params: { embedding_set: "restricted_corpora", max_similarity: 0.92 }
  - name: cost_limit
    type: numerical_threshold
    severity: major
    params: { metric_key: "cost_usd", threshold: 1.0, comparison: "<=" }
```

**Features:**
- ✅ Semantic similarity matching with sentence-transformers
- ✅ LLM-based classification with confidence thresholds
- ✅ Numerical threshold checks for resources
- ✅ Configurable severity levels and actions (allow, block, transform, log, require_approval)
- ✅ Rule metadata and context tracking

## 16. Security & Isolation

| Aspect | Mechanism |
|--------|-----------|
| Auth | OIDC (Keycloak/Auth0) → JWT (tenant claim) |
| Rate Limiting | Redis token bucket |
| Secrets | Vault / AWS Secrets Manager |
| Tool Sandbox | Firecracker / gVisor |
| Prompt Injection | Pre-filter + macro whitelist |
| Data Isolation | RLS in Postgres |
| Export Controls | Redaction pipeline |
| Supply Chain | Snyk/Trivy in CI |

## 17. Deployment Topology

Kubernetes services:
- api-gateway
- orchestrator-service
- inference-workers (HPA)
- tool-runner
- evaluation-runner
- vector-store
- postgres
- redis
- kafka/nats (optional)
- prometheus/grafana/jaeger/loki

Ingress: NGINX / Envoy  
CDN: optional for dashboard.

## 18. Observability Metrics

| Metric | Labels | Source |
|--------|--------|--------|
| inference_latency_ms | model, provider, agent | Adapter |
| tokens_consumed_total | model, provider, tenant | Adapter |
| violations_total | severity, rule_name | Monitor |
| cost_usd_total | provider, model, tenant | Cost engine |
| session_rounds | mode, converged | Orchestrator |
| tool_invocations_total | tool_name, status | Tool runner |
| dag_node_duration_ms | agent, node_type | Scheduler |

Spans: `session.round.turn`, `agent.inference`, `monitor.sequence`, `tool.invoke`  
Logs: JSON with `correlation_id = session_id`.

## 19. Evaluation Harness

Example:
```yaml
suite: "baseline_design_eval"
scenarios:
  - name: "resilience_planning"
    mode: "brainstorm"
    agents: ["Athena", "Apollo"]
    topic: "Edge network partition recovery"
    success_criteria:
      - type: keyword_presence
        must_include: ["redundancy", "failover"]
      - type: semantic_score
        reference_id: "ref_doc_12"
        min_score: 0.78
  - name: "ethical_debate"
    mode: "debate"
    agents: ["Athena", "Hermes"]
    topic: "Autonomous swarm decision hierarchy"
```

Per-scenario metrics: coverage_score, novelty_score, coherence_score, rule_violations_count.

## 20. Cost Tracking Flow

**Implementation Status:** ✅ **COMPLETED** (Step 3 - Cost Tracking Integration)  
Automatic cost recording with analytics via `get_cost_summary()`. See `agentnet/core/cost/` and `docs/STEP3_IMPLEMENTATION_SUMMARY.md`.

1. Adapter returns token counts.
2. Pricing table (JSON) applied.
3. Persist `cost_event`.
4. Aggregator rolls up hourly/daily per tenant & model.
5. Alert on spend velocity.

## 21. CI/CD Pipeline

1. Lint + type check (ruff + mypy)
2. Unit tests (pytest) coverage > 85%
3. Security scan (bandit / trivy)
4. Contract tests (adapters)
5. Integration tests (ephemeral DB + vector)
6. Evaluation regression gate
7. Docker build (sha + semver)
8. Deploy via ArgoCD/Flux (staging → prod)
9. Smoke tests

## 22. Risk Register

**Implementation Status:** ✅ **COMPLETED**  
Full implementation with automated risk management, event tracking, and mitigation workflows. See `agentnet/risk/__init__.py` (663 lines of production code).

**Risk Categories:**
- Operational (provider outage, convergence stall)
- Security (tool injection, prompt leakage)
- Performance (memory bloat, latency)
- Financial (token cost spike)
- Compliance (policy violations, data handling)
- Technical (API failures, system errors)

**Risk Management Features:**
- ✅ Automated risk detection and classification
- ✅ Risk event registration with metadata tracking
- ✅ Mitigation strategy definitions and automated execution
- ✅ Risk level assessment (low, medium, high, critical)
- ✅ Persistent storage and retrieval of risk events
- ✅ Risk dashboard and analytics
- ✅ Tenant-specific risk tracking

| Risk | Impact | Mitigation | Status |
|------|--------|------------|--------|
| Provider outage | Degraded service | Fallback + circuit breaker | ✅ Implemented |
| Policy false positives | Frustration | Severity tiers + override token | ✅ Implemented |
| Token cost spike | Budget overrun | Spend alerts + downgrade | ✅ Implemented |
| Memory bloat | Latency | Summaries + pruning | ✅ Implemented |
| Tool injection | Data exfiltration | Schema validation + sandbox | ✅ Implemented |
| Convergence stall | Long sessions | Hard caps + stagnation detection | ✅ Implemented |
| Prompt leakage | Compliance breach | Secret scanning + redaction | ✅ Implemented |

## 23. Phase Roadmap

| Phase | Duration | Goals |
|-------|----------|-------|
| P0 | Weeks 1–4 | Core agent runtime, adapters, sessions, monitors v1, persistence |
| P1 | Weeks 5–10 | Debate/brainstorm, convergence engine, policy bundles, dashboard MVP, cost v1 |
| P2 | Weeks 11–16 | Tool system, semantic memory |
| P3 | Weeks 17–22 | DAG planner/scheduler, evaluation harness |
| P4 | Weeks 23–28 | Advanced monitors, RBAC, multi-tenancy |
| P5 | Weeks 29–34 | UX polish, streaming, plugin SDK, fallback refinement |
| P6 | Weeks 35–40 | Auditing dashboards, spend anomaly detection, export controls, SOC2 logging |

## 24. First 6-Week Sprint Breakdown

**Week 1**
- Extract `DuetMindAgent` → `core/agent.py`
- ProviderAdapter + OpenAI adapter
- FastAPI skeleton + auth stub
- Alembic migrations

**Week 2**
- `POST /agents/{name}/infer`
- Monitor pipeline abstraction
- Cost event recording

**Week 3**
- `/sessions` endpoints
- Round orchestration + convergence detection
- Persist turns & violations
- Metrics + structured logging

**Week 4**
- Resource monitor + `/policies` loader
- Dashboard skeleton
- Fallback provider config + circuit breaker

**Week 5**
- Convergence visualization
- Embedding store setup (pgvector?)
- Memory retrieval prototype (flagged)

**Week 6**
- Error handling hardening
- Evaluation harness scaffold
- Load test + perf tuning
- P1 planning retrospective

## 25. Implemented Optimizations & Enhancements

### Performance Optimization ✅
- **Asynchronous memory operations with batching**: Batch processing in embedding system (`agentnet/deeplearning/embeddings.py`)
- **Model response caching with smart invalidation**: TTL-based cache with LRU eviction (`agentnet/core/cache.py`)
- **Distributed agent execution across multiple nodes**: Kubernetes operator with multi-node orchestration (`agentnet/enterprise/deployment.py`)
- **GPU acceleration for inference and embeddings**: Device selection (CPU/CUDA) in embedding generator
- **Memory usage optimization**: Cache limits, TTL expiration, LRU eviction strategies

### Security Enhancements ✅
- **End-to-end encryption**: TLS/SSL support with cert-manager integration in K8s deployments
- **Zero-trust architecture**: Security policies, sandboxing, permission management (`agentnet/plugins/security.py`)
- **Advanced threat detection**: Policy violation detection, audit logging, security monitoring
- **Secure multi-tenancy**: Plugin sandboxing with resource limits and environment isolation
- **Compliance automation**: SOC2 reporting, export controls, audit trails (`agentnet/compliance/`)

### Scalability Improvements ✅
- **Horizontal scaling**: Complete HPA implementation with CPU, memory, and custom metrics
- **Database sharding**: Multi-region deployment with data locality and compliance support
- **Content delivery network (CDN)**: Ingress configuration with CDN-ready architecture
- **Edge computing**: Multi-region deployment with zone-based distribution for low latency
- **Auto-scaling policies**: Full HPA/VPA with custom metrics, stabilization windows, and pod disruption budgets

## 26. Future Enhancements

- Streaming tool invocation with adaptive reasoning
- Agent reputation scoring
- Federated memory across clusters
- Exposed reasoning graphs API
- Adaptive strategy selection (topic classification)

## 26. Proposed File / Module Layout

```
/src
  /api
    agents.py
    sessions.py
    policies.py
    tools.py
    eval.py
  /core
    agent.py
    monitors/
      base.py
      keyword.py
      regex.py
      resource.py
      semantic.py
    policies/
      loader.py
    memory/
      store.py
      retriever.py
    provider/
      base.py
      openai_adapter.py
      local_adapter.py
    orchestration/
      session_manager.py
      round_engine.py
      dag_planner.py
      scheduler.py
    tools/
      registry.py
      executor.py
    cost/
      pricing.py
      recorder.py
    eval/
      runner.py
      metrics.py
  /infra
    db.py
    migrations/
  /observability
    metrics.py
    tracing.py
  /cli
    main.py
/tests
  /unit
  /integration
  /perf
```

## 27. Extensions to Current DuetMindAgent

| Current | Extension |
|---------|-----------|
| Keyword/regex/resource monitors | Add semantic + classifier |
| Sync/async reasoning | Add streaming + incremental monitors |
| Basic dialogue | Pluggable strategy interface |
| Style influence | Style transformers registry |
| JSON persistence | DB abstraction |
| File-based policy | Remote versioned store + hot reload |
| Lexical convergence | Add semantic similarity + stagnation detection |

## 28. Testing Strategy

| Layer | Focus |
|-------|-------|
| Unit | Monitors, policy parsing, memory retrieval |
| Contract | Provider adapters |
| Integration | Multi-agent lifecycle |
| Load | 500 concurrent sessions (stub model) |
| Chaos | Provider latency, tool failure |
| Security | Injection, sandbox escapes |
| Regression | Evaluation suite diffs |

## 29. Immediate Action Checklist

1. Create repo structure.
2. Isolate `DuetMindAgent` core & interfaces.
3. Implement ProviderAdapter base + OpenAI adapter.
4. FastAPI skeleton (`/health`, `/agents`).
5. Alembic migrations (schema above).
6. Evaluation scenario YAML + harness stub.
7. Prometheus metrics exporter.
8. Basic cost accumulator.

## 30. Next Step Inputs Needed

Please provide:
- Preferred vector DB: (pgvector / Qdrant / other)
- Primary initial model provider: (OpenAI / Anthropic / Local vLLM)
- Target deployment (AWS / GCP / on‑prem / other)
- Do you want an initial docker-compose (dev) with PG + vector store + Redis?

Once you confirm, I can:
- Generate `docker-compose.dev.yml`
- Produce initial code skeleton files
- Provide sample policy bundle + agent configs
- Optionally open a PR adding this roadmap

---

Let me know if you’d like any of these sections split into separate documents (e.g., SECURITY.md, ARCHITECTURE.md, EVALUATION.md) or if you prefer a slimmer public version.
