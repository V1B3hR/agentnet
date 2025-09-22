# AgentNet Integrations Guide

AgentNet provides seamless integrations with popular ML/AI frameworks and services, enabling easy migration and interoperability.

## Overview

The integrations module provides:

- **LangChain Compatibility Layer** - Migrate existing LangChain projects
- **OpenAI Assistants API** - Native support for OpenAI's assistant framework
- **Hugging Face Hub** - Direct model loading and fine-tuning
- **Vector Databases** - Pinecone, Weaviate, and Milvus support
- **Monitoring Stack** - Grafana dashboards and Prometheus metrics

## Installation

### All Integrations
```bash
pip install agentnet[integrations]
```

### Specific Integrations
```bash
pip install agentnet[langchain]          # LangChain compatibility
pip install agentnet[openai]             # OpenAI Assistants API
pip install agentnet[huggingface]        # Hugging Face Hub
pip install agentnet[vector_databases]   # All vector databases
pip install agentnet[monitoring]         # Grafana + Prometheus
```

### Individual Vector Databases
```bash
pip install agentnet[pinecone]          # Pinecone only
pip install agentnet[weaviate]          # Weaviate only
pip install agentnet[milvus]            # Milvus only
```

## LangChain Compatibility Layer

### Quick Migration

```python
from agentnet.integrations import get_langchain_compatibility
from langchain.chat_models import ChatOpenAI

# Your existing LangChain LLM
llm = ChatOpenAI()

# Create compatibility layer
compat = get_langchain_compatibility()()

# Wrap LLM for AgentNet
provider = compat.wrap_langchain_llm(llm)

# Use with AgentNet
from agentnet import AgentNet
agent = AgentNet(
    name="LangChainAgent",
    style={"analytical": 0.8},
    engine=provider
)

response = agent.reason("What are the benefits of AI?")
```

### Message Format Conversion

```python
# Convert between message formats
langchain_msg = compat.convert_message_from_langchain(agent_message)
agent_msg = compat.convert_message_to_langchain(langchain_message)
```

### Tool Migration

```python
from langchain.tools import DuckDuckGoSearchRun

# Convert LangChain tools
langchain_tools = [DuckDuckGoSearchRun()]
agentnet_tools = compat.migrate_langchain_tools(langchain_tools)
```

### Migration Guide Generation

```python
# Get automated migration suggestions
langchain_code = """
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI()
response = llm.predict("Hello")
"""

guide = compat.create_migration_guide(langchain_code)
print(guide)
```

## OpenAI Assistants API

### Basic Setup

```python
from agentnet.integrations import get_openai_assistants

AssistantsAdapter = get_openai_assistants()
assistant = AssistantsAdapter(
    api_key="your-openai-api-key",
    assistant_config={
        "name": "AgentNet Assistant",
        "instructions": "You are a helpful AI assistant.",
        "model": "gpt-4-1106-preview",
        "tools": [{"type": "code_interpreter"}]
    }
)

# Use as AgentNet provider
response = assistant.infer("Calculate the fibonacci sequence up to 100")
```

### Thread Management

```python
# Create conversation thread
thread_id = assistant.create_thread()

# Add messages to thread
assistant.add_message_to_thread(
    thread_id, 
    "Help me analyze this data",
    file_ids=["file-abc123"]
)

# Run assistant
run_id = assistant.run_assistant(thread_id)
run = assistant.wait_for_run_completion(thread_id, run_id)

# Get response messages
messages = assistant.get_thread_messages(thread_id)
```

### File Handling

```python
# Upload file for processing
file_id = assistant.upload_file("data.csv", purpose="assistants")

# Use in conversation
assistant.add_message_to_thread(
    thread_id,
    "Analyze this CSV file",
    file_ids=[file_id]
)
```

## Hugging Face Hub

### Model Loading

```python
from agentnet.integrations import get_huggingface_hub

HFAdapter = get_huggingface_hub()

# Load a conversational model
model = HFAdapter(
    model_name_or_path="microsoft/DialoGPT-medium",
    task="text-generation",
    device="auto"  # Use GPU if available
)

response = model.infer("Hello, how are you today?")
```

### Custom Models

```python
# Load your own fine-tuned model
model = HFAdapter(
    model_name_or_path="your-username/your-model",
    token="your-hf-token",
    trust_remote_code=True
)
```

### Model Search

```python
# Search for models
models = model.search_models(
    query="conversational AI",
    task="text-generation",
    limit=10
)

for model_info in models:
    print(f"{model_info['id']} - Downloads: {model_info['downloads']}")
```

### Fine-tuning

```python
# Fine-tune on your data
model.fine_tune(
    dataset_path="training_data.json",
    output_dir="./fine-tuned-model",
    training_args={
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "warmup_steps": 100,
    }
)
```

### Upload to Hub

```python
# Upload your fine-tuned model
repo_url = model.upload_to_hub(
    repo_name="your-username/your-fine-tuned-model",
    private=False
)
```

## Vector Databases

### Pinecone

```python
from agentnet.integrations import get_vector_database_adapter

PineconeAdapter = get_vector_database_adapter("pinecone")
pinecone = PineconeAdapter(
    api_key="your-pinecone-api-key",
    environment="your-environment"
)

# Connect and create index
pinecone.connect()
pinecone.create_collection("documents", dimension=1536)

# Insert vectors
vectors = [[0.1, 0.2, ...], [0.3, 0.4, ...]]  # Your embeddings
metadata = [{"text": "Document 1"}, {"text": "Document 2"}]
pinecone.insert("documents", vectors, metadata=metadata)

# Search
query_vector = [0.1, 0.2, ...]  # Your query embedding
results = pinecone.search("documents", query_vector, top_k=5)
```

### Weaviate

```python
WeaviateAdapter = get_vector_database_adapter("weaviate")
weaviate = WeaviateAdapter(
    url="http://localhost:8080",
    api_key="your-api-key"  # Optional
)

# Connect and create schema
weaviate.connect()
weaviate.create_collection(
    name="Documents",
    dimension=1536,
    properties=[
        {
            "name": "title",
            "dataType": ["string"],
            "description": "Document title"
        }
    ]
)

# Insert and search
weaviate.insert("Documents", vectors, metadata=metadata)
results = weaviate.search("Documents", query_vector, top_k=5)
```

### Milvus

```python
MilvusAdapter = get_vector_database_adapter("milvus")
milvus = MilvusAdapter(
    host="localhost",
    port=19530,
    user="your-username",  # Optional
    password="your-password"  # Optional
)

# Connect and create collection
milvus.connect()
milvus.create_collection("documents", dimension=1536)

# Insert and search
milvus.insert("documents", vectors, metadata=metadata)
results = milvus.search("documents", query_vector, top_k=5)
```

### Universal Vector Store Usage

```python
from agentnet.integrations.vector_databases import embed_and_store

# Universal function for any vector database
def embed_text(text):
    # Your embedding function
    return [0.1, 0.2, ...]  # Vector representation

texts = ["Document 1 content", "Document 2 content"]
success = embed_and_store(
    adapter=pinecone,  # or weaviate, milvus
    collection_name="documents",
    texts=texts,
    embedding_function=embed_text
)
```

## Monitoring Stack

### Prometheus Metrics

```python
from agentnet.integrations import get_monitoring_integration

PrometheusIntegration = get_monitoring_integration("prometheus")
prometheus = PrometheusIntegration(
    pushgateway_url="http://localhost:9091",
    job_name="agentnet-app"
)

# Record agent metrics
prometheus.record_inference(
    agent_name="my-agent",
    provider="openai",
    model="gpt-4",
    duration=1.5,
    tokens_input=100,
    tokens_output=50,
    cost=0.02
)

# Record violations
prometheus.record_violation(
    monitor_type="keyword_filter",
    severity="warning",
    agent_name="my-agent"
)

# Custom metrics
custom_metric = prometheus.create_custom_metric(
    name="custom_metric_total",
    metric_type="counter",
    description="Custom application metric",
    labels=["app_name", "version"]
)
custom_metric.labels("myapp", "1.0").inc()
```

### Grafana Dashboards

```python
GrafanaIntegration = get_monitoring_integration("grafana")
grafana = GrafanaIntegration(
    url="http://localhost:3000",
    api_key="your-grafana-api-key"
)

# Create Prometheus data source
grafana.create_data_source(
    name="Prometheus",
    ds_type="prometheus",
    url="http://localhost:9090"
)

# Create AgentNet dashboard automatically
dashboard_info = grafana.create_agentnet_dashboard()

# Create notification channel
grafana.create_notification_channel(
    name="slack-alerts",
    channel_type="slack",
    settings={
        "url": "your-slack-webhook-url",
        "channel": "#alerts"
    }
)
```

### Complete Monitoring Setup

```python
from agentnet.integrations.monitoring import setup_agentnet_monitoring

# Set up complete monitoring stack
prometheus, grafana = setup_agentnet_monitoring(
    prometheus_config={
        "pushgateway_url": "http://localhost:9091",
        "prometheus_url": "http://localhost:9090"
    },
    grafana_config={
        "url": "http://localhost:3000",
        "api_key": "your-api-key"
    }
)
```

## Best Practices

### Error Handling

```python
from agentnet.integrations import get_langchain_compatibility

try:
    LangChainCompatibilityLayer = get_langchain_compatibility()
    # Use the integration
except ImportError as e:
    print(f"LangChain not available: {e}")
    # Fallback to default AgentNet behavior
```

### Performance Optimization

```python
# Use appropriate device for HuggingFace models
model = HFAdapter(
    model_name_or_path="model-name",
    device="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype="float16",  # Use half precision for speed
    load_in_8bit=True  # Enable quantization
)
```

### Configuration Management

```python
import os

# Use environment variables for sensitive data
config = {
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
    "grafana_api_key": os.getenv("GRAFANA_API_KEY"),
}
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Install the specific integration dependencies
2. **Memory Issues**: Use quantization for large models
3. **Network Timeouts**: Configure appropriate timeout values
4. **API Rate Limits**: Implement retry logic with backoff

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug logging for integrations
logger = logging.getLogger("agentnet.integrations")
logger.setLevel(logging.DEBUG)
```

## Contributing

To add a new integration:

1. Create a new module in `agentnet/integrations/`
2. Implement the required adapter interface
3. Add lazy import functions to `__init__.py`
4. Update `pyproject.toml` with dependencies
5. Add documentation and tests

See the existing integrations as examples of the expected patterns and interfaces.