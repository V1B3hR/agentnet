# AgentNet Deep Learning Module (Phase 9)

The deep learning module brings neural network capabilities, fine-tuning infrastructure, and semantic embeddings to AgentNet, enabling state-of-the-art AI agent development.

## 🎯 Overview

Phase 9 integrates deep learning directly into AgentNet, providing:
- **Model Registry**: Version control and lifecycle management for trained models
- **Training Pipeline**: Scalable infrastructure for model training and fine-tuning
- **Fine-Tuning System**: LoRA/QLoRA for memory-efficient LLM adaptation
- **Embedding System**: Semantic understanding with vector search
- **Neural Reasoning**: AI-enhanced cognition integrated with Phase 7

## 📦 Installation

```bash
# Install with deep learning support
pip install agentnet[deeplearning]

# Or install specific frameworks
pip install agentnet[pytorch]      # PyTorch only (recommended)
pip install agentnet[tensorflow]   # TensorFlow support
pip install agentnet[deeplearning_full]  # Everything including wandb
```

### Dependencies

The deep learning module requires:
- **PyTorch** (>=2.0.0) - Primary deep learning framework
- **transformers** (>=4.30.0) - Hugging Face transformers library
- **sentence-transformers** (>=2.2.0) - Semantic embeddings
- **peft** (>=0.4.0) - Parameter-efficient fine-tuning
- **accelerate** (>=0.20.0) - Distributed training support
- **datasets** (>=2.0.0) - Dataset management

## 🚀 Quick Start

### Check Availability

```python
import agentnet
from agentnet import deeplearning

# Check if deep learning is available
print(f"Deep Learning Available: {deeplearning.is_available()}")
print(f"PyTorch Available: {deeplearning.pytorch_available()}")

# Get framework information
print(deeplearning.get_framework_info())
```

### Model Registry

```python
from agentnet.deeplearning import ModelRegistry, ModelMetadata, ModelType, ModelStatus

# Create a registry
registry = ModelRegistry()

# Register a model
metadata = ModelMetadata(
    model_id="my-model-v1",
    name="My Fine-Tuned Model",
    version="1.0.0",
    model_type=ModelType.LANGUAGE_MODEL,
    status=ModelStatus.READY,
    base_model="gpt2",
    metrics={"accuracy": 0.95, "loss": 0.23}
)
registry.register(metadata)

# List models
models = registry.list_models(status=ModelStatus.READY)
```

### Fine-Tuning with LoRA

```python
from agentnet.deeplearning import FineTuner, LoRAConfig

# Configure LoRA
config = LoRAConfig(
    r=8,  # LoRA rank
    lora_alpha=32,  # Scaling parameter
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)

# Fine-tune a model
finetuner = FineTuner(
    base_model="gpt2",
    config=config,
    training_data="path/to/instructions.jsonl",
    use_lora=True
)

# Train (requires PyTorch + peft installed)
model = finetuner.train(num_epochs=3, batch_size=4)
```

### Training Configuration

```python
from agentnet.deeplearning import TrainingConfig

config = TrainingConfig(
    model_name="my-model",
    output_dir="./output",
    learning_rate=2e-5,
    batch_size=8,
    num_epochs=3,
    fp16=True,  # Mixed precision training
    save_steps=500,
    logging_steps=10
)

# Convert to dict for use with trainers
config_dict = config.to_dict()
```

### Embedding Generation

```python
from agentnet.deeplearning import EmbeddingGenerator, EmbeddingCache, SemanticSearch

# Create embedding generator
embedder = EmbeddingGenerator(
    model="all-MiniLM-L6-v2",
    device="cuda"  # or "cpu"
)

# Generate embeddings
texts = ["AI is transforming society", "Machine learning is powerful"]
embeddings = embedder.encode(texts, batch_size=32)

# Semantic search
search = SemanticSearch(embedder)
search.add_documents(texts)
results = search.search("artificial intelligence", top_k=5)
```

### Neural Reasoning

```python
from agentnet.deeplearning import NeuralReasoner, AttentionReasoning

# Create neural reasoner
reasoner = NeuralReasoner(device="cuda")

# Perform reasoning
result = reasoner.reason(
    task="Analyze the impact of AI on society",
    reasoning_type="attention"
)
```

## 🏗️ Architecture

```
agentnet/deeplearning/
├── __init__.py           # Package initialization and exports
├── registry.py           # Model registry and metadata
├── trainer.py            # Training pipeline infrastructure
├── finetuning.py        # LLM fine-tuning utilities
├── embeddings.py        # Embedding generation and search
├── neural_reasoning.py  # Neural reasoning modules
├── model_adapters.py    # Framework adapters (planned)
├── datasets.py          # Dataset utilities (planned)
├── optimization.py      # Training optimization (planned)
└── evaluation.py        # Model evaluation (planned)
```

## 📚 Documentation

- **Planning Document**: [`docs/PHASE9_DEEP_LEARNING_PLAN.md`](../../docs/PHASE9_DEEP_LEARNING_PLAN.md)
- **Implementation Summary**: [`docs/P9_IMPLEMENTATION_SUMMARY.md`](../../docs/P9_IMPLEMENTATION_SUMMARY.md)
- **Demo Script**: [`examples/phase9_demo.py`](../../examples/phase9_demo.py)
- **Tests**: [`tests/test_phase9_deeplearning.py`](../../tests/test_phase9_deeplearning.py)

## 🔗 Integration with Other Phases

### Phase 7 - Advanced Intelligence & Reasoning
- Neural reasoning enhances chain-of-thought and multi-hop reasoning
- Embeddings improve episodic memory retrieval
- Integration with agent evolution for neural skill learning

### Phase 8 - Ecosystem & Integration
- Leverages cloud deployment for distributed training
- Uses enterprise connectors for training data sources
- Integrates with MLOps workflows

### Existing Integrations
- Extends Hugging Face integration with fine-tuning
- Enhances vector database integrations with neural embeddings
- Adds monitoring integration for training metrics

## 🧪 Testing

Run the Phase 9 test suite:

```bash
# Run all Phase 9 tests
pytest tests/test_phase9_deeplearning.py -v

# Run specific test class
pytest tests/test_phase9_deeplearning.py::TestModelRegistry -v
```

## 📊 Performance Goals

- **Training Throughput**: >1000 samples/second on GPU
- **Memory Efficiency**: Support 7B models on 24GB GPU with quantization
- **Model Loading**: <5 seconds for typical models
- **Embedding Generation**: >1000 texts/second
- **Semantic Search**: <100ms for 1M vectors

## 🛠️ Development

### Adding New Components

1. Create new module in `agentnet/deeplearning/`
2. Add imports to `__init__.py`
3. Update `__all__` export list
4. Add tests in `tests/test_phase9_deeplearning.py`
5. Document in this README

### Guidelines

- **Optional Dependencies**: Deep learning features should gracefully degrade
- **Framework Agnostic**: Support both PyTorch and TensorFlow where practical
- **Production Ready**: Focus on memory efficiency and scalability
- **Integration First**: Seamlessly extend existing AgentNet capabilities

## 🚧 Current Status

Phase 9 is in **KICKOFF** stage:

✅ **Completed**:
- Core module structure and scaffolding
- Model registry with versioning
- Training configuration classes
- LoRA/QLoRA configuration
- Embedding framework
- Neural reasoning stubs
- Comprehensive test suite (18 tests)
- Documentation

🚧 **In Progress**:
- Full PyTorch trainer implementation
- Actual embedding generation with sentence-transformers
- Neural reasoning implementations
- Integration with Phase 7 reasoning

📋 **Planned**:
- TensorFlow adapter
- Advanced training optimizations
- Model serving infrastructure
- Online learning capabilities

## 💡 Examples

See [`examples/phase9_demo.py`](../../examples/phase9_demo.py) for a complete demonstration of all Phase 9 features.

## 🤝 Contributing

Contributions to Phase 9 are welcome! Priority areas:
1. Full trainer implementations
2. Neural reasoning architectures
3. Performance optimizations
4. Integration examples
5. Documentation improvements

## 📄 License

Phase 9 follows AgentNet's main license (GPL-3.0).

---

**Version**: 0.5.0 (Phase 9 Kickoff)  
**Last Updated**: 2024  
**Status**: 🚧 Under Active Development
