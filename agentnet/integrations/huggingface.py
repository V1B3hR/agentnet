"""
Hugging Face Hub Integration

This module provides direct model loading and fine-tuning integration
with Hugging Face Hub, enabling AgentNet to leverage the vast ecosystem
of open-source models and datasets.
"""

from typing import Any, Dict, List, Optional, Union, AsyncIterator, Iterator, Tuple
import asyncio
import json
import logging
import os
from pathlib import Path

try:
    from huggingface_hub import HfApi, HfFolder, Repository, snapshot_download
    from huggingface_hub.utils import HfHubHTTPError
    import transformers
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        pipeline,
        Pipeline,
        Trainer,
        TrainingArguments,
    )
    import torch

    HUGGINGFACE_AVAILABLE = True
except ImportError:
    # Define minimal interface for type hints
    class HfApi:
        pass

    class HfFolder:
        pass

    class Repository:
        pass

    class AutoTokenizer:
        pass

    class AutoModelForCausalLM:
        pass

    class AutoModelForSeq2SeqLM:
        pass

    class Pipeline:
        pass

    class Trainer:
        pass

    class TrainingArguments:
        pass

    HUGGINGFACE_AVAILABLE = False
    torch = None

from ..providers.base import ProviderAdapter
from ..core.types import InferenceResult

logger = logging.getLogger(__name__)


class HuggingFaceHubAdapter(ProviderAdapter):
    """
    AgentNet provider adapter for Hugging Face Hub models.

    Features:
    - Direct model loading from Hub
    - Local model caching
    - Fine-tuning support
    - Pipeline-based inference
    - Custom model registration
    - Quantization support
    """

    def __init__(
        self,
        model_name_or_path: str,
        task: Optional[str] = None,
        token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
        **model_kwargs,
    ):
        """
        Initialize Hugging Face Hub adapter.

        Args:
            model_name_or_path: HF model identifier or local path
            task: Task type (text-generation, text2text-generation, etc.)
            token: Hugging Face token for private models
            cache_dir: Local cache directory
            device: Device to run model on (cuda, cpu, auto)
            torch_dtype: Torch data type (float16, bfloat16, etc.)
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization
            trust_remote_code: Allow custom code execution
            **model_kwargs: Additional model arguments
        """
        super().__init__()

        if not HUGGINGFACE_AVAILABLE:
            raise ImportError(
                "Hugging Face integration requires: pip install transformers huggingface_hub torch"
            )

        self.model_name_or_path = model_name_or_path
        self.task = task or self._infer_task()
        self.token = token or HfFolder.get_token()
        self.cache_dir = cache_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = getattr(torch, torch_dtype) if torch_dtype else None
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.trust_remote_code = trust_remote_code
        self.model_kwargs = model_kwargs

        # Initialize components
        self.hf_api = HfApi(token=self.token)
        self.tokenizer = None
        self.model = None
        self.pipeline = None

        self._initialize_model()

    def _infer_task(self) -> str:
        """Infer task type from model name."""
        model_name_lower = self.model_name_or_path.lower()

        if any(
            keyword in model_name_lower
            for keyword in ["gpt", "llama", "falcon", "mistral"]
        ):
            return "text-generation"
        elif any(keyword in model_name_lower for keyword in ["t5", "bart", "pegasus"]):
            return "text2text-generation"
        elif "embedding" in model_name_lower:
            return "feature-extraction"
        else:
            return "text-generation"  # Default

    def _initialize_model(self):
        """Initialize tokenizer, model, and pipeline."""
        try:
            logger.info(f"Loading model: {self.model_name_or_path}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                token=self.token,
                cache_dir=self.cache_dir,
                trust_remote_code=self.trust_remote_code,
            )

            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Prepare model loading arguments
            model_args = {
                "token": self.token,
                "cache_dir": self.cache_dir,
                "trust_remote_code": self.trust_remote_code,
                **self.model_kwargs,
            }

            # Add quantization arguments
            if self.load_in_8bit:
                model_args["load_in_8bit"] = True
            elif self.load_in_4bit:
                model_args["load_in_4bit"] = True

            if self.torch_dtype:
                model_args["torch_dtype"] = self.torch_dtype

            # Load model based on task
            if self.task == "text2text-generation":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name_or_path, **model_args
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path, **model_args
                )

            # Move to device if not using quantization
            if not (self.load_in_8bit or self.load_in_4bit):
                self.model = self.model.to(self.device)

            # Create pipeline
            self.pipeline = pipeline(
                self.task,
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=self.torch_dtype,
                trust_remote_code=self.trust_remote_code,
            )

            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    def infer(self, prompt: str, **kwargs) -> InferenceResult:
        """
        Synchronous inference using Hugging Face model.

        Args:
            prompt: Input prompt
            **kwargs: Generation parameters

        Returns:
            InferenceResult with model response
        """
        try:
            # Set default generation parameters
            generation_params = {
                "max_new_tokens": kwargs.get("max_tokens", 512),
                "temperature": kwargs.get("temperature", 0.7),
                "do_sample": kwargs.get("do_sample", True),
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 50),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False,
            }

            # Override with user parameters
            generation_params.update(
                {k: v for k, v in kwargs.items() if k in generation_params}
            )

            # Generate response
            if self.task == "text-generation":
                outputs = self.pipeline(prompt, **generation_params)
                if isinstance(outputs, list) and len(outputs) > 0:
                    generated_text = outputs[0].get("generated_text", "")
                else:
                    generated_text = str(outputs)
            elif self.task == "text2text-generation":
                outputs = self.pipeline(prompt, **generation_params)
                if isinstance(outputs, list) and len(outputs) > 0:
                    generated_text = outputs[0].get("generated_text", "")
                else:
                    generated_text = str(outputs)
            else:
                # Fallback for other tasks
                outputs = self.pipeline(prompt, **generation_params)
                generated_text = str(outputs)

            # Calculate approximate token usage
            input_tokens = len(self.tokenizer.encode(prompt))
            output_tokens = len(self.tokenizer.encode(generated_text))
            total_tokens = input_tokens + output_tokens

            return InferenceResult(
                content=generated_text,
                confidence=0.8,  # Default confidence for HF models
                tokens_used=total_tokens,
                cost=0.0,  # Open source models have no API cost
                model=self.model_name_or_path,
                provider="huggingface",
                metadata={
                    "task": self.task,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "device": self.device,
                    "generation_params": generation_params,
                },
            )

        except Exception as e:
            logger.error(f"Hugging Face inference error: {e}")
            return InferenceResult(
                content=f"Error: {str(e)}",
                confidence=0.0,
                tokens_used=0,
                cost=0.0,
                model=self.model_name_or_path,
                provider="huggingface",
                error=str(e),
            )

    async def ainfer(self, prompt: str, **kwargs) -> InferenceResult:
        """Asynchronous inference using Hugging Face model."""
        # Run inference in thread pool to avoid blocking
        return await asyncio.get_event_loop().run_in_executor(
            None, self.infer, prompt, **kwargs
        )

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Stream inference using Hugging Face model.
        Note: True streaming requires special handling.
        """
        try:
            # For now, implement pseudo-streaming by yielding tokens
            generation_params = {
                "max_new_tokens": kwargs.get("max_tokens", 512),
                "temperature": kwargs.get("temperature", 0.7),
                "do_sample": kwargs.get("do_sample", True),
                "return_full_text": False,
            }

            # Generate full response first (streaming would require lower-level API)
            result = self.infer(prompt, **kwargs)

            # Yield response word by word to simulate streaming
            words = result.content.split()
            for word in words:
                yield word + " "

        except Exception as e:
            logger.error(f"Hugging Face streaming error: {e}")
            yield f"Error: {str(e)}"

    async def astream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Async streaming inference."""
        # Convert sync generator to async
        for chunk in self.stream(prompt, **kwargs):
            yield chunk
            await asyncio.sleep(0.01)  # Small delay for async behavior

    def fine_tune(
        self,
        dataset_path: str,
        output_dir: str,
        training_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """
        Fine-tune the model on custom data.

        Args:
            dataset_path: Path to training dataset
            output_dir: Directory to save fine-tuned model
            training_args: Training configuration
            **kwargs: Additional training parameters

        Returns:
            Path to fine-tuned model
        """
        try:
            from datasets import load_dataset

            # Load dataset
            if dataset_path.endswith((".json", ".jsonl")):
                dataset = load_dataset("json", data_files=dataset_path)
            elif dataset_path.endswith(".csv"):
                dataset = load_dataset("csv", data_files=dataset_path)
            else:
                dataset = load_dataset(dataset_path)

            # Default training arguments
            default_args = {
                "output_dir": output_dir,
                "per_device_train_batch_size": 4,
                "per_device_eval_batch_size": 4,
                "num_train_epochs": 3,
                "warmup_steps": 500,
                "logging_steps": 10,
                "save_steps": 1000,
                "eval_steps": 1000,
                "evaluation_strategy": "steps",
                "save_total_limit": 2,
                "remove_unused_columns": False,
                "push_to_hub": False,
            }

            if training_args:
                default_args.update(training_args)

            # Create training arguments
            train_args = TrainingArguments(**default_args)

            # Prepare data collator
            from transformers import DataCollatorForLanguageModeling

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # Causal LM
            )

            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=train_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset.get("validation"),
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )

            # Start training
            logger.info(f"Starting fine-tuning of {self.model_name_or_path}")
            trainer.train()

            # Save model
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)

            logger.info(f"Fine-tuning completed. Model saved to: {output_dir}")
            return output_dir

        except Exception as e:
            logger.error(f"Fine-tuning error: {e}")
            raise

    def upload_to_hub(
        self,
        repo_name: str,
        model_path: Optional[str] = None,
        private: bool = False,
        commit_message: str = "Upload model via AgentNet",
        **kwargs,
    ) -> str:
        """
        Upload model to Hugging Face Hub.

        Args:
            repo_name: Repository name on Hub
            model_path: Local model path (uses current model if None)
            private: Whether to create private repository
            commit_message: Commit message
            **kwargs: Additional upload parameters

        Returns:
            Repository URL
        """
        try:
            # Create repository
            repo_url = self.hf_api.create_repo(
                repo_id=repo_name, private=private, exist_ok=True, token=self.token
            )

            if model_path:
                # Upload from specified path
                self.hf_api.upload_folder(
                    folder_path=model_path,
                    repo_id=repo_name,
                    commit_message=commit_message,
                    token=self.token,
                )
            else:
                # Save current model to temp directory and upload
                import tempfile

                with tempfile.TemporaryDirectory() as temp_dir:
                    self.model.save_pretrained(temp_dir)
                    self.tokenizer.save_pretrained(temp_dir)

                    self.hf_api.upload_folder(
                        folder_path=temp_dir,
                        repo_id=repo_name,
                        commit_message=commit_message,
                        token=self.token,
                    )

            logger.info(f"Model uploaded to: {repo_url}")
            return repo_url

        except Exception as e:
            logger.error(f"Upload error: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        try:
            # Get model info from Hub
            model_info = self.hf_api.model_info(
                repo_id=self.model_name_or_path, token=self.token
            )

            # Get local model info
            num_parameters = sum(p.numel() for p in self.model.parameters())

            return {
                "name": self.model_name_or_path,
                "provider": "huggingface",
                "task": self.task,
                "device": self.device,
                "num_parameters": num_parameters,
                "torch_dtype": str(self.torch_dtype) if self.torch_dtype else None,
                "quantization": {
                    "8bit": self.load_in_8bit,
                    "4bit": self.load_in_4bit,
                },
                "hub_info": {
                    "downloads": model_info.downloads if model_info else 0,
                    "likes": model_info.likes if model_info else 0,
                    "library_name": model_info.library_name if model_info else None,
                    "tags": model_info.tags if model_info else [],
                    "pipeline_tag": model_info.pipeline_tag if model_info else None,
                },
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {
                "name": self.model_name_or_path,
                "provider": "huggingface",
                "error": str(e),
            }

    def search_models(
        self,
        query: Optional[str] = None,
        task: Optional[str] = None,
        sort: str = "downloads",
        limit: int = 10,
        **filters,
    ) -> List[Dict[str, Any]]:
        """
        Search for models on Hugging Face Hub.

        Args:
            query: Search query
            task: Filter by task
            sort: Sort criteria (downloads, modified, etc.)
            limit: Maximum number of results
            **filters: Additional filters

        Returns:
            List of model information dictionaries
        """
        try:
            models = self.hf_api.list_models(
                search=query, task=task, sort=sort, limit=limit, **filters
            )

            model_list = []
            for model in models:
                model_list.append(
                    {
                        "id": model.modelId,
                        "author": model.author,
                        "downloads": model.downloads,
                        "likes": model.likes,
                        "tags": model.tags,
                        "pipeline_tag": model.pipeline_tag,
                        "library_name": model.library_name,
                        "created_at": (
                            model.created_at.isoformat() if model.created_at else None
                        ),
                        "last_modified": (
                            model.last_modified.isoformat()
                            if model.last_modified
                            else None
                        ),
                    }
                )

            return model_list

        except Exception as e:
            logger.error(f"Model search error: {e}")
            return []


# Utility functions
def load_huggingface_model(
    model_name: str, task: Optional[str] = None, **kwargs
) -> HuggingFaceHubAdapter:
    """
    Quick utility to load a Hugging Face model.

    Args:
        model_name: Model name or path
        task: Task type
        **kwargs: Additional model parameters

    Returns:
        Configured HuggingFaceHubAdapter
    """
    return HuggingFaceHubAdapter(model_name_or_path=model_name, task=task, **kwargs)


def search_huggingface_models(
    query: str, task: Optional[str] = None, limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search for models on Hugging Face Hub.

    Args:
        query: Search query
        task: Filter by task
        limit: Maximum results

    Returns:
        List of model information
    """
    # Create temporary adapter to access search functionality
    adapter = HuggingFaceHubAdapter("gpt2")  # Minimal model for API access
    return adapter.search_models(query=query, task=task, limit=limit)
