"""
OpenAI Assistants API Integration

This module provides native support for OpenAI's Assistants API,
allowing AgentNet to leverage OpenAI's assistant framework with
file handling, code execution, and function calling capabilities.
"""

from typing import Any, Dict, List, Optional, Union, AsyncIterator, Iterator
import asyncio
import json
import logging
import time
from enum import Enum

try:
    import openai
    from openai.types.beta import Assistant, Thread, ThreadMessage, Run
    from openai.types.beta.threads import TextContentBlock, ImageFileContentBlock

    OPENAI_AVAILABLE = True
except ImportError:
    # Define minimal interface for type hints when OpenAI is not available
    class Assistant:
        pass

    class Thread:
        pass

    class ThreadMessage:
        pass

    class Run:
        pass

    class TextContentBlock:
        pass

    class ImageFileContentBlock:
        pass

    OPENAI_AVAILABLE = False

from ..providers.base import ProviderAdapter
from ..core.types import InferenceResult

logger = logging.getLogger(__name__)


class RunStatus(Enum):
    """OpenAI Assistant Run status enumeration."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    REQUIRES_ACTION = "requires_action"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    FAILED = "failed"
    COMPLETED = "completed"
    EXPIRED = "expired"


class OpenAIAssistantsAdapter(ProviderAdapter):
    """
    AgentNet provider adapter for OpenAI Assistants API.

    Features:
    - Assistant creation and management
    - Thread-based conversations
    - File upload and processing
    - Code interpreter integration
    - Function calling support
    - Streaming responses
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        assistant_id: Optional[str] = None,
        assistant_config: Optional[Dict[str, Any]] = None,
        client: Optional[Any] = None,
    ):
        """
        Initialize OpenAI Assistants adapter.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            assistant_id: Existing assistant ID to use
            assistant_config: Configuration for creating new assistant
            client: Pre-configured OpenAI client
        """
        super().__init__()

        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package is required. Install with: pip install openai>=1.0.0"
            )

        # Initialize OpenAI client
        if client:
            self.client = client
        else:
            self.client = openai.OpenAI(api_key=api_key)

        self.assistant_id = assistant_id
        self.assistant_config = assistant_config or {}
        self.assistant = None
        self.threads = {}  # Thread management
        self.default_model = "gpt-4-1106-preview"

        # Initialize assistant
        if assistant_id:
            self._load_assistant()
        else:
            self._create_assistant()

    def _load_assistant(self):
        """Load existing assistant by ID."""
        try:
            self.assistant = self.client.beta.assistants.retrieve(self.assistant_id)
            logger.info(
                f"Loaded assistant: {self.assistant.name} ({self.assistant.id})"
            )
        except Exception as e:
            logger.error(f"Failed to load assistant {self.assistant_id}: {e}")
            raise

    def _create_assistant(self):
        """Create new assistant with configuration."""
        config = {
            "name": "AgentNet Assistant",
            "instructions": "You are a helpful AI assistant integrated with AgentNet.",
            "model": self.default_model,
            "tools": [{"type": "code_interpreter"}],
            **self.assistant_config,
        }

        try:
            self.assistant = self.client.beta.assistants.create(**config)
            self.assistant_id = self.assistant.id
            logger.info(
                f"Created assistant: {self.assistant.name} ({self.assistant.id})"
            )
        except Exception as e:
            logger.error(f"Failed to create assistant: {e}")
            raise

    def create_thread(self, thread_id: Optional[str] = None) -> str:
        """
        Create or retrieve a conversation thread.

        Args:
            thread_id: Existing thread ID (optional)

        Returns:
            Thread ID
        """
        if thread_id and thread_id in self.threads:
            return thread_id

        try:
            if thread_id:
                # Retrieve existing thread
                thread = self.client.beta.threads.retrieve(thread_id)
            else:
                # Create new thread
                thread = self.client.beta.threads.create()

            self.threads[thread.id] = thread
            return thread.id
        except Exception as e:
            logger.error(f"Failed to create/retrieve thread: {e}")
            raise

    def add_message_to_thread(
        self,
        thread_id: str,
        content: str,
        role: str = "user",
        file_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Add a message to a thread.

        Args:
            thread_id: Thread ID
            content: Message content
            role: Message role (user/assistant)
            file_ids: List of file IDs to attach

        Returns:
            Message ID
        """
        try:
            message_data = {
                "thread_id": thread_id,
                "role": role,
                "content": content,
            }

            if file_ids:
                message_data["file_ids"] = file_ids

            message = self.client.beta.threads.messages.create(**message_data)
            return message.id
        except Exception as e:
            logger.error(f"Failed to add message to thread: {e}")
            raise

    def run_assistant(
        self,
        thread_id: str,
        instructions: Optional[str] = None,
        additional_instructions: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Run the assistant on a thread.

        Args:
            thread_id: Thread ID
            instructions: Override instructions for this run
            additional_instructions: Additional instructions
            tools: Override tools for this run

        Returns:
            Run ID
        """
        try:
            run_data = {
                "thread_id": thread_id,
                "assistant_id": self.assistant_id,
            }

            if instructions:
                run_data["instructions"] = instructions
            if additional_instructions:
                run_data["additional_instructions"] = additional_instructions
            if tools:
                run_data["tools"] = tools

            run = self.client.beta.threads.runs.create(**run_data)
            return run.id
        except Exception as e:
            logger.error(f"Failed to run assistant: {e}")
            raise

    def wait_for_run_completion(
        self, thread_id: str, run_id: str, timeout: int = 30, poll_interval: float = 1.0
    ) -> Run:
        """
        Wait for a run to complete.

        Args:
            thread_id: Thread ID
            run_id: Run ID
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds

        Returns:
            Completed Run object
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id, run_id=run_id
                )

                if run.status in [
                    RunStatus.COMPLETED.value,
                    RunStatus.FAILED.value,
                    RunStatus.CANCELLED.value,
                    RunStatus.EXPIRED.value,
                ]:
                    return run
                elif run.status == RunStatus.REQUIRES_ACTION.value:
                    # Handle function calls if needed
                    self._handle_required_actions(thread_id, run_id, run)

                time.sleep(poll_interval)
            except Exception as e:
                logger.error(f"Error polling run status: {e}")
                break

        raise TimeoutError(f"Run {run_id} did not complete within {timeout} seconds")

    def _handle_required_actions(self, thread_id: str, run_id: str, run: Run):
        """Handle required actions (function calls) during run."""
        if not run.required_action:
            return

        tool_outputs = []

        for tool_call in run.required_action.submit_tool_outputs.tool_calls:
            try:
                # Execute the function call
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                # This would be where you'd implement your function calling logic
                # For now, we'll just return a placeholder
                result = f"Function {function_name} called with args: {function_args}"

                tool_outputs.append({"tool_call_id": tool_call.id, "output": result})
            except Exception as e:
                tool_outputs.append(
                    {"tool_call_id": tool_call.id, "output": f"Error: {str(e)}"}
                )

        # Submit tool outputs
        if tool_outputs:
            self.client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id, run_id=run_id, tool_outputs=tool_outputs
            )

    def get_thread_messages(
        self, thread_id: str, limit: int = 20
    ) -> List[ThreadMessage]:
        """
        Get messages from a thread.

        Args:
            thread_id: Thread ID
            limit: Maximum number of messages to retrieve

        Returns:
            List of thread messages
        """
        try:
            messages = self.client.beta.threads.messages.list(
                thread_id=thread_id, limit=limit
            )
            return list(messages.data)
        except Exception as e:
            logger.error(f"Failed to get thread messages: {e}")
            return []

    def upload_file(self, file_path: str, purpose: str = "assistants") -> str:
        """
        Upload a file for use with assistants.

        Args:
            file_path: Path to the file
            purpose: File purpose (assistants, fine-tune, etc.)

        Returns:
            File ID
        """
        try:
            with open(file_path, "rb") as file:
                uploaded_file = self.client.files.create(file=file, purpose=purpose)
            return uploaded_file.id
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            raise

    def infer(self, prompt: str, **kwargs) -> InferenceResult:
        """
        Synchronous inference using OpenAI Assistant.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters

        Returns:
            InferenceResult with assistant response
        """
        try:
            # Create or get thread
            thread_id = kwargs.get("thread_id")
            if not thread_id:
                thread_id = self.create_thread()

            # Add user message
            self.add_message_to_thread(thread_id, prompt)

            # Run assistant
            run_id = self.run_assistant(
                thread_id,
                instructions=kwargs.get("instructions"),
                additional_instructions=kwargs.get("additional_instructions"),
                tools=kwargs.get("tools"),
            )

            # Wait for completion
            run = self.wait_for_run_completion(
                thread_id, run_id, timeout=kwargs.get("timeout", 30)
            )

            if run.status != RunStatus.COMPLETED.value:
                return InferenceResult(
                    content=f"Run failed with status: {run.status}",
                    confidence=0.0,
                    tokens_used=0,
                    cost=0.0,
                    model=self.assistant.model,
                    provider="openai_assistants",
                    error=f"Run status: {run.status}",
                )

            # Get messages
            messages = self.get_thread_messages(thread_id, limit=1)
            if not messages:
                return InferenceResult(
                    content="No response from assistant",
                    confidence=0.0,
                    tokens_used=0,
                    cost=0.0,
                    model=self.assistant.model,
                    provider="openai_assistants",
                    error="No messages returned",
                )

            # Extract content from the latest message
            latest_message = messages[0]
            content = ""

            for content_block in latest_message.content:
                if isinstance(content_block, TextContentBlock):
                    content += content_block.text.value
                elif hasattr(content_block, "text"):
                    content += content_block.text.value

            return InferenceResult(
                content=content,
                confidence=0.9,  # High confidence for OpenAI Assistants
                tokens_used=run.usage.total_tokens if run.usage else 0,
                cost=self._estimate_cost(run.usage) if run.usage else 0.0,
                model=self.assistant.model,
                provider="openai_assistants",
                metadata={
                    "thread_id": thread_id,
                    "run_id": run_id,
                    "assistant_id": self.assistant_id,
                    "run_status": run.status,
                    "usage": run.usage.model_dump() if run.usage else None,
                },
            )

        except Exception as e:
            logger.error(f"OpenAI Assistants inference error: {e}")
            return InferenceResult(
                content=f"Error: {str(e)}",
                confidence=0.0,
                tokens_used=0,
                cost=0.0,
                model=self.assistant.model if self.assistant else "unknown",
                provider="openai_assistants",
                error=str(e),
            )

    async def ainfer(self, prompt: str, **kwargs) -> InferenceResult:
        """
        Asynchronous inference using OpenAI Assistant.
        Currently falls back to synchronous method.
        """
        # OpenAI Python client doesn't have full async support for Assistants API yet
        # This would be implemented when available
        return await asyncio.get_event_loop().run_in_executor(
            None, self.infer, prompt, **kwargs
        )

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Stream inference - OpenAI Assistants API doesn't support streaming yet.
        Falls back to regular inference.
        """
        result = self.infer(prompt, **kwargs)
        yield result.content

    async def astream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Async streaming - falls back to regular inference."""
        result = await self.ainfer(prompt, **kwargs)
        yield result.content

    def _estimate_cost(self, usage: Any) -> float:
        """
        Estimate cost based on token usage.
        This is a rough estimate and should be updated with actual pricing.
        """
        if not usage:
            return 0.0

        # GPT-4 Turbo pricing (as of late 2023)
        input_cost_per_token = 0.00001  # $0.01 per 1K tokens
        output_cost_per_token = 0.00003  # $0.03 per 1K tokens

        input_cost = usage.prompt_tokens * input_cost_per_token
        output_cost = usage.completion_tokens * output_cost_per_token

        return input_cost + output_cost

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the assistant."""
        if not self.assistant:
            return {"error": "No assistant loaded"}

        return {
            "assistant_id": self.assistant.id,
            "name": self.assistant.name,
            "model": self.assistant.model,
            "provider": "openai_assistants",
            "description": self.assistant.description,
            "instructions": self.assistant.instructions,
            "tools": (
                [tool.type for tool in self.assistant.tools]
                if self.assistant.tools
                else []
            ),
            "file_ids": self.assistant.file_ids,
            "created_at": self.assistant.created_at,
        }

    def update_assistant(self, **updates) -> Assistant:
        """
        Update assistant configuration.

        Args:
            **updates: Fields to update

        Returns:
            Updated Assistant object
        """
        try:
            self.assistant = self.client.beta.assistants.update(
                self.assistant_id, **updates
            )
            return self.assistant
        except Exception as e:
            logger.error(f"Failed to update assistant: {e}")
            raise

    def delete_assistant(self) -> bool:
        """
        Delete the assistant.

        Returns:
            True if successful
        """
        try:
            self.client.beta.assistants.delete(self.assistant_id)
            self.assistant = None
            self.assistant_id = None
            return True
        except Exception as e:
            logger.error(f"Failed to delete assistant: {e}")
            return False

    def list_assistants(self, limit: int = 20) -> List[Assistant]:
        """
        List available assistants.

        Args:
            limit: Maximum number of assistants to return

        Returns:
            List of Assistant objects
        """
        try:
            assistants = self.client.beta.assistants.list(limit=limit)
            return list(assistants.data)
        except Exception as e:
            logger.error(f"Failed to list assistants: {e}")
            return []


# Utility functions
def create_openai_assistant(
    name: str,
    instructions: str,
    model: str = "gpt-4-1106-preview",
    tools: Optional[List[Dict[str, Any]]] = None,
    file_ids: Optional[List[str]] = None,
    api_key: Optional[str] = None,
) -> OpenAIAssistantsAdapter:
    """
    Quick utility to create an OpenAI Assistant adapter.

    Args:
        name: Assistant name
        instructions: Assistant instructions
        model: Model to use
        tools: List of tools to enable
        file_ids: List of file IDs to attach
        api_key: OpenAI API key

    Returns:
        Configured OpenAIAssistantsAdapter
    """
    config = {
        "name": name,
        "instructions": instructions,
        "model": model,
    }

    if tools:
        config["tools"] = tools
    if file_ids:
        config["file_ids"] = file_ids

    return OpenAIAssistantsAdapter(api_key=api_key, assistant_config=config)
