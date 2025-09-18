"""
Observability-instrumented provider adapter mixin.

Provides automatic metrics collection, tracing, and logging for provider operations.
Can be mixed with any provider adapter to add observability features.
"""

import logging
import time
from functools import wraps
from typing import Any, Dict, Optional

from ..observability.dashboard import get_global_dashboard_collector
from ..observability.logging import get_correlation_logger
from ..observability.metrics import MetricsCollector, get_global_metrics
from ..observability.tracing import get_global_tracer, trace_agent_operation
from .base import ProviderAdapter

logger = get_correlation_logger("agentnet.providers.instrumented")


class InstrumentedProviderMixin:
    """
    Mixin class that adds observability instrumentation to provider adapters.

    Automatically collects metrics, traces, and logs for all provider operations.
    Should be mixed with a concrete provider adapter.
    """

    def __init__(self, *args, **kwargs):
        """Initialize instrumented provider with observability components."""
        super().__init__(*args, **kwargs)
        self.metrics = get_global_metrics()
        self.metrics_collector = MetricsCollector(self.metrics)
        self.tracer = get_global_tracer()
        self.dashboard_collector = get_global_dashboard_collector()

        # Provider identification
        self.provider_name = getattr(self, "name", self.__class__.__name__)
        self.model_name = getattr(self, "model", "default-model")

        logger.info(f"Initialized instrumented provider: {self.provider_name}")

    def _instrument_inference(self, method_name: str, sync: bool = True):
        """Decorator factory for instrumenting inference methods."""

        def decorator(func):
            if sync:

                @wraps(func)
                def wrapper(
                    prompt: str,
                    agent_name: str = "Agent",
                    session_id: Optional[str] = None,
                    **kwargs,
                ):
                    return self._execute_instrumented_inference(
                        func, method_name, prompt, agent_name, session_id, **kwargs
                    )

                return wrapper
            else:

                @wraps(func)
                async def async_wrapper(
                    prompt: str,
                    agent_name: str = "Agent",
                    session_id: Optional[str] = None,
                    **kwargs,
                ):
                    return await self._execute_instrumented_inference_async(
                        func, method_name, prompt, agent_name, session_id, **kwargs
                    )

                return async_wrapper

        return decorator

    def _execute_instrumented_inference(
        self,
        func,
        method_name: str,
        prompt: str,
        agent_name: str,
        session_id: Optional[str],
        **kwargs,
    ):
        """Execute instrumented synchronous inference."""
        start_time = time.time()

        # Set up logging context
        logger.set_correlation_context(
            session_id=session_id,
            agent_name=agent_name,
            operation=f"{self.provider_name}.{method_name}",
        )

        try:
            # Execute with tracing
            with trace_agent_operation(
                agent_name, self.model_name, self.provider_name, session_id
            ) as span:
                # Add span attributes
                span.set_attribute("method", method_name)
                span.set_attribute("prompt_length", len(prompt))

                # Execute the actual inference
                result = func(prompt, agent_name=agent_name, **kwargs)

                # Calculate duration
                duration_seconds = time.time() - start_time
                duration_ms = duration_seconds * 1000

                # Extract token information
                tokens_input = len(prompt.split())  # Simple approximation
                tokens_output = len(result.get("content", "").split())
                total_tokens = tokens_input + tokens_output

                # Add result attributes to span
                span.set_attribute("tokens_input", tokens_input)
                span.set_attribute("tokens_output", tokens_output)
                span.set_attribute("confidence", result.get("confidence", 0.0))
                span.set_attribute("duration_ms", duration_ms)

                # Record metrics
                self.metrics.record_inference_latency(
                    duration_seconds, self.model_name, self.provider_name, agent_name
                )
                self.metrics.record_tokens_consumed(
                    total_tokens, self.model_name, self.provider_name
                )

                # Record dashboard data
                self.dashboard_collector.add_performance_event(
                    agent_name, self.model_name, self.provider_name, duration_ms, True
                )

                # Log operation
                logger.log_agent_inference(
                    self.model_name,
                    self.provider_name,
                    total_tokens,
                    duration_ms,
                    prompt_length=len(prompt),
                    confidence=result.get("confidence", 0.0),
                )

                # Add instrumentation metadata to result
                result["_instrumentation"] = {
                    "duration_ms": duration_ms,
                    "tokens_input": tokens_input,
                    "tokens_output": tokens_output,
                    "provider": self.provider_name,
                    "model": self.model_name,
                    "agent_name": agent_name,
                    "session_id": session_id,
                }

                return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            # Record failure metrics
            self.dashboard_collector.add_performance_event(
                agent_name, self.model_name, self.provider_name, duration_ms, False
            )

            logger.error(
                f"Inference failed: {str(e)}",
                error_type=type(e).__name__,
                duration_ms=duration_ms,
            )
            raise
        finally:
            logger.clear_context()

    async def _execute_instrumented_inference_async(
        self,
        func,
        method_name: str,
        prompt: str,
        agent_name: str,
        session_id: Optional[str],
        **kwargs,
    ):
        """Execute instrumented asynchronous inference."""
        start_time = time.time()

        # Set up logging context
        logger.set_correlation_context(
            session_id=session_id,
            agent_name=agent_name,
            operation=f"{self.provider_name}.{method_name}_async",
        )

        try:
            # Execute with tracing
            with trace_agent_operation(
                agent_name, self.model_name, self.provider_name, session_id
            ) as span:
                # Add span attributes
                span.set_attribute("method", f"{method_name}_async")
                span.set_attribute("prompt_length", len(prompt))

                # Execute the actual inference
                result = await func(prompt, agent_name=agent_name, **kwargs)

                # Calculate duration
                duration_seconds = time.time() - start_time
                duration_ms = duration_seconds * 1000

                # Extract token information
                tokens_input = len(prompt.split())
                tokens_output = len(result.get("content", "").split())
                total_tokens = tokens_input + tokens_output

                # Add result attributes to span
                span.set_attribute("tokens_input", tokens_input)
                span.set_attribute("tokens_output", tokens_output)
                span.set_attribute("confidence", result.get("confidence", 0.0))
                span.set_attribute("duration_ms", duration_ms)

                # Record metrics
                self.metrics.record_inference_latency(
                    duration_seconds, self.model_name, self.provider_name, agent_name
                )
                self.metrics.record_tokens_consumed(
                    total_tokens, self.model_name, self.provider_name
                )

                # Record dashboard data
                self.dashboard_collector.add_performance_event(
                    agent_name, self.model_name, self.provider_name, duration_ms, True
                )

                # Log operation
                logger.log_agent_inference(
                    self.model_name,
                    self.provider_name,
                    total_tokens,
                    duration_ms,
                    prompt_length=len(prompt),
                    confidence=result.get("confidence", 0.0),
                    async_operation=True,
                )

                # Add instrumentation metadata to result
                result["_instrumentation"] = {
                    "duration_ms": duration_ms,
                    "tokens_input": tokens_input,
                    "tokens_output": tokens_output,
                    "provider": self.provider_name,
                    "model": self.model_name,
                    "agent_name": agent_name,
                    "session_id": session_id,
                }

                return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            # Record failure metrics
            self.dashboard_collector.add_performance_event(
                agent_name, self.model_name, self.provider_name, duration_ms, False
            )

            logger.error(
                f"Async inference failed: {str(e)}",
                error_type=type(e).__name__,
                duration_ms=duration_ms,
            )
            raise
        finally:
            logger.clear_context()


class InstrumentedProviderAdapter(InstrumentedProviderMixin, ProviderAdapter):
    """
    Base class for instrumented provider adapters.

    Combines the ProviderAdapter interface with observability instrumentation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize instrumented provider adapter."""
        super().__init__(config)

    def infer(
        self,
        prompt: str,
        agent_name: str = "Agent",
        session_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Instrumented synchronous inference - to be implemented by subclass."""
        raise NotImplementedError("Subclass must implement infer method")

    async def async_infer(
        self,
        prompt: str,
        agent_name: str = "Agent",
        session_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Instrumented asynchronous inference - to be implemented by subclass."""
        raise NotImplementedError("Subclass must implement async_infer method")


def instrument_provider(provider_class):
    """
    Class decorator to automatically instrument a provider adapter.

    Args:
        provider_class: Provider adapter class to instrument

    Returns:
        Instrumented provider class
    """

    class InstrumentedProvider(InstrumentedProviderMixin, provider_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Instrument the infer method if it exists
            if hasattr(self, "infer"):
                original_infer = self.infer
                self.infer = self._instrument_inference("infer", sync=True)(
                    original_infer
                )

            # Instrument the async_infer method if it exists
            if hasattr(self, "async_infer"):
                original_async_infer = self.async_infer
                self.async_infer = self._instrument_inference(
                    "async_infer", sync=False
                )(original_async_infer)

    InstrumentedProvider.__name__ = f"Instrumented{provider_class.__name__}"
    InstrumentedProvider.__qualname__ = f"Instrumented{provider_class.__qualname__}"

    return InstrumentedProvider
