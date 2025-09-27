"""
Enhanced Message/Turn Schema Implementation

Implements the complete JSON contract specified in the AgentNet roadmap
with comprehensive validation, serialization, and documentation.
"""

import json
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator, ConfigDict


class MessageType(str, Enum):
    """Types of messages in the AgentNet system."""
    
    TURN = "turn"
    SYSTEM = "system" 
    USER = "user"
    TOOL = "tool"
    ERROR = "error"


class MonitorStatus(str, Enum):
    """Status of monitor execution."""
    
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


class CostProvider(str, Enum):
    """LLM cost providers."""
    
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    LOCAL = "local"
    EXAMPLE = "example"


# Pydantic models for strict validation

class ContextModel(BaseModel):
    """Context information for agent reasoning."""
    
    short_term: List[str] = Field(default_factory=list, description="Short-term memory items")
    semantic_refs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Semantic memory references with scores"
    )
    episodic_refs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Episodic memory references"
    )
    additional_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context data"
    )

    @validator('semantic_refs')
    def validate_semantic_refs(cls, v):
        """Validate semantic reference structure."""
        for ref in v:
            if 'id' not in ref or 'score' not in ref:
                raise ValueError("Semantic refs must have 'id' and 'score' fields")
            if not isinstance(ref['score'], (int, float)) or not 0 <= ref['score'] <= 1:
                raise ValueError("Semantic ref score must be a number between 0 and 1")
        return v


class InputModel(BaseModel):
    """Input data for agent processing."""
    
    prompt: str = Field(..., description="Main prompt for the agent")
    context: ContextModel = Field(default_factory=ContextModel, description="Contextual information")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional input metadata")


class TokensModel(BaseModel):
    """Token usage information."""
    
    input: int = Field(ge=0, description="Input tokens consumed")
    output: int = Field(ge=0, description="Output tokens generated")
    total: int = Field(ge=0, description="Total tokens used")
    
    @validator('total')
    def validate_total(cls, v, values):
        """Ensure total equals input + output."""
        if 'input' in values and 'output' in values:
            expected_total = values['input'] + values['output']
            if v != expected_total:
                return expected_total  # Auto-correct
        return v


class OutputModel(BaseModel):
    """Output data from agent processing."""
    
    content: str = Field(..., description="Generated content")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    style_insights: List[str] = Field(
        default_factory=list,
        description="Insights about applied reasoning style"
    )
    tokens: TokensModel = Field(..., description="Token usage information")
    reasoning_type: Optional[str] = Field(None, description="Type of reasoning applied")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional output metadata")


class MonitorResultModel(BaseModel):
    """Result from a monitor execution."""
    
    name: str = Field(..., description="Monitor name")
    status: MonitorStatus = Field(..., description="Monitor execution status")
    elapsed_ms: float = Field(ge=0, description="Execution time in milliseconds")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional monitor details")
    violations: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Any policy violations detected"
    )


class CostModel(BaseModel):
    """Cost tracking information."""
    
    provider: CostProvider = Field(..., description="LLM provider")
    model: str = Field(..., description="Model name")
    usd: float = Field(ge=0, description="Cost in USD")
    tokens_per_dollar: Optional[float] = Field(None, ge=0, description="Token efficiency")
    estimated: bool = Field(default=False, description="Whether cost is estimated")


class TimingModel(BaseModel):
    """Timing information for the turn."""
    
    started: float = Field(..., description="Start timestamp (Unix time)")
    completed: float = Field(..., description="Completion timestamp (Unix time)")
    latency_ms: float = Field(ge=0, description="Total latency in milliseconds")
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of latency by component"
    )
    
    @validator('latency_ms')
    def validate_latency(cls, v, values):
        """Validate latency matches timing difference."""
        if 'started' in values and 'completed' in values:
            expected_latency = (values['completed'] - values['started']) * 1000
            # Allow some tolerance for rounding
            if abs(v - expected_latency) > 10:  # 10ms tolerance
                return expected_latency
        return v


class TurnMessage(BaseModel):
    """
    Complete turn message schema implementing the AgentNet JSON contract.
    
    This is the primary message format for agent communications,
    implementing the schema defined in docs/RoadmapAgentNet.md.
    """
    
    model_config = ConfigDict(
        extra='allow',  # Allow additional fields for extensibility
        validate_assignment=True,  # Validate on assignment
        use_enum_values=True  # Use enum values in serialization
    )
    
    # Core identification
    task_id: str = Field(..., description="Unique task identifier")
    agent: str = Field(..., description="Agent name/identifier")
    message_type: MessageType = Field(default=MessageType.TURN, description="Message type")
    version: str = Field(default="1.0.0", description="Schema version")
    
    # Input/Output
    input: InputModel = Field(..., description="Input data")
    output: OutputModel = Field(..., description="Output data")
    
    # Monitoring and governance
    monitors: List[MonitorResultModel] = Field(
        default_factory=list,
        description="Monitor execution results"
    )
    
    # Cost and resource tracking
    cost: Optional[CostModel] = Field(None, description="Cost information")
    timing: TimingModel = Field(..., description="Timing information")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('task_id')
    def validate_task_id(cls, v):
        """Ensure task_id is not empty."""
        if not v or not v.strip():
            return str(uuid.uuid4())
        return v
    
    @validator('timing')
    def validate_timing_consistency(cls, v):
        """Validate timing consistency."""
        if v.completed < v.started:
            raise ValueError("Completion time cannot be before start time")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return self.model_dump()
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TurnMessage':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TurnMessage':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def add_monitor_result(self, name: str, status: MonitorStatus, 
                          elapsed_ms: float, **kwargs) -> None:
        """Add a monitor result to the message."""
        monitor_result = MonitorResultModel(
            name=name,
            status=status,
            elapsed_ms=elapsed_ms,
            **kwargs
        )
        self.monitors.append(monitor_result)
    
    def calculate_total_cost(self) -> float:
        """Calculate total cost for this turn."""
        return self.cost.usd if self.cost else 0.0
    
    def get_latency_breakdown(self) -> Dict[str, float]:
        """Get latency breakdown by component."""
        breakdown = self.timing.breakdown.copy()
        breakdown['total'] = self.timing.latency_ms
        return breakdown
    
    def is_successful(self) -> bool:
        """Check if the turn was successful (no failed monitors)."""
        return not any(monitor.status == MonitorStatus.FAIL for monitor in self.monitors)


class MessageSchemaValidator:
    """Utility class for message schema validation and compliance checking."""
    
    @staticmethod
    def validate_message(message: Union[TurnMessage, Dict[str, Any]]) -> bool:
        """Validate a message against the schema."""
        try:
            if isinstance(message, dict):
                TurnMessage.from_dict(message)
            elif not isinstance(message, TurnMessage):
                return False
            return True
        except Exception:
            return False
    
    @staticmethod
    def validate_json_schema(json_str: str) -> bool:
        """Validate JSON string against schema."""
        try:
            TurnMessage.from_json(json_str)
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_schema_compliance_report(message: Union[TurnMessage, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate compliance report for a message."""
        report = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "completeness": 0.0
        }
        
        try:
            if isinstance(message, dict):
                msg = TurnMessage.from_dict(message)
            else:
                msg = message
            
            report["valid"] = True
            
            # Check completeness
            required_fields = ['task_id', 'agent', 'input', 'output', 'timing']
            present_fields = 0
            
            for field in required_fields:
                if hasattr(msg, field) and getattr(msg, field) is not None:
                    present_fields += 1
                else:
                    report["warnings"].append(f"Missing or null required field: {field}")
            
            report["completeness"] = present_fields / len(required_fields)
            
            # Check optional but recommended fields
            recommended_fields = ['monitors', 'cost', 'metadata']
            for field in recommended_fields:
                if not hasattr(msg, field) or getattr(msg, field) is None:
                    report["warnings"].append(f"Missing recommended field: {field}")
            
        except Exception as e:
            report["errors"].append(str(e))
        
        return report


class MessageFactory:
    """Factory for creating standardized messages."""
    
    @staticmethod
    def create_turn_message(
        agent_name: str,
        prompt: str,
        content: str,
        confidence: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        task_id: Optional[str] = None,
        **kwargs
    ) -> TurnMessage:
        """Create a standard turn message."""
        
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        start_time = time.time()
        end_time = start_time + kwargs.get('duration', 0.1)
        
        return TurnMessage(
            task_id=task_id,
            agent=agent_name,
            input=InputModel(prompt=prompt),
            output=OutputModel(
                content=content,
                confidence=confidence,
                tokens=TokensModel(
                    input=input_tokens,
                    output=output_tokens,
                    total=input_tokens + output_tokens
                )
            ),
            timing=TimingModel(
                started=start_time,
                completed=end_time,
                latency_ms=(end_time - start_time) * 1000
            ),
            **kwargs
        )
    
    @staticmethod
    def create_from_agent_result(
        agent_name: str,
        agent_result: Dict[str, Any],
        task_id: Optional[str] = None,
        **kwargs
    ) -> TurnMessage:
        """Create message from AgentNet result."""
        
        result = agent_result.get('result', {})
        content = result.get('content', '')
        confidence = float(result.get('confidence', 0.0))
        
        # Extract token information if available
        tokens_info = result.get('tokens', {})
        input_tokens = tokens_info.get('input', 0)
        output_tokens = tokens_info.get('output', 0)
        
        return MessageFactory.create_turn_message(
            agent_name=agent_name,
            prompt=kwargs.get('prompt', ''),
            content=content,
            confidence=confidence,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            task_id=task_id,
            **kwargs
        )


def create_example_message() -> TurnMessage:
    """Create an example message demonstrating the schema."""
    
    return TurnMessage(
        task_id="example-task-123",
        agent="Athena",
        input=InputModel(
            prompt="Analyze the benefits of renewable energy",
            context=ContextModel(
                short_term=["Previous discussion about climate change"],
                semantic_refs=[{"id": "e1", "score": 0.83}],
                episodic_refs=[]
            )
        ),
        output=OutputModel(
            content="Renewable energy offers numerous benefits including environmental sustainability, energy independence, and long-term cost savings...",
            confidence=0.87,
            style_insights=["Applying rigorous logical validation"],
            tokens=TokensModel(input=324, output=512, total=836)
        ),
        monitors=[
            MonitorResultModel(
                name="keyword_guard",
                status=MonitorStatus.PASS,
                elapsed_ms=2.1
            )
        ],
        cost=CostModel(
            provider=CostProvider.OPENAI,
            model="gpt-4o",
            usd=0.01234
        ),
        timing=TimingModel(
            started=1736981000.123,
            completed=1736981001.001,
            latency_ms=878
        ),
        version="agent:Athena@1.0.0"
    )


# Export main classes
__all__ = [
    'TurnMessage',
    'MessageType', 
    'MonitorStatus',
    'CostProvider',
    'ContextModel',
    'InputModel',
    'OutputModel',
    'TokensModel',
    'MonitorResultModel',
    'CostModel',
    'TimingModel',
    'MessageSchemaValidator',
    'MessageFactory',
    'create_example_message',
]


if __name__ == "__main__":
    # Demonstrate schema functionality
    print("📋 AgentNet Message Schema Validation")
    print("=" * 50)
    
    example = create_example_message()
    json_str = example.to_json(indent=2)
    
    validator = MessageSchemaValidator()
    report = validator.get_schema_compliance_report(example)
    
    print("✅ Schema validation passed")
    print(f"📊 Completeness: {report['completeness']:.1%}")
    print(f"⚠️ Warnings: {len(report['warnings'])}")
    print(f"❌ Errors: {len(report['errors'])}")
    
    print("\n📄 Example JSON (truncated):")
    print(json_str[:500] + "..." if len(json_str) > 500 else json_str)
    
    print(f"\n🎯 Message successful: {example.is_successful()}")
    print(f"💰 Total cost: ${example.calculate_total_cost():.4f}")
    print(f"⏱️ Latency breakdown: {example.get_latency_breakdown()}")