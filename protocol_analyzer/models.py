"""
Core data models for protocol analysis and comparison.
"""

from typing import Dict, List, Optional, Union, Any
from enum import Enum
from pydantic import BaseModel, Field
import numpy as np


class ProtocolType(str, Enum):
    """Protocol categories for classification."""
    IOT_FIRST = "iot_first"
    AGENT_FIRST = "agent_first"
    EDGE_NATIVE_AGENT = "edge_native_agent"


class VerbType(str, Enum):
    """Communication verb types."""
    PING = "ping"
    TELL = "tell"
    ASK = "ask"
    OBSERVE = "observe"
    INFORM = "inform"
    REQUEST = "request"
    SUBSCRIBE = "subscribe"
    QUERY = "query"
    NEGOTIATE = "negotiate"


class QoSLevel(int, Enum):
    """Quality of Service levels."""
    AT_MOST_ONCE = 0
    AT_LEAST_ONCE = 1
    EXACTLY_ONCE = 2


class ContentType(str, Enum):
    """Content type encodings."""
    CBOR = "cbor"
    JSON = "json"
    PROTOBUF = "protobuf"
    TEXT = "text"
    BINARY = "binary"


class Protocol(BaseModel):
    """Represents a communication protocol with its characteristics."""
    
    name: str
    type: ProtocolType
    description: str
    
    # Header characteristics
    header_size_min: int = Field(..., description="Minimum header size in bytes")
    header_size_max: int = Field(..., description="Maximum header size in bytes")
    
    # Supported features
    supports_pubsub: bool = False
    supports_rpc: bool = False
    supports_streaming: bool = False
    supports_qos: bool = False
    supports_discovery: bool = False
    
    # Supported verbs
    supported_verbs: List[VerbType] = []
    
    # Content types
    content_types: List[ContentType] = []
    
    # State complexity
    state_complexity: str = "unknown"  # O(1), O(n), O(n²), etc.
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProtocolMetrics(BaseModel):
    """Quantitative metrics for protocol performance."""
    
    # Efficiency metrics
    header_efficiency: float = Field(..., description="Header efficiency ratio (0-1)")
    message_overhead: float = Field(..., description="Message overhead ratio")
    
    # Performance metrics
    latency_rtt: float = Field(..., description="Latency in RTTs")
    throughput: float = Field(..., description="Throughput in messages/sec")
    energy_per_message: float = Field(..., description="Energy consumption per message (µJ)")
    
    # Scalability metrics
    max_agents: int = Field(..., description="Maximum supported agents")
    memory_per_agent: float = Field(..., description="Memory usage per agent (bytes)")
    
    # Memory breakdown
    memory_breakdown: Dict[str, int] = Field(default_factory=dict, description="Detailed memory breakdown by component")
    
    # Reliability metrics
    delivery_guarantee: str = Field(..., description="Delivery guarantee description")
    fault_tolerance: float = Field(..., description="Fault tolerance score (0-1)")


class AgentInteraction(BaseModel):
    """Represents an agent interaction pattern."""
    
    name: str
    description: str
    required_verbs: List[VerbType]
    complexity: str  # "simple", "moderate", "complex"
    rtt_count: int
    state_requirements: str
    
    # Example scenarios
    examples: List[str] = []


class ProtocolComparison(BaseModel):
    """Results of comparing two protocols."""
    
    protocol_a: str
    protocol_b: str
    
    # Comparative metrics
    efficiency_ratio: float  # A/B ratio
    performance_ratio: float
    scalability_ratio: float
    
    # Feature comparison
    feature_advantages: Dict[str, str]  # feature -> advantage description
    use_case_recommendations: Dict[str, str]  # use case -> recommended protocol
    
    # Detailed analysis
    analysis: str
