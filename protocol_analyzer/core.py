"""
Core protocol analysis and comparison engine.

This module provides the main analysis functionality for comparing protocols
and making recommendations based on use cases and requirements.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import time
from .models import Protocol, ProtocolMetrics, ProtocolComparison, AgentInteraction
from .protocols import ProtocolDatabase
# Import from µACP library
try:
    from uacp_lib import UACPProtocol, UACPVerb, UACPOptionType, UACPMessage
    UACP_LIB_AVAILABLE = True
except ImportError:
    # Fallback to local implementation
    from .uacp import UACPProtocol, UACPVerb, UACPOptionType
    UACP_LIB_AVAILABLE = False


class ProtocolAnalyzer:
    """Main protocol analysis engine."""
    
    def __init__(self):
        self.protocol_db = ProtocolDatabase()
        self.uacp = UACPProtocol()
        
        # Define common agent interaction patterns
        self.interaction_patterns = self._initialize_interaction_patterns()
    
    def _initialize_interaction_patterns(self) -> Dict[str, AgentInteraction]:
        """Initialize common agent interaction patterns."""
        patterns = {}
        
        # Simple sensor reading
        patterns["sensor_reading"] = AgentInteraction(
            name="Sensor Reading",
            description="Periodic sensor data collection",
            required_verbs=["tell"],
            complexity="simple",
            rtt_count=1,
            state_requirements="O(1)",
            examples=[
                "Temperature sensor publishing readings every minute",
                "IoT device sending status updates"
            ]
        )
        
        # Request-response query
        patterns["request_response"] = AgentInteraction(
            name="Request-Response Query",
            description="Synchronous request with immediate response",
            required_verbs=["ask"],
            complexity="simple",
            rtt_count=1,
            state_requirements="O(1)",
            examples=[
                "Agent asking for current weather data",
                "Service requesting user authentication"
            ]
        )
        
        # Subscription to updates
        patterns["subscription"] = AgentInteraction(
            name="Subscription to Updates",
            description="Subscribe to future information updates",
            required_verbs=["observe"],
            complexity="moderate",
            rtt_count=1,
            state_requirements="O(subscriptions)",
            examples=[
                "Agent subscribing to stock price changes",
                "Monitoring system for alert notifications"
            ]
        )
        
        # Multi-turn conversation
        patterns["conversation"] = AgentInteraction(
            name="Multi-turn Conversation",
            description="Extended dialogue with context",
            required_verbs=["ask", "tell", "observe"],
            complexity="complex",
            rtt_count=3,
            state_requirements="O(conversation_history)",
            examples=[
                "AI agent helping user plan a trip",
                "Negotiation between autonomous vehicles"
            ]
        )
        
        # Coordination and planning
        patterns["coordination"] = AgentInteraction(
            name="Coordination and Planning",
            description="Multi-agent task coordination",
            required_verbs=["ask", "tell", "observe"],
            complexity="complex",
            rtt_count=5,
            state_requirements="O(agent_count * tasks)",
            examples=[
                "Robot swarm coordinating movement",
                "Smart grid load balancing"
            ]
        )
        
        return patterns
    
    def analyze_protocol_efficiency(self, protocol: Protocol, payload_sizes: List[int]) -> Dict[str, Any]:
        """Analyze protocol efficiency across different payload sizes."""
        results = {
            "protocol": protocol.name,
            "header_efficiency": [],
            "message_overhead": [],
            "payload_sizes": payload_sizes
        }
        
        for payload_size in payload_sizes:
            if protocol.name == "µACP":
                # Use µACP-specific calculation
                efficiency = self.uacp.calculate_header_efficiency(payload_size)
            else:
                # Generic calculation
                header_size = protocol.header_size_min
                efficiency = header_size / (header_size + payload_size)
            
            overhead = 1 - efficiency
            
            results["header_efficiency"].append(efficiency)
            results["message_overhead"].append(overhead)
        
        return results
    
    def calculate_protocol_metrics(self, protocol: Protocol, 
                                 rtt: float = 0.001,
                                 serialization_time: float = 0.0001,
                                 loss_prob: float = 0.01) -> ProtocolMetrics:
        """Calculate comprehensive metrics for a protocol."""
        
        # Header efficiency (for 16-byte payload as baseline)
        payload_size = 16
        if protocol.name == "µACP":
            header_efficiency = self.uacp.calculate_header_efficiency(payload_size)
        else:
            header_efficiency = protocol.header_size_min / (protocol.header_size_min + payload_size)
        
        # Message overhead
        message_overhead = 1 - header_efficiency
        
        # Latency analysis
        if protocol.supports_qos:
            latency_rtt = 1.0  # QoS can reduce effective RTT
        else:
            latency_rtt = 1.0
        
        # Throughput estimation (messages per second)
        # Simplified model: throughput = 1 / (RTT + processing_time)
        processing_time = serialization_time + (protocol.header_size_min * 0.000001)  # Rough estimate
        throughput = 1 / (rtt + processing_time)
        
        # Energy per message (µJ) - rough estimate
        # Energy ∝ message_size * transmission_time
        message_size = protocol.header_size_min + payload_size
        energy_per_message = message_size * 0.1  # 0.1 µJ per byte
        
        # Scalability estimates (realistic with real-world factors)
        memory_per_agent, memory_breakdown = self._calculate_realistic_memory(protocol)
        
        if protocol.state_complexity == "O(1)":
            max_agents = 1000000
        elif protocol.state_complexity == "O(W)":
            max_agents = 100000
        elif protocol.state_complexity == "O(topics)":
            max_agents = 10000
        else:  # App-level
            max_agents = 1000
        
        # Reliability metrics
        if protocol.supports_qos:
            delivery_guarantee = "QoS-based delivery guarantees"
            fault_tolerance = 0.9
        else:
            delivery_guarantee = "Best-effort delivery"
            fault_tolerance = 0.7
        
        return ProtocolMetrics(
            header_efficiency=header_efficiency,
            message_overhead=message_overhead,
            latency_rtt=latency_rtt,
            throughput=throughput,
            energy_per_message=energy_per_message,
            max_agents=max_agents,
            memory_per_agent=memory_per_agent,
            memory_breakdown=memory_breakdown,
            delivery_guarantee=delivery_guarantee,
            fault_tolerance=fault_tolerance
        )
    
    def _calculate_realistic_memory(self, protocol: Protocol) -> int:
        """Calculate realistic memory per agent including all real-world factors."""
        
        # Base protocol memory
        if protocol.state_complexity == "O(1)":
            base_memory = 1024  # 1 KB
        elif protocol.state_complexity == "O(W)":
            base_memory = 8192  # 8 KB
        elif protocol.state_complexity == "O(topics)":
            base_memory = 32768  # 32 KB
        else:  # App-level
            base_memory = 262144  # 256 KB
        
        # Add message buffers
        message_buffers = self._estimate_message_buffers(protocol)
        
        # Add connection overhead
        connection_overhead = self._estimate_connection_overhead(protocol)
        
        # Add security context
        security_overhead = self._estimate_security_overhead(protocol)
        
        # Add application state
        app_state = self._estimate_application_state(protocol)
        
        # Add OS overhead
        os_overhead = self._estimate_os_overhead(protocol)
        
        # Add routing & addressing state
        routing_state = self._estimate_routing_state(protocol)
        
        # Add subscription & dialogue state
        subscription_state = self._estimate_subscription_state(protocol)
        
        # Add reliability & QoS state
        reliability_state = self._estimate_reliability_state(protocol)
        
        # Add timers & scheduling state
        timer_state = self._estimate_timer_state(protocol)
        
        # Add broker/middleware state
        broker_state = self._estimate_broker_state(protocol)
        
        # Add instrumentation & control state
        instrumentation_state = self._estimate_instrumentation_state(protocol)
        
        # Add resource binding state
        resource_state = self._estimate_resource_binding_state(protocol)
        
        total_memory = (base_memory + message_buffers + connection_overhead + 
                       security_overhead + app_state + os_overhead + routing_state +
                       subscription_state + reliability_state + timer_state + 
                       broker_state + instrumentation_state + resource_state)
        
        # Create detailed memory breakdown
        memory_breakdown = {
            "base_protocol": base_memory,
            "message_buffers": message_buffers,
            "connection_overhead": connection_overhead,
            "security_context": security_overhead,
            "application_state": app_state,
            "os_overhead": os_overhead,
            "routing_addressing": routing_state,
            "subscription_dialogue": subscription_state,
            "reliability_qos": reliability_state,
            "timers_scheduling": timer_state,
            "broker_middleware": broker_state,
            "instrumentation_control": instrumentation_state,
            "resource_binding": resource_state,
            "total": total_memory
        }
        
        return total_memory, memory_breakdown
    
    def test_uacp_protocol(self) -> Dict[str, Any]:
        """Test µACP protocol using the actual library implementation."""
        if not UACP_LIB_AVAILABLE:
            return {"error": "µACP library not available"}
        
        try:
            results = {}
            
            # Test message creation
            start_time = time.time()
            ping_msg = UACPProtocol.create_ping(1, 0)
            ask_msg = UACPProtocol.create_ask(2, "test/topic", {"data": "test"}, 1)
            tell_msg = UACPProtocol.create_tell(3, "test/topic", {"data": "test"}, 0)
            observe_msg = UACPProtocol.create_observe(4, "test/topic", 1)
            
            creation_time = time.time() - start_time
            results["message_creation"] = {
                "time": creation_time,
                "messages_created": 4,
                "throughput": 4 / creation_time if creation_time > 0 else 0
            }
            
            # Test message packing/unpacking
            start_time = time.time()
            ping_data = ping_msg.pack()
            ask_data = ask_msg.pack()
            tell_data = tell_msg.pack()
            observe_data = observe_msg.pack()
            
            # Unpack messages
            ping_unpacked = UACPMessage.unpack(ping_data)
            ask_unpacked = UACPMessage.unpack(ask_data)
            tell_unpacked = UACPMessage.unpack(tell_data)
            observe_unpacked = UACPMessage.unpack(observe_data)
            
            pack_unpack_time = time.time() - start_time
            results["pack_unpack"] = {
                "time": pack_unpack_time,
                "operations": 8,
                "throughput": 8 / pack_unpack_time if pack_unpack_time > 0 else 0
            }
            
            # Test message validation
            start_time = time.time()
            ping_valid = UACPProtocol.validate_message(ping_msg)
            ask_valid = UACPProtocol.validate_message(ask_msg)
            tell_valid = UACPProtocol.validate_message(tell_msg)
            observe_valid = UACPProtocol.validate_message(observe_msg)
            
            validation_time = time.time() - start_time
            results["validation"] = {
                "time": validation_time,
                "messages_validated": 4,
                "all_valid": all([ping_valid, ask_valid, tell_valid, observe_valid]),
                "throughput": 4 / validation_time if validation_time > 0 else 0
            }
            
            # Test message sizes
            results["message_sizes"] = {
                "ping": len(ping_data),
                "ask": len(ask_data),
                "tell": len(tell_data),
                "observe": len(observe_data),
                "average": (len(ping_data) + len(ask_data) + len(tell_data) + len(observe_data)) / 4
            }
            
            # Test header efficiency
            header_size = 8  # Fixed header size
            results["efficiency"] = {
                "header_size": header_size,
                "ping_efficiency": header_size / len(ping_data) if len(ping_data) > 0 else 0,
                "ask_efficiency": header_size / len(ask_data) if len(ask_data) > 0 else 0,
                "tell_efficiency": header_size / len(tell_data) if len(tell_data) > 0 else 0,
                "observe_efficiency": header_size / len(observe_data) if len(observe_data) > 0 else 0
            }
            
            # Test option handling
            results["options"] = {
                "ping_options": len(ping_msg.options),
                "ask_options": len(ask_msg.options),
                "tell_options": len(tell_msg.options),
                "observe_options": len(observe_msg.options)
            }
            
            return results
            
        except Exception as e:
            return {"error": f"Protocol test failed: {str(e)}"}
    
    def _estimate_message_buffers(self, protocol: Protocol) -> int:
        """Estimate memory for message buffers."""
        # Send/receive buffers + retransmission buffers
        if protocol.supports_qos:
            # QoS requires retransmission buffers
            buffer_size = 4096  # 4 KB for reliability
        else:
            buffer_size = 2048  # 2 KB basic
        
        # Add flow control buffers
        flow_control = 2048  # 2 KB
        
        return buffer_size + flow_control
    
    def _estimate_connection_overhead(self, protocol: Protocol) -> int:
        """Estimate memory for connection management."""
        if protocol.name == "MQTT":
            # TCP connection + keepalive
            return 4096  # 4 KB
        elif protocol.name == "CoAP":
            # UDP socket + DTLS context
            return 3072  # 3 KB
        elif protocol.name == "µACP":
            # UDP socket + minimal connection tracking
            return 2048  # 2 KB
        elif protocol.name in ["MCP", "FIPA-ACL"]:
            # HTTP/WebSocket + connection pools
            return 8192  # 8 KB
        else:
            return 4096  # 4 KB default
    
    def _estimate_security_overhead(self, protocol: Protocol) -> int:
        """Estimate memory for security contexts."""
        # Authentication + encryption + certificates
        if protocol.name in ["MQTT", "CoAP"]:
            # IoT protocols with DTLS/TLS
            return 4096  # 4 KB
        elif protocol.name == "µACP":
            # Lightweight security (COSE tokens)
            return 2048  # 2 KB
        elif protocol.name in ["MCP", "FIPA-ACL"]:
            # Rich security (JWT, OAuth, certificates)
            return 8192  # 8 KB
        else:
            return 4096  # 4 KB default
    
    def _estimate_application_state(self, protocol: Protocol) -> int:
        """Estimate memory for application state."""
        # User sessions + business logic + caches
        if protocol.type.value == "iot_first":
            # IoT devices have minimal app state
            return 2048  # 2 KB
        elif protocol.type.value == "edge_native_agent":
            # Edge agents have moderate app state
            return 8192  # 8 KB
        elif protocol.type.value == "agent_first":
            # Rich agents have extensive app state
            return 32768  # 32 KB
        else:
            return 8192  # 8 KB default
    
    def _estimate_os_overhead(self, protocol: Protocol) -> int:
        """Estimate memory for operating system overhead."""
        # Process stacks + system calls + memory management
        if protocol.name in ["MQTT", "CoAP"]:
            # IoT protocols on embedded systems
            return 2048  # 2 KB
        elif protocol.name == "µACP":
            # Edge protocols on lightweight OS
            return 4096  # 4 KB
        elif protocol.name in ["MCP", "FIPA-ACL"]:
            # Rich protocols on full OS
            return 8192  # 8 KB
        else:
            return 4096  # 4 KB default
    
    def compare_protocols_comprehensive(self, protocol_a: str, protocol_b: str) -> ProtocolComparison:
        """Perform comprehensive comparison between two protocols."""
        
        # Get basic comparison
        basic_comp = self.protocol_db.compare_protocols(protocol_a, protocol_b)
        
        # Get protocols
        a = self.protocol_db.get_protocol(protocol_a)
        b = self.protocol_db.get_protocol(protocol_b)
        
        # Calculate metrics
        metrics_a = self.calculate_protocol_metrics(a)
        metrics_b = self.calculate_protocol_metrics(b)
        
        # Calculate ratios
        efficiency_ratio = metrics_b.header_efficiency / metrics_a.header_efficiency if metrics_a.header_efficiency > 0 else float('inf')
        performance_ratio = metrics_b.throughput / metrics_a.throughput if metrics_a.throughput > 0 else float('inf')
        scalability_ratio = metrics_b.max_agents / metrics_a.max_agents if metrics_a.max_agents > 0 else float('inf')
        
        # Feature advantages
        feature_advantages = {}
        for feature in ["pubsub", "rpc", "streaming", "qos", "discovery"]:
            a_supports = getattr(a, f"supports_{feature}")
            b_supports = getattr(b, f"supports_{feature}")
            
            if a_supports and not b_supports:
                feature_advantages[feature] = f"{protocol_a} supports {feature}, {protocol_b} does not"
            elif b_supports and not a_supports:
                feature_advantages[feature] = f"{protocol_b} supports {feature}, {protocol_a} does not"
        
        # Use case recommendations
        use_case_recommendations = self._generate_use_case_recommendations(a, b, metrics_a, metrics_b)
        
        # Detailed analysis
        analysis = self._generate_comparison_analysis(a, b, metrics_a, metrics_b, basic_comp)
        
        return ProtocolComparison(
            protocol_a=protocol_a,
            protocol_b=protocol_b,
            efficiency_ratio=efficiency_ratio,
            performance_ratio=performance_ratio,
            scalability_ratio=scalability_ratio,
            feature_advantages=feature_advantages,
            use_case_recommendations=use_case_recommendations,
            analysis=analysis
        )
    
    def _generate_use_case_recommendations(self, a: Protocol, b: Protocol, 
                                         metrics_a: ProtocolMetrics, 
                                         metrics_b: ProtocolMetrics) -> Dict[str, str]:
        """Generate use case recommendations."""
        recommendations = {}
        
        # Resource-constrained environments
        if metrics_a.energy_per_message < metrics_b.energy_per_message:
            recommendations["resource_constrained"] = f"{a.name} for lower energy consumption"
        else:
            recommendations["resource_constrained"] = f"{b.name} for lower energy consumption"
        
        # High-throughput scenarios
        if metrics_a.throughput > metrics_b.throughput:
            recommendations["high_throughput"] = f"{a.name} for higher message throughput"
        else:
            recommendations["high_throughput"] = f"{b.name} for higher message throughput"
        
        # Large-scale deployments
        if metrics_a.max_agents > metrics_b.max_agents:
            recommendations["large_scale"] = f"{a.name} for better scalability"
        else:
            recommendations["large_scale"] = f"{b.name} for better scalability"
        
        # Reliability requirements
        if metrics_a.fault_tolerance > metrics_b.fault_tolerance:
            recommendations["high_reliability"] = f"{a.name} for better fault tolerance"
        else:
            recommendations["high_reliability"] = f"{b.name} for better fault tolerance"
        
        return recommendations
    
    def _generate_comparison_analysis(self, a: Protocol, b: Protocol,
                                    metrics_a: ProtocolMetrics,
                                    metrics_b: ProtocolMetrics,
                                    basic_comp: Dict[str, Any]) -> str:
        """Generate detailed analysis text."""
        
        analysis = f"Comprehensive comparison between {a.name} and {b.name}:\n\n"
        
        # Header efficiency analysis
        if metrics_a.header_efficiency > metrics_b.header_efficiency:
            analysis += f"• {a.name} has better header efficiency ({metrics_a.header_efficiency:.3f} vs {metrics_b.header_efficiency:.3f})\n"
        else:
            analysis += f"• {b.name} has better header efficiency ({metrics_b.header_efficiency:.3f} vs {metrics_a.header_efficiency:.3f})\n"
        
        # Performance analysis
        if metrics_a.throughput > metrics_b.throughput:
            analysis += f"• {a.name} achieves higher throughput ({metrics_a.throughput:.0f} vs {metrics_b.throughput:.0f} msg/s)\n"
        else:
            analysis += f"• {b.name} achieves higher throughput ({metrics_b.throughput:.0f} vs {metrics_a.throughput:.0f} msg/s)\n"
        
        # Scalability analysis
        if metrics_a.max_agents > metrics_b.max_agents:
            analysis += f"• {a.name} scales to more agents ({metrics_a.max_agents:,} vs {metrics_b.max_agents:,})\n"
        else:
            analysis += f"• {b.name} scales to more agents ({metrics_b.max_agents:,} vs {metrics_a.max_agents:,})\n"
        
        # Feature analysis
        analysis += f"\nFeature comparison:\n"
        analysis += f"• Pub/Sub: {a.name} {'✓' if a.supports_pubsub else '✗'} vs {b.name} {'✓' if b.supports_pubsub else '✗'}\n"
        analysis += f"• RPC: {a.name} {'✓' if a.supports_rpc else '✗'} vs {b.name} {'✓' if b.supports_rpc else '✗'}\n"
        analysis += f"• Streaming: {a.name} {'✓' if a.supports_streaming else '✗'} vs {b.name} {'✓' if b.supports_streaming else '✗'}\n"
        analysis += f"• QoS: {a.name} {'✓' if a.supports_qos else '✗'} vs {b.name} {'✓' if b.supports_qos else '✗'}\n"
        analysis += f"• Discovery: {a.name} {'✓' if a.supports_discovery else '✗'} vs {b.name} {'✓' if b.supports_discovery else '✗'}\n"
        
        # Protocol type positioning
        analysis += f"\nProtocol positioning:\n"
        analysis += f"• {a.name}: {a.type.value.replace('_', ' ').title()}\n"
        analysis += f"• {b.name}: {b.type.value.replace('_', ' ').title()}\n"
        
        return analysis
    
    def _estimate_routing_state(self, protocol: Protocol) -> int:
        """Estimate memory for routing & addressing state."""
        # Neighbor tables, NAT traversal, multicast, forwarding caches
        if protocol.name == "µACP":
            # Edge agents need neighbor discovery and routing
            return 4096  # 4 KB
        elif protocol.name in ["MQTT", "CoAP"]:
            # IoT protocols have minimal routing (usually single-hop)
            return 1024  # 1 KB
        elif protocol.name in ["MCP", "FIPA-ACL"]:
            # Rich agents may have complex routing and addressing
            return 8192  # 8 KB
        else:
            return 2048  # 2 KB default
    
    def _estimate_subscription_state(self, protocol: Protocol) -> int:
        """Estimate memory for subscription & dialogue state."""
        # OBSERVE tables, conversation state, correlation IDs, contracts
        if protocol.supports_pubsub:
            if protocol.name == "MQTT":
                # MQTT has extensive topic subscription management
                return 8192  # 8 KB
            elif protocol.name == "µACP":
                # µACP has OBSERVE and conversation state
                return 4096  # 4 KB
            elif protocol.name in ["MCP", "FIPA-ACL"]:
                # Rich agents have complex dialogue management
                return 16384  # 16 KB
            else:
                return 4096  # 4 KB default
        else:
            # No pub/sub support
            return 1024  # 1 KB minimal
    
    def _estimate_reliability_state(self, protocol: Protocol) -> int:
        """Estimate memory for reliability & QoS state."""
        # Duplicate suppression, ACK timers, reassembly, sliding windows
        if protocol.supports_qos:
            if protocol.name == "MQTT":
                # MQTT QoS 1&2 require extensive reliability state
                return 6144  # 6 KB
            elif protocol.name == "µACP":
                # µACP has QoS levels and reliability windows
                return 4096  # 4 KB
            elif protocol.name == "CoAP":
                # CoAP has blockwise transfer and reliability
                return 3072  # 3 KB
            else:
                return 4096  # 4 KB default
        else:
            # No QoS support
            return 1024  # 1 KB minimal
    
    def _estimate_timer_state(self, protocol: Protocol) -> int:
        """Estimate memory for timers & scheduling state."""
        # Retransmission, heartbeat, session, priority queues
        if protocol.name == "µACP":
            # Edge agents need various timers for reliability
            return 3072  # 3 KB
        elif protocol.name == "MQTT":
            # MQTT has keepalive and session timers
            return 2048  # 2 KB
        elif protocol.name in ["MCP", "FIPA-ACL"]:
            # Rich agents have complex scheduling
            return 4096  # 4 KB
        else:
            return 2048  # 2 KB default
    
    def _estimate_broker_state(self, protocol: Protocol) -> int:
        """Estimate memory for broker/middleware state."""
        # Topic trees, retained messages, flow control, load balancing
        if protocol.name == "MQTT":
            # MQTT brokers have extensive topic management
            return 8192  # 8 KB
        elif protocol.name == "µACP":
            # µACP may have lightweight brokers
            return 2048  # 2 KB
        elif protocol.name in ["MCP", "FIPA-ACL"]:
            # Rich protocols may have middleware
            return 4096  # 4 KB
        else:
            return 1024  # 1 KB default
    
    def _estimate_instrumentation_state(self, protocol: Protocol) -> int:
        """Estimate memory for instrumentation & control state."""
        # Logging, metrics, debug, policy enforcement
        if protocol.type.value == "iot_first":
            # IoT devices have minimal instrumentation
            return 1024  # 1 KB
        elif protocol.type.value == "edge_native_agent":
            # Edge agents need moderate instrumentation
            return 2048  # 2 KB
        elif protocol.type.value == "agent_first":
            # Rich agents have extensive instrumentation
            return 4096  # 4 KB
        else:
            return 2048  # 2 KB default
    
    def _estimate_resource_binding_state(self, protocol: Protocol) -> int:
        """Estimate memory for resource binding state."""
        # File descriptors, DMA buffers, crypto contexts, storage handles
        if protocol.name in ["MQTT", "CoAP"]:
            # IoT protocols on embedded systems
            return 2048  # 2 KB
        elif protocol.name == "µACP":
            # Edge protocols on lightweight systems
            return 3072  # 3 KB
        elif protocol.name in ["MCP", "FIPA-ACL"]:
            # Rich protocols on full systems
            return 6144  # 6 KB
        else:
            return 3072  # 3 KB default
    
    def recommend_protocol_for_use_case(self, 
                                       interaction_pattern: str,
                                       constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend the best protocol for a specific use case."""
        
        if interaction_pattern not in self.interaction_patterns:
            raise ValueError(f"Unknown interaction pattern: {interaction_pattern}")
        
        pattern = self.interaction_patterns[interaction_pattern]
        required_verbs = set(pattern.required_verbs)
        
        # Score each protocol
        protocol_scores = {}
        
        for protocol_name in self.protocol_db.get_protocol_names():
            protocol = self.protocol_db.get_protocol(protocol_name)
            metrics = self.calculate_protocol_metrics(protocol)
            
            score = 0
            
            # Verb support scoring
            supported_verbs = set(protocol.supported_verbs)
            verb_score = len(required_verbs & supported_verbs) / len(required_verbs)
            score += verb_score * 0.4  # 40% weight
            
            # Efficiency scoring
            efficiency_score = 1 - metrics.message_overhead
            score += efficiency_score * 0.3  # 30% weight
            
            # Performance scoring
            performance_score = min(metrics.throughput / 1000, 1.0)  # Normalize to 1000 msg/s
            score += performance_score * 0.2  # 20% weight
            
            # Scalability scoring
            scalability_score = min(metrics.max_agents / 10000, 1.0)  # Normalize to 10k agents
            score += scalability_score * 0.1  # 10% weight
            
            protocol_scores[protocol_name] = {
                "score": score,
                "protocol": protocol,
                "metrics": metrics,
                "verb_support": verb_score,
                "efficiency": efficiency_score,
                "performance": performance_score,
                "scalability": scalability_score
            }
        
        # Sort by score
        sorted_protocols = sorted(protocol_scores.items(), 
                                key=lambda x: x[1]["score"], 
                                reverse=True)
        
        # Generate recommendations
        recommendations = {
            "interaction_pattern": interaction_pattern,
            "constraints": constraints,
            "protocols": sorted_protocols,
            "top_recommendation": sorted_protocols[0] if sorted_protocols else None,
            "analysis": self._generate_recommendation_analysis(pattern, sorted_protocols)
        }
        
        return recommendations
    
    def _generate_recommendation_analysis(self, pattern: AgentInteraction, 
                                        sorted_protocols: List[Tuple[str, Dict]]) -> str:
        """Generate analysis for protocol recommendations."""
        
        analysis = f"Protocol recommendations for '{pattern.name}' interaction pattern:\n\n"
        analysis += f"Pattern complexity: {pattern.complexity}\n"
        analysis += f"Required RTTs: {pattern.rtt_count}\n"
        analysis += f"State requirements: {pattern.state_requirements}\n\n"
        
        if sorted_protocols:
            top_protocol = sorted_protocols[0]
            analysis += f"Top recommendation: {top_protocol[0]} (score: {top_protocol[1]['score']:.3f})\n"
            analysis += f"• Verb support: {top_protocol[1]['verb_support']:.1%}\n"
            analysis += f"• Efficiency: {top_protocol[1]['efficiency']:.1%}\n"
            analysis += f"• Performance: {top_protocol[1]['performance']:.1%}\n"
            analysis += f"• Scalability: {top_protocol[1]['scalability']:.1%}\n\n"
            
            analysis += "Alternative options:\n"
            for i, (name, data) in enumerate(sorted_protocols[1:4], 1):  # Show top 4
                analysis += f"{i}. {name}: score {data['score']:.3f}\n"
        
        return analysis
