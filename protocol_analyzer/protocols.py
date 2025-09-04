"""
Protocol database for comparison and analysis.

This module contains definitions of existing protocols (MQTT, CoAP, MCP, FIPA-ACL)
for comparison with µACP.
"""

from typing import Dict, List
from .models import Protocol, ProtocolType, VerbType, ContentType


class ProtocolDatabase:
    """Database of existing protocols for comparison."""
    
    def __init__(self):
        self.protocols = self._initialize_protocols()
    
    def _initialize_protocols(self) -> Dict[str, Protocol]:
        """Initialize the protocol database."""
        protocols = {}
        
        # µACP - Edge-native agent protocol
        protocols["µACP"] = Protocol(
            name="µACP",
            type=ProtocolType.EDGE_NATIVE_AGENT,
            description="Micro Agent Communication Protocol - lightweight agent communication for edge-native systems",
            header_size_min=8,
            header_size_max=8,
            supports_pubsub=True,
            supports_rpc=True,
            supports_streaming=True,
            supports_qos=True,
            supports_discovery=True,
            supported_verbs=[
                VerbType.PING, VerbType.TELL, VerbType.ASK, VerbType.OBSERVE
            ],
            content_types=[ContentType.CBOR, ContentType.JSON, ContentType.PROTOBUF, ContentType.TEXT],
            state_complexity="O(W)",
            metadata={
                "uacp_version": "1.0",
                "transport": "UDP/TCP",
                "use_cases": ["Edge computing", "Multi-agent systems", "Resource-constrained agents"]
            }
        )
        
        # MQTT - IoT-first protocol
        protocols["MQTT"] = Protocol(
            name="MQTT",
            type=ProtocolType.IOT_FIRST,
            description="Message Queuing Telemetry Transport - lightweight pub/sub for IoT",
            header_size_min=2,
            header_size_max=4,
            supports_pubsub=True,
            supports_rpc=False,
            supports_streaming=False,
            supports_qos=True,
            supports_discovery=False,
            supported_verbs=[
                VerbType.TELL,  # PUBLISH
            ],
            content_types=[ContentType.BINARY],
            state_complexity="O(topics)",
            metadata={
                "mqtt_version": "5.0",
                "transport": "TCP/TLS",
                "use_cases": ["IoT sensors", "Home automation", "Industrial monitoring"]
            }
        )
        
        # CoAP - IoT-first protocol
        protocols["CoAP"] = Protocol(
            name="CoAP",
            type=ProtocolType.IOT_FIRST,
            description="Constrained Application Protocol - RESTful protocol for constrained devices",
            header_size_min=4,
            header_size_max=8,
            supports_pubsub=False,
            supports_rpc=True,
            supports_streaming=False,
            supports_qos=False,
            supports_discovery=True,
            supported_verbs=[
                VerbType.ASK,  # GET, POST, PUT, DELETE
            ],
            content_types=[ContentType.CBOR, ContentType.JSON, ContentType.TEXT],
            state_complexity="O(W)",
            metadata={
                "coap_version": "RFC 7252",
                "transport": "UDP/DTLS",
                "use_cases": ["Smart cities", "Wearables", "Sensor networks"]
            }
        )
        
        # MCP - Agent-first protocol
        protocols["MCP"] = Protocol(
            name="MCP",
            type=ProtocolType.AGENT_FIRST,
            description="Model Context Protocol - agent communication with rich context",
            header_size_min=10,
            header_size_max=40,
            supports_pubsub=True,
            supports_rpc=True,
            supports_streaming=True,
            supports_qos=False,
            supports_discovery=True,
            supported_verbs=[
                VerbType.TELL, VerbType.ASK, VerbType.OBSERVE,
                VerbType.INFORM, VerbType.REQUEST, VerbType.SUBSCRIBE
            ],
            content_types=[ContentType.JSON],
            state_complexity="App-level",
            metadata={
                "mcp_version": "1.0",
                "transport": "WebSocket/HTTP",
                "use_cases": ["AI agents", "Chatbots", "Knowledge systems"]
            }
        )
        
        # FIPA-ACL - Agent-first protocol
        protocols["FIPA-ACL"] = Protocol(
            name="FIPA-ACL",
            type=ProtocolType.AGENT_FIRST,
            description="Foundation for Intelligent Physical Agents - Agent Communication Language",
            header_size_min=200,
            header_size_max=800,
            supports_pubsub=True,
            supports_rpc=True,
            supports_streaming=True,
            supports_qos=False,
            supports_discovery=True,
            supported_verbs=[
                VerbType.TELL, VerbType.ASK, VerbType.OBSERVE,
                VerbType.INFORM, VerbType.REQUEST, VerbType.SUBSCRIBE,
                VerbType.QUERY, VerbType.NEGOTIATE
            ],
            content_types=[ContentType.JSON, ContentType.TEXT],
            state_complexity="App-level",
            metadata={
                "fipa_version": "2000",
                "transport": "HTTP/TCP",
                "use_cases": ["Multi-agent systems", "Distributed AI", "Autonomous systems"]
            }
        )
        
        # HTTP/2 - General purpose
        protocols["HTTP/2"] = Protocol(
            name="HTTP/2",
            type=ProtocolType.AGENT_FIRST,
            description="Hypertext Transfer Protocol version 2 - general purpose web protocol",
            header_size_min=10,
            header_size_max=50,
            supports_pubsub=False,
            supports_rpc=True,
            supports_streaming=True,
            supports_qos=False,
            supports_discovery=False,
            supported_verbs=[
                VerbType.ASK,  # GET, POST, PUT, DELETE
            ],
            content_types=[ContentType.JSON, ContentType.TEXT, ContentType.BINARY],
            state_complexity="O(streams)",
            metadata={
                "http_version": "2.0",
                "transport": "TCP/TLS",
                "use_cases": ["Web APIs", "Microservices", "General communication"]
            }
        )
        
        # gRPC - RPC-focused
        protocols["gRPC"] = Protocol(
            name="gRPC",
            type=ProtocolType.AGENT_FIRST,
            description="Google Remote Procedure Call - high-performance RPC framework",
            header_size_min=15,
            header_size_max=30,
            supports_pubsub=False,
            supports_rpc=True,
            supports_streaming=True,
            supports_qos=False,
            supports_discovery=False,
            supported_verbs=[
                VerbType.ASK,  # Unary, Server streaming, Client streaming, Bidirectional
            ],
            content_types=[ContentType.PROTOBUF],
            state_complexity="O(streams)",
            metadata={
                "grpc_version": "1.0",
                "transport": "HTTP/2",
                "use_cases": ["Microservices", "High-performance APIs", "Real-time systems"]
            }
        )
        
        return protocols
    
    def get_protocol(self, name: str) -> Protocol:
        """Get a protocol by name."""
        if name not in self.protocols:
            raise ValueError(f"Protocol '{name}' not found")
        return self.protocols[name]
    
    def get_all_protocols(self) -> List[Protocol]:
        """Get all protocols."""
        return list(self.protocols.values())
    
    def get_protocols_by_type(self, protocol_type: ProtocolType) -> List[Protocol]:
        """Get protocols by type."""
        return [p for p in self.protocols.values() if p.type == protocol_type]
    
    def get_protocol_names(self) -> List[str]:
        """Get all protocol names."""
        return list(self.protocols.keys())
    
    def compare_protocols(self, protocol_a: str, protocol_b: str) -> Dict[str, any]:
        """Compare two protocols."""
        if protocol_a not in self.protocols or protocol_b not in self.protocols:
            raise ValueError("One or both protocols not found")
        
        a = self.protocols[protocol_a]
        b = self.protocols[protocol_b]
        
        comparison = {
            "protocol_a": protocol_a,
            "protocol_b": protocol_b,
            "header_efficiency": {
                "a": a.header_size_min,
                "b": b.header_size_min,
                "ratio": b.header_size_min / a.header_size_min if a.header_size_min > 0 else float('inf')
            },
            "features": {
                "pubsub": {"a": a.supports_pubsub, "b": b.supports_pubsub},
                "rpc": {"a": a.supports_rpc, "b": b.supports_rpc},
                "streaming": {"a": a.supports_streaming, "b": b.supports_streaming},
                "qos": {"a": a.supports_qos, "b": b.supports_qos},
                "discovery": {"a": a.supports_discovery, "b": b.supports_discovery}
            },
            "verb_support": {
                "a": len(a.supported_verbs),
                "b": len(b.supported_verbs),
                "common": len(set(a.supported_verbs) & set(b.supported_verbs))
            },
            "content_types": {
                "a": len(a.content_types),
                "b": len(b.content_types),
                "common": len(set(a.content_types) & set(b.content_types))
            },
            "state_complexity": {
                "a": a.state_complexity,
                "b": b.state_complexity
            }
        }
        
        return comparison
    
    def get_protocol_summary(self) -> Dict[str, any]:
        """Get a summary of all protocols."""
        summary = {
            "total_protocols": len(self.protocols),
            "by_type": {},
            "header_size_ranges": {},
            "feature_support": {
                "pubsub": 0,
                "rpc": 0,
                "streaming": 0,
                "qos": 0,
                "discovery": 0
            }
        }
        
        # Count by type
        for protocol in self.protocols.values():
            protocol_type = protocol.type.value
            summary["by_type"][protocol_type] = summary["by_type"].get(protocol_type, 0) + 1
            
            # Count features
            if protocol.supports_pubsub:
                summary["feature_support"]["pubsub"] += 1
            if protocol.supports_rpc:
                summary["feature_support"]["rpc"] += 1
            if protocol.supports_streaming:
                summary["feature_support"]["streaming"] += 1
            if protocol.supports_qos:
                summary["feature_support"]["qos"] += 1
            if protocol.supports_discovery:
                summary["feature_support"]["discovery"] += 1
        
        # Header size analysis
        header_sizes = [p.header_size_min for p in self.protocols.values()]
        summary["header_size_ranges"] = {
            "min": min(header_sizes),
            "max": max(header_sizes),
            "avg": sum(header_sizes) / len(header_sizes)
        }
        
        return summary
