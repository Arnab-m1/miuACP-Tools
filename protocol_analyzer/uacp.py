"""
µACP (Micro Agent Communication Protocol) Implementation

This module implements the µACP protocol specification including:
- Fixed 8-byte header parsing
- TLV options handling
- Protocol verbs and semantics
- Mathematical analysis functions
"""

import struct
from typing import Dict, List, Optional, Tuple, Union
from enum import IntEnum
import cbor2
from .models import Protocol, ProtocolType, VerbType, ContentType, QoSLevel


class UACPVerb(IntEnum):
    """µACP protocol verbs."""
    PING = 0
    TELL = 1
    ASK = 2
    OBSERVE = 3


class UACPOptionType(IntEnum):
    """µACP TLV option types."""
    CONV_ID = 0x01
    CORR_ID = 0x02
    TOPIC_PATH = 0x03
    CONTENT_TYPE = 0x04
    ETAG = 0x05
    MAX_AGE = 0x06
    BLOCK = 0x07
    AUTH = 0x08
    PRIORITY = 0x09


class UACPHeader:
    """µACP fixed 8-byte header implementation."""
    
    def __init__(self, 
                 version: int = 1,
                 verb: UACPVerb = UACPVerb.PING,
                 qos: QoSLevel = QoSLevel.AT_MOST_ONCE,
                 code: int = 0,
                 msg_id: int = 0,
                 options_count: int = 0):
        self.version = version
        self.verb = verb
        self.qos = qos
        self.code = code
        self.msg_id = msg_id
        self.options_count = options_count
    
    def pack(self) -> bytes:
        """Pack header into 8 bytes."""
        # V(2b) + Type(2b) + QoS(2b) + Code(8b) + MsgID(24b) + Opts(8b)
        # Pack into 8 bytes using bit manipulation
        header = 0
        
        # Version (2 bits)
        header |= (self.version & 0x3) << 62
        
        # Verb type (2 bits)
        header |= (self.verb & 0x3) << 60
        
        # QoS (2 bits)
        header |= (self.qos & 0x3) << 58
        
        # Code (8 bits)
        header |= (self.code & 0xFF) << 50
        
        # Message ID (24 bits)
        header |= (self.msg_id & 0xFFFFFF) << 26
        
        # Options count (8 bits)
        header |= (self.options_count & 0xFF) << 18
        
        return header.to_bytes(8, byteorder='big')
    
    @classmethod
    def unpack(cls, data: bytes) -> 'UACPHeader':
        """Unpack header from 8 bytes."""
        if len(data) < 8:
            raise ValueError("Header must be at least 8 bytes")
        
        header_int = int.from_bytes(data[:8], byteorder='big')
        
        version = (header_int >> 62) & 0x3
        verb = (header_int >> 60) & 0x3
        qos = (header_int >> 58) & 0x3
        code = (header_int >> 50) & 0xFF
        msg_id = (header_int >> 26) & 0xFFFFFF
        options_count = (header_int >> 18) & 0xFF
        
        return cls(version, UACPVerb(verb), QoSLevel(qos), code, msg_id, options_count)
    
    def __str__(self) -> str:
        return (f"UACPHeader(v{self.version}, {self.verb.name}, "
                f"QoS{self.qos.value}, code={self.code}, "
                f"msg_id={self.msg_id:06x}, opts={self.options_count})")


class UACPOption:
    """µACP TLV option implementation."""
    
    def __init__(self, opt_type: UACPOptionType, value: Union[bytes, str, int]):
        self.opt_type = opt_type
        self.value = value
    
    def pack(self) -> bytes:
        """Pack option into TLV format."""
        if isinstance(self.value, str):
            value_bytes = self.value.encode('utf-8')
        elif isinstance(self.value, int):
            value_bytes = self.value.to_bytes(3, byteorder='big')
        else:
            value_bytes = self.value
        
        # Type (1 byte) + Length (1 byte) + Value
        return struct.pack('BB', self.opt_type, len(value_bytes)) + value_bytes
    
    @classmethod
    def unpack(cls, data: bytes, offset: int = 0) -> Tuple['UACPOption', int]:
        """Unpack option from TLV format."""
        if offset + 2 > len(data):
            raise ValueError("Insufficient data for option header")
        
        opt_type, length = struct.unpack('BB', data[offset:offset+2])
        offset += 2
        
        if offset + length > len(data):
            raise ValueError("Insufficient data for option value")
        
        value_bytes = data[offset:offset+length]
        offset += length
        
        # Parse value based on type
        if opt_type == UACPOptionType.CONTENT_TYPE:
            value = int.from_bytes(value_bytes, byteorder='big')
        elif opt_type == UACPOptionType.MAX_AGE:
            value = int.from_bytes(value_bytes, byteorder='big')
        elif opt_type == UACPOptionType.PRIORITY:
            value = int.from_bytes(value_bytes, byteorder='big')
        else:
            value = value_bytes
        
        return cls(UACPOptionType(opt_type), value), offset


class UACPProtocol:
    """µACP protocol implementation with analysis capabilities."""
    
    def __init__(self):
        self.name = "µACP"
        self.type = ProtocolType.EDGE_NATIVE_AGENT
        self.description = "Micro Agent Communication Protocol - lightweight agent communication for edge-native systems"
        
        # Protocol characteristics
        self.header_size_min = 8
        self.header_size_max = 8  # Fixed header size
        
        # Supported features
        self.supports_pubsub = True
        self.supports_rpc = True
        self.supports_streaming = True
        self.supports_qos = True
        self.supports_discovery = True
        
        # Supported verbs
        self.supported_verbs = [
            VerbType.PING, VerbType.TELL, VerbType.ASK, VerbType.OBSERVE
        ]
        
        # Content types
        self.content_types = [
            ContentType.CBOR, ContentType.JSON, ContentType.PROTOBUF, ContentType.TEXT
        ]
        
        # State complexity
        self.state_complexity = "O(W)"  # W = reliability window size
    
    def create_message(self, 
                      verb: UACPVerb,
                      payload: Optional[bytes] = None,
                      qos: QoSLevel = QoSLevel.AT_MOST_ONCE,
                      msg_id: int = 0,
                      options: Optional[List[UACPOption]] = None) -> bytes:
        """Create a complete µACP message."""
        options = options or []
        
        # Create header
        header = UACPHeader(
            version=1,
            verb=verb,
            qos=qos,
            code=0,
            msg_id=msg_id,
            options_count=len(options)
        )
        
        # Pack header and options
        message = header.pack()
        
        for option in options:
            message += option.pack()
        
        # Add payload if present
        if payload:
            message += payload
        
        return message
    
    def parse_message(self, data: bytes) -> Tuple[UACPHeader, List[UACPOption], Optional[bytes]]:
        """Parse a complete µACP message."""
        if len(data) < 8:
            raise ValueError("Message too short")
        
        # Parse header
        header = UACPHeader.unpack(data)
        
        # Parse options
        options = []
        offset = 8
        
        for _ in range(header.options_count):
            if offset >= len(data):
                break
            option, offset = UACPOption.unpack(data, offset)
            options.append(option)
        
        # Remaining data is payload
        payload = data[offset:] if offset < len(data) else None
        
        return header, options, payload
    
    def calculate_header_efficiency(self, payload_size: int) -> float:
        """Calculate header efficiency ratio η_h = H/(H+P)."""
        header_size = 8  # Fixed header size
        return header_size / (header_size + payload_size)
    
    def calculate_latency_bound(self, rtt: float, serialization_time: float, loss_prob: float = 0.0) -> float:
        """Calculate latency bound T_ASK ≤ RTT + t_ser."""
        if loss_prob == 0:
            return rtt + serialization_time
        else:
            # Expected time with retries: (RTT + t_ser) / (1-p)
            return (rtt + serialization_time) / (1 - loss_prob)
    
    def calculate_energy_model(self, 
                             tx_power: float, 
                             rx_power: float,
                             tx_time: float,
                             rx_time: float,
                             message_size: int) -> float:
        """Calculate energy consumption: E ≈ P_tx * t_tx + P_rx * t_rx."""
        # Energy is proportional to bytes on air
        energy_factor = message_size / 1024  # Normalize to 1KB
        return (tx_power * tx_time + rx_power * rx_time) * energy_factor
    
    def get_protocol_model(self) -> Protocol:
        """Get the protocol model for analysis."""
        return Protocol(
            name=self.name,
            type=self.type,
            description=self.description,
            header_size_min=self.header_size_min,
            header_size_max=self.header_size_max,
            supports_pubsub=self.supports_pubsub,
            supports_rpc=self.supports_rpc,
            supports_streaming=self.supports_streaming,
            supports_qos=self.supports_qos,
            supports_discovery=self.supports_discovery,
            supported_verbs=self.supported_verbs,
            content_types=self.content_types,
            state_complexity=self.state_complexity
        )
    
    def encode_fipa_acl(self, fipa_act: str, content: str) -> bytes:
        """Encode FIPA-ACL performative into µACP."""
        if fipa_act == "inform":
            return self.create_message(UACPVerb.TELL, content.encode())
        elif fipa_act == "request":
            return self.create_message(UACPVerb.ASK, content.encode())
        elif fipa_act == "subscribe":
            return self.create_message(UACPVerb.OBSERVE, content.encode())
        else:
            # Default to ASK for other performatives
            return self.create_message(UACPVerb.ASK, content.encode())
    
    def create_discovery_response(self, 
                                agent_id: str,
                                capabilities: List[str],
                                topics: List[str],
                                content_types: List[ContentType],
                                max_block_size: int = 1024) -> bytes:
        """Create agent discovery response in CBOR format."""
        discovery_data = {
            "id": agent_id,
            "caps": capabilities,
            "topics": topics,
            "ct": [ct.value for ct in content_types],
            "max_block": max_block_size
        }
        
        cbor_data = cbor2.dumps(discovery_data)
        return self.create_message(UACPVerb.TELL, cbor_data)
