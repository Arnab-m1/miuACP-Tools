"""
µACP Protocol Analyzer Library

A comprehensive framework for analyzing and comparing agent communication protocols,
with special focus on µACP (Micro Agent Communication Protocol) and its positioning
against existing solutions.
"""

__version__ = "0.1.0"
__author__ = "Protocol Analysis Team"

from .core import ProtocolAnalyzer, ProtocolComparison
from .models import Protocol, ProtocolMetrics, AgentInteraction
from .benchmarks import BenchmarkSuite
from .visualization import ProtocolVisualizer

# Import from the standalone µACP library
try:
    from uacp_lib import (
        UACPProtocol, UACPHeader, UACPOption, 
        UACPOptionType, UACPVerb, UACPContentType,
        UACPMessage
    )
except ImportError:
    # Fallback to local implementation if library not available
    from .uacp import UACPProtocol, UACPHeader, UACPOption, UACPOptionType, UACPVerb

__all__ = [
    "ProtocolAnalyzer",
    "ProtocolComparison", 
    "Protocol",
    "ProtocolMetrics",
    "AgentInteraction",
    "UACPProtocol",
    "UACPHeader",
    "UACPOptions",
    "BenchmarkSuite",
    "ProtocolVisualizer"
]
