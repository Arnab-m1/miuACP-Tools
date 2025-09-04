"""
Performance Benchmarking Suite for µACP

Comprehensive benchmarking tools to validate µACP performance against
critical targets and industry standards.

Target Performance Metrics:
- Latency: <1ms end-to-end on ESP32-C3
- Memory: <1KB per agent, <100 bytes per message  
- Scalability: Maintain <5ms latency under 20% packet loss
- Energy: <1mJ per message
"""

from .latency_benchmark import LatencyBenchmark, LatencyProfiler
from .memory_benchmark import MemoryBenchmark, MemoryProfiler
from .performance_analyzer import PerformanceAnalyzer, BenchmarkRunner



__all__ = [
    'LatencyBenchmark',
    'LatencyProfiler',
    'MemoryBenchmark', 
    'MemoryProfiler',
    'PerformanceAnalyzer',
    'BenchmarkRunner'
]

__version__ = "1.0.1"
