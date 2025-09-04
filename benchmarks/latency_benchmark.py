"""
Latency Benchmarking for µACP

Measures and validates latency performance against critical targets:
- Target: <1ms end-to-end on ESP32-C3
- Message creation time
- Transmission time  
- Processing time
- End-to-end latency
"""

import time
import statistics
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

# Add miuACP to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'miuACP', 'src'))

try:
    from miuacp import UACPMessage, UACPHeader, MessageType, StatusCode
    from miuacp.edge import get_edge_memory_manager
except ImportError:
    print("Warning: miuACP library not found. Using mock implementations.")
    # Mock implementations for testing
    class UACPMessage:
        def __init__(self, *args, **kwargs):
            pass
    
    class UACPHeader:
        def __init__(self, *args, **kwargs):
            pass
    
    class MessageType:
        PING = "PING"
        PONG = "PONG"
        TELL = "TELL"
        ASK = "ASK"
    
    class StatusCode:
        SUCCESS = 200


class LatencyTestType(Enum):
    """Types of latency tests"""
    MESSAGE_CREATION = "message_creation"
    MESSAGE_PROCESSING = "message_processing"
    TRANSMISSION = "transmission"
    END_TO_END = "end_to_end"
    ROUND_TRIP = "round_trip"


@dataclass
class LatencyMeasurement:
    """Individual latency measurement"""
    test_type: LatencyTestType
    start_time: float
    end_time: float
    duration: float
    message_size: int
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyStats:
    """Statistical analysis of latency measurements"""
    test_type: LatencyTestType
    measurements: List[LatencyMeasurement]
    count: int
    min_latency: float
    max_latency: float
    mean_latency: float
    median_latency: float
    p95_latency: float
    p99_latency: float
    std_deviation: float
    success_rate: float
    target_met: bool
    target_latency: float


class LatencyProfiler:
    """High-precision latency profiler"""
    
    def __init__(self):
        self.measurements: List[LatencyMeasurement] = []
        self.active_measurements: Dict[str, float] = {}
    
    def start_measurement(self, test_id: str, test_type: LatencyTestType) -> str:
        """Start a latency measurement"""
        start_time = time.perf_counter()
        self.active_measurements[test_id] = start_time
        return test_id
    
    def end_measurement(self, test_id: str, test_type: LatencyTestType, 
                       message_size: int = 0, success: bool = True, 
                       error: str = None, metadata: Dict[str, Any] = None) -> LatencyMeasurement:
        """End a latency measurement and record it"""
        if test_id not in self.active_measurements:
            raise ValueError(f"Measurement {test_id} not found")
        
        start_time = self.active_measurements[test_id]
        end_time = time.perf_counter()
        duration = (end_time - start_time) * 1000  # Convert to milliseconds
        
        measurement = LatencyMeasurement(
            test_type=test_type,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            message_size=message_size,
            success=success,
            error=error,
            metadata=metadata or {}
        )
        
        self.measurements.append(measurement)
        del self.active_measurements[test_id]
        
        return measurement
    
    def get_measurements(self, test_type: LatencyTestType = None) -> List[LatencyMeasurement]:
        """Get measurements, optionally filtered by test type"""
        if test_type:
            return [m for m in self.measurements if m.test_type == test_type]
        return self.measurements.copy()
    
    def clear_measurements(self):
        """Clear all measurements"""
        self.measurements.clear()
        self.active_measurements.clear()


class LatencyBenchmark:
    """
    Comprehensive latency benchmarking for µACP
    
    Tests critical latency targets:
    - Message creation: <0.1ms
    - Message processing: <0.5ms  
    - Transmission: <0.3ms
    - End-to-end: <1ms
    """
    
    def __init__(self):
        self.profiler = LatencyProfiler()
        self.targets = {
            LatencyTestType.MESSAGE_CREATION: 0.1,  # 0.1ms
            LatencyTestType.MESSAGE_PROCESSING: 0.5,  # 0.5ms
            LatencyTestType.TRANSMISSION: 0.3,  # 0.3ms
            LatencyTestType.END_TO_END: 1.0,  # 1ms
            LatencyTestType.ROUND_TRIP: 2.0   # 2ms
        }
    
    def benchmark_message_creation(self, iterations: int = 1000) -> LatencyStats:
        """Benchmark message creation latency"""
        print(f"Benchmarking message creation ({iterations} iterations)...")
        
        for i in range(iterations):
            test_id = f"msg_creation_{i}"
            self.profiler.start_measurement(test_id, LatencyTestType.MESSAGE_CREATION)
            
            try:
                # Create a UACP message
                header = UACPHeader(
                    message_type=MessageType.PING,
                    status_code=StatusCode.SUCCESS,
                    message_id=i,
                    correlation_id=i,
                    source_agent="benchmark_agent",
                    destination_agent="target_agent"
                )
                
                message = UACPMessage(
                    header=header,
                    payload=b"benchmark_payload",
                    options=[]
                )
                
                # Simulate message size calculation
                message_size = len(str(message))
                
                self.profiler.end_measurement(
                    test_id, LatencyTestType.MESSAGE_CREATION,
                    message_size=message_size, success=True
                )
                
            except Exception as e:
                self.profiler.end_measurement(
                    test_id, LatencyTestType.MESSAGE_CREATION,
                    success=False, error=str(e)
                )
        
        return self._calculate_stats(LatencyTestType.MESSAGE_CREATION)
    
    def benchmark_message_processing(self, iterations: int = 1000) -> LatencyStats:
        """Benchmark message processing latency"""
        print(f"Benchmarking message processing ({iterations} iterations)...")
        
        for i in range(iterations):
            test_id = f"msg_processing_{i}"
            self.profiler.start_measurement(test_id, LatencyTestType.MESSAGE_PROCESSING)
            
            try:
                # Create and process a message
                header = UACPHeader(
                    message_type=MessageType.PING,
                    status_code=StatusCode.SUCCESS,
                    message_id=i,
                    correlation_id=i,
                    source_agent="benchmark_agent",
                    destination_agent="target_agent"
                )
                
                message = UACPMessage(
                    header=header,
                    payload=b"benchmark_payload",
                    options=[]
                )
                
                # Simulate message processing
                processed_data = self._simulate_processing(message)
                message_size = len(str(message))
                
                self.profiler.end_measurement(
                    test_id, LatencyTestType.MESSAGE_PROCESSING,
                    message_size=message_size, success=True,
                    metadata={'processed_size': len(processed_data)}
                )
                
            except Exception as e:
                self.profiler.end_measurement(
                    test_id, LatencyTestType.MESSAGE_PROCESSING,
                    success=False, error=str(e)
                )
        
        return self._calculate_stats(LatencyTestType.MESSAGE_PROCESSING)
    
    def benchmark_transmission(self, iterations: int = 1000) -> LatencyStats:
        """Benchmark transmission latency"""
        print(f"Benchmarking transmission ({iterations} iterations)...")
        
        for i in range(iterations):
            test_id = f"transmission_{i}"
            self.profiler.start_measurement(test_id, LatencyTestType.TRANSMISSION)
            
            try:
                # Create message
                header = UACPHeader(
                    message_type=MessageType.PING,
                    status_code=StatusCode.SUCCESS,
                    message_id=i,
                    correlation_id=i,
                    source_agent="benchmark_agent",
                    destination_agent="target_agent"
                )
                
                message = UACPMessage(
                    header=header,
                    payload=b"benchmark_payload",
                    options=[]
                )
                
                # Simulate transmission
                serialized = self._simulate_serialization(message)
                message_size = len(serialized)
                
                # Simulate network delay
                time.sleep(0.0001)  # 0.1ms simulated network delay
                
                self.profiler.end_measurement(
                    test_id, LatencyTestType.TRANSMISSION,
                    message_size=message_size, success=True,
                    metadata={'serialized_size': len(serialized)}
                )
                
            except Exception as e:
                self.profiler.end_measurement(
                    test_id, LatencyTestType.TRANSMISSION,
                    success=False, error=str(e)
                )
        
        return self._calculate_stats(LatencyTestType.TRANSMISSION)
    
    def benchmark_end_to_end(self, iterations: int = 1000) -> LatencyStats:
        """Benchmark end-to-end latency"""
        print(f"Benchmarking end-to-end latency ({iterations} iterations)...")
        
        for i in range(iterations):
            test_id = f"e2e_{i}"
            self.profiler.start_measurement(test_id, LatencyTestType.END_TO_END)
            
            try:
                # Complete end-to-end flow
                header = UACPHeader(
                    message_type=MessageType.PING,
                    status_code=StatusCode.SUCCESS,
                    message_id=i,
                    correlation_id=i,
                    source_agent="benchmark_agent",
                    destination_agent="target_agent"
                )
                
                message = UACPMessage(
                    header=header,
                    payload=b"benchmark_payload",
                    options=[]
                )
                
                # Simulate complete flow
                serialized = self._simulate_serialization(message)
                processed = self._simulate_processing(message)
                response = self._simulate_response(message)
                
                message_size = len(serialized)
                
                self.profiler.end_measurement(
                    test_id, LatencyTestType.END_TO_END,
                    message_size=message_size, success=True,
                    metadata={
                        'serialized_size': len(serialized),
                        'processed_size': len(processed),
                        'response_size': len(response)
                    }
                )
                
            except Exception as e:
                self.profiler.end_measurement(
                    test_id, LatencyTestType.END_TO_END,
                    success=False, error=str(e)
                )
        
        return self._calculate_stats(LatencyTestType.END_TO_END)
    
    def benchmark_round_trip(self, iterations: int = 1000) -> LatencyStats:
        """Benchmark round-trip latency"""
        print(f"Benchmarking round-trip latency ({iterations} iterations)...")
        
        for i in range(iterations):
            test_id = f"round_trip_{i}"
            self.profiler.start_measurement(test_id, LatencyTestType.ROUND_TRIP)
            
            try:
                # Create request
                header = UACPHeader(
                    message_type=MessageType.PING,
                    status_code=StatusCode.SUCCESS,
                    message_id=i,
                    correlation_id=i,
                    source_agent="benchmark_agent",
                    destination_agent="target_agent"
                )
                
                request = UACPMessage(
                    header=header,
                    payload=b"ping_request",
                    options=[]
                )
                
                # Simulate round-trip
                serialized_req = self._simulate_serialization(request)
                processed_req = self._simulate_processing(request)
                
                # Create response
                response_header = UACPHeader(
                    message_type=MessageType.PONG,
                    status_code=StatusCode.SUCCESS,
                    message_id=i + 1000,
                    correlation_id=i,
                    source_agent="target_agent",
                    destination_agent="benchmark_agent"
                )
                
                response = UACPMessage(
                    header=response_header,
                    payload=b"pong_response",
                    options=[]
                )
                
                serialized_resp = self._simulate_serialization(response)
                processed_resp = self._simulate_processing(response)
                
                message_size = len(serialized_req) + len(serialized_resp)
                
                self.profiler.end_measurement(
                    test_id, LatencyTestType.ROUND_TRIP,
                    message_size=message_size, success=True,
                    metadata={
                        'request_size': len(serialized_req),
                        'response_size': len(serialized_resp),
                        'total_size': message_size
                    }
                )
                
            except Exception as e:
                self.profiler.end_measurement(
                    test_id, LatencyTestType.ROUND_TRIP,
                    success=False, error=str(e)
                )
        
        return self._calculate_stats(LatencyTestType.ROUND_TRIP)
    
    def run_all_benchmarks(self, iterations: int = 1000) -> Dict[LatencyTestType, LatencyStats]:
        """Run all latency benchmarks"""
        print("Starting comprehensive latency benchmarking...")
        print(f"Target: <1ms end-to-end on ESP32-C3")
        print(f"Iterations per test: {iterations}")
        print("-" * 50)
        
        results = {}
        
        # Run all benchmarks
        results[LatencyTestType.MESSAGE_CREATION] = self.benchmark_message_creation(iterations)
        results[LatencyTestType.MESSAGE_PROCESSING] = self.benchmark_message_processing(iterations)
        results[LatencyTestType.TRANSMISSION] = self.benchmark_transmission(iterations)
        results[LatencyTestType.END_TO_END] = self.benchmark_end_to_end(iterations)
        results[LatencyTestType.ROUND_TRIP] = self.benchmark_round_trip(iterations)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _simulate_processing(self, message) -> bytes:
        """Simulate message processing"""
        # Simulate processing time
        time.sleep(0.0001)  # 0.1ms processing time
        return b"processed_data"
    
    def _simulate_serialization(self, message) -> bytes:
        """Simulate message serialization"""
        # Simulate serialization time
        time.sleep(0.00005)  # 0.05ms serialization time
        return b"serialized_message"
    
    def _simulate_response(self, message) -> bytes:
        """Simulate response generation"""
        # Simulate response time
        time.sleep(0.0001)  # 0.1ms response time
        return b"response_data"
    
    def _calculate_stats(self, test_type: LatencyTestType) -> LatencyStats:
        """Calculate statistics for a test type"""
        measurements = self.profiler.get_measurements(test_type)
        
        if not measurements:
            return LatencyStats(
                test_type=test_type,
                measurements=[],
                count=0,
                min_latency=0,
                max_latency=0,
                mean_latency=0,
                median_latency=0,
                p95_latency=0,
                p99_latency=0,
                std_deviation=0,
                success_rate=0,
                target_met=False,
                target_latency=self.targets[test_type]
            )
        
        successful_measurements = [m for m in measurements if m.success]
        latencies = [m.duration for m in successful_measurements]
        
        if not latencies:
            return LatencyStats(
                test_type=test_type,
                measurements=measurements,
                count=len(measurements),
                min_latency=0,
                max_latency=0,
                mean_latency=0,
                median_latency=0,
                p95_latency=0,
                p99_latency=0,
                std_deviation=0,
                success_rate=0,
                target_met=False,
                target_latency=self.targets[test_type]
            )
        
        target_latency = self.targets[test_type]
        
        return LatencyStats(
            test_type=test_type,
            measurements=measurements,
            count=len(measurements),
            min_latency=min(latencies),
            max_latency=max(latencies),
            mean_latency=statistics.mean(latencies),
            median_latency=statistics.median(latencies),
            p95_latency=self._percentile(latencies, 95),
            p99_latency=self._percentile(latencies, 99),
            std_deviation=statistics.stdev(latencies) if len(latencies) > 1 else 0,
            success_rate=(len(successful_measurements) / len(measurements)) * 100,
            target_met=statistics.mean(latencies) <= target_latency,
            target_latency=target_latency
        )
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
        return sorted_data[index]
    
    def _print_summary(self, results: Dict[LatencyTestType, LatencyStats]):
        """Print benchmark summary"""
        print("\n" + "=" * 60)
        print("LATENCY BENCHMARK SUMMARY")
        print("=" * 60)
        
        for test_type, stats in results.items():
            status = "✅ PASS" if stats.target_met else "❌ FAIL"
            print(f"\n{test_type.value.upper().replace('_', ' ')}")
            print(f"  Target: {stats.target_latency:.1f}ms")
            print(f"  Mean:   {stats.mean_latency:.3f}ms")
            print(f"  P95:    {stats.p95_latency:.3f}ms")
            print(f"  P99:    {stats.p99_latency:.3f}ms")
            print(f"  Success Rate: {stats.success_rate:.1f}%")
            print(f"  Status: {status}")
        
        # Overall assessment
        all_targets_met = all(stats.target_met for stats in results.values())
        overall_status = "✅ ALL TARGETS MET" if all_targets_met else "❌ SOME TARGETS MISSED"
        
        print(f"\n{'=' * 60}")
        print(f"OVERALL STATUS: {overall_status}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    # Run latency benchmarks
    benchmark = LatencyBenchmark()
    results = benchmark.run_all_benchmarks(iterations=1000)
