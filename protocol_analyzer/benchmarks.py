"""
Benchmark tools for protocol performance testing and comparison.

This module provides tools to measure and compare protocol performance
across different metrics and scenarios.
"""

import time
import asyncio
import statistics
from typing import Dict, List, Any, Optional, Tuple
import json
import csv
from dataclasses import dataclass, asdict
from .models import Protocol, ProtocolMetrics
from .protocols import ProtocolDatabase
# Import from µACP library
try:
    from miuacp import UACPProtocol, UACPHeader, UACPOption, UACPOptionType, UACPVerb, UACPMessage
    UACP_LIB_AVAILABLE = True
except ImportError:
    # Fallback to local implementation
    from .uacp import UACPProtocol, UACPHeader, UACPOption, UACPOptionType, UACPVerb
    UACP_LIB_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    protocol_name: str
    test_name: str
    duration_ms: float
    memory_bytes: int
    message_count: int
    error_count: int
    throughput_msg_per_sec: float
    latency_ms: float
    energy_estimate_uj: float
    timestamp: str


class BenchmarkSuite:
    """Comprehensive benchmark suite for protocol comparison."""
    
    def __init__(self):
        self.protocol_db = ProtocolDatabase()
        self.uacp = UACPProtocol()
        self.results: List[BenchmarkResult] = []
    
    def benchmark_message_creation(self, protocol_name: str, message_count: int = 1000) -> BenchmarkResult:
        """Benchmark message creation performance."""
        protocol = self.protocol_db.get_protocol(protocol_name)
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        messages = []
        errors = 0
        
        try:
            for i in range(message_count):
                if protocol_name == "µACP":
                    # Create µACP message
                    message = self.uacp.create_message(
                        verb=UACPVerb.TELL,
                        payload=f"Test message {i}".encode(),
                        msg_id=i
                    )
                else:
                    # Simulate message creation for other protocols
                    header_size = protocol.header_size_min
                    payload_size = 16
                    message = b"0" * (header_size + payload_size)
                
                messages.append(message)
                
        except Exception as e:
            errors += 1
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        duration_ms = (end_time - start_time) * 1000
        memory_bytes = end_memory - start_memory
        throughput = message_count / (duration_ms / 1000)
        latency = duration_ms / message_count
        
        # Estimate energy consumption
        total_bytes = 0
        for msg in messages:
            if hasattr(msg, 'pack'):
                # µACP message - pack it to get size
                total_bytes += len(msg.pack())
            else:
                # Other message types
                total_bytes += len(msg)
        energy_estimate = total_bytes * 0.1  # 0.1 µJ per byte
        
        result = BenchmarkResult(
            protocol_name=protocol_name,
            test_name="message_creation",
            duration_ms=duration_ms,
            memory_bytes=memory_bytes,
            message_count=message_count,
            error_count=errors,
            throughput_msg_per_sec=throughput,
            latency_ms=latency,
            energy_estimate_uj=energy_estimate,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.results.append(result)
        return result
    
    def benchmark_header_parsing(self, protocol_name: str, message_count: int = 1000) -> BenchmarkResult:
        """Benchmark header parsing performance."""
        protocol = self.protocol_db.get_protocol(protocol_name)
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        errors = 0
        
        try:
            for i in range(message_count):
                if protocol_name == "µACP":
                    # Create and parse µACP message
                    message = self.uacp.create_message(
                        verb=UACPVerb.TELL,
                        payload=f"Test message {i}".encode(),
                        msg_id=i
                    )
                    header, options, payload = self.uacp.parse_message(message)
                else:
                    # Simulate header parsing for other protocols
                    header_size = protocol.header_size_min
                    message = b"0" * (header_size + 16)
                    # Simulate parsing time
                    time.sleep(0.000001)  # 1 µs
                
        except Exception as e:
            errors += 1
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        duration_ms = (end_time - start_time) * 1000
        memory_bytes = end_memory - start_memory
        throughput = message_count / (duration_ms / 1000)
        latency = duration_ms / message_count
        
        # Estimate energy consumption
        energy_estimate = message_count * protocol.header_size_min * 0.1
        
        result = BenchmarkResult(
            protocol_name=protocol_name,
            test_name="header_parsing",
            duration_ms=duration_ms,
            memory_bytes=memory_bytes,
            message_count=message_count,
            error_count=errors,
            throughput_msg_per_sec=throughput,
            latency_ms=latency,
            energy_estimate_uj=energy_estimate,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.results.append(result)
        return result
    
    def benchmark_memory_efficiency(self, protocol_name: str, message_count: int = 1000) -> BenchmarkResult:
        """Benchmark memory efficiency."""
        protocol = self.protocol_db.get_protocol(protocol_name)
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        messages = []
        errors = 0
        
        try:
            for i in range(message_count):
                if protocol_name == "µACP":
                    message = self.uacp.create_message(
                        verb=UACPVerb.TELL,
                        payload=f"Test message {i}".encode(),
                        msg_id=i
                    )
                else:
                    # Simulate message for other protocols
                    header_size = protocol.header_size_min
                    payload_size = 16
                    message = b"0" * (header_size + payload_size)
                
                messages.append(message)
                
        except Exception as e:
            errors += 1
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        duration_ms = (end_time - start_time) * 1000
        memory_bytes = end_memory - start_memory
        throughput = message_count / (duration_ms / 1000)
        latency = duration_ms / message_count
        
        # Calculate memory efficiency
        total_message_size = 0
        for msg in messages:
            if hasattr(msg, 'pack'):
                # µACP message - pack it to get size
                total_message_size += len(msg.pack())
            else:
                # Other message types
                total_message_size += len(msg)
        memory_efficiency = total_message_size / max(memory_bytes, 1)
        energy_estimate = total_message_size * 0.1
        
        result = BenchmarkResult(
            protocol_name=protocol_name,
            test_name="memory_efficiency",
            duration_ms=duration_ms,
            memory_bytes=memory_bytes,
            message_count=message_count,
            error_count=errors,
            throughput_msg_per_sec=throughput,
            latency_ms=latency,
            energy_estimate_uj=energy_estimate,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.results.append(result)
        return result
    
    def benchmark_uacp_library(self, message_count: int = 1000) -> Dict[str, Any]:
        """Benchmark µACP library using real implementation."""
        if not UACP_LIB_AVAILABLE:
            return {"error": "µACP library not available"}
        
        try:
            results = {}
            
            # Benchmark message creation
            start_time = time.time()
            messages = []
            for i in range(message_count):
                if i % 4 == 0:
                    msg = UACPProtocol.create_ping(i, 0)
                elif i % 4 == 1:
                    msg = UACPProtocol.create_tell(i, f"topic/{i}", {"data": f"message_{i}"}, 0)
                elif i % 4 == 2:
                    msg = UACPProtocol.create_ask(i, f"topic/{i}", {"request": f"data_{i}"}, 1)
                else:
                    msg = UACPProtocol.create_observe(i, f"topic/{i}", 1)
                messages.append(msg)
            
            creation_time = time.time() - start_time
            results["message_creation"] = {
                "time": creation_time,
                "messages": message_count,
                "throughput": message_count / creation_time if creation_time > 0 else 0
            }
            
            # Benchmark packing
            start_time = time.time()
            packed_messages = []
            for msg in messages:
                packed = msg.pack()
                packed_messages.append(packed)
            
            pack_time = time.time() - start_time
            results["packing"] = {
                "time": pack_time,
                "messages": message_count,
                "throughput": message_count / pack_time if pack_time > 0 else 0
            }
            
            # Benchmark unpacking
            start_time = time.time()
            for packed in packed_messages:
                unpacked = UACPMessage.unpack(packed)
            
            unpack_time = time.time() - start_time
            results["unpacking"] = {
                "time": unpack_time,
                "messages": message_count,
                "throughput": message_count / unpack_time if unpack_time > 0 else 0
            }
            
            # Benchmark validation
            start_time = time.time()
            valid_count = 0
            for msg in messages:
                if UACPProtocol.validate_message(msg):
                    valid_count += 1
            
            validation_time = time.time() - start_time
            results["validation"] = {
                "time": validation_time,
                "messages": message_count,
                "valid_count": valid_count,
                "throughput": message_count / validation_time if validation_time > 0 else 0
            }
            
            # Calculate message sizes
            total_size = sum(len(packed) for packed in packed_messages)
            avg_size = total_size / message_count if message_count > 0 else 0
            
            results["sizes"] = {
                "total_bytes": total_size,
                "average_bytes": avg_size,
                "header_efficiency": 8 / avg_size if avg_size > 0 else 0
            }
            
            # Overall performance
            total_time = creation_time + pack_time + unpack_time + validation_time
            results["overall"] = {
                "total_time": total_time,
                "total_operations": message_count * 4,  # create, pack, unpack, validate
                "overall_throughput": (message_count * 4) / total_time if total_time > 0 else 0
            }
            
            return results
            
        except Exception as e:
            return {"error": f"µACP benchmark failed: {str(e)}"}
    
    def run_comprehensive_benchmark(self, protocols: Optional[List[str]] = None, 
                                  message_counts: Optional[List[int]] = None) -> Dict[str, Any]:
        """Run comprehensive benchmark suite across multiple protocols."""
        
        if protocols is None:
            protocols = self.protocol_db.get_protocol_names()
        
        if message_counts is None:
            message_counts = [100, 1000, 10000]
        
        benchmark_results = {}
        
        for protocol_name in protocols:
            print(f"Benchmarking {protocol_name}...")
            protocol_results = {}
            
            for message_count in message_counts:
                print(f"  Testing with {message_count} messages...")
                
                # Run all benchmark types
                creation_result = self.benchmark_message_creation(protocol_name, message_count)
                parsing_result = self.benchmark_header_parsing(protocol_name, message_count)
                memory_result = self.benchmark_memory_efficiency(protocol_name, message_count)
                
                protocol_results[message_count] = {
                    "creation": creation_result,
                    "parsing": parsing_result,
                    "memory": memory_result
                }
            
            benchmark_results[protocol_name] = protocol_results
        
        return benchmark_results
    
    def generate_benchmark_report(self, results: Dict[str, Any], 
                                output_format: str = "json") -> str:
        """Generate a comprehensive benchmark report."""
        
        if output_format == "json":
            return self._generate_json_report(results)
        elif output_format == "csv":
            return self._generate_csv_report(results)
        else:
            return self._generate_text_report(results)
    
    def _generate_json_report(self, results: Dict[str, Any]) -> str:
        """Generate JSON benchmark report."""
        report = {
            "benchmark_summary": {
                "total_protocols": len(results),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "protocols_tested": list(results.keys())
            },
            "detailed_results": {}
        }
        
        for protocol_name, protocol_results in results.items():
            report["detailed_results"][protocol_name] = {}
            
            for message_count, test_results in protocol_results.items():
                report["detailed_results"][protocol_name][str(message_count)] = {}
                
                for test_name, result in test_results.items():
                    report["detailed_results"][protocol_name][str(message_count)][test_name] = asdict(result)
        
        return json.dumps(report, indent=2)
    
    def _generate_csv_report(self, results: Dict[str, Any]) -> str:
        """Generate CSV benchmark report."""
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            "Protocol", "Message Count", "Test Type", "Duration (ms)", 
            "Memory (bytes)", "Throughput (msg/s)", "Latency (ms)", 
            "Energy (µJ)", "Errors", "Timestamp"
        ])
        
        # Write data
        for protocol_name, protocol_results in results.items():
            for message_count, test_results in protocol_results.items():
                for test_name, result in test_results.items():
                    writer.writerow([
                        protocol_name, message_count, test_name,
                        result.duration_ms, result.memory_bytes,
                        result.throughput_msg_per_sec, result.latency_ms,
                        result.energy_estimate_uj, result.error_count,
                        result.timestamp
                    ])
        
        return output.getvalue()
    
    def _generate_text_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable text report."""
        report = []
        report.append("=" * 80)
        report.append("µACP PROTOCOL BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Protocols tested: {', '.join(results.keys())}")
        report.append("")
        
        for protocol_name, protocol_results in results.items():
            report.append(f"PROTOCOL: {protocol_name}")
            report.append("-" * 40)
            
            for message_count, test_results in protocol_results.items():
                report.append(f"  Message Count: {message_count:,}")
                
                for test_name, result in test_results.items():
                    report.append(f"    {test_name.title()}:")
                    report.append(f"      Duration: {result.duration_ms:.2f} ms")
                    report.append(f"      Throughput: {result.throughput_msg_per_sec:.0f} msg/s")
                    report.append(f"      Latency: {result.latency_ms:.3f} ms")
                    report.append(f"      Memory: {result.memory_bytes:,} bytes")
                    report.append(f"      Energy: {result.energy_estimate_uj:.1f} µJ")
                    report.append(f"      Errors: {result.error_count}")
                    report.append("")
            
            report.append("")
        
        return "\n".join(report)
    
    def compare_benchmark_results(self, protocol_a: str, protocol_b: str, 
                                message_count: int = 1000) -> Dict[str, Any]:
        """Compare benchmark results between two protocols."""
        
        if protocol_a not in self.results or protocol_b not in self.results:
            raise ValueError("Benchmark results not found for one or both protocols")
        
        # Get results for the specified message count
        a_results = [r for r in self.results if r.protocol_name == protocol_a and r.message_count == message_count]
        b_results = [r for r in self.results if r.protocol_name == protocol_b and r.message_count == message_count]
        
        if not a_results or not b_results:
            raise ValueError(f"No results found for message count {message_count}")
        
        comparison = {
            "protocol_a": protocol_a,
            "protocol_b": protocol_b,
            "message_count": message_count,
            "comparison": {}
        }
        
        # Compare each test type
        test_types = set(r.test_name for r in a_results)
        
        for test_type in test_types:
            a_result = next(r for r in a_results if r.test_name == test_type)
            b_result = next(r for r in b_results if r.test_name == test_type)
            
            comparison["comparison"][test_type] = {
                "throughput_ratio": b_result.throughput_msg_per_sec / a_result.throughput_msg_per_sec,
                "latency_ratio": b_result.latency_ms / a_result.latency_ms,
                "memory_ratio": b_result.memory_bytes / a_result.memory_bytes,
                "energy_ratio": b_result.energy_estimate_uj / a_result.energy_estimate_uj,
                "a_performance": {
                    "throughput": a_result.throughput_msg_per_sec,
                    "latency": a_result.latency_ms,
                    "memory": a_result.memory_bytes,
                    "energy": a_result.energy_estimate_uj
                },
                "b_performance": {
                    "throughput": b_result.throughput_msg_per_sec,
                    "latency": b_result.latency_ms,
                    "memory": b_result.memory_bytes,
                    "energy": b_result.energy_estimate_uj
                }
            }
        
        return comparison
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage (simplified)."""
        # This is a simplified memory measurement
        # In a real implementation, you might use psutil or similar
        import gc
        gc.collect()
        return len(gc.get_objects()) * 64  # Rough estimate
    
    def export_results(self, filename: str, format: str = "json"):
        """Export benchmark results to file."""
        if format == "json":
            with open(filename, 'w') as f:
                json.dump([asdict(r) for r in self.results], f, indent=2)
        elif format == "csv":
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Protocol", "Test", "Duration (ms)", "Memory (bytes)",
                    "Message Count", "Error Count", "Throughput (msg/s)",
                    "Latency (ms)", "Energy (µJ)", "Timestamp"
                ])
                for result in self.results:
                    writer.writerow([
                        result.protocol_name, result.test_name,
                        result.duration_ms, result.memory_bytes,
                        result.message_count, result.error_count,
                        result.throughput_msg_per_sec, result.latency_ms,
                        result.energy_estimate_uj, result.timestamp
                    ])
        else:
            with open(filename, 'w') as f:
                f.write(self._generate_text_report({"results": self.results}))
        
        print(f"Results exported to {filename}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of all benchmark results."""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        summary = {
            "total_runs": len(self.results),
            "protocols_tested": list(set(r.protocol_name for r in self.results)),
            "test_types": list(set(r.test_name for r in self.results)),
            "performance_ranking": {},
            "best_performers": {}
        }
        
        # Calculate performance rankings
        for test_type in summary["test_types"]:
            test_results = [r for r in self.results if r.test_name == test_type]
            
            # Rank by throughput
            throughput_ranking = sorted(test_results, key=lambda x: x.throughput_msg_per_sec, reverse=True)
            summary["performance_ranking"][f"{test_type}_throughput"] = [
                {"protocol": r.protocol_name, "throughput": r.throughput_msg_per_sec}
                for r in throughput_ranking
            ]
            
            # Rank by latency (lower is better)
            latency_ranking = sorted(test_results, key=lambda x: x.latency_ms)
            summary["performance_ranking"][f"{test_type}_latency"] = [
                {"protocol": r.protocol_name, "latency": r.latency_ms}
                for r in latency_ranking
            ]
            
            # Best performer for this test type
            best_throughput = throughput_ranking[0]
            summary["best_performers"][test_type] = {
                "best_throughput": best_throughput.protocol_name,
                "best_latency": latency_ranking[0].protocol_name
            }
        
        return summary
