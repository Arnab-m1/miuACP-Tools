"""
Memory Benchmarking for µACP

Measures and validates memory usage against critical targets:
- Target: <1KB per agent, <100 bytes per message
- RAM usage per agent
- RAM usage per message
- Flash usage
- Memory fragmentation analysis
"""

import gc
import sys
import tracemalloc
import statistics
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import os
import time

# Optional imports with fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Using fallback memory monitoring.")

# Add miuACP to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'miuACP', 'src'))

try:
    from miuacp import UACPMessage, UACPHeader, MessageType, StatusCode
    from miuacp.edge import get_edge_memory_manager, EdgeMemoryManager
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
    
    class EdgeMemoryManager:
        def __init__(self):
            pass
        
        def get_memory_stats(self):
            return {'total_memory': 0, 'used_memory': 0}


class MemoryTestType(Enum):
    """Types of memory tests"""
    AGENT_MEMORY = "agent_memory"
    MESSAGE_MEMORY = "message_memory"
    FLASH_USAGE = "flash_usage"
    HEAP_FRAGMENTATION = "heap_fragmentation"
    MEMORY_LEAK = "memory_leak"
    POOL_EFFICIENCY = "pool_efficiency"


@dataclass
class MemoryMeasurement:
    """Individual memory measurement"""
    test_type: MemoryTestType
    timestamp: float
    memory_used: int
    memory_available: int
    memory_percentage: float
    heap_size: int
    heap_used: int
    heap_free: int
    fragmentation: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryStats:
    """Statistical analysis of memory measurements"""
    test_type: MemoryTestType
    measurements: List[MemoryMeasurement]
    count: int
    min_memory: int
    max_memory: int
    mean_memory: int
    median_memory: int
    memory_growth: int
    fragmentation_avg: float
    fragmentation_max: float
    success_rate: float
    target_met: bool
    target_memory: int


class MemoryProfiler:
    """Memory usage profiler with detailed tracking"""
    
    def __init__(self):
        self.measurements: List[MemoryMeasurement] = []
        self.baseline_memory = 0
        self.tracemalloc_started = False
    
    def start_profiling(self):
        """Start memory profiling"""
        if not self.tracemalloc_started:
            tracemalloc.start()
            self.tracemalloc_started = True
        
        # Force garbage collection and record baseline
        gc.collect()
        self.baseline_memory = self._get_current_memory()
    
    def stop_profiling(self):
        """Stop memory profiling"""
        if self.tracemalloc_started:
            tracemalloc.stop()
            self.tracemalloc_started = False
    
    def measure_memory(self, test_type: MemoryTestType, 
                      success: bool = True, error: str = None,
                      metadata: Dict[str, Any] = None) -> MemoryMeasurement:
        """Take a memory measurement"""
        current_memory = self._get_current_memory()
        memory_available = self._get_available_memory()
        memory_percentage = (current_memory / memory_available) * 100 if memory_available > 0 else 0
        
        heap_info = self._get_heap_info()
        fragmentation = self._calculate_fragmentation()
        
        measurement = MemoryMeasurement(
            test_type=test_type,
            timestamp=time.time(),
            memory_used=current_memory,
            memory_available=memory_available,
            memory_percentage=memory_percentage,
            heap_size=heap_info['size'],
            heap_used=heap_info['used'],
            heap_free=heap_info['free'],
            fragmentation=fragmentation,
            success=success,
            error=error,
            metadata=metadata or {}
        )
        
        self.measurements.append(measurement)
        return measurement
    
    def _get_current_memory(self) -> int:
        """Get current memory usage in bytes"""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                return process.memory_info().rss
            except:
                pass
        
        # Fallback for systems without psutil
        return sys.getsizeof(gc.get_objects())
    
    def _get_available_memory(self) -> int:
        """Get available system memory in bytes"""
        if PSUTIL_AVAILABLE:
            try:
                return psutil.virtual_memory().total
            except:
                pass
        
        # Fallback - assume 1GB for testing
        return 1024 * 1024 * 1024
    
    def _get_heap_info(self) -> Dict[str, int]:
        """Get heap information"""
        try:
            # Try to get heap info from gc module
            heap_stats = gc.get_stats()
            if heap_stats:
                total_objects = sum(stat['collections'] for stat in heap_stats)
                return {
                    'size': total_objects,
                    'used': total_objects,
                    'free': 0  # Simplified
                }
        except:
            pass
        
        # Fallback
        return {
            'size': 0,
            'used': 0,
            'free': 0
        }
    
    def _calculate_fragmentation(self) -> float:
        """Calculate memory fragmentation percentage"""
        try:
            if self.tracemalloc_started:
                current, peak = tracemalloc.get_traced_memory()
                if peak > 0:
                    return ((peak - current) / peak) * 100
        except:
            pass
        
        return 0.0
    
    def get_measurements(self, test_type: MemoryTestType = None) -> List[MemoryMeasurement]:
        """Get measurements, optionally filtered by test type"""
        if test_type:
            return [m for m in self.measurements if m.test_type == test_type]
        return self.measurements.copy()
    
    def clear_measurements(self):
        """Clear all measurements"""
        self.measurements.clear()


class MemoryBenchmark:
    """
    Comprehensive memory benchmarking for µACP
    
    Tests critical memory targets:
    - Agent memory: <1KB per agent
    - Message memory: <100 bytes per message
    - Flash usage: Optimized for 4MB ESP32-C3
    - Heap fragmentation: <10%
    """
    
    def __init__(self):
        self.profiler = MemoryProfiler()
        self.targets = {
            MemoryTestType.AGENT_MEMORY: 1024,      # 1KB per agent
            MemoryTestType.MESSAGE_MEMORY: 100,     # 100 bytes per message
            MemoryTestType.FLASH_USAGE: 3 * 1024 * 1024,  # 3MB for ESP32-C3
            MemoryTestType.HEAP_FRAGMENTATION: 10.0,  # 10% fragmentation
            MemoryTestType.MEMORY_LEAK: 0,          # No memory leaks
            MemoryTestType.POOL_EFFICIENCY: 80.0    # 80% pool efficiency
        }
    
    def benchmark_agent_memory(self, num_agents: int = 100) -> MemoryStats:
        """Benchmark memory usage per agent"""
        print(f"Benchmarking agent memory usage ({num_agents} agents)...")
        
        self.profiler.start_profiling()
        
        # Initial measurement
        self.profiler.measure_memory(
            MemoryTestType.AGENT_MEMORY,
            metadata={'agents': 0, 'phase': 'baseline'}
        )
        
        agents = []
        
        try:
            for i in range(num_agents):
                # Create agent (mock implementation)
                agent = self._create_mock_agent(f"agent_{i}")
                agents.append(agent)
                
                # Measure memory every 10 agents
                if (i + 1) % 10 == 0:
                    self.profiler.measure_memory(
                        MemoryTestType.AGENT_MEMORY,
                        metadata={'agents': i + 1, 'phase': 'creation'}
                    )
            
            # Final measurement
            self.profiler.measure_memory(
                MemoryTestType.AGENT_MEMORY,
                metadata={'agents': num_agents, 'phase': 'complete'}
            )
            
        except Exception as e:
            self.profiler.measure_memory(
                MemoryTestType.AGENT_MEMORY,
                success=False, error=str(e)
            )
        
        finally:
            self.profiler.stop_profiling()
        
        return self._calculate_stats(MemoryTestType.AGENT_MEMORY)
    
    def benchmark_message_memory(self, num_messages: int = 1000) -> MemoryStats:
        """Benchmark memory usage per message"""
        print(f"Benchmarking message memory usage ({num_messages} messages)...")
        
        self.profiler.start_profiling()
        
        # Initial measurement
        self.profiler.measure_memory(
            MemoryTestType.MESSAGE_MEMORY,
            metadata={'messages': 0, 'phase': 'baseline'}
        )
        
        messages = []
        
        try:
            for i in range(num_messages):
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
                    payload=f"benchmark_payload_{i}".encode(),
                    options=[]
                )
                
                messages.append(message)
                
                # Measure memory every 100 messages
                if (i + 1) % 100 == 0:
                    self.profiler.measure_memory(
                        MemoryTestType.MESSAGE_MEMORY,
                        metadata={'messages': i + 1, 'phase': 'creation'}
                    )
            
            # Final measurement
            self.profiler.measure_memory(
                MemoryTestType.MESSAGE_MEMORY,
                metadata={'messages': num_messages, 'phase': 'complete'}
            )
            
        except Exception as e:
            self.profiler.measure_memory(
                MemoryTestType.MESSAGE_MEMORY,
                success=False, error=str(e)
            )
        
        finally:
            self.profiler.stop_profiling()
        
        return self._calculate_stats(MemoryTestType.MESSAGE_MEMORY)
    
    def benchmark_flash_usage(self) -> MemoryStats:
        """Benchmark flash storage usage"""
        print("Benchmarking flash storage usage...")
        
        self.profiler.start_profiling()
        
        try:
            # Simulate flash operations
            flash_data = []
            
            # Write data to flash (simulated)
            for i in range(100):
                data = f"flash_data_block_{i}" * 100  # 1KB per block
                flash_data.append(data)
                
                if (i + 1) % 20 == 0:
                    self.profiler.measure_memory(
                        MemoryTestType.FLASH_USAGE,
                        metadata={'blocks': i + 1, 'size': len(data)}
                    )
            
            # Final measurement
            total_flash_size = sum(len(data) for data in flash_data)
            self.profiler.measure_memory(
                MemoryTestType.FLASH_USAGE,
                metadata={'total_blocks': len(flash_data), 'total_size': total_flash_size}
            )
            
        except Exception as e:
            self.profiler.measure_memory(
                MemoryTestType.FLASH_USAGE,
                success=False, error=str(e)
            )
        
        finally:
            self.profiler.stop_profiling()
        
        return self._calculate_stats(MemoryTestType.FLASH_USAGE)
    
    def benchmark_heap_fragmentation(self, iterations: int = 100) -> MemoryStats:
        """Benchmark heap fragmentation"""
        print(f"Benchmarking heap fragmentation ({iterations} iterations)...")
        
        self.profiler.start_profiling()
        
        try:
            for i in range(iterations):
                # Allocate and deallocate memory to create fragmentation
                objects = []
                
                # Allocate objects of different sizes
                for j in range(10):
                    obj = [0] * (100 + j * 10)  # Different sizes
                    objects.append(obj)
                
                # Deallocate some objects
                for j in range(0, len(objects), 2):
                    objects[j] = None
                
                # Force garbage collection
                gc.collect()
                
                # Measure fragmentation
                if (i + 1) % 10 == 0:
                    self.profiler.measure_memory(
                        MemoryTestType.HEAP_FRAGMENTATION,
                        metadata={'iteration': i + 1, 'objects': len(objects)}
                    )
            
            # Final measurement
            self.profiler.measure_memory(
                MemoryTestType.HEAP_FRAGMENTATION,
                metadata={'iterations': iterations, 'phase': 'complete'}
            )
            
        except Exception as e:
            self.profiler.measure_memory(
                MemoryTestType.HEAP_FRAGMENTATION,
                success=False, error=str(e)
            )
        
        finally:
            self.profiler.stop_profiling()
        
        return self._calculate_stats(MemoryTestType.HEAP_FRAGMENTATION)
    
    def benchmark_memory_leak(self, duration_seconds: int = 60) -> MemoryStats:
        """Benchmark for memory leaks"""
        print(f"Benchmarking memory leaks ({duration_seconds} seconds)...")
        
        self.profiler.start_profiling()
        
        start_time = time.time()
        iteration = 0
        
        try:
            while time.time() - start_time < duration_seconds:
                # Create and destroy objects
                objects = []
                for i in range(100):
                    obj = {
                        'data': f"leak_test_{i}" * 10,
                        'timestamp': time.time(),
                        'id': i
                    }
                    objects.append(obj)
                
                # Process objects
                for obj in objects:
                    _ = obj['data'] + str(obj['timestamp'])
                
                # Clear objects
                objects.clear()
                gc.collect()
                
                iteration += 1
                
                # Measure every 10 seconds
                if iteration % 10 == 0:
                    self.profiler.measure_memory(
                        MemoryTestType.MEMORY_LEAK,
                        metadata={'iteration': iteration, 'elapsed': time.time() - start_time}
                    )
            
            # Final measurement
            self.profiler.measure_memory(
                MemoryTestType.MEMORY_LEAK,
                metadata={'iterations': iteration, 'duration': duration_seconds}
            )
            
        except Exception as e:
            self.profiler.measure_memory(
                MemoryTestType.MEMORY_LEAK,
                success=False, error=str(e)
            )
        
        finally:
            self.profiler.stop_profiling()
        
        return self._calculate_stats(MemoryTestType.MEMORY_LEAK)
    
    def benchmark_pool_efficiency(self) -> MemoryStats:
        """Benchmark memory pool efficiency"""
        print("Benchmarking memory pool efficiency...")
        
        self.profiler.start_profiling()
        
        try:
            # Test memory pool efficiency
            memory_manager = get_edge_memory_manager()
            
            # Allocate and deallocate from pools
            blocks = []
            
            for i in range(50):
                # Allocate from different pools
                if i % 4 == 0:
                    block = memory_manager.allocate_message(caller="benchmark")
                elif i % 4 == 1:
                    block = memory_manager.allocate_agent(caller="benchmark")
                elif i % 4 == 2:
                    block = memory_manager.allocate_buffer(caller="benchmark")
                else:
                    block = memory_manager.allocate_temporary(caller="benchmark")
                
                if block:
                    blocks.append(block)
                
                # Measure every 10 allocations
                if (i + 1) % 10 == 0:
                    pool_stats = memory_manager.get_memory_stats()
                    self.profiler.measure_memory(
                        MemoryTestType.POOL_EFFICIENCY,
                        metadata={'allocations': i + 1, 'pool_stats': pool_stats}
                    )
            
            # Deallocate half the blocks
            for i in range(0, len(blocks), 2):
                memory_manager.deallocate(blocks[i])
            
            # Final measurement
            pool_stats = memory_manager.get_memory_stats()
            self.profiler.measure_memory(
                MemoryTestType.POOL_EFFICIENCY,
                metadata={'total_allocations': len(blocks), 'pool_stats': pool_stats}
            )
            
        except Exception as e:
            self.profiler.measure_memory(
                MemoryTestType.POOL_EFFICIENCY,
                success=False, error=str(e)
            )
        
        finally:
            self.profiler.stop_profiling()
        
        return self._calculate_stats(MemoryTestType.POOL_EFFICIENCY)
    
    def run_all_benchmarks(self) -> Dict[MemoryTestType, MemoryStats]:
        """Run all memory benchmarks"""
        print("Starting comprehensive memory benchmarking...")
        print("Targets:")
        print(f"  - Agent memory: <{self.targets[MemoryTestType.AGENT_MEMORY]} bytes")
        print(f"  - Message memory: <{self.targets[MemoryTestType.MESSAGE_MEMORY]} bytes")
        print(f"  - Flash usage: <{self.targets[MemoryTestType.FLASH_USAGE] // (1024*1024)}MB")
        print(f"  - Heap fragmentation: <{self.targets[MemoryTestType.HEAP_FRAGMENTATION]}%")
        print("-" * 50)
        
        results = {}
        
        # Run all benchmarks
        results[MemoryTestType.AGENT_MEMORY] = self.benchmark_agent_memory(100)
        results[MemoryTestType.MESSAGE_MEMORY] = self.benchmark_message_memory(1000)
        results[MemoryTestType.FLASH_USAGE] = self.benchmark_flash_usage()
        results[MemoryTestType.HEAP_FRAGMENTATION] = self.benchmark_heap_fragmentation(100)
        results[MemoryTestType.MEMORY_LEAK] = self.benchmark_memory_leak(30)  # 30 seconds for testing
        results[MemoryTestType.POOL_EFFICIENCY] = self.benchmark_pool_efficiency()
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _create_mock_agent(self, agent_id: str) -> Dict[str, Any]:
        """Create a mock agent for testing"""
        return {
            'id': agent_id,
            'state': 'active',
            'connections': [],
            'messages': [],
            'timestamp': time.time()
        }
    
    def _calculate_stats(self, test_type: MemoryTestType) -> MemoryStats:
        """Calculate statistics for a test type"""
        measurements = self.profiler.get_measurements(test_type)
        
        if not measurements:
            return MemoryStats(
                test_type=test_type,
                measurements=[],
                count=0,
                min_memory=0,
                max_memory=0,
                mean_memory=0,
                median_memory=0,
                memory_growth=0,
                fragmentation_avg=0,
                fragmentation_max=0,
                success_rate=0,
                target_met=False,
                target_memory=self.targets[test_type]
            )
        
        successful_measurements = [m for m in measurements if m.success]
        memory_usage = [m.memory_used for m in successful_measurements]
        fragmentations = [m.fragmentation for m in successful_measurements]
        
        if not memory_usage:
            return MemoryStats(
                test_type=test_type,
                measurements=measurements,
                count=len(measurements),
                min_memory=0,
                max_memory=0,
                mean_memory=0,
                median_memory=0,
                memory_growth=0,
                fragmentation_avg=0,
                fragmentation_max=0,
                success_rate=0,
                target_met=False,
                target_memory=self.targets[test_type]
            )
        
        target_memory = self.targets[test_type]
        memory_growth = memory_usage[-1] - memory_usage[0] if len(memory_usage) > 1 else 0
        
        # Check if target is met based on test type
        if test_type == MemoryTestType.HEAP_FRAGMENTATION:
            target_met = statistics.mean(fragmentations) <= target_memory
        elif test_type == MemoryTestType.MEMORY_LEAK:
            target_met = memory_growth <= target_memory
        else:
            target_met = statistics.mean(memory_usage) <= target_memory
        
        return MemoryStats(
            test_type=test_type,
            measurements=measurements,
            count=len(measurements),
            min_memory=min(memory_usage),
            max_memory=max(memory_usage),
            mean_memory=int(statistics.mean(memory_usage)),
            median_memory=int(statistics.median(memory_usage)),
            memory_growth=memory_growth,
            fragmentation_avg=statistics.mean(fragmentations) if fragmentations else 0,
            fragmentation_max=max(fragmentations) if fragmentations else 0,
            success_rate=(len(successful_measurements) / len(measurements)) * 100,
            target_met=target_met,
            target_memory=target_memory
        )
    
    def _print_summary(self, results: Dict[MemoryTestType, MemoryStats]):
        """Print benchmark summary"""
        print("\n" + "=" * 60)
        print("MEMORY BENCHMARK SUMMARY")
        print("=" * 60)
        
        for test_type, stats in results.items():
            status = "✅ PASS" if stats.target_met else "❌ FAIL"
            
            if test_type == MemoryTestType.HEAP_FRAGMENTATION:
                print(f"\n{test_type.value.upper().replace('_', ' ')}")
                print(f"  Target: {stats.target_memory:.1f}%")
                print(f"  Average: {stats.fragmentation_avg:.2f}%")
                print(f"  Maximum: {stats.fragmentation_max:.2f}%")
            elif test_type == MemoryTestType.MEMORY_LEAK:
                print(f"\n{test_type.value.upper().replace('_', ' ')}")
                print(f"  Target: {stats.target_memory} bytes growth")
                print(f"  Growth: {stats.memory_growth} bytes")
                print(f"  Mean: {stats.mean_memory} bytes")
            else:
                print(f"\n{test_type.value.upper().replace('_', ' ')}")
                print(f"  Target: {stats.target_memory} bytes")
                print(f"  Mean: {stats.mean_memory} bytes")
                print(f"  Min: {stats.min_memory} bytes")
                print(f"  Max: {stats.max_memory} bytes")
            
            print(f"  Success Rate: {stats.success_rate:.1f}%")
            print(f"  Status: {status}")
        
        # Overall assessment
        all_targets_met = all(stats.target_met for stats in results.values())
        overall_status = "✅ ALL TARGETS MET" if all_targets_met else "❌ SOME TARGETS MISSED"
        
        print(f"\n{'=' * 60}")
        print(f"OVERALL STATUS: {overall_status}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    import time
    
    # Run memory benchmarks
    benchmark = MemoryBenchmark()
    results = benchmark.run_all_benchmarks()
