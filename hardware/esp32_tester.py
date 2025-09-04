"""
ESP32-C3 Hardware Testing for µACP

Implements hardware testing capabilities for ESP32-C3 devices including:
- Device connectivity testing
- Performance measurement
- Memory usage validation
- Real-time testing
"""

import time
import serial
import threading
import subprocess
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import os


class TestType(Enum):
    """Types of hardware tests"""
    CONNECTIVITY = "connectivity"
    PERFORMANCE = "performance"
    MEMORY = "memory"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    STRESS = "stress"
    ENDURANCE = "endurance"


class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ESP32Config:
    """ESP32-C3 configuration"""
    port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    timeout: float = 5.0
    device_id: str = "ESP32-C3"
    flash_size: int = 4 * 1024 * 1024  # 4MB
    sram_size: int = 520 * 1024  # 520KB
    cpu_frequency: int = 160  # MHz
    wifi_enabled: bool = True
    bluetooth_enabled: bool = True


@dataclass
class ESP32TestResult:
    """ESP32 test result"""
    test_type: TestType
    test_name: str
    status: TestStatus
    start_time: float
    end_time: float
    duration: float
    success: bool
    metrics: Dict[str, Any]
    error_message: Optional[str] = None
    raw_data: List[str] = field(default_factory=list)


class ESP32Tester:
    """
    ESP32-C3 hardware tester for µACP
    
    Provides comprehensive testing capabilities for ESP32-C3 devices
    running µACP protocol implementation.
    """
    
    def __init__(self, config: ESP32Config = None):
        self.config = config or ESP32Config()
        self.serial_connection: Optional[serial.Serial] = None
        self.test_results: List[ESP32TestResult] = []
        self.is_connected = False
        self.test_thread: Optional[threading.Thread] = None
        self.stop_testing = False
    
    def connect(self) -> bool:
        """Connect to ESP32-C3 device"""
        try:
            self.serial_connection = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baudrate,
                timeout=self.config.timeout
            )
            
            # Wait for connection to stabilize
            time.sleep(2)
            
            # Test connection
            if self._test_connection():
                self.is_connected = True
                return True
            else:
                self.disconnect()
                return False
                
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from ESP32-C3 device"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        self.is_connected = False
    
    def _test_connection(self) -> bool:
        """Test basic connection to device"""
        try:
            # Send test command
            self._send_command("ping")
            response = self._read_response(timeout=2.0)
            return "pong" in response.lower()
        except Exception:
            return False
    
    def _send_command(self, command: str):
        """Send command to ESP32-C3"""
        if not self.serial_connection or not self.serial_connection.is_open:
            raise Exception("Not connected to device")
        
        command_bytes = (command + "\n").encode('utf-8')
        self.serial_connection.write(command_bytes)
        self.serial_connection.flush()
    
    def _read_response(self, timeout: float = None) -> str:
        """Read response from ESP32-C3"""
        if not self.serial_connection or not self.serial_connection.is_open:
            raise Exception("Not connected to device")
        
        if timeout:
            self.serial_connection.timeout = timeout
        
        response = self.serial_connection.readline().decode('utf-8').strip()
        return response
    
    def run_connectivity_test(self) -> ESP32TestResult:
        """Run connectivity test"""
        test_name = "ESP32 Connectivity Test"
        start_time = time.time()
        
        result = ESP32TestResult(
            test_type=TestType.CONNECTIVITY,
            test_name=test_name,
            status=TestStatus.RUNNING,
            start_time=start_time,
            end_time=0,
            duration=0,
            success=False,
            metrics={}
        )
        
        try:
            # Test basic connectivity
            self._send_command("status")
            status_response = self._read_response()
            
            # Test WiFi connectivity
            self._send_command("wifi_status")
            wifi_response = self._read_response()
            
            # Test Bluetooth connectivity
            self._send_command("bt_status")
            bt_response = self._read_response()
            
            # Parse results
            connectivity_ok = "connected" in status_response.lower()
            wifi_ok = "connected" in wifi_response.lower()
            bt_ok = "enabled" in bt_response.lower()
            
            result.metrics = {
                'basic_connectivity': connectivity_ok,
                'wifi_connectivity': wifi_ok,
                'bluetooth_connectivity': bt_ok,
                'status_response': status_response,
                'wifi_response': wifi_response,
                'bt_response': bt_response
            }
            
            result.success = connectivity_ok
            result.status = TestStatus.COMPLETED
            
        except Exception as e:
            result.error_message = str(e)
            result.status = TestStatus.FAILED
        
        result.end_time = time.time()
        result.duration = result.end_time - result.start_time
        self.test_results.append(result)
        
        return result
    
    def run_performance_test(self) -> ESP32TestResult:
        """Run performance test"""
        test_name = "ESP32 Performance Test"
        start_time = time.time()
        
        result = ESP32TestResult(
            test_type=TestType.PERFORMANCE,
            test_name=test_name,
            status=TestStatus.RUNNING,
            start_time=start_time,
            end_time=0,
            duration=0,
            success=False,
            metrics={}
        )
        
        try:
            # Test CPU performance
            self._send_command("cpu_benchmark")
            cpu_response = self._read_response()
            
            # Test memory performance
            self._send_command("memory_benchmark")
            memory_response = self._read_response()
            
            # Test network performance
            self._send_command("network_benchmark")
            network_response = self._read_response()
            
            # Parse performance metrics
            cpu_metrics = self._parse_performance_metrics(cpu_response)
            memory_metrics = self._parse_performance_metrics(memory_response)
            network_metrics = self._parse_performance_metrics(network_response)
            
            result.metrics = {
                'cpu_performance': cpu_metrics,
                'memory_performance': memory_metrics,
                'network_performance': network_metrics,
                'raw_cpu_response': cpu_response,
                'raw_memory_response': memory_response,
                'raw_network_response': network_response
            }
            
            # Determine success based on performance thresholds
            result.success = self._evaluate_performance_success(result.metrics)
            result.status = TestStatus.COMPLETED
            
        except Exception as e:
            result.error_message = str(e)
            result.status = TestStatus.FAILED
        
        result.end_time = time.time()
        result.duration = result.end_time - result.start_time
        self.test_results.append(result)
        
        return result
    
    def run_memory_test(self) -> ESP32TestResult:
        """Run memory usage test"""
        test_name = "ESP32 Memory Test"
        start_time = time.time()
        
        result = ESP32TestResult(
            test_type=TestType.MEMORY,
            test_name=test_name,
            status=TestStatus.RUNNING,
            start_time=start_time,
            end_time=0,
            duration=0,
            success=False,
            metrics={}
        )
        
        try:
            # Get memory usage
            self._send_command("memory_usage")
            memory_response = self._read_response()
            
            # Get heap information
            self._send_command("heap_info")
            heap_response = self._read_response()
            
            # Get flash usage
            self._send_command("flash_usage")
            flash_response = self._read_response()
            
            # Parse memory metrics
            memory_metrics = self._parse_memory_metrics(memory_response)
            heap_metrics = self._parse_memory_metrics(heap_response)
            flash_metrics = self._parse_memory_metrics(flash_response)
            
            result.metrics = {
                'memory_usage': memory_metrics,
                'heap_usage': heap_metrics,
                'flash_usage': flash_metrics,
                'total_sram': self.config.sram_size,
                'total_flash': self.config.flash_size,
                'raw_memory_response': memory_response,
                'raw_heap_response': heap_response,
                'raw_flash_response': flash_response
            }
            
            # Check memory constraints
            result.success = self._evaluate_memory_success(result.metrics)
            result.status = TestStatus.COMPLETED
            
        except Exception as e:
            result.error_message = str(e)
            result.status = TestStatus.FAILED
        
        result.end_time = time.time()
        result.duration = result.end_time - result.start_time
        self.test_results.append(result)
        
        return result
    
    def run_latency_test(self, iterations: int = 100) -> ESP32TestResult:
        """Run latency test"""
        test_name = "ESP32 Latency Test"
        start_time = time.time()
        
        result = ESP32TestResult(
            test_type=TestType.LATENCY,
            test_name=test_name,
            status=TestStatus.RUNNING,
            start_time=start_time,
            end_time=0,
            duration=0,
            success=False,
            metrics={}
        )
        
        try:
            latencies = []
            
            for i in range(iterations):
                # Measure round-trip latency
                cmd_start = time.time()
                self._send_command(f"ping_{i}")
                response = self._read_response()
                cmd_end = time.time()
                
                latency = (cmd_end - cmd_start) * 1000  # Convert to milliseconds
                latencies.append(latency)
                
                # Small delay between iterations
                time.sleep(0.01)
            
            # Calculate statistics
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            result.metrics = {
                'iterations': iterations,
                'average_latency_ms': avg_latency,
                'min_latency_ms': min_latency,
                'max_latency_ms': max_latency,
                'all_latencies': latencies,
                'target_latency_ms': 1.0  # Target: <1ms
            }
            
            # Success if average latency is below target
            result.success = avg_latency < 1.0
            result.status = TestStatus.COMPLETED
            
        except Exception as e:
            result.error_message = str(e)
            result.status = TestStatus.FAILED
        
        result.end_time = time.time()
        result.duration = result.end_time - result.start_time
        self.test_results.append(result)
        
        return result
    
    def run_stress_test(self, duration_seconds: int = 60) -> ESP32TestResult:
        """Run stress test"""
        test_name = "ESP32 Stress Test"
        start_time = time.time()
        
        result = ESP32TestResult(
            test_type=TestType.STRESS,
            test_name=test_name,
            status=TestStatus.RUNNING,
            start_time=start_time,
            end_time=0,
            duration=0,
            success=False,
            metrics={}
        )
        
        try:
            commands_sent = 0
            responses_received = 0
            errors = 0
            end_time = start_time + duration_seconds
            
            while time.time() < end_time and not self.stop_testing:
                try:
                    # Send random command
                    command = f"stress_test_{commands_sent}"
                    self._send_command(command)
                    
                    # Try to read response
                    response = self._read_response(timeout=1.0)
                    if response:
                        responses_received += 1
                    else:
                        errors += 1
                    
                    commands_sent += 1
                    
                    # Small delay
                    time.sleep(0.1)
                    
                except Exception:
                    errors += 1
                    commands_sent += 1
            
            result.metrics = {
                'duration_seconds': duration_seconds,
                'commands_sent': commands_sent,
                'responses_received': responses_received,
                'errors': errors,
                'success_rate': responses_received / max(1, commands_sent),
                'commands_per_second': commands_sent / duration_seconds
            }
            
            # Success if success rate is above 90%
            result.success = result.metrics['success_rate'] > 0.9
            result.status = TestStatus.COMPLETED
            
        except Exception as e:
            result.error_message = str(e)
            result.status = TestStatus.FAILED
        
        result.end_time = time.time()
        result.duration = result.end_time - result.start_time
        self.test_results.append(result)
        
        return result
    
    def run_comprehensive_test_suite(self) -> List[ESP32TestResult]:
        """Run comprehensive test suite"""
        if not self.is_connected:
            if not self.connect():
                return []
        
        test_results = []
        
        # Run all tests
        tests = [
            self.run_connectivity_test,
            self.run_performance_test,
            self.run_memory_test,
            self.run_latency_test,
            self.run_stress_test
        ]
        
        for test_func in tests:
            try:
                result = test_func()
                test_results.append(result)
                
                # Stop if critical test fails
                if not result.success and result.test_type in [TestType.CONNECTIVITY, TestType.MEMORY]:
                    break
                    
            except Exception as e:
                print(f"Test {test_func.__name__} failed: {e}")
        
        return test_results
    
    def _parse_performance_metrics(self, response: str) -> Dict[str, Any]:
        """Parse performance metrics from response"""
        metrics = {}
        
        try:
            # Try to parse JSON response
            if response.startswith('{'):
                metrics = json.loads(response)
            else:
                # Parse text response
                lines = response.split('\n')
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower().replace(' ', '_')
                        try:
                            metrics[key] = float(value.strip())
                        except ValueError:
                            metrics[key] = value.strip()
        except Exception:
            metrics['raw_response'] = response
        
        return metrics
    
    def _parse_memory_metrics(self, response: str) -> Dict[str, Any]:
        """Parse memory metrics from response"""
        metrics = {}
        
        try:
            # Try to parse JSON response
            if response.startswith('{'):
                metrics = json.loads(response)
            else:
                # Parse text response for memory information
                lines = response.split('\n')
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower().replace(' ', '_')
                        try:
                            # Extract numeric values
                            numeric_value = ''.join(filter(str.isdigit, value))
                            if numeric_value:
                                metrics[key] = int(numeric_value)
                            else:
                                metrics[key] = value.strip()
                        except ValueError:
                            metrics[key] = value.strip()
        except Exception:
            metrics['raw_response'] = response
        
        return metrics
    
    def _evaluate_performance_success(self, metrics: Dict[str, Any]) -> bool:
        """Evaluate if performance test was successful"""
        # Check CPU performance
        cpu_perf = metrics.get('cpu_performance', {})
        if 'cpu_usage' in cpu_perf and cpu_perf['cpu_usage'] > 90:
            return False
        
        # Check memory performance
        memory_perf = metrics.get('memory_performance', {})
        if 'memory_usage' in memory_perf and memory_perf['memory_usage'] > 80:
            return False
        
        return True
    
    def _evaluate_memory_success(self, metrics: Dict[str, Any]) -> bool:
        """Evaluate if memory test was successful"""
        # Check SRAM usage
        memory_usage = metrics.get('memory_usage', {})
        if 'used_sram' in memory_usage:
            used_sram = memory_usage['used_sram']
            if used_sram > self.config.sram_size * 0.9:  # 90% threshold
                return False
        
        # Check flash usage
        flash_usage = metrics.get('flash_usage', {})
        if 'used_flash' in flash_usage:
            used_flash = flash_usage['used_flash']
            if used_flash > self.config.flash_size * 0.9:  # 90% threshold
                return False
        
        return True
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get test summary"""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result.success)
        
        test_types = {}
        for result in self.test_results:
            test_type = result.test_type.value
            test_types[test_type] = test_types.get(test_type, 0) + 1
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': successful_tests / max(1, total_tests),
            'test_types': test_types,
            'connection_status': self.is_connected,
            'device_config': {
                'port': self.config.port,
                'baudrate': self.config.baudrate,
                'sram_size': self.config.sram_size,
                'flash_size': self.config.flash_size
            }
        }
    
    def export_test_results(self, filename: str):
        """Export test results to file"""
        results_data = {
            'test_summary': self.get_test_summary(),
            'test_results': [
                {
                    'test_type': result.test_type.value,
                    'test_name': result.test_name,
                    'status': result.status.value,
                    'success': result.success,
                    'duration': result.duration,
                    'metrics': result.metrics,
                    'error_message': result.error_message
                }
                for result in self.test_results
            ],
            'timestamp': time.time()
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
