#!/usr/bin/env python3
"""
Comprehensive µACP Library Testing Script

This script tests every function in the miuACP library to ensure
complete functionality and integration.
"""

import sys
import os
import time
import traceback
import importlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Add miuACP to path
sys.path.insert(0, '/home/arnab/Projects/agentcom/miuACP/src')

@dataclass
class TestResult:
    """Test result for a single function"""
    module_name: str
    function_name: str
    success: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0
    return_value: Any = None

class TestStatus(Enum):
    """Test status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"

class ComprehensiveLibraryTester:
    """Comprehensive tester for the entire µACP library"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.test_status: Dict[str, TestStatus] = {}
        self.start_time = time.time()
        
        # Define test modules and their functions
        self.test_modules = {
            # Core Protocol
            'miuacp.protocol': [
                'Message', 'MessageType', 'MessagePriority', 'MessageHeader',
                'create_message', 'parse_message', 'validate_message',
                'serialize_message', 'deserialize_message'
            ],
            'miuacp.client': [
                'Client', 'connect', 'disconnect', 'send_message',
                'receive_message', 'subscribe', 'unsubscribe'
            ],
            'miuacp.server': [
                'Server', 'start', 'stop', 'handle_message',
                'route_message', 'manage_connections'
            ],
            'miuacp.agent': [
                'Agent', 'start_agent', 'stop_agent', 'process_message',
                'update_state', 'handle_events'
            ],
            'miuacp.discovery': [
                'ServiceDiscovery', 'discover_services', 'register_service',
                'unregister_service', 'find_peers'
            ],
            'miuacp.utils': [
                'generate_id', 'calculate_checksum', 'compress_data',
                'decompress_data', 'encrypt_data', 'decrypt_data'
            ],
            
            # Edge Optimization
            'miuacp.edge.memory_pool': [
                'MemoryPool', 'EdgeMemoryManager', 'PooledObject',
                'allocate_memory', 'deallocate_memory', 'get_pool_stats'
            ],
            'miuacp.edge.flash_optimizer': [
                'FlashOptimizer', 'EdgeStorageManager', 'FlashBlockType',
                'CompressionType', 'optimize_flash', 'compress_data',
                'decompress_data', 'wear_leveling'
            ],
            'miuacp.edge.heap_manager': [
                'HeapManager', 'FragmentationPreventer', 'HeapBlockType',
                'manage_heap', 'prevent_fragmentation', 'get_heap_stats'
            ],
            'miuacp.edge.resource_constraints': [
                'ResourceConstraints', 'EdgeDeviceProfile', 'DeviceType',
                'check_constraints', 'optimize_resources'
            ],
            
            # Novel Behaviors
            'miuacp.behaviors.swarm_coordination': [
                'SwarmCoordinator', 'DecentralizedCoordination',
                'coordinate_swarm', 'handle_topology_change'
            ],
            'miuacp.behaviors.edge_rl': [
                'EdgeRLCollective', 'DistributedLearning', 'EdgeRLAgent',
                'RLAlgorithm', 'train_agent', 'update_policy'
            ],
            'miuacp.behaviors.anomaly_detection': [
                'AnomalyDetector', 'CollaborativeAlerting', 'StatisticalDetector',
                'BehavioralDetector', 'AnomalyType', 'AlertLevel',
                'detect_anomaly', 'collaborate_alert'
            ],
            
            # Formal Verification
            'miuacp.verification.security_verification': [
                'SecurityVerifier', 'verify_authentication', 'verify_confidentiality',
                'verify_integrity', 'generate_proof'
            ],
            'miuacp.verification.protocol_verification': [
                'ProtocolVerifier', 'verify_deadlock_freedom', 'verify_liveness',
                'verify_termination', 'verify_message_ordering'
            ],
            'miuacp.verification.model_checker': [
                'ModelChecker', 'check_property', 'explore_states',
                'generate_counterexample'
            ],
            'miuacp.verification.specification_generator': [
                'SpecificationGenerator', 'generate_tla_spec', 'generate_coq_spec',
                'generate_alloy_spec', 'generate_promela_spec'
            ]
        }
    
    def test_module_import(self, module_name: str) -> bool:
        """Test if a module can be imported"""
        try:
            importlib.import_module(module_name)
            return True
        except Exception as e:
            print(f"❌ Failed to import {module_name}: {e}")
            return False
    
    def test_function_execution(self, module_name: str, function_name: str) -> TestResult:
        """Test execution of a specific function"""
        start_time = time.time()
        
        try:
            # Import the module
            module = importlib.import_module(module_name)
            
            # Get the function/class
            if hasattr(module, function_name):
                func_or_class = getattr(module, function_name)
                
                # Test different types of objects
                if callable(func_or_class):
                    # Test function or class constructor
                    if isinstance(func_or_class, type):
                        # It's a class, test instantiation
                        try:
                            # Try to instantiate with minimal parameters
                            if function_name in ['Message', 'MessageType', 'MessagePriority']:
                                instance = func_or_class()
                            elif function_name in ['Client', 'Server', 'Agent']:
                                instance = func_or_class("test_id")
                            elif function_name in ['MemoryPool', 'FlashOptimizer', 'HeapManager']:
                                instance = func_or_class()
                            else:
                                instance = func_or_class()
                            
                            execution_time = time.time() - start_time
                            return TestResult(
                                module_name=module_name,
                                function_name=function_name,
                                success=True,
                                execution_time=execution_time,
                                return_value=f"Class instantiated: {type(instance).__name__}"
                            )
                        except Exception as e:
                            # If instantiation fails, try calling as function
                            try:
                                result = func_or_class()
                                execution_time = time.time() - start_time
                                return TestResult(
                                    module_name=module_name,
                                    function_name=function_name,
                                    success=True,
                                    execution_time=execution_time,
                                    return_value=f"Function result: {type(result).__name__}"
                                )
                            except Exception as e2:
                                execution_time = time.time() - start_time
                                return TestResult(
                                    module_name=module_name,
                                    function_name=function_name,
                                    success=False,
                                    error_message=f"Both instantiation and function call failed: {e}, {e2}",
                                    execution_time=execution_time
                                )
                    else:
                        # It's a function, try to call it
                        try:
                            result = func_or_class()
                            execution_time = time.time() - start_time
                            return TestResult(
                                module_name=module_name,
                                function_name=function_name,
                                success=True,
                                execution_time=execution_time,
                                return_value=f"Function result: {type(result).__name__}"
                            )
                        except Exception as e:
                            execution_time = time.time() - start_time
                            return TestResult(
                                module_name=module_name,
                                function_name=function_name,
                                success=False,
                                error_message=f"Function call failed: {e}",
                                execution_time=execution_time
                            )
                else:
                    # It's a constant or enum
                    execution_time = time.time() - start_time
                    return TestResult(
                        module_name=module_name,
                        function_name=function_name,
                        success=True,
                        execution_time=execution_time,
                        return_value=f"Constant/Enum: {func_or_class}"
                    )
            else:
                execution_time = time.time() - start_time
                return TestResult(
                    module_name=module_name,
                    function_name=function_name,
                    success=False,
                    error_message=f"Function/class {function_name} not found in module",
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                module_name=module_name,
                function_name=function_name,
                success=False,
                error_message=f"Module import or execution failed: {e}",
                execution_time=execution_time
            )
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive tests on all modules and functions"""
        print("🚀 Starting Comprehensive µACP Library Testing")
        print("=" * 60)
        
        total_tests = sum(len(functions) for functions in self.test_modules.values())
        current_test = 0
        passed_tests = 0
        failed_tests = 0
        
        for module_name, functions in self.test_modules.items():
            print(f"\n📦 Testing Module: {module_name}")
            print("-" * 40)
            
            # Test module import first
            if not self.test_module_import(module_name):
                print(f"❌ Module {module_name} failed to import - skipping all functions")
                for func_name in functions:
                    current_test += 1
                    failed_tests += 1
                    result = TestResult(
                        module_name=module_name,
                        function_name=func_name,
                        success=False,
                        error_message="Module import failed"
                    )
                    self.results.append(result)
                continue
            
            print(f"✅ Module {module_name} imported successfully")
            
            # Test each function in the module
            for func_name in functions:
                current_test += 1
                print(f"  🧪 Testing {func_name} ({current_test}/{total_tests})")
                
                result = self.test_function_execution(module_name, func_name)
                self.results.append(result)
                
                if result.success:
                    passed_tests += 1
                    print(f"    ✅ {func_name}: PASSED ({result.execution_time:.3f}s)")
                else:
                    failed_tests += 1
                    print(f"    ❌ {func_name}: FAILED - {result.error_message}")
        
        # Calculate final statistics
        total_time = time.time() - self.start_time
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print("\n" + "=" * 60)
        print("🎯 COMPREHENSIVE TEST RESULTS")
        print("=" * 60)
        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        print(f"⏱️  Total Time: {total_time:.2f} seconds")
        
        # Group results by module
        module_results = {}
        for result in self.results:
            if result.module_name not in module_results:
                module_results[result.module_name] = {'passed': 0, 'failed': 0, 'total': 0}
            module_results[result.module_name]['total'] += 1
            if result.success:
                module_results[result.module_name]['passed'] += 1
            else:
                module_results[result.module_name]['failed'] += 1
        
        print(f"\n📋 Module Results:")
        for module, stats in module_results.items():
            module_success_rate = (stats['passed'] / stats['total']) * 100 if stats['total'] > 0 else 0
            status = "✅" if module_success_rate == 100 else "⚠️" if module_success_rate >= 80 else "❌"
            print(f"  {status} {module}: {stats['passed']}/{stats['total']} ({module_success_rate:.1f}%)")
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': success_rate,
            'total_time': total_time,
            'module_results': module_results,
            'detailed_results': self.results
        }
    
    def generate_report(self, results: Dict[str, Any], output_file: str = "library_test_report.txt"):
        """Generate detailed test report"""
        with open(output_file, 'w') as f:
            f.write("µACP COMPREHENSIVE LIBRARY TEST REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Tests: {results['total_tests']}\n")
            f.write(f"Passed: {results['passed_tests']}\n")
            f.write(f"Failed: {results['failed_tests']}\n")
            f.write(f"Success Rate: {results['success_rate']:.1f}%\n")
            f.write(f"Total Time: {results['total_time']:.2f} seconds\n\n")
            
            f.write("MODULE RESULTS:\n")
            f.write("-" * 20 + "\n")
            for module, stats in results['module_results'].items():
                module_success_rate = (stats['passed'] / stats['total']) * 100 if stats['total'] > 0 else 0
                f.write(f"{module}: {stats['passed']}/{stats['total']} ({module_success_rate:.1f}%)\n")
            
            f.write("\nDETAILED RESULTS:\n")
            f.write("-" * 20 + "\n")
            for result in results['detailed_results']:
                status = "PASS" if result.success else "FAIL"
                f.write(f"{status}: {result.module_name}.{result.function_name}\n")
                if not result.success and result.error_message:
                    f.write(f"  Error: {result.error_message}\n")
                f.write(f"  Time: {result.execution_time:.3f}s\n\n")
        
        print(f"📄 Detailed report saved to: {output_file}")

def main():
    """Main function to run comprehensive library testing"""
    print("🧪 µACP Comprehensive Library Tester")
    print("Testing every function in the miuACP library")
    print("=" * 60)
    
    tester = ComprehensiveLibraryTester()
    results = tester.run_comprehensive_tests()
    
    # Generate report
    timestamp = int(time.time())
    report_file = f"results/library_test_report_{timestamp}.txt"
    os.makedirs("results", exist_ok=True)
    tester.generate_report(results, report_file)
    
    # Save JSON results
    import json
    json_file = f"results/library_test_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        # Convert results to JSON-serializable format
        json_results = {
            'total_tests': results['total_tests'],
            'passed_tests': results['passed_tests'],
            'failed_tests': results['failed_tests'],
            'success_rate': results['success_rate'],
            'total_time': results['total_time'],
            'module_results': results['module_results'],
            'detailed_results': [
                {
                    'module_name': r.module_name,
                    'function_name': r.function_name,
                    'success': r.success,
                    'error_message': r.error_message,
                    'execution_time': r.execution_time,
                    'return_value': str(r.return_value) if r.return_value is not None else None
                }
                for r in results['detailed_results']
            ]
        }
        json.dump(json_results, f, indent=2)
    
    print(f"📊 JSON results saved to: {json_file}")
    
    # Final status
    if results['success_rate'] >= 90:
        print("\n🎉 EXCELLENT: Library is highly functional!")
    elif results['success_rate'] >= 80:
        print("\n✅ GOOD: Library is mostly functional with minor issues")
    elif results['success_rate'] >= 70:
        print("\n⚠️  FAIR: Library has some issues that need attention")
    else:
        print("\n❌ POOR: Library has significant issues that need fixing")
    
    return results['success_rate']

if __name__ == "__main__":
    success_rate = main()
    sys.exit(0 if success_rate >= 80 else 1)
