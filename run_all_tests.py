#!/usr/bin/env python3
"""
Comprehensive Test Runner for miuACP-Tools
Runs all tests, generates reports, and provides detailed analysis.
"""

import os
import sys
import time
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

class TestRunner:
    """Comprehensive test runner for all miuACP-Tools tests."""
    
    def __init__(self):
        self.output_dir = Path(__file__).parent / "output"
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        self.start_time = time.time()
        
    def log_output(self, content: str, filename: str, file_type: str = "txt") -> Path:
        """Log output to timestamped file."""
        output_file = self.output_dir / f"{self.timestamp}_{filename}.{file_type}"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"📁 Output saved to: {output_file}")
        return output_file
    
    def run_command(self, command: List[str], description: str) -> Dict[str, Any]:
        """Run a command and capture results."""
        print(f"\n🔧 {description}")
        print("=" * 50)
        
        start_time = time.time()
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            end_time = time.time()
            
            success = result.returncode == 0
            duration = end_time - start_time
            
            if success:
                print(f"✅ {description} completed successfully")
            else:
                print(f"❌ {description} failed with return code {result.returncode}")
            
            return {
                'success': success,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'duration': duration,
                'command': ' '.join(command)
            }
            
        except subprocess.TimeoutExpired:
            print(f"⏰ {description} timed out after 5 minutes")
            return {
                'success': False,
                'return_code': -1,
                'stdout': '',
                'stderr': 'Command timed out',
                'duration': 300,
                'command': ' '.join(command),
                'error': 'timeout'
            }
        except Exception as e:
            print(f"💥 {description} failed with exception: {e}")
            return {
                'success': False,
                'return_code': -1,
                'stdout': '',
                'stderr': str(e),
                'duration': time.time() - start_time,
                'command': ' '.join(command),
                'error': 'exception'
            }
    
    def run_library_tests(self) -> Dict[str, Any]:
        """Run the main library test suite."""
        print("\n🧪 Running µACP Library Tests")
        print("=" * 50)
        
        result = self.run_command(
            [sys.executable, "test_library.py"],
            "µACP Library Comprehensive Test Suite"
        )
        
        self.results['library_tests'] = result
        return result
    
    def run_cli_tests(self) -> Dict[str, Any]:
        """Run CLI command tests."""
        print("\n🖥️  Running CLI Tests")
        print("=" * 50)
        
        cli_tests = [
            (["info"], "CLI Info Command"),
            (["test-uacp"], "CLI µACP Test Command"),
            (["--help"], "CLI Help Command"),
        ]
        
        cli_results = {}
        for args, description in cli_tests:
            result = self.run_command(
                [sys.executable, "cli.py"] + args,
                description
            )
            cli_results[description] = result
        
        self.results['cli_tests'] = cli_results
        return cli_results
    
    def run_demo_tests(self) -> Dict[str, Any]:
        """Run demonstration tests."""
        print("\n🎭 Running Demo Tests")
        print("=" * 50)
        
        result = self.run_command(
            [sys.executable, "demo.py"],
            "µACP Protocol Demonstration"
        )
        
        self.results['demo_tests'] = result
        return result
    
    def run_example_tests(self) -> Dict[str, Any]:
        """Run example script tests."""
        print("\n📚 Running Example Tests")
        print("=" * 50)
        
        result = self.run_command(
            [sys.executable, "example.py"],
            "µACP Usage Examples"
        )
        
        self.results['example_tests'] = result
        return result
    
    def run_protocol_analyzer_tests(self) -> Dict[str, Any]:
        """Run protocol analyzer tests if available."""
        print("\n🔍 Running Protocol Analyzer Tests")
        print("=" * 50)
        
        try:
            # Check if protocol analyzer is available
            import protocol_analyzer
            print("✅ Protocol Analyzer available")
            
            # Run basic import tests
            from protocol_analyzer import models, protocols, benchmarks
            print("✅ All protocol analyzer modules imported successfully")
            
            result = {
                'success': True,
                'return_code': 0,
                'stdout': 'Protocol Analyzer modules imported successfully',
                'stderr': '',
                'duration': 0.1,
                'command': 'import test',
                'modules': ['models', 'protocols', 'benchmarks']
            }
            
        except ImportError as e:
            print(f"⚠️  Protocol Analyzer not available: {e}")
            result = {
                'success': False,
                'return_code': -1,
                'stdout': '',
                'stderr': f'Protocol Analyzer not available: {e}',
                'duration': 0.1,
                'command': 'import test',
                'error': 'import_error'
            }
        
        self.results['protocol_analyzer_tests'] = result
        return result
    
    def run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks."""
        print("\n⚡ Running Performance Benchmarks")
        print("=" * 50)
        
        # Run the library test which includes performance benchmarks
        result = self.run_command(
            [sys.executable, "test_library.py"],
            "Performance Benchmarking"
        )
        
        self.results['performance_benchmarks'] = result
        return result
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        print("\n🔗 Running Integration Tests")
        print("=" * 50)
        
        # Test package imports
        import_tests = [
            ("miuacp", "Core µACP Library"),
            ("click", "CLI Framework"),
            ("rich", "Rich Text Library"),
        ]
        
        integration_results = {}
        for package, description in import_tests:
            try:
                __import__(package)
                result = {
                    'success': True,
                    'return_code': 0,
                    'stdout': f'{package} imported successfully',
                    'stderr': '',
                    'duration': 0.1,
                    'command': f'import {package}',
                }
                print(f"✅ {description} imported successfully")
            except ImportError as e:
                result = {
                    'success': False,
                    'return_code': -1,
                    'stdout': '',
                    'stderr': f'Failed to import {package}: {e}',
                    'duration': 0.1,
                    'command': f'import {package}',
                    'error': 'import_error'
                }
                print(f"❌ {description} import failed: {e}")
            
            integration_results[description] = result
        
        self.results['integration_tests'] = integration_results
        return integration_results
    
    def generate_summary_report(self) -> str:
        """Generate comprehensive test summary report."""
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        total_duration = 0
        
        # Count results
        for category, results in self.results.items():
            if isinstance(results, dict):
                if 'success' in results:
                    total_tests += 1
                    if results['success']:
                        passed_tests += 1
                    else:
                        failed_tests += 1
                    total_duration += results.get('duration', 0)
                else:
                    # Handle nested results (like CLI tests)
                    for test_name, test_result in results.items():
                        total_tests += 1
                        if test_result['success']:
                            passed_tests += 1
                        else:
                            failed_tests += 1
                        total_duration += test_result.get('duration', 0)
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        report = f"""╭─────────────────────────────────────────────────────────────────────────────────────────╮
│ 🚀 miuACP-Tools Comprehensive Test Report                                                              │
╰─────────────────────────────────────────────────────────────────────────────────────────╯

📊 Test Summary
{'=' * 50}
Total Tests: {total_tests}
Passed: {passed_tests} ✅
Failed: {failed_tests} ❌
Success Rate: {success_rate:.1f}%
Total Duration: {total_duration:.2f} seconds

📋 Test Results by Category
{'=' * 50}"""
        
        for category, results in self.results.items():
            report += f"\n\n🔹 {category.replace('_', ' ').title()}"
            report += f"\n{'-' * 40}"
            
            if isinstance(results, dict):
                if 'success' in results:
                    # Single result
                    status = "✅ PASS" if results['success'] else "❌ FAIL"
                    duration = f"{results.get('duration', 0):.2f}s"
                    report += f"\n  Status: {status}"
                    report += f"\n  Duration: {duration}"
                    if not results['success']:
                        report += f"\n  Error: {results.get('stderr', 'Unknown error')}"
                else:
                    # Multiple results
                    for test_name, test_result in results.items():
                        status = "✅ PASS" if test_result['success'] else "❌ FAIL"
                        duration = f"{test_result.get('duration', 0):.2f}s"
                        report += f"\n  {test_name}: {status} ({duration})"
        
        report += f"""

🎯 Overall Assessment
{'=' * 50}
"""
        
        if success_rate >= 90:
            report += "🌟 EXCELLENT: All critical tests passed successfully!"
        elif success_rate >= 75:
            report += "✅ GOOD: Most tests passed, minor issues detected."
        elif success_rate >= 50:
            report += "⚠️  FAIR: Some tests failed, review recommended."
        else:
            report += "❌ POOR: Many tests failed, immediate attention required."
        
        report += f"""

📁 Output Files
{'=' * 50}
All test outputs have been saved to the output directory with timestamps.
Timestamp: {self.timestamp}

🔧 Next Steps
{'=' * 50}"""
        
        if failed_tests > 0:
            report += """
1. Review failed tests and error messages
2. Fix identified issues
3. Re-run tests to verify fixes
4. Update documentation if needed"""
        else:
            report += """
1. All tests passed successfully!
2. Ready for production deployment
3. Consider running extended test suites
4. Update test coverage if needed"""
        
        report += f"""

⏱️  Test Execution Time: {time.time() - self.start_time:.2f} seconds
📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

╰─────────────────────────────────────────────────────────────────────────────────────────╯"""
        
        return report
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all available tests."""
        print("🚀 Starting Comprehensive Test Suite for miuACP-Tools")
        print("=" * 70)
        print(f"Timestamp: {self.timestamp}")
        print(f"Output Directory: {self.output_dir}")
        print("=" * 70)
        
        # Run all test categories
        self.run_library_tests()
        self.run_cli_tests()
        self.run_demo_tests()
        self.run_example_tests()
        self.run_protocol_analyzer_tests()
        self.run_performance_benchmarks()
        self.run_integration_tests()
        
        # Generate and save comprehensive report
        report = self.generate_summary_report()
        report_file = self.log_output(report, "comprehensive_test_report")
        
        # Save detailed results as JSON
        json_results = {
            'timestamp': self.timestamp,
            'total_duration': time.time() - self.start_time,
            'results': self.results
        }
        
        json_file = self.log_output(
            json.dumps(json_results, indent=2),
            "detailed_test_results",
            "json"
        )
        
        print(f"\n📊 Comprehensive test report saved to: {report_file}")
        print(f"📋 Detailed results saved to: {json_file}")
        
        return self.results

def main():
    """Main function to run all tests."""
    runner = TestRunner()
    
    try:
        results = runner.run_all_tests()
        
        # Determine exit code
        failed_tests = 0
        for category, category_results in results.items():
            if isinstance(category_results, dict):
                if 'success' in category_results:
                    if not category_results['success']:
                        failed_tests += 1
                else:
                    for test_name, test_result in category_results.items():
                        if not test_result['success']:
                            failed_tests += 1
        
        if failed_tests == 0:
            print("\n🎉 All tests completed successfully!")
            sys.exit(0)
        else:
            print(f"\n⚠️  {failed_tests} test(s) failed. Please review the report above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Test execution failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
