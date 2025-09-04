#!/usr/bin/env python3
"""
Comprehensive Benchmark Runner for µACP

Runs all performance benchmarks and generates detailed reports.
This is the main entry point for performance validation.
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Add benchmarks to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'benchmarks'))

from benchmarks.performance_analyzer import BenchmarkRunner


def main():
    """Main benchmark runner"""
    parser = argparse.ArgumentParser(
        description="Run comprehensive µACP performance benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmarks.py                    # Run with default settings
  python run_benchmarks.py --iterations 5000  # Run with 5000 iterations
  python run_benchmarks.py --output results/  # Save to custom directory
  python run_benchmarks.py --quick           # Quick test with 100 iterations
        """
    )
    
    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=1000,
        help='Number of iterations per benchmark (default: 1000)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='benchmark_results',
        help='Output directory for results (default: benchmark_results)'
    )
    
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick test with 100 iterations'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Adjust iterations for quick mode
    if args.quick:
        args.iterations = 100
        print("🚀 Quick mode: Running with 100 iterations")
    
    print("=" * 80)
    print("µACP COMPREHENSIVE PERFORMANCE BENCHMARK SUITE")
    print("=" * 80)
    print(f"Target Performance:")
    print(f"  - Latency: <1ms end-to-end on ESP32-C3")
    print(f"  - Memory: <1KB per agent, <100 bytes per message")
    print(f"  - Scalability: Maintain <5ms latency under 20% packet loss")
    print(f"  - Energy: <1mJ per message")
    print(f"")
    print(f"Configuration:")
    print(f"  - Iterations: {args.iterations}")
    print(f"  - Output: {args.output}")
    print(f"  - Verbose: {args.verbose}")
    print("=" * 80)
    
    try:
        # Create benchmark runner
        runner = BenchmarkRunner()
        
        # Run comprehensive benchmark suite
        start_time = time.time()
        analysis = runner.run_full_benchmark_suite(
            iterations=args.iterations,
            output_dir=args.output
        )
        end_time = time.time()
        
        # Print final summary
        print("\n" + "=" * 80)
        print("BENCHMARK EXECUTION COMPLETED")
        print("=" * 80)
        print(f"Total Execution Time: {end_time - start_time:.2f} seconds")
        print(f"Overall Performance: {analysis.overall_level.value.upper()}")
        print(f"Targets Met: {analysis.summary['targets_met']}/{analysis.summary['total_targets']}")
        print(f"Critical Targets Met: {analysis.summary['critical_targets_met']}/{analysis.summary['critical_targets']}")
        
        # Performance distribution
        dist = analysis.summary['performance_distribution']
        print(f"\nPerformance Distribution:")
        print(f"  Excellent: {dist['excellent']}")
        print(f"  Good: {dist['good']}")
        print(f"  Acceptable: {dist['acceptable']}")
        print(f"  Poor: {dist['poor']}")
        print(f"  Critical: {dist['critical']}")
        
        # Key recommendations
        if analysis.recommendations:
            print(f"\nKey Recommendations:")
            for i, rec in enumerate(analysis.recommendations[:5], 1):
                print(f"  {i}. {rec}")
        
        print("=" * 80)
        
        # Exit with appropriate code
        if analysis.overall_level.value in ['excellent', 'good']:
            print("✅ BENCHMARKS PASSED - Performance targets met!")
            sys.exit(0)
        elif analysis.overall_level.value == 'acceptable':
            print("⚠️  BENCHMARKS ACCEPTABLE - Some optimization needed")
            sys.exit(1)
        else:
            print("❌ BENCHMARKS FAILED - Significant optimization required")
            sys.exit(2)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Benchmark execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
