"""
Performance Analyzer for µACP

Comprehensive performance analysis and benchmarking tool that combines
all performance metrics and provides detailed analysis and recommendations.

Features:
- Combined latency, memory, scalability, and energy analysis
- Performance trend analysis
- Optimization recommendations
- Industry comparison
- Report generation
"""

import json
import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import os
import sys

# Import benchmark modules
from .latency_benchmark import LatencyBenchmark, LatencyTestType
from .memory_benchmark import MemoryBenchmark, MemoryTestType


class PerformanceLevel(Enum):
    """Performance levels for analysis"""
    EXCELLENT = "excellent"    # Exceeds all targets
    GOOD = "good"             # Meets all targets
    ACCEPTABLE = "acceptable"  # Meets most targets
    POOR = "poor"             # Fails multiple targets
    CRITICAL = "critical"     # Fails critical targets


@dataclass
class PerformanceTarget:
    """Performance target definition"""
    name: str
    target_value: float
    unit: str
    critical: bool = False
    description: str = ""


@dataclass
class PerformanceResult:
    """Performance result for a specific metric"""
    target: PerformanceTarget
    actual_value: float
    target_met: bool
    percentage_of_target: float
    level: PerformanceLevel
    recommendations: List[str] = field(default_factory=list)


@dataclass
class PerformanceAnalysis:
    """Complete performance analysis"""
    timestamp: float
    overall_level: PerformanceLevel
    results: Dict[str, PerformanceResult]
    summary: Dict[str, Any]
    recommendations: List[str]
    industry_comparison: Dict[str, Any]
    trends: Dict[str, Any]


class PerformanceAnalyzer:
    """
    Comprehensive performance analyzer for µACP
    
    Analyzes performance across all dimensions and provides
    actionable insights and optimization recommendations.
    """
    
    def __init__(self):
        self.latency_benchmark = LatencyBenchmark()
        self.memory_benchmark = MemoryBenchmark()
        
        # Define performance targets
        self.targets = {
            # Latency targets (milliseconds)
            'message_creation_latency': PerformanceTarget(
                name="Message Creation Latency",
                target_value=0.1,
                unit="ms",
                critical=True,
                description="Time to create a UACP message"
            ),
            'message_processing_latency': PerformanceTarget(
                name="Message Processing Latency", 
                target_value=0.5,
                unit="ms",
                critical=True,
                description="Time to process a UACP message"
            ),
            'transmission_latency': PerformanceTarget(
                name="Transmission Latency",
                target_value=0.3,
                unit="ms",
                critical=True,
                description="Time to transmit a message"
            ),
            'end_to_end_latency': PerformanceTarget(
                name="End-to-End Latency",
                target_value=1.0,
                unit="ms",
                critical=True,
                description="Complete message round-trip time"
            ),
            'round_trip_latency': PerformanceTarget(
                name="Round-Trip Latency",
                target_value=2.0,
                unit="ms",
                critical=False,
                description="Request-response cycle time"
            ),
            
            # Memory targets (bytes)
            'agent_memory': PerformanceTarget(
                name="Agent Memory Usage",
                target_value=1024,
                unit="bytes",
                critical=True,
                description="Memory usage per agent"
            ),
            'message_memory': PerformanceTarget(
                name="Message Memory Usage",
                target_value=100,
                unit="bytes",
                critical=True,
                description="Memory usage per message"
            ),
            'flash_usage': PerformanceTarget(
                name="Flash Storage Usage",
                target_value=3 * 1024 * 1024,  # 3MB
                unit="bytes",
                critical=False,
                description="Flash storage usage"
            ),
            'heap_fragmentation': PerformanceTarget(
                name="Heap Fragmentation",
                target_value=10.0,
                unit="%",
                critical=False,
                description="Memory heap fragmentation"
            ),
            'memory_leak': PerformanceTarget(
                name="Memory Leak",
                target_value=0,
                unit="bytes",
                critical=True,
                description="Memory growth over time"
            ),
            
            # Scalability targets
            'max_agents': PerformanceTarget(
                name="Maximum Agents",
                target_value=100,
                unit="agents",
                critical=False,
                description="Maximum concurrent agents"
            ),
            'messages_per_second': PerformanceTarget(
                name="Messages Per Second",
                target_value=1000,
                unit="msg/s",
                critical=False,
                description="Message processing rate"
            ),
            
            # Energy targets (millijoules)
            'energy_per_message': PerformanceTarget(
                name="Energy Per Message",
                target_value=1.0,
                unit="mJ",
                critical=False,
                description="Energy consumption per message"
            )
        }
        
        # Industry benchmarks for comparison
        self.industry_benchmarks = {
            'MQTT': {
                'end_to_end_latency': 5.0,  # ms
                'message_memory': 200,      # bytes
                'max_connections': 1000,
                'messages_per_second': 10000
            },
            'CoAP': {
                'end_to_end_latency': 10.0,  # ms
                'message_memory': 150,       # bytes
                'max_connections': 100,
                'messages_per_second': 1000
            },
            'ACP': {
                'end_to_end_latency': 2.0,   # ms
                'message_memory': 120,       # bytes
                'max_connections': 500,
                'messages_per_second': 5000
            }
        }
    
    def run_comprehensive_analysis(self, iterations: int = 1000) -> PerformanceAnalysis:
        """Run comprehensive performance analysis"""
        print("Starting comprehensive performance analysis...")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all benchmarks
        print("Running latency benchmarks...")
        latency_results = self.latency_benchmark.run_all_benchmarks(iterations)
        
        print("\nRunning memory benchmarks...")
        memory_results = self.memory_benchmark.run_all_benchmarks()
        
        # Analyze results
        print("\nAnalyzing results...")
        analysis = self._analyze_results(latency_results, memory_results)
        
        # Generate recommendations
        print("Generating recommendations...")
        analysis.recommendations = self._generate_recommendations(analysis)
        
        # Industry comparison
        print("Performing industry comparison...")
        analysis.industry_comparison = self._compare_with_industry(analysis)
        
        # Calculate trends (simplified for now)
        analysis.trends = self._calculate_trends(analysis)
        
        end_time = time.time()
        analysis.summary['analysis_duration'] = end_time - start_time
        
        print(f"\nAnalysis completed in {analysis.summary['analysis_duration']:.2f} seconds")
        
        return analysis
    
    def _analyze_results(self, latency_results: Dict, memory_results: Dict) -> PerformanceAnalysis:
        """Analyze benchmark results and create performance analysis"""
        results = {}
        
        # Analyze latency results
        for test_type, stats in latency_results.items():
            target_name = test_type.value
            if target_name in self.targets:
                target = self.targets[target_name]
                actual_value = stats.mean_latency
                target_met = stats.target_met
                percentage = (actual_value / target.target_value) * 100 if target.target_value > 0 else 0
                
                level = self._determine_performance_level(percentage, target.critical)
                
                results[target_name] = PerformanceResult(
                    target=target,
                    actual_value=actual_value,
                    target_met=target_met,
                    percentage_of_target=percentage,
                    level=level,
                    recommendations=self._get_metric_recommendations(target_name, actual_value, target.target_value)
                )
        
        # Analyze memory results
        for test_type, stats in memory_results.items():
            target_name = test_type.value
            if target_name in self.targets:
                target = self.targets[target_name]
                
                if test_type == MemoryTestType.HEAP_FRAGMENTATION:
                    actual_value = stats.fragmentation_avg
                elif test_type == MemoryTestType.MEMORY_LEAK:
                    actual_value = stats.memory_growth
                else:
                    actual_value = stats.mean_memory
                
                target_met = stats.target_met
                percentage = (actual_value / target.target_value) * 100 if target.target_value > 0 else 0
                
                level = self._determine_performance_level(percentage, target.critical)
                
                results[target_name] = PerformanceResult(
                    target=target,
                    actual_value=actual_value,
                    target_met=target_met,
                    percentage_of_target=percentage,
                    level=level,
                    recommendations=self._get_metric_recommendations(target_name, actual_value, target.target_value)
                )
        
        # Calculate overall performance level
        overall_level = self._calculate_overall_level(results)
        
        # Create summary
        summary = self._create_summary(results, latency_results, memory_results)
        
        return PerformanceAnalysis(
            timestamp=time.time(),
            overall_level=overall_level,
            results=results,
            summary=summary,
            recommendations=[],  # Will be filled later
            industry_comparison={},
            trends={}
        )
    
    def _determine_performance_level(self, percentage: float, critical: bool) -> PerformanceLevel:
        """Determine performance level based on percentage of target"""
        if percentage <= 50:
            return PerformanceLevel.EXCELLENT
        elif percentage <= 100:
            return PerformanceLevel.GOOD
        elif percentage <= 150:
            return PerformanceLevel.ACCEPTABLE
        elif percentage <= 200:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.CRITICAL if critical else PerformanceLevel.POOR
    
    def _calculate_overall_level(self, results: Dict[str, PerformanceResult]) -> PerformanceLevel:
        """Calculate overall performance level"""
        critical_results = [r for r in results.values() if r.target.critical]
        all_results = list(results.values())
        
        if not critical_results:
            critical_results = all_results
        
        # Check if any critical targets are failed
        critical_failures = [r for r in critical_results if r.level in [PerformanceLevel.POOR, PerformanceLevel.CRITICAL]]
        if critical_failures:
            return PerformanceLevel.CRITICAL
        
        # Check overall performance distribution
        excellent_count = sum(1 for r in all_results if r.level == PerformanceLevel.EXCELLENT)
        good_count = sum(1 for r in all_results if r.level == PerformanceLevel.GOOD)
        acceptable_count = sum(1 for r in all_results if r.level == PerformanceLevel.ACCEPTABLE)
        
        total_count = len(all_results)
        
        if excellent_count / total_count >= 0.7:
            return PerformanceLevel.EXCELLENT
        elif (excellent_count + good_count) / total_count >= 0.7:
            return PerformanceLevel.GOOD
        elif (excellent_count + good_count + acceptable_count) / total_count >= 0.7:
            return PerformanceLevel.ACCEPTABLE
        else:
            return PerformanceLevel.POOR
    
    def _create_summary(self, results: Dict[str, PerformanceResult], 
                       latency_results: Dict, memory_results: Dict) -> Dict[str, Any]:
        """Create performance summary"""
        total_targets = len(self.targets)
        targets_met = sum(1 for r in results.values() if r.target_met)
        critical_targets = [t for t in self.targets.values() if t.critical]
        critical_met = sum(1 for t in critical_targets 
                          if t.name in results and results[t.name].target_met)
        
        return {
            'total_targets': total_targets,
            'targets_met': targets_met,
            'targets_met_percentage': (targets_met / total_targets) * 100,
            'critical_targets': len(critical_targets),
            'critical_targets_met': critical_met,
            'critical_targets_met_percentage': (critical_met / len(critical_targets)) * 100 if critical_targets else 100,
            'performance_distribution': {
                'excellent': sum(1 for r in results.values() if r.level == PerformanceLevel.EXCELLENT),
                'good': sum(1 for r in results.values() if r.level == PerformanceLevel.GOOD),
                'acceptable': sum(1 for r in results.values() if r.level == PerformanceLevel.ACCEPTABLE),
                'poor': sum(1 for r in results.values() if r.level == PerformanceLevel.POOR),
                'critical': sum(1 for r in results.values() if r.level == PerformanceLevel.CRITICAL)
            }
        }
    
    def _get_metric_recommendations(self, metric_name: str, actual: float, target: float) -> List[str]:
        """Get specific recommendations for a metric"""
        recommendations = []
        
        if metric_name.endswith('_latency'):
            if actual > target:
                recommendations.extend([
                    "Optimize message serialization/deserialization",
                    "Reduce memory allocations in hot paths",
                    "Consider using memory pools for frequent allocations",
                    "Profile and optimize critical code paths"
                ])
        
        elif metric_name == 'agent_memory':
            if actual > target:
                recommendations.extend([
                    "Implement agent state compression",
                    "Use lazy loading for agent data",
                    "Optimize agent data structures",
                    "Consider agent state externalization"
                ])
        
        elif metric_name == 'message_memory':
            if actual > target:
                recommendations.extend([
                    "Optimize message header structure",
                    "Use compact binary encoding",
                    "Implement message compression",
                    "Reduce message metadata overhead"
                ])
        
        elif metric_name == 'heap_fragmentation':
            if actual > target:
                recommendations.extend([
                    "Implement memory pooling",
                    "Use fixed-size allocations where possible",
                    "Optimize garbage collection frequency",
                    "Consider custom memory allocator"
                ])
        
        elif metric_name == 'memory_leak':
            if actual > target:
                recommendations.extend([
                    "Review object lifecycle management",
                    "Implement proper resource cleanup",
                    "Use weak references where appropriate",
                    "Add memory leak detection tools"
                ])
        
        return recommendations
    
    def _generate_recommendations(self, analysis: PerformanceAnalysis) -> List[str]:
        """Generate overall performance recommendations"""
        recommendations = []
        
        # Overall performance recommendations
        if analysis.overall_level == PerformanceLevel.CRITICAL:
            recommendations.extend([
                "CRITICAL: Multiple performance targets failed",
                "Immediate optimization required for production readiness",
                "Focus on critical latency and memory targets first",
                "Consider architectural changes for better performance"
            ])
        elif analysis.overall_level == PerformanceLevel.POOR:
            recommendations.extend([
                "Significant performance improvements needed",
                "Focus on the worst-performing metrics",
                "Consider performance optimization sprint",
                "Review and optimize critical code paths"
            ])
        elif analysis.overall_level == PerformanceLevel.ACCEPTABLE:
            recommendations.extend([
                "Performance is acceptable but can be improved",
                "Focus on optimization opportunities",
                "Consider performance monitoring in production",
                "Plan for future performance improvements"
            ])
        elif analysis.overall_level == PerformanceLevel.GOOD:
            recommendations.extend([
                "Performance meets targets - good job!",
                "Consider fine-tuning for even better performance",
                "Monitor performance in production",
                "Plan for scalability improvements"
            ])
        else:  # EXCELLENT
            recommendations.extend([
                "Excellent performance - exceeds targets!",
                "Consider sharing optimization techniques",
                "Monitor for performance regressions",
                "Plan for next-generation improvements"
            ])
        
        # Specific metric recommendations
        for result in analysis.results.values():
            if result.level in [PerformanceLevel.POOR, PerformanceLevel.CRITICAL]:
                recommendations.extend([f"Priority: {rec}" for rec in result.recommendations])
        
        return recommendations
    
    def _compare_with_industry(self, analysis: PerformanceAnalysis) -> Dict[str, Any]:
        """Compare performance with industry standards"""
        comparison = {}
        
        for protocol, benchmarks in self.industry_benchmarks.items():
            comparison[protocol] = {}
            
            for metric, industry_value in benchmarks.items():
                if metric in analysis.results:
                    actual_value = analysis.results[metric].actual_value
                    target_value = analysis.results[metric].target.target_value
                    
                    # Calculate relative performance
                    vs_industry = (industry_value / actual_value) if actual_value > 0 else float('inf')
                    vs_target = (target_value / actual_value) if actual_value > 0 else float('inf')
                    
                    comparison[protocol][metric] = {
                        'industry_value': industry_value,
                        'actual_value': actual_value,
                        'target_value': target_value,
                        'vs_industry': vs_industry,
                        'vs_target': vs_target,
                        'better_than_industry': actual_value < industry_value,
                        'meets_target': actual_value <= target_value
                    }
        
        return comparison
    
    def _calculate_trends(self, analysis: PerformanceAnalysis) -> Dict[str, Any]:
        """Calculate performance trends (simplified implementation)"""
        # This would typically compare with historical data
        # For now, we'll provide a basic structure
        return {
            'trend_analysis': 'Historical data not available',
            'improvement_areas': [r.target.name for r in analysis.results.values() 
                                if r.level in [PerformanceLevel.POOR, PerformanceLevel.CRITICAL]],
            'strength_areas': [r.target.name for r in analysis.results.values() 
                             if r.level == PerformanceLevel.EXCELLENT]
        }
    
    def generate_report(self, analysis: PerformanceAnalysis, output_file: str = None) -> str:
        """Generate comprehensive performance report"""
        report = []
        
        # Header
        report.append("=" * 80)
        report.append("µACP PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Analysis Duration: {analysis.summary['analysis_duration']:.2f} seconds")
        report.append("")
        
        # Overall Performance
        report.append("OVERALL PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Level: {analysis.overall_level.value.upper()}")
        report.append(f"Targets Met: {analysis.summary['targets_met']}/{analysis.summary['total_targets']} ({analysis.summary['targets_met_percentage']:.1f}%)")
        report.append(f"Critical Targets Met: {analysis.summary['critical_targets_met']}/{analysis.summary['critical_targets']} ({analysis.summary['critical_targets_met_percentage']:.1f}%)")
        report.append("")
        
        # Performance Distribution
        dist = analysis.summary['performance_distribution']
        report.append("PERFORMANCE DISTRIBUTION")
        report.append("-" * 40)
        report.append(f"Excellent: {dist['excellent']}")
        report.append(f"Good: {dist['good']}")
        report.append(f"Acceptable: {dist['acceptable']}")
        report.append(f"Poor: {dist['poor']}")
        report.append(f"Critical: {dist['critical']}")
        report.append("")
        
        # Detailed Results
        report.append("DETAILED RESULTS")
        report.append("-" * 40)
        for name, result in analysis.results.items():
            status = "✅ PASS" if result.target_met else "❌ FAIL"
            report.append(f"{result.target.name}:")
            report.append(f"  Target: {result.target.target_value} {result.target.unit}")
            report.append(f"  Actual: {result.actual_value:.3f} {result.target.unit}")
            report.append(f"  Level: {result.level.value.upper()}")
            report.append(f"  Status: {status}")
            if result.recommendations:
                report.append(f"  Recommendations: {', '.join(result.recommendations[:2])}")
            report.append("")
        
        # Industry Comparison
        report.append("INDUSTRY COMPARISON")
        report.append("-" * 40)
        for protocol, metrics in analysis.industry_comparison.items():
            report.append(f"{protocol}:")
            for metric, comparison in metrics.items():
                better = "✅" if comparison['better_than_industry'] else "❌"
                report.append(f"  {metric}: {better} {comparison['vs_industry']:.1f}x vs industry")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        for i, rec in enumerate(analysis.recommendations, 1):
            report.append(f"{i}. {rec}")
        report.append("")
        
        # Footer
        report.append("=" * 80)
        report.append("End of Report")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {output_file}")
        
        return report_text
    
    def save_analysis_json(self, analysis: PerformanceAnalysis, output_file: str):
        """Save analysis results as JSON"""
        # Convert analysis to JSON-serializable format
        json_data = {
            'timestamp': analysis.timestamp,
            'overall_level': analysis.overall_level.value,
            'summary': analysis.summary,
            'results': {
                name: {
                    'target_name': result.target.name,
                    'target_value': result.target.target_value,
                    'target_unit': result.target.unit,
                    'target_critical': result.target.critical,
                    'actual_value': result.actual_value,
                    'target_met': result.target_met,
                    'percentage_of_target': result.percentage_of_target,
                    'level': result.level.value,
                    'recommendations': result.recommendations
                }
                for name, result in analysis.results.items()
            },
            'recommendations': analysis.recommendations,
            'industry_comparison': analysis.industry_comparison,
            'trends': analysis.trends
        }
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Analysis JSON saved to: {output_file}")


class BenchmarkRunner:
    """Convenience class to run all benchmarks and generate reports"""
    
    def __init__(self):
        self.analyzer = PerformanceAnalyzer()
    
    def run_full_benchmark_suite(self, iterations: int = 1000, 
                                output_dir: str = "benchmark_results") -> PerformanceAnalysis:
        """Run complete benchmark suite and generate reports"""
        print("Starting full µACP benchmark suite...")
        print("=" * 60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Run analysis
        analysis = self.analyzer.run_comprehensive_analysis(iterations)
        
        # Generate reports
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Text report
        report_file = os.path.join(output_dir, f"performance_report_{timestamp}.txt")
        self.analyzer.generate_report(analysis, report_file)
        
        # JSON report
        json_file = os.path.join(output_dir, f"performance_analysis_{timestamp}.json")
        self.analyzer.save_analysis_json(analysis, json_file)
        
        # Print summary
        print("\n" + "=" * 60)
        print("BENCHMARK SUITE COMPLETED")
        print("=" * 60)
        print(f"Overall Performance: {analysis.overall_level.value.upper()}")
        print(f"Targets Met: {analysis.summary['targets_met']}/{analysis.summary['total_targets']}")
        print(f"Reports Generated:")
        print(f"  - Text Report: {report_file}")
        print(f"  - JSON Data: {json_file}")
        print("=" * 60)
        
        return analysis


if __name__ == "__main__":
    # Run full benchmark suite
    runner = BenchmarkRunner()
    analysis = runner.run_full_benchmark_suite(iterations=1000)
