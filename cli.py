#!/usr/bin/env python3
"""
Command-line interface for µACP protocol analysis and benchmarking.
"""

import click
import sys
import os
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

# Add the parent directory to the path to import miuacp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from miuacp import UACPProtocol, UACPVerb, UACPMessage
    UACP_AVAILABLE = True
except ImportError:
    UACP_AVAILABLE = False
    print("❌ µACP Library not available - install with: pip install miuacp")

# Import protocol analyzer components
try:
    # Add current directory to path for protocol_analyzer imports
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from protocol_analyzer.models import Protocol, ProtocolMetrics
    from protocol_analyzer.protocols import ProtocolDatabase
    from protocol_analyzer.benchmarks import BenchmarkSuite
    from protocol_analyzer.visualization import ProtocolVisualizer
    from protocol_analyzer.core import ProtocolAnalyzer
    PROTOCOL_ANALYZER_AVAILABLE = True
except ImportError as e:
    PROTOCOL_ANALYZER_AVAILABLE = False
    print(f"⚠️  Protocol Analyzer not available - some features may be limited: {e}")

class OutputManager:
    """Manages timestamped output generation."""
    
    def __init__(self):
        self.output_dir = Path(__file__).parent / "output"
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_output(self, content, filename, file_type="txt"):
        """Save output to timestamped file."""
        output_file = self.output_dir / f"{self.timestamp}_{filename}.{file_type}"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"📁 Output saved to: {output_file}")
        return output_file
    
    def get_output_path(self, filename, file_type="txt"):
        """Get output file path."""
        return self.output_dir / f"{self.timestamp}_{filename}.{file_type}"

# Global output manager
output_manager = OutputManager()

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """µACP Protocol Analysis and Benchmarking CLI."""
    pass

@cli.command()
@click.argument('protocol_name')
def analyze(protocol_name):
    """Analyze a specific protocol."""
    if not PROTOCOL_ANALYZER_AVAILABLE:
        click.echo("❌ Protocol Analyzer not available")
        return
    
    try:
        analyzer = ProtocolAnalyzer()
        result = analyzer.analyze_protocol(protocol_name)
        
        # Generate output
        output_content = f"Protocol Analysis: {protocol_name}\n"
        output_content += "=" * 50 + "\n"
        output_content += str(result)
        
        output_file = output_manager.save_output(output_content, f"analysis_{protocol_name.lower()}")
        click.echo(f"✅ Analysis completed and saved to: {output_file}")
        
    except Exception as e:
        click.echo(f"❌ Analysis failed: {e}")

@cli.command()
@click.option('--protocols', '-p', multiple=True, default=['µACP', 'MQTT', 'CoAP'], 
              help='Protocols to benchmark')
@click.option('--message-counts', '-m', multiple=True, default=[100, 1000, 10000], 
              help='Message counts for benchmarking')
def benchmark(protocols, message_counts):
    """Run performance benchmarks on protocols."""
    if not PROTOCOL_ANALYZER_AVAILABLE:
        click.echo("❌ Protocol Analyzer not available")
        return
    
    try:
        suite = BenchmarkSuite()
        
        # Convert message counts to integers
        message_counts = [int(m) for m in message_counts]
        
        results = suite.run_benchmarks(protocols, message_counts)
        
        # Generate output
        output_content = f"Benchmark Results\n"
        output_content += "=" * 50 + "\n"
        output_content += f"Protocols: {', '.join(protocols)}\n"
        output_content += f"Message Counts: {', '.join(map(str, message_counts))}\n\n"
        output_content += str(results)
        
        output_file = output_manager.save_output(output_content, "benchmark_results")
        click.echo(f"✅ Benchmark completed and saved to: {output_file}")
        
    except Exception as e:
        click.echo(f"❌ Benchmark failed: {e}")

@cli.command()
@click.argument('protocol1')
@click.argument('protocol2')
def compare(protocol1, protocol2):
    """Compare two protocols."""
    if not PROTOCOL_ANALYZER_AVAILABLE:
        click.echo("❌ Protocol Analyzer not available")
        return
    
    try:
        analyzer = ProtocolAnalyzer()
        comparison = analyzer.compare_protocols(protocol1, protocol2)
        
        # Generate output
        output_content = f"Protocol Comparison: {protocol1} vs {protocol2}\n"
        output_content += "=" * 50 + "\n"
        output_content += str(comparison)
        
        output_file = output_manager.save_output(output_content, f"comparison_{protocol1.lower()}_vs_{protocol2.lower()}")
        click.echo(f"✅ Comparison completed and saved to: {output_file}")
        
    except Exception as e:
        click.echo(f"❌ Comparison failed: {e}")

@cli.command()
def demo():
    """Run comprehensive demonstration."""
    try:
        # Import and run demo
        from demo import main as run_demo
        result = run_demo()
        
        # Generate output
        output_content = f"Demo Execution Results\n"
        output_content += "=" * 50 + "\n"
        output_content += str(result) if result else "Demo completed successfully"
        
        output_file = output_manager.save_output(output_content, "demo_results")
        click.echo(f"✅ Demo completed and saved to: {output_file}")
        
    except Exception as e:
        click.echo(f"❌ Demo failed: {e}")

@cli.command()
@click.option('--protocol', '-p', default='µACP', help='Protocol for mathematical analysis')
def math(protocol):
    """Perform mathematical analysis on protocol."""
    if not PROTOCOL_ANALYZER_AVAILABLE:
        click.echo("❌ Protocol Analyzer not available")
        return
    
    try:
        analyzer = ProtocolAnalyzer()
        math_result = analyzer.mathematical_analysis(protocol)
        
        # Generate output
        output_content = f"Mathematical Analysis: {protocol}\n"
        output_content += "=" * 50 + "\n"
        output_content += str(math_result)
        
        output_file = output_manager.save_output(output_content, f"math_analysis_{protocol.lower()}")
        click.echo(f"✅ Mathematical analysis completed and saved to: {output_file}")
        
    except Exception as e:
        click.echo(f"❌ Mathematical analysis failed: {e}")

@cli.command()
@click.option('--use-case', '-u', required=True, help='Use case for protocol recommendation')
def recommend(use_case):
    """Get protocol recommendations for a use case."""
    if not PROTOCOL_ANALYZER_AVAILABLE:
        click.echo("❌ Protocol Analyzer not available")
        return
    
    try:
        analyzer = ProtocolAnalyzer()
        recommendation = analyzer.recommend_protocol(use_case)
        
        # Generate output
        output_content = f"Protocol Recommendation for: {use_case}\n"
        output_content += "=" * 50 + "\n"
        output_content += str(recommendation)
        
        output_file = output_manager.save_output(output_content, f"recommendation_{use_case.lower().replace(' ', '_')}")
        click.echo(f"✅ Recommendation generated and saved to: {output_file}")
        
    except Exception as e:
        click.echo(f"❌ Recommendation failed: {e}")

@cli.command()
def test_uacp():
    """Test µACP library functionality."""
    if not UACP_AVAILABLE:
        click.echo("❌ µACP Library not available")
        return
    
    try:
        # Import and run test
        from test_library import UACPLibraryTester
        tester = UACPLibraryTester()
        results = tester.run_all_tests()
        
        # Generate output
        output_content = f"µACP Library Test Results\n"
        output_content += "=" * 50 + "\n"
        output_content += f"Results: {results}\n"
        output_content += f"Status: {'PASS' if all(r.get('status') == 'PASS' for r in results.values() if isinstance(r, dict) and 'status' in r) else 'FAIL'}"
        
        output_file = output_manager.save_output(output_content, "uacp_test_results")
        click.echo(f"✅ µACP library test completed and saved to: {output_file}")
        
    except Exception as e:
        click.echo(f"❌ µACP library test failed: {e}")

@cli.command()
def visualize():
    """Create protocol visualizations."""
    if not PROTOCOL_ANALYZER_AVAILABLE:
        click.echo("❌ Protocol Analyzer not available")
        return
    
    try:
        visualizer = ProtocolVisualizer()
        
        # Create various visualizations and save them to output folder
        charts = {}
        
        # Header efficiency comparison
        try:
            fig = visualizer.plot_header_efficiency_comparison()
            chart_path = output_manager.get_output_path("header_efficiency", "png")
            fig.savefig(chart_path, bbox_inches='tight', dpi=300)
            charts["Header Efficiency"] = str(chart_path)
            plt.close(fig)
            click.echo(f"✅ Created header efficiency chart: {chart_path}")
        except Exception as e:
            click.echo(f"⚠️  Header efficiency chart failed: {e}")
        
        # Feature comparison matrix
        try:
            fig = visualizer.plot_feature_comparison_matrix()
            chart_path = output_manager.get_output_path("feature_matrix", "png")
            fig.savefig(chart_path, bbox_inches='tight', dpi=300)
            charts["Feature Matrix"] = str(chart_path)
            plt.close(fig)
            click.echo(f"✅ Created feature matrix chart: {chart_path}")
        except Exception as e:
            click.echo(f"⚠️  Feature matrix chart failed: {e}")
        
        # Scalability comparison
        try:
            fig = visualizer.plot_scalability_comparison()
            chart_path = output_manager.get_output_path("scalability", "png")
            fig.savefig(chart_path, bbox_inches='tight', dpi=300)
            charts["Scalability"] = str(chart_path)
            plt.close(fig)
            click.echo(f"✅ Created scalability chart: {chart_path}")
        except Exception as e:
            click.echo(f"⚠️  Scalability chart failed: {e}")
        
        # Energy efficiency comparison
        try:
            fig = visualizer.plot_energy_efficiency_comparison()
            chart_path = output_manager.get_output_path("energy_efficiency", "png")
            fig.savefig(chart_path, bbox_inches='tight', dpi=300)
            charts["Energy Efficiency"] = str(chart_path)
            plt.close(fig)
            click.echo(f"✅ Created energy efficiency chart: {chart_path}")
        except Exception as e:
            click.echo(f"⚠️  Energy efficiency chart failed: {e}")
        
        # Generate output summary
        output_content = f"Visualization Results\n"
        output_content += "=" * 50 + "\n"
        output_content += f"Charts created: {len(charts)}\n"
        for chart_name, chart_path in charts.items():
            output_content += f"- {chart_name}: {chart_path}\n"
        
        output_file = output_manager.save_output(output_content, "visualization_results")
        click.echo(f"✅ Visualizations completed and saved to: {output_file}")
        click.echo(f"📁 Charts saved to output directory: {output_manager.output_dir}")
        
    except Exception as e:
        click.echo(f"❌ Visualization failed: {e}")

@cli.command()
def info():
    """Show system information and available components."""
    info_content = f"µACP CLI System Information\n"
    info_content += "=" * 50 + "\n"
    info_content += f"µACP Library Available: {'✅ Yes' if UACP_AVAILABLE else '❌ No'}\n"
    info_content += f"Protocol Analyzer Available: {'✅ Yes' if PROTOCOL_ANALYZER_AVAILABLE else '❌ No'}\n"
    info_content += f"Output Directory: {output_manager.output_dir}\n"
    info_content += f"Timestamp: {output_manager.timestamp}\n"
    
    if UACP_AVAILABLE:
        from miuacp import __version__
        info_content += f"µACP Version: {__version__}\n"
    
    info_content += "\nAvailable Commands:\n"
    info_content += "- analyze: Analyze a specific protocol\n"
    info_content += "- benchmark: Run performance benchmarks\n"
    info_content += "- compare: Compare two protocols\n"
    info_content += "- demo: Run comprehensive demonstration\n"
    info_content += "- math: Perform mathematical analysis\n"
    info_content += "- recommend: Get protocol recommendations\n"
    info_content += "- visualize: Create protocol visualizations\n"
    info_content += "- test_uacp: Test µACP library functionality\n"
    info_content += "- info: Show system information\n"
    
    click.echo(info_content)
    
    # Save info to file
    output_file = output_manager.save_output(info_content, "system_info")
    click.echo(f"📁 System information saved to: {output_file}")

if __name__ == '__main__':
    cli()
