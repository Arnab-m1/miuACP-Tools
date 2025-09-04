#!/usr/bin/env python3
"""
µACP Protocol Analyzer - Usage Examples

This file demonstrates how to use the protocol analyzer
and the µACP library for various tasks.
"""

from protocol_analyzer import ProtocolAnalyzer, ProtocolComparison
from protocol_analyzer.benchmarks import BenchmarkSuite
from protocol_analyzer.visualization import ProtocolVisualizer

# Try to import µACP library
try:
    from miuacp import UACPProtocol, UACPVerb, UACPMessage
    UACP_AVAILABLE = True
    print("✅ µACP Library available")
except ImportError:
    UACP_AVAILABLE = False
    print("❌ µACP Library not available - install with: pip install -e miuacp/")

def basic_analysis_example():
    """Basic protocol analysis example."""
    print("\n" + "="*50)
    print("BASIC PROTOCOL ANALYSIS")
    print("="*50)
    
    analyzer = ProtocolAnalyzer()
    
    # Get protocol information
    print("\nProtocol Database Summary:")
    protocols = analyzer.protocol_db.get_protocol_names()
    print(f"  Available protocols: {', '.join(protocols)}")
    
    # Get µACP protocol details
    uacp_protocol = analyzer.protocol_db.get_protocol("µACP")
    print(f"\nµACP Protocol Details:")
    print(f"  Type: {uacp_protocol.type.value}")
    print(f"  Description: {uacp_protocol.description}")
    print(f"  Header Size: {uacp_protocol.header_size_min}-{uacp_protocol.header_size_max} bytes")
    print(f"  State Complexity: {uacp_protocol.state_complexity}")
    
    # Show feature support
    print(f"\nFeature Support:")
    print(f"  Pub/Sub: {'✓' if uacp_protocol.supports_pubsub else '✗'}")
    print(f"  RPC: {'✓' if uacp_protocol.supports_rpc else '✗'}")
    print(f"  Streaming: {'✓' if uacp_protocol.supports_streaming else '✗'}")
    print(f"  QoS: {'✓' if uacp_protocol.supports_qos else '✗'}")
    print(f"  Discovery: {'✓' if uacp_protocol.supports_discovery else '✗'}")
    
    # Compare with MQTT
    print(f"\nComparing µACP vs MQTT:")
    try:
        comparison = analyzer.compare_protocols("µACP", "MQTT")
        print(comparison)
    except Exception as e:
        print(f"  Comparison failed: {e}")

def uacp_library_example():
    """µACP library usage example."""
    if not UACP_AVAILABLE:
        print("\n❌ µACP Library not available for this example")
        return
    
    print("\n" + "="*50)
    print("µACP LIBRARY USAGE")
    print("="*50)
    
    # Create messages
    print("\nCreating µACP messages:")
    ping_msg = UACPProtocol.create_ping(1, 0)
    ask_msg = UACPProtocol.create_ask(2, "sensors/temperature", {"request": "current"}, 1)
    tell_msg = UACPProtocol.create_tell(3, "sensors/humidity", {"value": 45.2}, 0)
    observe_msg = UACPProtocol.create_observe(4, "sensors/alerts", 1)
    
    print(f"  ✓ PING: {ping_msg.header.verb.name}")
    print(f"  ✓ ASK: {ask_msg.header.verb.name} -> {ask_msg.header.verb.name}")
    print(f"  ✓ TELL: {tell_msg.header.verb.name}")
    print(f"  ✓ OBSERVE: {observe_msg.header.verb.name}")
    
    # Pack and unpack messages
    print("\nTesting pack/unpack:")
    ping_data = ping_msg.pack()
    ping_unpacked = UACPMessage.unpack(ping_data)
    print(f"  ✓ PING: {len(ping_data)} bytes")
    print(f"  ✓ Unpacked: {ping_unpacked.header.verb.name}")
    
    # Validate messages
    print("\nValidating messages:")
    ping_valid = UACPProtocol.validate_message(ping_msg)
    ask_valid = UACPProtocol.validate_message(ask_msg)
    tell_valid = UACPProtocol.validate_message(tell_msg)
    observe_valid = UACPProtocol.validate_message(observe_msg)
    
    all_valid = all([ping_valid, ask_valid, tell_valid, observe_valid])
    print(f"  ✓ All messages valid: {all_valid}")
    
    # Test protocol analyzer integration
    print("\nTesting with Protocol Analyzer:")
    analyzer = ProtocolAnalyzer()
    uacp_test = analyzer.test_uacp_protocol()
    
    if "error" not in uacp_test:
        print(f"  Message Creation: {uacp_test['message_creation']['throughput']:.0f} msg/s")
        print(f"  Pack/Unpack: {uacp_test['pack_unpack']['throughput']:.0f} ops/s")
        print(f"  Validation: {uacp_test['validation']['throughput']:.0f} msg/s")
        print(f"  Average Message Size: {uacp_test['message_sizes']['average']:.1f} bytes")

def benchmarking_example():
    """Benchmarking example."""
    print("\n" + "="*50)
    print("BENCHMARKING")
    print("="*50)
    
    benchmark_suite = BenchmarkSuite()
    
    # Run µACP library benchmark if available
    if UACP_AVAILABLE:
        print("\nRunning µACP Library Benchmark:")
        uacp_results = benchmark_suite.benchmark_uacp_library(1000)
        
        if "error" not in uacp_results:
            print(f"  Message Creation: {uacp_results['message_creation']['throughput']:.0f} msg/s")
            print(f"  Packing: {uacp_results['packing']['throughput']:.0f} msg/s")
            print(f"  Unpacking: {uacp_results['unpacking']['throughput']:.0f} msg/s")
            print(f"  Overall: {uacp_results['overall']['overall_throughput']:.0f} ops/s")
            print(f"  Header Efficiency: {uacp_results['sizes']['header_efficiency']:.1%}")
        else:
            print(f"  ❌ Benchmark failed: {uacp_results['error']}")
    
    # Run standard benchmarks
    print("\nRunning Standard Benchmarks:")
    protocols = ["µACP", "MQTT", "CoAP"]
    message_counts = [100, 1000]
    
    for protocol in protocols:
        print(f"\n  {protocol}:")
        for count in message_counts:
            result = benchmark_suite.benchmark_message_creation(protocol, count)
            print(f"    {count} messages: {result.throughput_msg_per_sec:.0f} msg/s")

def visualization_example():
    """Visualization example."""
    print("\n" + "="*50)
    print("VISUALIZATION")
    print("="*50)
    
    visualizer = ProtocolVisualizer()
    
    # Create various charts
    print("\nCreating visualizations:")
    
    try:
        # Header efficiency comparison
        chart = visualizer.plot_header_efficiency_comparison()
        chart.savefig("header_efficiency.png", bbox_inches='tight', dpi=300)
        print("  ✓ Header efficiency chart saved")
        
        # Feature comparison matrix
        chart = visualizer.plot_feature_comparison_matrix()
        chart.savefig("feature_matrix.png", bbox_inches='tight', dpi=300)
        print("  ✓ Feature matrix saved")
        
        # Scalability comparison
        chart = visualizer.plot_scalability_comparison()
        chart.savefig("scalability.png", bbox_inches='tight', dpi=300)
        print("  ✓ Scalability chart saved")
        
        # Comprehensive dashboard
        dashboard = visualizer.create_comprehensive_dashboard("dashboard.png")
        print("  ✓ Comprehensive dashboard saved")
        
    except Exception as e:
        print(f"  ❌ Visualization failed: {e}")

def main():
    """Main example function."""
    print("🚀 µACP Protocol Analyzer - Usage Examples")
    print("=" * 60)
    
    # Run examples
    basic_analysis_example()
    uacp_library_example()
    benchmarking_example()
    visualization_example()
    
    print("\n" + "="*60)
    print("✅ All examples completed!")
    print("="*60)

if __name__ == "__main__":
    main()
