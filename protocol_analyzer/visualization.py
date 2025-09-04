"""
Visualization tools for protocol analysis and comparison.

This module provides charts, graphs, and visual representations of protocol
characteristics, performance metrics, and comparisons.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import matplotlib.patches as mpatches
from .models import Protocol, ProtocolMetrics
from .protocols import ProtocolDatabase


class ProtocolVisualizer:
    """Visualization engine for protocol analysis."""
    
    def __init__(self):
        self.protocol_db = ProtocolDatabase()
        self.setup_style()
    
    def setup_style(self):
        """Setup matplotlib and seaborn styling."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Set figure size and DPI
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 100
        
        # Font settings
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
    
    def plot_protocol_comparison_radar(self, protocols: List[str], 
                                      save_path: Optional[str] = None) -> plt.Figure:
        """Create a radar chart comparing multiple protocols across key metrics."""
        
        # Get protocol data
        protocol_data = {}
        for name in protocols:
            protocol = self.protocol_db.get_protocol(name)
            protocol_data[name] = protocol
        
        # Define metrics for radar chart
        metrics = ['Header Efficiency', 'Pub/Sub Support', 'RPC Support', 
                  'Streaming Support', 'QoS Support', 'Discovery Support']
        
        # Calculate values for each metric
        values = {}
        for name, protocol in protocol_data.items():
            values[name] = []
            
            # Header efficiency (normalized to 0-1, higher is better)
            header_eff = 1 - (protocol.header_size_min / 100)  # Normalize to 100 bytes
            values[name].append(min(header_eff, 1.0))
            
            # Feature support (boolean to 0-1)
            values[name].append(1.0 if protocol.supports_pubsub else 0.0)
            values[name].append(1.0 if protocol.supports_rpc else 0.0)
            values[name].append(1.0 if protocol.supports_streaming else 0.0)
            values[name].append(1.0 if protocol.supports_qos else 0.0)
            values[name].append(1.0 if protocol.supports_discovery else 0.0)
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        # Calculate angles for each metric
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each protocol
        colors = plt.cm.Set3(np.linspace(0, 1, len(protocols)))
        for i, (name, protocol_values) in enumerate(values.items()):
            # Complete the circle
            values_circle = protocol_values + protocol_values[:1]
            
            ax.plot(angles, values_circle, 'o-', linewidth=2, 
                   label=name, color=colors[i], alpha=0.7)
            ax.fill(angles, values_circle, alpha=0.1, color=colors[i])
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        
        # Add legend and title
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.set_title('Protocol Comparison Radar Chart', size=16, pad=20)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig
    
    def plot_header_efficiency_comparison(self, payload_sizes: List[int] = None,
                                        save_path: Optional[str] = None) -> plt.Figure:
        """Plot header efficiency comparison across payload sizes."""
        
        if payload_sizes is None:
            payload_sizes = [1, 4, 16, 64, 256, 1024]
        
        protocols = self.protocol_db.get_protocol_names()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for protocol_name in protocols:
            protocol = self.protocol_db.get_protocol(protocol_name)
            efficiencies = []
            
            for payload_size in payload_sizes:
                if protocol_name == "µACP":
                    # Use µACP-specific calculation
                    efficiency = 8 / (8 + payload_size)
                else:
                    efficiency = protocol.header_size_min / (protocol.header_size_min + payload_size)
                efficiencies.append(efficiency)
            
            ax.plot(payload_sizes, efficiencies, 'o-', linewidth=2, 
                   label=protocol_name, markersize=6)
        
        ax.set_xscale('log')
        ax.set_xlabel('Payload Size (bytes)')
        ax.set_ylabel('Header Efficiency (η_h)')
        ax.set_title('Header Efficiency vs Payload Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add theoretical bounds
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect Efficiency')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig
    
    def plot_protocol_type_distribution(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot distribution of protocols by type."""
        
        summary = self.protocol_db.get_protocol_summary()
        type_counts = summary["by_type"]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart
        labels = [t.replace('_', ' ').title() for t in type_counts.keys()]
        sizes = list(type_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Protocol Distribution by Type')
        
        # Bar chart
        x_pos = np.arange(len(labels))
        bars = ax2.bar(x_pos, sizes, color=colors, alpha=0.7)
        ax2.set_xlabel('Protocol Type')
        ax2.set_ylabel('Number of Protocols')
        ax2.set_title('Protocol Count by Type')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    str(size), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig
    
    def plot_feature_comparison_matrix(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create a heatmap showing feature support across protocols."""
        
        protocols = self.protocol_db.get_protocol_names()
        features = ['Pub/Sub', 'RPC', 'Streaming', 'QoS', 'Discovery']
        
        # Create feature matrix
        feature_matrix = []
        for protocol_name in protocols:
            protocol = self.protocol_db.get_protocol(protocol_name)
            row = [
                protocol.supports_pubsub,
                protocol.supports_rpc,
                protocol.supports_streaming,
                protocol.supports_qos,
                protocol.supports_discovery
            ]
            feature_matrix.append(row)
        
        # Convert to DataFrame for better labeling
        df = pd.DataFrame(feature_matrix, index=protocols, columns=features)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(df, annot=True, cmap='RdYlGn', cbar_kws={'label': 'Feature Support'},
                   ax=ax, fmt='d', linewidths=0.5)
        
        ax.set_title('Protocol Feature Support Matrix')
        ax.set_xlabel('Features')
        ax.set_ylabel('Protocols')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig
    
    def plot_scalability_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot scalability comparison across protocols."""
        
        protocols = self.protocol_db.get_protocol_names()
        
        # Get scalability data
        names = []
        max_agents = []
        memory_per_agent = []
        
        for protocol_name in protocols:
            protocol = self.protocol_db.get_protocol(protocol_name)
            
            # Estimate scalability based on state complexity (realistic)
            if protocol.state_complexity == "O(1)":
                max_agents.append(1000000)
                memory_per_agent.append(1024)  # 1 KB
            elif protocol.state_complexity == "O(W)":
                max_agents.append(100000)
                memory_per_agent.append(8192)  # 8 KB
            elif protocol.state_complexity == "O(topics)":
                max_agents.append(10000)
                memory_per_agent.append(32768)  # 32 KB
            else:  # App-level
                max_agents.append(1000)
                memory_per_agent.append(262144)  # 256 KB
            
            names.append(protocol_name)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Max agents comparison
        bars1 = ax1.bar(names, max_agents, color=plt.cm.Set3(np.linspace(0, 1, len(names))))
        ax1.set_ylabel('Maximum Agents')
        ax1.set_title('Scalability: Maximum Agents')
        ax1.set_yscale('log')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_xticklabels(ax1.get_xticklabels(), ha='right')
        
        # Memory per agent comparison
        bars2 = ax2.bar(names, memory_per_agent, color=plt.cm.Set3(np.linspace(0, 1, len(names))))
        ax2.set_ylabel('Memory per Agent (bytes)')
        ax2.set_title('Scalability: Memory per Agent')
        ax2.set_yscale('log')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_xticklabels(ax2.get_xticklabels(), ha='right')
        
        # Add value labels
        for bar, value in zip(bars1, max_agents):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                    f'{value:,}', ha='center', va='bottom', fontsize=8)
        
        for bar, value in zip(bars2, memory_per_agent):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                    f'{value}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig
    
    def plot_energy_efficiency_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot energy efficiency comparison across protocols."""
        
        protocols = self.protocol_db.get_protocol_names()
        
        # Calculate energy per message for each protocol
        energy_data = []
        names = []
        
        for protocol_name in protocols:
            protocol = self.protocol_db.get_protocol(protocol_name)
            
            # Estimate energy consumption
            # Energy ∝ message_size * transmission_time
            message_size = protocol.header_size_min + 16  # 16-byte payload
            energy_per_message = message_size * 0.1  # 0.1 µJ per byte
            
            energy_data.append(energy_per_message)
            names.append(protocol_name)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bar chart
        bars = ax.bar(names, energy_data, color=plt.cm.Set3(np.linspace(0, 1, len(names))))
        ax.set_ylabel('Energy per Message (µJ)')
        ax.set_title('Energy Efficiency Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.set_xticklabels(ax.get_xticklabels(), ha='right')
        
        # Add value labels
        for bar, value in zip(bars, energy_data):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # Add horizontal line for µACP
        if "µACP" in names:
            uacp_idx = names.index("µACP")
            uacp_energy = energy_data[uacp_idx]
            ax.axhline(y=uacp_energy, color='red', linestyle='--', alpha=0.7, 
                      label=f'µACP baseline ({uacp_energy:.1f} µJ)')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig
    
    def create_comprehensive_dashboard(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create a comprehensive dashboard with multiple visualizations."""
        
        fig = plt.figure(figsize=(20, 16))
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Protocol type distribution (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        summary = self.protocol_db.get_protocol_summary()
        type_counts = summary["by_type"]
        labels = [t.replace('_', ' ').title() for t in type_counts.keys()]
        sizes = list(type_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Protocol Distribution by Type')
        
        # 2. Feature support matrix (top center)
        ax2 = fig.add_subplot(gs[0, 1:])
        protocols = self.protocol_db.get_protocol_names()
        features = ['Pub/Sub', 'RPC', 'Streaming', 'QoS', 'Discovery']
        feature_matrix = []
        for protocol_name in protocols:
            protocol = self.protocol_db.get_protocol(protocol_name)
            row = [
                protocol.supports_pubsub,
                protocol.supports_rpc,
                protocol.supports_streaming,
                protocol.supports_qos,
                protocol.supports_discovery
            ]
            feature_matrix.append(row)
        df = pd.DataFrame(feature_matrix, index=protocols, columns=features)
        sns.heatmap(df, annot=True, cmap='RdYlGn', cbar_kws={'label': 'Feature Support'},
                   ax=ax2, fmt='d', linewidths=0.5)
        ax2.set_title('Feature Support Matrix')
        
        # 3. Header efficiency comparison (middle row)
        ax3 = fig.add_subplot(gs[1, :])
        payload_sizes = [1, 4, 16, 64, 256, 1024]
        for protocol_name in protocols:
            protocol = self.protocol_db.get_protocol(protocol_name)
            efficiencies = []
            for payload_size in payload_sizes:
                if protocol_name == "µACP":
                    efficiency = 8 / (8 + payload_size)
                else:
                    efficiency = protocol.header_size_min / (protocol.header_size_min + payload_size)
                efficiencies.append(efficiency)
            ax3.plot(payload_sizes, efficiencies, 'o-', linewidth=2, 
                    label=protocol_name, markersize=6)
        ax3.set_xscale('log')
        ax3.set_xlabel('Payload Size (bytes)')
        ax3.set_ylabel('Header Efficiency (η_h)')
        ax3.set_title('Header Efficiency vs Payload Size')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. Scalability comparison (bottom left)
        ax4 = fig.add_subplot(gs[2, 0])
        max_agents = []
        for protocol_name in protocols:
            protocol = self.protocol_db.get_protocol(protocol_name)
            if protocol.state_complexity == "O(1)":
                max_agents.append(1000000)
            elif protocol.state_complexity == "O(W)":
                max_agents.append(100000)
            elif protocol.state_complexity == "O(topics)":
                max_agents.append(10000)
            else:
                max_agents.append(1000)
        bars = ax4.bar(protocols, max_agents, color=plt.cm.Set3(np.linspace(0, 1, len(protocols))))
        ax4.set_ylabel('Maximum Agents')
        ax4.set_title('Scalability: Maximum Agents')
        ax4.set_yscale('log')
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_xticklabels(ax4.get_xticklabels(), ha='right')
        
        # 5. Energy efficiency (bottom center)
        ax5 = fig.add_subplot(gs[2, 1])
        energy_data = []
        for protocol_name in protocols:
            protocol = self.protocol_db.get_protocol(protocol_name)
            message_size = protocol.header_size_min + 16
            energy_per_message = message_size * 0.1
            energy_data.append(energy_per_message)
        bars = ax5.bar(protocols, energy_data, color=plt.cm.Set3(np.linspace(0, 1, len(protocols))))
        ax5.set_ylabel('Energy per Message (µJ)')
        ax5.set_title('Energy Efficiency')
        ax5.tick_params(axis='x', rotation=45)
        ax5.set_xticklabels(ax5.get_xticklabels(), ha='right')
        
        # 6. Protocol positioning (bottom right)
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        # Create positioning text
        positioning_text = "Protocol Positioning:\n\n"
        positioning_text += "• IoT-First: MQTT, CoAP\n"
        positioning_text += "  - Device-oriented, lightweight\n"
        positioning_text += "  - Limited agent semantics\n\n"
        positioning_text += "• Agent-First: MCP, FIPA-ACL\n"
        positioning_text += "  - Rich agent interactions\n"
        positioning_text += "  - Heavyweight protocols\n\n"
        positioning_text += "• Edge-Native Agent: µACP\n"
        positioning_text += "  - Agent-oriented + lightweight\n"
        positioning_text += "  - Best of both worlds"
        
        ax6.text(0.1, 0.5, positioning_text, transform=ax6.transAxes, 
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        fig.suptitle('µACP Protocol Analysis Dashboard', fontsize=20, y=0.98)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig
