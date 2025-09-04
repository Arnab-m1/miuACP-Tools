"""
Advanced Features for µACP-Tools

Provides advanced features for µACP development including:
- TinyML integration
- Intermittent connectivity handling
- Advanced visualization
- CLI improvements
"""

from .tinyml_integration import TinyMLIntegration, MLModel, EdgeInference
from .connectivity_manager import ConnectivityManager, ConnectionState, OfflineQueue
from .advanced_visualization import AdvancedVisualization, NetworkTopology, PerformanceDashboard
from .cli_enhancements import CLIEnhancements, InteractiveMode, ProgressIndicator

__all__ = [
    'TinyMLIntegration',
    'MLModel',
    'EdgeInference',
    'ConnectivityManager',
    'ConnectionState',
    'OfflineQueue',
    'AdvancedVisualization',
    'NetworkTopology',
    'PerformanceDashboard',
    'CLIEnhancements',
    'InteractiveMode',
    'ProgressIndicator'
]

__version__ = "1.0.0"
