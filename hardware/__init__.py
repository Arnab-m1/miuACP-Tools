"""
Hardware Testing Framework for µACP

Provides hardware testing capabilities for µACP protocol including:
- ESP32-C3 device testing
- Hardware-in-the-loop testing
- Performance measurement
- Real device validation
"""

from .esp32_tester import ESP32Tester, ESP32TestResult, ESP32Config
from .hardware_validator import HardwareValidator, ValidationResult, DeviceProfile
from .performance_measurement import PerformanceMeasurement, MeasurementResult, HardwareMetrics

__all__ = [
    'ESP32Tester',
    'ESP32TestResult', 
    'ESP32Config',
    'HardwareValidator',
    'ValidationResult',
    'DeviceProfile',
    'PerformanceMeasurement',
    'MeasurementResult',
    'HardwareMetrics'
]

__version__ = "1.0.0"
