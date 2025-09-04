"""
Industry Validation Tools for µACP

Provides industry validation capabilities for µACP protocol including:
- Use case validation
- Performance certification
- Industry standard compliance
- Real-world deployment testing
"""

from .use_case_validator import UseCaseValidator, UseCase, ValidationResult
from .performance_certifier import PerformanceCertifier, CertificationResult, PerformanceStandard
from .industry_compliance import IndustryCompliance, ComplianceResult, ComplianceStandard
from .deployment_tester import DeploymentTester, DeploymentResult, DeploymentScenario

__all__ = [
    'UseCaseValidator',
    'UseCase',
    'ValidationResult',
    'PerformanceCertifier',
    'CertificationResult',
    'PerformanceStandard',
    'IndustryCompliance',
    'ComplianceResult',
    'ComplianceStandard',
    'DeploymentTester',
    'DeploymentResult',
    'DeploymentScenario'
]

__version__ = "1.0.0"
