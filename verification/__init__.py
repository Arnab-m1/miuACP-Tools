"""
Formal Verification for µACP

Provides formal verification capabilities for µACP protocol including:
- Security property verification
- Protocol correctness proofs
- Model checking
- Formal specification generation
"""

from .security_verification import SecurityVerifier, SecurityProperty, SecurityProof
from .protocol_verification import ProtocolVerifier, ProtocolProperty, CorrectnessProof
from .model_checker import ModelChecker, ModelState, VerificationResult
from .specification_generator import SpecificationGenerator, TLAplusSpec, CoqSpec

__all__ = [
    'SecurityVerifier',
    'SecurityProperty', 
    'SecurityProof',
    'ProtocolVerifier',
    'ProtocolProperty',
    'CorrectnessProof',
    'ModelChecker',
    'ModelState',
    'VerificationResult',
    'SpecificationGenerator',
    'TLAplusSpec',
    'CoqSpec'
]

__version__ = "1.0.0"
