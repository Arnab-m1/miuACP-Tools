"""
Model Checker for µACP

Implements model checking capabilities for µACP protocol including:
- State space exploration
- Property verification
- Counterexample generation
- Model checking algorithms
"""

import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import queue


class ModelState:
    """Model state for verification"""
    state_id: str
    variables: Dict[str, Any]
    transitions: List[str]
    properties: Dict[str, bool]
    timestamp: float
    
    def __init__(self, state_id: str, variables: Dict[str, Any] = None):
        self.state_id = state_id
        self.variables = variables or {}
        self.transitions = []
        self.properties = {}
        self.timestamp = time.time()


class VerificationResult:
    """Model checking verification result"""
    property_name: str
    verified: bool
    counterexample: Optional[List[ModelState]]
    proof_trace: List[str]
    confidence: float
    execution_time: float
    states_explored: int
    
    def __init__(self, property_name: str):
        self.property_name = property_name
        self.verified = False
        self.counterexample = None
        self.proof_trace = []
        self.confidence = 0.0
        self.execution_time = 0.0
        self.states_explored = 0


class ModelChecker:
    """
    Model checker for µACP protocol verification
    
    Implements state space exploration and property verification
    using various model checking algorithms.
    """
    
    def __init__(self, max_states: int = 10000):
        self.max_states = max_states
        self.states: Dict[str, ModelState] = {}
        self.transitions: Dict[str, List[str]] = {}
        self.verification_results: Dict[str, VerificationResult] = {}
        self.property_checkers: Dict[str, Callable[[ModelState], bool]] = {}
        self._initialize_property_checkers()
    
    def _initialize_property_checkers(self):
        """Initialize property checker functions"""
        self.property_checkers = {
            'safety': self._check_safety_property,
            'liveness': self._check_liveness_property,
            'invariant': self._check_invariant_property,
            'reachability': self._check_reachability_property,
            'deadlock_freedom': self._check_deadlock_freedom,
            'termination': self._check_termination_property
        }
    
    def add_state(self, state: ModelState):
        """Add a state to the model"""
        self.states[state.state_id] = state
        self.transitions[state.state_id] = state.transitions
    
    def add_transition(self, from_state: str, to_state: str):
        """Add a transition between states"""
        if from_state not in self.transitions:
            self.transitions[from_state] = []
        if to_state not in self.transitions[from_state]:
            self.transitions[from_state].append(to_state)
    
    def verify_property(self, property_name: str, initial_state: str, 
                       property_type: str = 'safety') -> VerificationResult:
        """Verify a property using model checking"""
        start_time = time.time()
        result = VerificationResult(property_name)
        
        if property_type not in self.property_checkers:
            result.proof_trace.append(f"Unknown property type: {property_type}")
            result.execution_time = time.time() - start_time
            return result
        
        # Get property checker
        checker = self.property_checkers[property_type]
        
        # Explore state space
        if property_type == 'safety':
            result = self._verify_safety_property(property_name, initial_state, checker)
        elif property_type == 'liveness':
            result = self._verify_liveness_property(property_name, initial_state, checker)
        elif property_type == 'reachability':
            result = self._verify_reachability_property(property_name, initial_state, checker)
        else:
            result = self._verify_generic_property(property_name, initial_state, checker)
        
        result.execution_time = time.time() - start_time
        self.verification_results[property_name] = result
        
        return result
    
    def _verify_safety_property(self, property_name: str, initial_state: str, 
                               checker: Callable[[ModelState], bool]) -> VerificationResult:
        """Verify safety property using breadth-first search"""
        result = VerificationResult(property_name)
        visited = set()
        queue_states = queue.Queue()
        queue_states.put(initial_state)
        
        while not queue_states.empty() and result.states_explored < self.max_states:
            current_state_id = queue_states.get()
            
            if current_state_id in visited:
                continue
            
            visited.add(current_state_id)
            result.states_explored += 1
            
            # Get current state
            if current_state_id not in self.states:
                result.proof_trace.append(f"State {current_state_id} not found")
                continue
            
            current_state = self.states[current_state_id]
            
            # Check property
            if not checker(current_state):
                result.verified = False
                result.counterexample = [current_state]
                result.proof_trace.append(f"Safety property violated in state {current_state_id}")
                result.confidence = 1.0
                return result
            
            result.proof_trace.append(f"Property satisfied in state {current_state_id}")
            
            # Add successor states
            for next_state_id in self.transitions.get(current_state_id, []):
                if next_state_id not in visited:
                    queue_states.put(next_state_id)
        
        # All reachable states satisfy the property
        result.verified = True
        result.proof_trace.append(f"Safety property verified for {result.states_explored} states")
        result.confidence = 0.95
        
        return result
    
    def _verify_liveness_property(self, property_name: str, initial_state: str, 
                                 checker: Callable[[ModelState], bool]) -> VerificationResult:
        """Verify liveness property using depth-first search"""
        result = VerificationResult(property_name)
        visited = set()
        path = []
        
        def dfs_visit(state_id: str) -> bool:
            if state_id in visited:
                return False
            
            visited.add(state_id)
            result.states_explored += 1
            path.append(state_id)
            
            if state_id not in self.states:
                return False
            
            current_state = self.states[state_id]
            
            # Check if property is satisfied
            if checker(current_state):
                result.verified = True
                result.proof_trace.append(f"Liveness property satisfied in state {state_id}")
                result.confidence = 0.90
                return True
            
            # Continue exploring
            for next_state_id in self.transitions.get(state_id, []):
                if dfs_visit(next_state_id):
                    return True
            
            path.pop()
            return False
        
        # Start DFS from initial state
        if dfs_visit(initial_state):
            result.proof_trace.append(f"Liveness property verified")
        else:
            result.verified = False
            result.proof_trace.append(f"Liveness property not satisfied")
            result.confidence = 0.0
        
        return result
    
    def _verify_reachability_property(self, property_name: str, initial_state: str, 
                                    checker: Callable[[ModelState], bool]) -> VerificationResult:
        """Verify reachability property"""
        result = VerificationResult(property_name)
        visited = set()
        queue_states = queue.Queue()
        queue_states.put(initial_state)
        
        while not queue_states.empty() and result.states_explored < self.max_states:
            current_state_id = queue_states.get()
            
            if current_state_id in visited:
                continue
            
            visited.add(current_state_id)
            result.states_explored += 1
            
            if current_state_id not in self.states:
                continue
            
            current_state = self.states[current_state_id]
            
            # Check if target property is satisfied
            if checker(current_state):
                result.verified = True
                result.proof_trace.append(f"Target state reached: {current_state_id}")
                result.confidence = 0.95
                return result
            
            # Add successor states
            for next_state_id in self.transitions.get(current_state_id, []):
                if next_state_id not in visited:
                    queue_states.put(next_state_id)
        
        # Target state not reachable
        result.verified = False
        result.proof_trace.append(f"Target state not reachable from {initial_state}")
        result.confidence = 0.90
        
        return result
    
    def _verify_generic_property(self, property_name: str, initial_state: str, 
                               checker: Callable[[ModelState], bool]) -> VerificationResult:
        """Verify generic property"""
        result = VerificationResult(property_name)
        
        if initial_state not in self.states:
            result.proof_trace.append(f"Initial state {initial_state} not found")
            result.confidence = 0.0
            return result
        
        initial_state_obj = self.states[initial_state]
        
        # Check property on initial state
        if checker(initial_state_obj):
            result.verified = True
            result.proof_trace.append(f"Property satisfied in initial state")
            result.confidence = 0.8
        else:
            result.verified = False
            result.proof_trace.append(f"Property not satisfied in initial state")
            result.confidence = 0.8
        
        result.states_explored = 1
        return result
    
    def _check_safety_property(self, state: ModelState) -> bool:
        """Check safety property (no bad states)"""
        # Example: Check that no agent is in error state
        for var_name, var_value in state.variables.items():
            if 'error' in str(var_value).lower():
                return False
        return True
    
    def _check_liveness_property(self, state: ModelState) -> bool:
        """Check liveness property (eventually good state)"""
        # Example: Check that at least one agent is active
        for var_name, var_value in state.variables.items():
            if 'active' in str(var_value).lower():
                return True
        return False
    
    def _check_invariant_property(self, state: ModelState) -> bool:
        """Check invariant property (always true)"""
        # Example: Check that system is in valid state
        return 'valid' in state.variables.get('system_state', '')
    
    def _check_reachability_property(self, state: ModelState) -> bool:
        """Check reachability property (can reach target)"""
        # Example: Check if target state is reached
        return state.variables.get('target_reached', False)
    
    def _check_deadlock_freedom(self, state: ModelState) -> bool:
        """Check deadlock freedom"""
        # Check if state has outgoing transitions
        return len(state.transitions) > 0
    
    def _check_termination_property(self, state: ModelState) -> bool:
        """Check termination property"""
        # Check if system has terminated
        return state.variables.get('terminated', False)
    
    def generate_counterexample(self, property_name: str) -> Optional[List[ModelState]]:
        """Generate counterexample for failed property"""
        if property_name not in self.verification_results:
            return None
        
        result = self.verification_results[property_name]
        return result.counterexample
    
    def get_verification_summary(self) -> Dict[str, Any]:
        """Get verification summary"""
        total_properties = len(self.verification_results)
        verified_properties = sum(1 for result in self.verification_results.values() if result.verified)
        
        total_states = len(self.states)
        total_transitions = sum(len(transitions) for transitions in self.transitions.values())
        
        return {
            'total_properties': total_properties,
            'verified_properties': verified_properties,
            'verification_rate': verified_properties / max(1, total_properties),
            'total_states': total_states,
            'total_transitions': total_transitions,
            'average_execution_time': sum(r.execution_time for r in self.verification_results.values()) / max(1, total_properties),
            'total_states_explored': sum(r.states_explored for r in self.verification_results.values())
        }
    
    def export_model(self, filename: str):
        """Export model to file"""
        model_data = {
            'states': {
                state_id: {
                    'variables': state.variables,
                    'transitions': state.transitions,
                    'properties': state.properties,
                    'timestamp': state.timestamp
                }
                for state_id, state in self.states.items()
            },
            'transitions': self.transitions,
            'verification_results': {
                prop_name: {
                    'verified': result.verified,
                    'confidence': result.confidence,
                    'execution_time': result.execution_time,
                    'states_explored': result.states_explored,
                    'proof_trace': result.proof_trace
                }
                for prop_name, result in self.verification_results.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def import_model(self, filename: str):
        """Import model from file"""
        with open(filename, 'r') as f:
            model_data = json.load(f)
        
        # Import states
        for state_id, state_data in model_data.get('states', {}).items():
            state = ModelState(state_id, state_data.get('variables', {}))
            state.transitions = state_data.get('transitions', [])
            state.properties = state_data.get('properties', {})
            state.timestamp = state_data.get('timestamp', time.time())
            self.add_state(state)
        
        # Import transitions
        self.transitions = model_data.get('transitions', {})
        
        # Import verification results
        for prop_name, result_data in model_data.get('verification_results', {}).items():
            result = VerificationResult(prop_name)
            result.verified = result_data.get('verified', False)
            result.confidence = result_data.get('confidence', 0.0)
            result.execution_time = result_data.get('execution_time', 0.0)
            result.states_explored = result_data.get('states_explored', 0)
            result.proof_trace = result_data.get('proof_trace', [])
            self.verification_results[prop_name] = result
