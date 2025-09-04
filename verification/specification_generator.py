"""
Specification Generator for µACP

Generates formal specifications for µACP protocol including:
- TLA+ specifications
- Coq formalizations
- Model specifications
- Formal documentation
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json


class SpecificationType(Enum):
    """Types of formal specifications"""
    TLA_PLUS = "tla_plus"
    COQ = "coq"
    ALLOY = "alloy"
    PROMELA = "promela"
    CSP = "csp"


class SpecificationLanguage(Enum):
    """Formal specification languages"""
    TLA_PLUS = "tla+"
    COQ = "coq"
    ALLOY = "alloy"
    PROMELA = "promela"
    CSP = "csp"


@dataclass
class TLAplusSpec:
    """TLA+ specification"""
    spec_name: str
    variables: List[str]
    invariants: List[str]
    actions: List[str]
    temporal_formulas: List[str]
    assumptions: List[str]
    theorems: List[str]
    timestamp: float


@dataclass
class CoqSpec:
    """Coq specification"""
    spec_name: str
    definitions: List[str]
    lemmas: List[str]
    theorems: List[str]
    proofs: List[str]
    imports: List[str]
    timestamp: float


class SpecificationGenerator:
    """
    Formal specification generator for µACP
    
    Generates formal specifications in various languages
    for protocol verification and documentation.
    """
    
    def __init__(self):
        self.specifications: Dict[str, Any] = {}
        self.templates: Dict[SpecificationType, str] = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize specification templates"""
        self.templates = {
            SpecificationType.TLA_PLUS: self._get_tla_plus_template(),
            SpecificationType.COQ: self._get_coq_template(),
            SpecificationType.ALLOY: self._get_alloy_template(),
            SpecificationType.PROMELA: self._get_promela_template(),
            SpecificationType.CSP: self._get_csp_template()
        }
    
    def generate_tla_plus_spec(self, protocol_model: Dict[str, Any]) -> TLAplusSpec:
        """Generate TLA+ specification"""
        spec_name = protocol_model.get('name', 'miuACP')
        
        # Extract variables from protocol model
        variables = self._extract_tla_variables(protocol_model)
        
        # Generate invariants
        invariants = self._generate_tla_invariants(protocol_model)
        
        # Generate actions
        actions = self._generate_tla_actions(protocol_model)
        
        # Generate temporal formulas
        temporal_formulas = self._generate_tla_temporal_formulas(protocol_model)
        
        # Generate assumptions
        assumptions = self._generate_tla_assumptions(protocol_model)
        
        # Generate theorems
        theorems = self._generate_tla_theorems(protocol_model)
        
        spec = TLAplusSpec(
            spec_name=spec_name,
            variables=variables,
            invariants=invariants,
            actions=actions,
            temporal_formulas=temporal_formulas,
            assumptions=assumptions,
            theorems=theorems,
            timestamp=time.time()
        )
        
        self.specifications[f"{spec_name}_tla"] = spec
        return spec
    
    def generate_coq_spec(self, protocol_model: Dict[str, Any]) -> CoqSpec:
        """Generate Coq specification"""
        spec_name = protocol_model.get('name', 'miuACP')
        
        # Generate definitions
        definitions = self._generate_coq_definitions(protocol_model)
        
        # Generate lemmas
        lemmas = self._generate_coq_lemmas(protocol_model)
        
        # Generate theorems
        theorems = self._generate_coq_theorems(protocol_model)
        
        # Generate proofs
        proofs = self._generate_coq_proofs(protocol_model)
        
        # Generate imports
        imports = self._generate_coq_imports(protocol_model)
        
        spec = CoqSpec(
            spec_name=spec_name,
            definitions=definitions,
            lemmas=lemmas,
            theorems=theorems,
            proofs=proofs,
            imports=imports,
            timestamp=time.time()
        )
        
        self.specifications[f"{spec_name}_coq"] = spec
        return spec
    
    def _extract_tla_variables(self, protocol_model: Dict[str, Any]) -> List[str]:
        """Extract TLA+ variables from protocol model"""
        variables = []
        
        # Add basic protocol variables
        variables.extend([
            "agents",
            "messages", 
            "channels",
            "state",
            "clock"
        ])
        
        # Add agent-specific variables
        if 'agents' in protocol_model:
            for agent in protocol_model['agents']:
                agent_id = agent.get('id', 'agent')
                variables.extend([
                    f"{agent_id}_state",
                    f"{agent_id}_messages",
                    f"{agent_id}_clock"
                ])
        
        return variables
    
    def _generate_tla_invariants(self, protocol_model: Dict[str, Any]) -> List[str]:
        """Generate TLA+ invariants"""
        invariants = [
            "TypeOK",
            "MessageIntegrity",
            "AgentConsistency",
            "ChannelIntegrity"
        ]
        
        # Add protocol-specific invariants
        if 'invariants' in protocol_model:
            invariants.extend(protocol_model['invariants'])
        
        return invariants
    
    def _generate_tla_actions(self, protocol_model: Dict[str, Any]) -> List[str]:
        """Generate TLA+ actions"""
        actions = [
            "SendMessage",
            "ReceiveMessage", 
            "ProcessMessage",
            "UpdateState",
            "Tick"
        ]
        
        # Add protocol-specific actions
        if 'actions' in protocol_model:
            actions.extend(protocol_model['actions'])
        
        return actions
    
    def _generate_tla_temporal_formulas(self, protocol_model: Dict[str, Any]) -> List[str]:
        """Generate TLA+ temporal formulas"""
        temporal_formulas = [
            "Spec == Init /\\ [][Next]_vars /\\ WF_vars(Tick)",
            "Liveness == <>[]Terminated",
            "Safety == []Invariant"
        ]
        
        return temporal_formulas
    
    def _generate_tla_assumptions(self, protocol_model: Dict[str, Any]) -> List[str]:
        """Generate TLA+ assumptions"""
        assumptions = [
            "ASSUME Cardinality(agents) > 0",
            "ASSUME Cardinality(channels) > 0",
            "ASSUME \\A a \\in agents : a.state \\in ValidStates"
        ]
        
        return assumptions
    
    def _generate_tla_theorems(self, protocol_model: Dict[str, Any]) -> List[str]:
        """Generate TLA+ theorems"""
        theorems = [
            "THEOREM Spec => []TypeOK",
            "THEOREM Spec => []MessageIntegrity",
            "THEOREM Spec => Liveness"
        ]
        
        return theorems
    
    def _generate_coq_definitions(self, protocol_model: Dict[str, Any]) -> List[str]:
        """Generate Coq definitions"""
        definitions = [
            "Inductive AgentState : Type :=",
            "  | Active : AgentState",
            "  | Inactive : AgentState",
            "  | Error : AgentState.",
            "",
            "Inductive MessageType : Type :=",
            "  | TELL : MessageType",
            "  | ASK : MessageType",
            "  | OBSERVE : MessageType.",
            "",
            "Record Message : Type :=",
            "  { msg_type : MessageType;",
            "    msg_sender : nat;",
            "    msg_receiver : nat;",
            "    msg_content : string;",
            "    msg_timestamp : nat }."
        ]
        
        return definitions
    
    def _generate_coq_lemmas(self, protocol_model: Dict[str, Any]) -> List[str]:
        """Generate Coq lemmas"""
        lemmas = [
            "Lemma message_integrity : forall m : Message,",
            "  msg_timestamp m > 0.",
            "",
            "Lemma agent_consistency : forall a : Agent,",
            "  agent_state a \\in ValidStates."
        ]
        
        return lemmas
    
    def _generate_coq_theorems(self, protocol_model: Dict[str, Any]) -> List[str]:
        """Generate Coq theorems"""
        theorems = [
            "Theorem protocol_safety : forall s : SystemState,",
            "  Invariant s -> Safe s.",
            "",
            "Theorem protocol_liveness : forall s : SystemState,",
            "  Reachable s -> Eventually Terminated s."
        ]
        
        return theorems
    
    def _generate_coq_proofs(self, protocol_model: Dict[str, Any]) -> List[str]:
        """Generate Coq proofs"""
        proofs = [
            "Proof.",
            "  intros s H.",
            "  unfold Invariant in H.",
            "  apply H.",
            "Qed.",
            "",
            "Proof.",
            "  intros s H.",
            "  induction H.",
            "  - apply eventually_terminated.",
            "  - apply IHReachable.",
            "Qed."
        ]
        
        return proofs
    
    def _generate_coq_imports(self, protocol_model: Dict[str, Any]) -> List[str]:
        """Generate Coq imports"""
        imports = [
            "Require Import Coq.Arith.Arith.",
            "Require Import Coq.Strings.String.",
            "Require Import Coq.Lists.List."
        ]
        
        return imports
    
    def _get_tla_plus_template(self) -> str:
        """Get TLA+ template"""
        return """
---- MODULE {spec_name} ----
EXTENDS Naturals, Sequences, FiniteSets

VARIABLES {variables}

TypeOK == 
    /\\ agents \\in SUBSET Agent
    /\\ messages \\in Seq(Message)
    /\\ channels \\in SUBSET Channel
    /\\ state \\in SystemState
    /\\ clock \\in Nat

Init == 
    /\\ agents = {{}}
    /\\ messages = <<>>
    /\\ channels = {{}}
    /\\ state = InitialState
    /\\ clock = 0

Next == 
    \\/ SendMessage
    \\/ ReceiveMessage
    \\/ ProcessMessage
    \\/ UpdateState
    \\/ Tick

SendMessage == 
    /\\ \\E a \\in agents : 
        /\\ a.state = Active
        /\\ messages' = Append(messages, CreateMessage(a))
        /\\ UNCHANGED <<agents, channels, state, clock>>

ReceiveMessage == 
    /\\ Len(messages) > 0
    /\\ \\E a \\in agents :
        /\\ a.state = Active
        /\\ ProcessMessage(Head(messages), a)
        /\\ messages' = Tail(messages)
        /\\ UNCHANGED <<agents, channels, state, clock>>

Tick == 
    /\\ clock' = clock + 1
    /\\ UNCHANGED <<agents, messages, channels, state>>

Spec == Init /\\ [][Next]_vars /\\ WF_vars(Tick)

THEOREM Spec => []TypeOK

====
"""
    
    def _get_coq_template(self) -> str:
        """Get Coq template"""
        return """
Require Import Coq.Arith.Arith.
Require Import Coq.Strings.String.
Require Import Coq.Lists.List.

{definitions}

{lemmas}

{theorems}

{proofs}
"""
    
    def _get_alloy_template(self) -> str:
        """Get Alloy template"""
        return """
module {spec_name}

sig Agent {{
    state: State,
    messages: set Message,
    clock: Int
}}

sig Message {{
    sender: Agent,
    receiver: Agent,
    content: String,
    timestamp: Int
}}

sig Channel {{
    agents: set Agent
}}

fact MessageIntegrity {{
    all m: Message | m.timestamp > 0
}}

fact AgentConsistency {{
    all a: Agent | a.state in ValidStates
}}

pred ProtocolSafety {{
    all a: Agent | a.state != Error
}}

pred ProtocolLiveness {{
    some a: Agent | a.state = Terminated
}}
"""
    
    def _get_promela_template(self) -> str:
        """Get Promela template"""
        return """
mtype = {{ TELL, ASK, OBSERVE }};

chan channels[N] = [1] of {{ mtype, int, string }};

active proctype Agent(int id) {{
    mtype msg_type;
    int sender;
    string content;
    
    do
    :: channels[id] ? msg_type, sender, content ->
        if
        :: msg_type == TELL -> process_tell(sender, content)
        :: msg_type == ASK -> process_ask(sender, content)
        :: msg_type == OBSERVE -> process_observe(sender, content)
        fi
    od
}}

proctype process_tell(int sender, string content) {{
    // Process TELL message
}}

proctype process_ask(int sender, string content) {{
    // Process ASK message
}}

proctype process_observe(int sender, string content) {{
    // Process OBSERVE message
}}
"""
    
    def _get_csp_template(self) -> str:
        """Get CSP template"""
        return """
channel tell, ask, observe

Agent(id) = 
    tell?from!content -> Agent(id)
    []
    ask?from!content -> Agent(id)
    []
    observe?from!content -> Agent(id)

System = ||| id:{{0..N}} @ Agent(id)

assert System [T= System
"""
    
    def export_specification(self, spec_name: str, filename: str, 
                           spec_type: SpecificationType):
        """Export specification to file"""
        if spec_name not in self.specifications:
            return False
        
        spec = self.specifications[spec_name]
        
        if spec_type == SpecificationType.TLA_PLUS:
            content = self._format_tla_plus_spec(spec)
        elif spec_type == SpecificationType.COQ:
            content = self._format_coq_spec(spec)
        else:
            content = self._format_generic_spec(spec, spec_type)
        
        with open(filename, 'w') as f:
            f.write(content)
        
        return True
    
    def _format_tla_plus_spec(self, spec: TLAplusSpec) -> str:
        """Format TLA+ specification"""
        template = self.templates[SpecificationType.TLA_PLUS]
        
        variables_str = ", ".join(spec.variables)
        invariants_str = " /\\ ".join(spec.invariants)
        actions_str = " \\/ ".join(spec.actions)
        temporal_str = " /\\ ".join(spec.temporal_formulas)
        assumptions_str = "\n".join(f"ASSUME {assumption}" for assumption in spec.assumptions)
        theorems_str = "\n".join(f"THEOREM {theorem}" for theorem in spec.theorems)
        
        return template.format(
            spec_name=spec.spec_name,
            variables=variables_str,
            invariants=invariants_str,
            actions=actions_str,
            temporal=temporal_str,
            assumptions=assumptions_str,
            theorems=theorems_str
        )
    
    def _format_coq_spec(self, spec: CoqSpec) -> str:
        """Format Coq specification"""
        template = self.templates[SpecificationType.COQ]
        
        definitions_str = "\n".join(spec.definitions)
        lemmas_str = "\n".join(spec.lemmas)
        theorems_str = "\n".join(spec.theorems)
        proofs_str = "\n".join(spec.proofs)
        
        return template.format(
            definitions=definitions_str,
            lemmas=lemmas_str,
            theorems=theorems_str,
            proofs=proofs_str
        )
    
    def _format_generic_spec(self, spec: Any, spec_type: SpecificationType) -> str:
        """Format generic specification"""
        template = self.templates[spec_type]
        return template.format(spec_name=getattr(spec, 'spec_name', 'miuACP'))
    
    def get_specification(self, spec_name: str) -> Optional[Any]:
        """Get specification by name"""
        return self.specifications.get(spec_name)
    
    def get_all_specifications(self) -> Dict[str, Any]:
        """Get all specifications"""
        return self.specifications.copy()
    
    def get_generation_summary(self) -> Dict[str, Any]:
        """Get specification generation summary"""
        total_specs = len(self.specifications)
        
        spec_types = {}
        for spec_name, spec in self.specifications.items():
            if 'tla' in spec_name:
                spec_types['TLA+'] = spec_types.get('TLA+', 0) + 1
            elif 'coq' in spec_name:
                spec_types['Coq'] = spec_types.get('Coq', 0) + 1
            elif 'alloy' in spec_name:
                spec_types['Alloy'] = spec_types.get('Alloy', 0) + 1
        
        return {
            'total_specifications': total_specs,
            'specification_types': spec_types,
            'available_templates': list(self.templates.keys())
        }
