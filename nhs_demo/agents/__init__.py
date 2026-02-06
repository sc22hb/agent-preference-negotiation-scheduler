"""Agent modules for the NHS-style MAS demonstrator."""

from nhs_demo.agents.master_allocator_agent import AllocationResult, MasterAllocatorAgent
from nhs_demo.agents.patient_agent import PatientAgent
from nhs_demo.agents.receptionist import ReceptionistAgent
from nhs_demo.agents.rota_agent import RotaAgent
from nhs_demo.agents.safety_gate import SafetyGateAgent
from nhs_demo.agents.triage_routing import TriageRoutingAgent

__all__ = [
    "SafetyGateAgent",
    "ReceptionistAgent",
    "TriageRoutingAgent",
    "RotaAgent",
    "PatientAgent",
    "MasterAllocatorAgent",
    "AllocationResult",
]
