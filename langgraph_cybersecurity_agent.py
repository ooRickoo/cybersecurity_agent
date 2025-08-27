#!/usr/bin/env python3
"""
LangGraph Cybersecurity Agent with MCP Tool Integration
A dynamic, multi-agent system for cybersecurity analysis and workflow management.
"""

import asyncio
import os
import sys
import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Annotated
from dataclasses import dataclass

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# MCP imports (commented out for now to avoid import errors)
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
# from mcp.types import TextContent, ImageContent, EmbeddedResource

# Pydantic for data validation
from pydantic import BaseModel, Field

# Environment and encryption
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Project imports
sys.path.append(str(Path(__file__).parent / 'bin'))

# Data manipulation and visualization
import pandas as pd

from mcp_tools import (
    KnowledgeGraphManager, 
    FrameworkProcessor, 
    SessionManager,
    EncryptionManager
)

# Memory management imports
from bin.context_memory_manager import ContextMemoryManager
from bin.memory_mcp_tools import MemoryMCPTools

# Workflow verification imports
from bin.workflow_verification_mcp_tools import get_workflow_verification_mcp_tools

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

# Default encryption to enabled
ENCRYPTION_ENABLED = os.getenv('ENCRYPTION_ENABLED', 'true').lower() == 'true'
ENCRYPTION_PASSWORD_HASH = os.getenv('ENCRYPTION_PASSWORD_HASH', '')

# If encryption is enabled but no password hash, prompt user
if ENCRYPTION_ENABLED and not ENCRYPTION_PASSWORD_HASH:
    print("🔐 Encryption enabled but no password hash provided")
    print("   Please provide your encryption password:")
    
    import getpass
    password = getpass.getpass("Encryption Password: ")
    if password:
        import hashlib
        ENCRYPTION_PASSWORD_HASH = hashlib.sha256(password.encode()).hexdigest()
        print("✅ Password hash generated")
        
        # Save to environment for this session
        os.environ['ENCRYPTION_PASSWORD_HASH'] = ENCRYPTION_PASSWORD_HASH
    else:
        print("❌ No password provided, disabling encryption")
        ENCRYPTION_ENABLED = False

# Host verification and salt management
try:
    import sys
    from pathlib import Path
    
    # Add the bin directory to the path
    bin_path = Path(__file__).parent / "bin"
    if str(bin_path) not in sys.path:
        sys.path.insert(0, str(bin_path))
    
    from host_verification import HostVerification
    from salt_manager import SaltManager
    
    # Verify host compatibility before proceeding
    host_verifier = HostVerification()
    if not host_verifier.verify_host_compatibility():
        print("❌ Host verification failed - exiting")
        sys.exit(1)
    
    # Use device-bound salt for knowledge graph context memory
    salt_manager = SaltManager()
    ENCRYPTION_SALT = salt_manager.get_or_create_device_bound_salt()
    print("🔐 Using device-bound salt for knowledge graph encryption")
    
except ImportError as e:
    print(f"⚠️  Host verification modules not available: {e}")
    # Fallback to environment variable if salt manager not available
    ENCRYPTION_SALT = os.getenv('ENCRYPTION_SALT', 'cybersecurity_agent_salt')
    print("⚠️  Salt manager not available, using environment variable")

# Default password hash for 'Vosteen2025' if none provided
if not ENCRYPTION_PASSWORD_HASH:
    DEFAULT_PASSWORD = ''  # No default password for security
    ENCRYPTION_PASSWORD_HASH = hashlib.sha256(DEFAULT_PASSWORD.encode()).hexdigest()

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class AgentState:
    """State management for the LangGraph agent."""
    messages: List[Dict[str, Any]] = None
    current_workflow: Optional[str] = None
    workflow_state: Dict[str, Any] = None
    session_id: Optional[str] = None
    knowledge_context: Dict[str, Any] = None
    available_tools: List[str] = None
    memory_context: Dict[str, Any] = None
    
    # Dynamic workflow adaptation fields
    user_context: Optional[Dict[str, Any]] = None
    pending_clarifications: Optional[List[str]] = None
    workflow_adaptations: Optional[List[Dict[str, Any]]] = None
    user_feedback: Optional[Dict[str, Any]] = None
    execution_monitoring: Optional[Dict[str, Any]] = None
    
    # Workflow verification fields
    verification_required: bool = False
    verification_result: Optional[Dict[str, Any]] = None
    workflow_steps: Optional[List[Dict[str, Any]]] = None
    original_question: Optional[str] = None
    final_answer: Optional[str] = None
    execution_id: Optional[str] = None
    needs_backtrack: bool = False
    backtrack_result: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self.workflow_state is None:
            self.workflow_state = {}
        if self.knowledge_context is None:
            self.knowledge_context = {}
        if self.available_tools is None:
            self.available_tools = []
        if self.memory_context is None:
            self.memory_context = {}
        if self.user_context is None:
            self.user_context = {}
        if self.pending_clarifications is None:
            self.pending_clarifications = []
        if self.workflow_adaptations is None:
            self.workflow_adaptations = []
        if self.user_feedback is None:
            self.user_feedback = {}
        if self.execution_monitoring is None:
            self.execution_monitoring = {}
        
        # Initialize verification fields
        if self.workflow_steps is None:
            self.workflow_steps = []
        if self.verification_result is None:
            self.verification_result = {}
        if self.backtrack_result is None:
            self.backtrack_result = {}

class WorkflowTemplate(BaseModel):
    """Template for defining workflows."""
    name: str
    description: str
    steps: List[Dict[str, Any]]
    required_tools: List[str]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]

# ============================================================================
# ENCRYPTION MANAGEMENT
# ============================================================================

class EncryptionManager:
    """Manages encryption/decryption of sensitive data."""
    
    def __init__(self, password_hash: str = None, salt: str = None):
        if password_hash is None:
            password_hash = ENCRYPTION_PASSWORD_HASH
        if salt is None:
            salt = ENCRYPTION_SALT
            
        self.password_hash = password_hash
        self.salt = salt.encode() if isinstance(salt, str) else salt
        self.key = self._derive_key_from_hash(password_hash)
        self.cipher = Fernet(self.key) if self.key else None
    
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password (legacy method)."""
        salt = b'cybersecurity_agent_salt'  # In production, use random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def _derive_key_from_hash(self, password_hash: str) -> bytes:
        """Derive encryption key from password hash using salt."""
        try:
            # Use the password hash directly with the salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=self.salt,
                iterations=100000,
            )
            # The hash is already a hex string, so we use it directly
            key = base64.urlsafe_b64encode(kdf.derive(password_hash.encode()))
            return key
        except Exception as e:
            print(f"Warning: Failed to derive key from hash: {e}")
            return None
    
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data."""
        if not ENCRYPTION_ENABLED:
            return data
        return self.cipher.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data."""
        if not ENCRYPTION_ENABLED:
            return encrypted_data
        return self.cipher.decrypt(encrypted_data)
    
    def encrypt_file(self, file_path: Path) -> Path:
        """Encrypt a file and return path to encrypted version."""
        if not ENCRYPTION_ENABLED:
            return file_path
        
        encrypted_path = file_path.with_suffix(file_path.suffix + '.encrypted')
        with open(file_path, 'rb') as f:
            data = f.read()
        
        encrypted_data = self.encrypt_data(data)
        with open(encrypted_path, 'wb') as f:
            f.write(encrypted_data)
        
        return encrypted_path
    
    def decrypt_file(self, encrypted_path: Path) -> Path:
        """Decrypt a file and return path to decrypted version."""
        if not ENCRYPTION_ENABLED:
            return encrypted_path
        
        decrypted_path = encrypted_path.with_suffix('').with_suffix(
            encrypted_path.suffix.replace('.encrypted', '')
        )
        
        with open(encrypted_path, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_data = self.decrypt_data(encrypted_data)
        with open(decrypted_path, 'wb') as f:
            f.write(decrypted_data)
        
        return decrypted_path

# ============================================================================
# KNOWLEDGE GRAPH & MEMORY MANAGEMENT
# ============================================================================

class KnowledgeGraphManager:
    """Manages multi-dimensional knowledge graphs and memory."""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
        self.knowledge_base_path = Path("knowledge-objects")
        self.knowledge_base_path.mkdir(exist_ok=True)
        
        # Memory dimensions
        self.short_term = {}      # Current session context
        self.running_term = {}    # Active workflows
        self.long_term = {}       # Persistent knowledge
        
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """Load existing knowledge base."""
        master_catalog = self.knowledge_base_path / "master_catalog.db"
        if master_catalog.exists():
            # Load encrypted database if encryption is enabled
            if ENCRYPTION_ENABLED:
                decrypted_path = self.encryption_manager.decrypt_file(master_catalog)
                # Load from decrypted path
                decrypted_path.unlink()  # Clean up
            else:
                # Load directly
                pass
    
    def add_framework(self, framework_name: str, framework_data: Dict[str, Any]) -> str:
        """Add a new framework to the knowledge base."""
        framework_id = str(uuid.uuid4())
        
        # Store in appropriate memory dimension
        if len(framework_data) < 1000:
            # Small framework - store in short-term
            self.short_term[framework_id] = {
                'name': framework_name,
                'type': 'framework',
                'data': framework_data,
                'timestamp': datetime.now().isoformat(),
                'dimension': 'short_term'
            }
        elif len(framework_data) < 10000:
            # Medium framework - store in running-term
            self.running_term[framework_id] = {
                'name': framework_name,
                'type': 'framework',
                'data': framework_data,
                'timestamp': datetime.now().isoformat(),
                'dimension': 'running_term'
            }
        else:
            # Large framework - store in long-term
            self.long_term[framework_id] = {
                'name': framework_name,
                'type': 'framework',
                'data': framework_data,
                'timestamp': datetime.now().isoformat(),
                'dimension': 'long_term'
            }
        
        return framework_id
    
    def query_knowledge(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Query knowledge across all dimensions."""
        results = {
            'short_term': [],
            'running_term': [],
            'long_term': [],
            'relevance_scores': {}
        }
        
        # Search across dimensions
        for dimension_name, dimension_data in [
            ('short_term', self.short_term),
            ('running_term', self.running_term),
            ('long_term', self.long_term)
        ]:
            for item_id, item_data in dimension_data.items():
                relevance = self._calculate_relevance(query, item_data, context)
                if relevance > 0.5:  # Threshold for relevance
                    results[dimension_name].append({
                        'id': item_id,
                        'data': item_data,
                        'relevance': relevance
                    })
                    results['relevance_scores'][item_id] = relevance
        
        # Sort by relevance
        for dimension in ['short_term', 'running_term', 'long_term']:
            results[dimension].sort(key=lambda x: x['relevance'], reverse=True)
        
        return results
    
    def _calculate_relevance(self, query: str, item_data: Dict[str, Any], context: Dict[str, Any] = None) -> float:
        """Calculate relevance score for a query against an item."""
        # Simple keyword matching for now
        query_lower = query.lower()
        item_name = item_data.get('name', '').lower()
        item_type = item_data.get('type', '').lower()
        
        score = 0.0
        
        # Exact name match
        if query_lower in item_name:
            score += 0.8
        
        # Type match
        if query_lower in item_type:
            score += 0.6
        
        # Context relevance
        if context:
            context_keywords = ' '.join(str(v) for v in context.values()).lower()
            if any(keyword in context_keywords for keyword in query_lower.split()):
                score += 0.4
        
        return min(score, 1.0)
    
    def rehydrate_context(self, context_keys: List[str]) -> Dict[str, Any]:
        """Rehydrate context from memory dimensions."""
        context = {}
        
        for key in context_keys:
            # Search across all dimensions
            for dimension_data in [self.short_term, self.running_term, self.long_term]:
                if key in dimension_data:
                    context[key] = dimension_data[key]
                    break
        
        return context

# ============================================================================
# CONTEXT MEMORY MANAGEMENT INTEGRATION
# ============================================================================

class ContextMemoryIntegration:
    """Integrates context memory management with the main agent."""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
        self.memory_manager = ContextMemoryManager()
        self.memory_tools = MemoryMCPTools()
        
        # Auto-import patterns for workflow data
        self.auto_import_patterns = {
            'mitre_attack': {
                'domain': 'mitre_attack',
                'tier': 'long_term',
                'ttl_days': 365,
                'description': 'MITRE ATT&CK framework data from workflow'
            },
            'mitre_d3fend': {
                'domain': 'mitre_d3fend',
                'tier': 'long_term',
                'ttl_days': 365,
                'description': 'MITRE D3fend framework data from workflow'
            },
            'nist_frameworks': {
                'domain': 'nist',
                'tier': 'long_term',
                'ttl_days': 365,
                'description': 'NIST framework data from workflow'
            },
            'grc_data': {
                'domain': 'grc_policies',
                'tier': 'long_term',
                'ttl_days': 365,
                'description': 'GRC policy data from workflow'
            },
            'host_inventory': {
                'domain': 'hosts',
                'tier': 'long_term',
                'ttl_days': 365,
                'description': 'Host inventory data from workflow'
            },
            'application_inventory': {
                'domain': 'applications',
                'tier': 'long_term',
                'ttl_days': 365,
                'description': 'Application inventory data from workflow'
            },
            'user_inventory': {
                'domain': 'users',
                'tier': 'long_term',
                'ttl_days': 365,
                'description': 'User inventory data from workflow'
            },
            'network_inventory': {
                'domain': 'networks',
                'tier': 'long_term',
                'ttl_days': 365,
                'description': 'Network inventory data from workflow'
            },
            'ioc_collection': {
                'domain': 'iocs',
                'tier': 'short_term',
                'ttl_days': 7,
                'description': 'IOC collection data from workflow'
            },
            'threat_actors': {
                'domain': 'threat_actors',
                'tier': 'medium_term',
                'ttl_days': 30,
                'description': 'Threat actor intelligence from workflow'
            },
            'investigation_entities': {
                'domain': 'investigation',
                'tier': 'short_term',
                'ttl_days': 7,
                'description': 'Investigation entities from workflow'
            },
            'splunk_schemas': {
                'domain': 'splunk_schemas',
                'tier': 'long_term',
                'ttl_days': 365,
                'description': 'Splunk schema data from workflow'
            }
        }
    
    def auto_import_workflow_data(self, workflow_data: Dict[str, Any], workflow_context: str = '') -> List[str]:
        """Automatically import relevant workflow data into memory."""
        imported_ids = []
        
        try:
            # Analyze workflow data for importable content
            for pattern_name, pattern_config in self.auto_import_patterns.items():
                if self._should_import_pattern(workflow_data, pattern_name):
                    # Extract relevant data
                    extracted_data = self._extract_data_for_pattern(workflow_data, pattern_name)
                    if extracted_data:
                        # Import into memory
                        memory_id = self.memory_manager.import_data(
                            domain=pattern_config['domain'],
                            data=extracted_data,
                            source=f'workflow_auto_import_{pattern_name}',
                            tier=pattern_config['tier'],
                            ttl_days=pattern_config['ttl_days'],
                            tags=[pattern_name, 'workflow_auto_import', workflow_context],
                            description=pattern_config['description'],
                            priority=7  # Medium priority for auto-imports
                        )
                        imported_ids.append(memory_id)
                        
                        print(f"🧠 Auto-imported {pattern_name} data into memory (ID: {memory_id})")
            
            return imported_ids
            
        except Exception as e:
            print(f"❌ Error in auto-import: {e}")
            return []
    
    def _should_import_pattern(self, workflow_data: Dict[str, Any], pattern_name: str) -> bool:
        """Determine if workflow data should trigger auto-import for a pattern."""
        if pattern_name == 'mitre_attack':
            return any(key in workflow_data for key in ['mitre_techniques', 'attack_patterns', 'tactics'])
        elif pattern_name == 'mitre_d3fend':
            return any(key in workflow_data for key in ['defense_techniques', 'd3fend_patterns'])
        elif pattern_name == 'nist_frameworks':
            return any(key in workflow_data for key in ['nist_controls', 'compliance_frameworks'])
        elif pattern_name == 'grc_data':
            return any(key in workflow_data for key in ['policies', 'controls', 'compliance'])
        elif pattern_name == 'host_inventory':
            return any(key in workflow_data for key in ['hosts', 'host_list', 'endpoints'])
        elif pattern_name == 'application_inventory':
            return any(key in workflow_data for key in ['applications', 'app_list', 'software'])
        elif pattern_name == 'user_inventory':
            return any(key in workflow_data for key in ['users', 'user_list', 'accounts'])
        elif pattern_name == 'network_inventory':
            return any(key in workflow_data for key in ['networks', 'network_list', 'subnets'])
        elif pattern_name == 'ioc_collection':
            return any(key in workflow_data for key in ['iocs', 'indicators', 'threat_data'])
        elif pattern_name == 'threat_actors':
            return any(key in workflow_data for key in ['threat_actors', 'adversaries', 'attackers'])
        elif pattern_name == 'investigation_entities':
            return any(key in workflow_data for key in ['investigation', 'entities', 'evidence'])
        elif pattern_name == 'splunk_schemas':
            return any(key in workflow_data for key in ['splunk', 'schemas', 'indexes', 'sourcetypes'])
        
        return False
    
    def _extract_data_for_pattern(self, workflow_data: Dict[str, Any], pattern_name: str) -> Any:
        """Extract relevant data for a specific import pattern."""
        if pattern_name == 'mitre_attack':
            return {k: v for k, v in workflow_data.items() if k in ['mitre_techniques', 'attack_patterns', 'tactics']}
        elif pattern_name == 'mitre_d3fend':
            return {k: v for k, v in workflow_data.items() if k in ['defense_techniques', 'd3fend_patterns']}
        elif pattern_name == 'nist_frameworks':
            return {k: v for k, v in workflow_data.items() if k in ['nist_controls', 'compliance_frameworks']}
        elif pattern_name == 'grc_data':
            return {k: v for k, v in workflow_data.items() if k in ['policies', 'controls', 'compliance']}
        elif pattern_name == 'host_inventory':
            return {k: v for k, v in workflow_data.items() if k in ['hosts', 'host_list', 'endpoints']}
        elif pattern_name == 'application_inventory':
            return {k: v for k, v in workflow_data.items() if k in ['applications', 'app_list', 'software']}
        elif pattern_name == 'user_inventory':
            return {k: v for k, v in workflow_data.items() if k in ['users', 'user_list', 'accounts']}
        elif pattern_name == 'network_inventory':
            return {k: v for k, v in workflow_data.items() if k in ['networks', 'network_list', 'subnets']}
        elif pattern_name == 'ioc_collection':
            return {k: v for k, v in workflow_data.items() if k in ['iocs', 'indicators', 'threat_data']}
        elif pattern_name == 'threat_actors':
            return {k: v for k, v in workflow_data.items() if k in ['threat_actors', 'adversaries', 'attackers']}
        elif pattern_name == 'investigation_entities':
            return {k: v for k, v in workflow_data.items() if k in ['investigation', 'entities', 'evidence']}
        elif pattern_name == 'splunk_schemas':
            return {k: v for k, v in workflow_data.items() if k in ['splunk', 'schemas', 'indexes', 'sourcetypes']}
        
        return None
    
    def get_memory_context(self, query: str, domains: List[str] = None, max_results: int = 10) -> List[Any]:
        """Retrieve relevant context from memory for a query."""
        try:
            results = self.memory_manager.retrieve_context(
                query=query,
                domains=domains,
                max_results=max_results
            )
            return results
        except Exception as e:
            print(f"❌ Error retrieving memory context: {e}")
            return []
    
    def add_workflow_relationships(self, workflow_entities: List[str], relationship_type: str = 'workflow_related'):
        """Add relationships between workflow entities."""
        try:
            if len(workflow_entities) < 2:
                return
            
            # Create relationships between all entities in the workflow
            for i in range(len(workflow_entities) - 1):
                source = workflow_entities[i]
                target = workflow_entities[i + 1]
                
                self.memory_manager.add_relationship(
                    source_entity=source,
                    target_entity=target,
                    relationship_type=relationship_type,
                    strength=0.8,
                    metadata={'workflow_context': True, 'timestamp': datetime.now().isoformat()}
                )
            
            print(f"🔗 Added {len(workflow_entities) - 1} workflow relationships")
            
        except Exception as e:
            print(f"❌ Error adding workflow relationships: {e}")
    
    def get_memory_tools(self) -> MemoryMCPTools:
        """Get memory MCP tools for agent use."""
        return self.memory_tools

# ============================================================================
# WORKFLOW TEMPLATE MANAGER
# ============================================================================

class WorkflowTemplateManager:
    """Manages workflow templates and execution."""
    
    def __init__(self):
        self.templates: Dict[str, WorkflowTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default workflow templates."""
        
        # Policy Analysis Workflow
        self.templates['policy_analysis'] = WorkflowTemplate(
            name="Policy Analysis",
            description="Analyze security policies and map to frameworks",
            steps=[
                {"name": "data_ingestion", "tool": "framework_processor", "description": "Load and validate policy data"},
                {"name": "framework_mapping", "tool": "policy_analyzer", "description": "Map policies to MITRE ATT&CK"},
                {"name": "risk_assessment", "tool": "risk_assessor", "description": "Assess policy risks"},
                {"name": "compliance_check", "tool": "compliance_checker", "description": "Check compliance status"},
                {"name": "report_generation", "tool": "report_generator", "description": "Generate analysis report"}
            ],
            required_tools=["framework_processor", "policy_analyzer", "risk_assessor", "compliance_checker", "report_generator"],
            input_schema={"csv_file": "string", "framework": "string"},
            output_schema={"analysis_report": "string", "risk_scores": "object", "compliance_status": "object"}
        )
        
        # Threat Intelligence Workflow
        self.templates['threat_intelligence'] = WorkflowTemplate(
            name="Threat Intelligence",
            description="Process and analyze threat intelligence data",
            steps=[
                {"name": "data_collection", "tool": "stix_processor", "description": "Collect STIX threat data"},
                {"name": "ioc_processing", "tool": "ioc_processor", "description": "Process IOCs and indicators"},
                {"name": "enrichment", "tool": "threat_analyzer", "description": "Enrich with context"},
                {"name": "correlation", "tool": "threat_hunter", "description": "Correlate threats and patterns"},
                {"name": "risk_assessment", "tool": "risk_assessor", "description": "Assess threat risks"},
                {"name": "reporting", "tool": "report_generator", "description": "Generate intelligence report"}
            ],
            required_tools=["stix_processor", "ioc_processor", "threat_analyzer", "threat_hunter", "risk_assessor", "report_generator"],
            input_schema={"threat_sources": "array", "timeframe": "string", "ioc_types": "array"},
            output_schema={"threat_report": "string", "risk_indicators": "array", "threat_landscape": "object"}
        )
        
        # Incident Response Workflow
        self.templates['incident_response'] = WorkflowTemplate(
            name="Incident Response",
            description="Manage and coordinate incident response",
            steps=[
                {"name": "incident_assessment", "tool": "incident_tracker", "description": "Assess incident severity"},
                {"name": "containment", "tool": "playbook_executor", "description": "Execute containment playbooks"},
                {"name": "investigation", "tool": "forensics_analyzer", "description": "Investigate root cause"},
                {"name": "threat_analysis", "tool": "threat_analyzer", "description": "Analyze threat context"},
                {"name": "remediation", "tool": "automation_engine", "description": "Plan and execute remediation"},
                {"name": "documentation", "tool": "report_generator", "description": "Document incident and lessons learned"}
            ],
            required_tools=["incident_tracker", "playbook_executor", "forensics_analyzer", "threat_analyzer", "automation_engine", "report_generator"],
            input_schema={"incident_type": "string", "severity": "string", "affected_systems": "array"},
            output_schema={"response_plan": "string", "timeline": "object", "remediation_status": "string"}
        )
        
        # Vulnerability Assessment Workflow
        self.templates['vulnerability_assessment'] = WorkflowTemplate(
            name="Vulnerability Assessment",
            description="Comprehensive vulnerability scanning and assessment",
            steps=[
                {"name": "network_scanning", "tool": "network_scanner", "description": "Scan networks for vulnerabilities"},
                {"name": "traffic_analysis", "tool": "traffic_analyzer", "description": "Analyze network traffic patterns"},
                {"name": "vulnerability_analysis", "tool": "risk_assessor", "description": "Assess vulnerability risks"},
                {"name": "patch_management", "tool": "automation_engine", "description": "Plan patch deployment"},
                {"name": "reporting", "tool": "report_generator", "description": "Generate vulnerability report"}
            ],
            required_tools=["network_scanner", "traffic_analyzer", "risk_assessor", "automation_engine", "report_generator"],
            input_schema={"target_networks": "array", "scan_depth": "string", "exclude_hosts": "array"},
            output_schema={"vulnerability_report": "string", "risk_matrix": "object", "remediation_plan": "string"}
        )
        
        # Data Analysis Workflow
        self.templates['data_analysis'] = WorkflowTemplate(
            name="Data Analysis",
            description="Analyze large security datasets for insights",
            steps=[
                {"name": "data_ingestion", "tool": "data_analyzer", "description": "Load and validate datasets"},
                {"name": "pattern_detection", "tool": "pattern_detector", "description": "Detect security patterns"},
                {"name": "anomaly_detection", "tool": "anomaly_detector", "description": "Identify anomalies"},
                {"name": "correlation_analysis", "tool": "threat_hunter", "description": "Correlate findings"},
                {"name": "visualization", "tool": "chart_generator", "description": "Create visualizations"},
                {"name": "reporting", "tool": "report_generator", "description": "Generate analysis report"}
            ],
            required_tools=["data_analyzer", "pattern_detector", "anomaly_detector", "threat_hunter", "chart_generator", "report_generator"],
            input_schema={"data_sources": "array", "analysis_type": "string", "time_range": "string"},
            output_schema={"analysis_report": "string", "visualizations": "array", "key_findings": "array"}
        )
        
        # Compliance Assessment Workflow
        self.templates['compliance_assessment'] = WorkflowTemplate(
            name="Compliance Assessment",
            description="Assess compliance against security standards",
            steps=[
                {"name": "framework_loading", "tool": "framework_processor", "description": "Load compliance frameworks"},
                {"name": "policy_mapping", "tool": "policy_analyzer", "description": "Map policies to frameworks"},
                {"name": "gap_analysis", "tool": "compliance_checker", "description": "Identify compliance gaps"},
                {"name": "risk_assessment", "tool": "risk_assessor", "description": "Assess compliance risks"},
                {"name": "remediation_planning", "tool": "automation_engine", "description": "Plan remediation actions"},
                {"name": "reporting", "tool": "report_generator", "description": "Generate compliance report"}
            ],
            required_tools=["framework_processor", "policy_analyzer", "compliance_checker", "risk_assessor", "automation_engine", "report_generator"],
            input_schema={"compliance_frameworks": "array", "policies": "array", "scope": "string"},
            output_schema={"compliance_report": "string", "gap_analysis": "object", "remediation_plan": "string"}
        )
        
        # Threat Hunting Workflow
        self.templates['threat_hunting'] = WorkflowTemplate(
            name="Threat Hunting",
            description="Proactive threat hunting and investigation",
            steps=[
                {"name": "hypothesis_generation", "tool": "threat_hunter", "description": "Generate hunting hypotheses"},
                {"name": "data_collection", "tool": "data_analyzer", "description": "Collect relevant data"},
                {"name": "pattern_analysis", "tool": "pattern_detector", "description": "Analyze patterns and behaviors"},
                {"name": "threat_correlation", "tool": "threat_analyzer", "description": "Correlate with threat intelligence"},
                {"name": "investigation", "tool": "forensics_analyzer", "description": "Investigate findings"},
                {"name": "reporting", "tool": "report_generator", "description": "Document hunting results"}
            ],
            required_tools=["threat_hunter", "data_analyzer", "pattern_detector", "threat_analyzer", "forensics_analyzer", "report_generator"],
            input_schema={"hunting_hypothesis": "string", "data_sources": "array", "time_range": "string"},
            output_schema={"hunting_report": "string", "findings": "array", "recommendations": "array"}
        )
        
        # Adaptive Incident Response Workflow
        self.templates['adaptive_incident_response'] = WorkflowTemplate(
            name="Adaptive Incident Response",
            description="Dynamic incident response with user interaction and adaptation",
            steps=[
                {"name": "initial_assessment", "tool": "incident_tracker", "description": "Initial incident assessment"},
                {"name": "user_clarification", "tool": "user_interaction", "description": "Ask clarifying questions"},
                {"name": "dynamic_planning", "tool": "workflow_adapter", "description": "Adapt response plan based on user input"},
                {"name": "conditional_execution", "tool": "conditional_branching", "description": "Execute conditional response steps"},
                {"name": "progress_monitoring", "tool": "execution_monitor", "description": "Monitor progress and suggest adaptations"},
                {"name": "user_feedback", "tool": "feedback_collector", "description": "Collect user feedback and adjust"},
                {"name": "final_response", "tool": "playbook_executor", "description": "Execute final response actions"},
                {"name": "documentation", "tool": "report_generator", "description": "Document response and lessons learned"}
            ],
            required_tools=["incident_tracker", "user_interaction", "workflow_adapter", "conditional_branching", "execution_monitor", "feedback_collector", "playbook_executor", "report_generator"],
            input_schema={"incident_type": "string", "severity": "string", "user_preferences": "object"},
            output_schema={"response_plan": "string", "adaptations_made": "array", "user_feedback": "object", "final_report": "string"}
        )
        
        # Interactive Policy Analysis Workflow
        self.templates['interactive_policy_analysis'] = WorkflowTemplate(
            name="Interactive Policy Analysis",
            description="Policy analysis with dynamic user interaction and clarification",
            steps=[
                {"name": "policy_ingestion", "tool": "framework_processor", "description": "Load and parse policy data"},
                {"name": "initial_analysis", "tool": "policy_analyzer", "description": "Perform initial policy analysis"},
                {"name": "gap_identification", "tool": "compliance_checker", "description": "Identify compliance gaps"},
                {"name": "user_clarification", "tool": "clarification_engine", "description": "Generate clarifying questions"},
                {"name": "context_management", "tool": "context_manager", "description": "Manage user context and preferences"},
                {"name": "adaptive_mapping", "tool": "workflow_adapter", "description": "Adapt framework mapping based on user input"},
                {"name": "risk_assessment", "tool": "risk_assessor", "description": "Assess risks with user context"},
                {"name": "report_generation", "tool": "report_generator", "description": "Generate personalized analysis report"}
            ],
            required_tools=["framework_processor", "policy_analyzer", "compliance_checker", "clarification_engine", "context_manager", "workflow_adapter", "risk_assessor", "report_generator"],
            input_schema={"policy_data": "string", "user_context": "object", "preferred_frameworks": "array"},
            output_schema={"analysis_report": "string", "clarification_questions": "array", "user_context": "object", "risk_assessment": "object"}
        )
        
        # File Processing & Analysis Workflow
        self.templates['file_processing_analysis'] = WorkflowTemplate(
            name="File Processing & Analysis",
            description="Process files from file system locations with comprehensive analysis",
            steps=[
                {"name": "path_validation", "tool": "path_validator", "description": "Validate file system paths and URLs"},
                {"name": "file_type_detection", "tool": "file_type_detector", "description": "Detect file types and suggest processors"},
                {"name": "file_processing", "tool": "file_processor", "description": "Process files based on type"},
                {"name": "metadata_extraction", "tool": "file_metadata_extractor", "description": "Extract file metadata for analysis"},
                {"name": "content_analysis", "tool": "data_analyzer", "description": "Analyze file content for security insights"},
                {"name": "pattern_detection", "tool": "pattern_detector", "description": "Detect security patterns in file content"},
                {"name": "report_generation", "tool": "report_generator", "description": "Generate comprehensive file analysis report"}
            ],
            required_tools=["path_validator", "file_type_detector", "file_processor", "file_metadata_extractor", "data_analyzer", "pattern_detector", "report_generator"],
            input_schema={"file_paths": "array", "analysis_type": "string", "output_format": "string"},
            output_schema={"file_analysis": "array", "security_findings": "array", "metadata_summary": "object", "analysis_report": "string"}
        )
        
        # CSV Processing & Analysis Workflow
        self.templates['csv_processing_analysis'] = WorkflowTemplate(
            name="CSV Processing & Analysis",
            description="Process CSV files with configurable parameters and security analysis",
            steps=[
                {"name": "csv_validation", "tool": "path_validator", "description": "Validate CSV file path and format"},
                {"name": "csv_processing", "tool": "csv_processor", "description": "Process CSV with configurable parameters"},
                {"name": "data_validation", "tool": "data_analyzer", "description": "Validate and analyze CSV data"},
                {"name": "security_analysis", "tool": "threat_analyzer", "description": "Analyze CSV data for security insights"},
                {"name": "pattern_detection", "tool": "pattern_detector", "description": "Detect patterns and anomalies"},
                {"name": "report_generation", "tool": "report_generator", "description": "Generate CSV analysis report"}
            ],
            required_tools=["path_validator", "csv_processor", "data_analyzer", "threat_analyzer", "pattern_detector", "report_generator"],
            input_schema={"csv_file_path": "string", "processing_parameters": "object", "analysis_focus": "string"},
            output_schema={"processed_data": "object", "security_insights": "array", "pattern_analysis": "object", "analysis_report": "string"}
        )
        
        # File Summarization Workflow
        self.templates['file_summarization'] = WorkflowTemplate(
            name="File Summarization",
            description="Summarize files using LLM tasks (no iteration, comprehensive summary)",
            steps=[
                {"name": "file_validation", "tool": "path_validator", "description": "Validate file path and accessibility"},
                {"name": "file_type_analysis", "tool": "file_type_detector", "description": "Analyze file type and content structure"},
                {"name": "content_extraction", "tool": "file_processor", "description": "Extract content for summarization"},
                {"name": "llm_summarization", "tool": "file_summarizer", "description": "Generate comprehensive summary using LLM"},
                {"name": "summary_validation", "tool": "workflow_validator", "description": "Validate summary quality and completeness"},
                {"name": "output_formatting", "tool": "output_formatter", "description": "Format summary in requested output format"},
                {"name": "report_generation", "tool": "report_generator", "description": "Generate summarization report"}
            ],
            required_tools=["path_validator", "file_type_detector", "file_processor", "file_summarizer", "workflow_validator", "output_formatter", "report_generator"],
            input_schema={"file_path": "string", "summary_type": "string", "output_format": "string", "focus_areas": "array"},
            output_schema={"file_summary": "string", "summary_metadata": "object", "key_insights": "array", "summary_report": "string"}
        )
        
        # URL & Web Data Processing Workflow
        self.templates['url_web_processing'] = WorkflowTemplate(
            name="URL & Web Data Processing",
            description="Process URLs and web-based data sources for security analysis",
            steps=[
                {"name": "url_validation", "tool": "path_validator", "description": "Validate URL format and accessibility"},
                {"name": "web_data_collection", "tool": "url_processor", "description": "Collect data from web sources"},
                {"name": "content_analysis", "tool": "data_analyzer", "description": "Analyze web content for security insights"},
                {"name": "threat_assessment", "tool": "threat_analyzer", "description": "Assess web sources for potential threats"},
                {"name": "security_validation", "tool": "risk_assessor", "description": "Validate security of web sources"},
                {"name": "report_generation", "tool": "report_generator", "description": "Generate web data analysis report"}
            ],
            required_tools=["path_validator", "url_processor", "data_analyzer", "threat_analyzer", "risk_assessor", "report_generator"],
            input_schema={"urls": "array", "analysis_depth": "string", "security_focus": "string"},
            output_schema={"web_analysis": "array", "security_assessment": "object", "threat_indicators": "array", "analysis_report": "string"}
        )
        
        # NLP Text Analysis & Categorization Workflow
        self.templates['nlp_text_analysis'] = WorkflowTemplate(
            name="NLP Text Analysis & Categorization",
            description="Comprehensive text analysis using spaCy NLP capabilities",
            steps=[
                {"name": "text_preprocessing", "tool": "text_preprocessor", "description": "Preprocess and clean text data"},
                {"name": "language_detection", "tool": "language_detector", "description": "Detect text language and select models"},
                {"name": "entity_extraction", "tool": "entity_extractor", "description": "Extract named entities and relationships"},
                {"name": "text_categorization", "tool": "text_categorizer", "description": "Categorize text by content and intent"},
                {"name": "sentiment_analysis", "tool": "sentiment_analyzer", "description": "Analyze sentiment and emotional context"},
                {"name": "topic_modeling", "tool": "topic_modeler", "description": "Extract and model topics from text"},
                {"name": "keyword_extraction", "tool": "keyword_extractor", "description": "Extract key terms and concepts"},
                {"name": "report_generation", "tool": "report_generator", "description": "Generate comprehensive NLP analysis report"}
            ],
            required_tools=["text_preprocessor", "language_detector", "entity_extractor", "text_categorizer", "sentiment_analyzer", "topic_modeler", "keyword_extractor", "report_generator"],
            input_schema={"text_data": "string", "analysis_type": "array", "output_format": "string"},
            output_schema={"nlp_analysis": "object", "entities": "array", "categories": "array", "sentiment": "object", "topics": "array", "keywords": "array", "analysis_report": "string"}
        )
        
        # Adaptive Security Analysis Workflow
        self.templates['adaptive_security_analysis'] = WorkflowTemplate(
            name="Adaptive Security Analysis",
            description="Dynamic security analysis with NLP-driven workflow adaptation",
            steps=[
                {"name": "initial_analysis", "tool": "nlp_processor", "description": "Initial NLP analysis of input data"},
                {"name": "intent_detection", "tool": "text_categorizer", "description": "Detect user intent and analysis goals"},
                {"name": "workflow_planning", "tool": "planner_agent", "description": "Plan analysis workflow based on intent"},
                {"name": "dynamic_execution", "tool": "workflow_navigator", "description": "Execute workflow with navigation control"},
                {"name": "progress_monitoring", "tool": "execution_monitor", "description": "Monitor progress and detect adaptation needs"},
                {"name": "workflow_adaptation", "tool": "workflow_adapter", "description": "Adapt workflow based on findings"},
                {"name": "phase_control", "tool": "phase_controller", "description": "Control phase transitions and loops"},
                {"name": "final_analysis", "tool": "data_analyzer", "description": "Final comprehensive analysis"},
                {"name": "report_generation", "tool": "report_generator", "description": "Generate adaptive analysis report"}
            ],
            required_tools=["nlp_processor", "text_categorizer", "planner_agent", "workflow_navigator", "execution_monitor", "workflow_adapter", "phase_controller", "data_analyzer", "report_generator"],
            input_schema={"input_data": "string", "analysis_goals": "array", "adaptation_preferences": "object"},
            output_schema={"analysis_results": "object", "workflow_adaptations": "array", "phase_transitions": "array", "final_report": "string"}
        )
        
        # Intelligent Threat Detection Workflow
        self.templates['intelligent_threat_detection'] = WorkflowTemplate(
            name="Intelligent Threat Detection",
            description="NLP-powered threat detection with dynamic workflow adaptation",
            steps=[
                {"name": "text_analysis", "tool": "nlp_processor", "description": "Analyze text for threat indicators"},
                {"name": "entity_extraction", "tool": "custom_ner", "description": "Extract security-relevant entities"},
                {"name": "threat_categorization", "tool": "text_categorizer", "description": "Categorize threats by type and severity"},
                {"name": "sentiment_assessment", "tool": "sentiment_analyzer", "description": "Assess threat sentiment and urgency"},
                {"name": "similarity_analysis", "tool": "text_similarity", "description": "Find similar threat patterns"},
                {"name": "workflow_adaptation", "tool": "workflow_adapter", "description": "Adapt workflow based on threat type"},
                {"name": "response_planning", "tool": "planner_agent", "description": "Plan response based on threat analysis"},
                {"name": "execution_monitoring", "tool": "execution_monitor", "description": "Monitor threat response execution"},
                {"name": "report_generation", "tool": "report_generator", "description": "Generate threat detection report"}
            ],
            required_tools=["nlp_processor", "custom_ner", "text_categorizer", "sentiment_analyzer", "text_similarity", "workflow_adapter", "planner_agent", "execution_monitor", "report_generator"],
            input_schema={"threat_data": "string", "detection_focus": "string", "response_urgency": "string"},
            output_schema={"threat_analysis": "object", "threat_categories": "array", "response_plan": "string", "threat_report": "string"}
        )
    
    def get_template(self, template_name: str) -> Optional[WorkflowTemplate]:
        """Get a workflow template by name."""
        return self.templates.get(template_name)
    
    def list_templates(self) -> List[str]:
        """List available workflow templates."""
        return list(self.templates.keys())
    
    def create_custom_template(self, template_data: Dict[str, Any]) -> str:
        """Create a custom workflow template."""
        template_id = str(uuid.uuid4())
        template = WorkflowTemplate(**template_data)
        self.templates[template_id] = template
        return template_id

# ============================================================================
# MAIN AGENT CLASS
# ============================================================================

class LangGraphCybersecurityAgent:
    """Main LangGraph-based cybersecurity agent."""
    
    def __init__(self):
        self.encryption_manager = EncryptionManager()
        self.knowledge_manager = KnowledgeGraphManager(self.encryption_manager)
        self.workflow_manager = WorkflowTemplateManager()
        self.session_manager = SessionManager()
        
        # Initialize context memory management
        self.memory_integration = ContextMemoryIntegration(self.encryption_manager)
        
        # Initialize workflow verification tools
        try:
            self.verification_tools = get_workflow_verification_mcp_tools()
            print("✅ Workflow verification tools initialized")
        except Exception as e:
            print(f"⚠️  Workflow verification tools not available: {e}")
            self.verification_tools = None
        
        # Initialize MCP client
        self.mcp_client = None
        self.mcp_tools = {}
        
        # Build the LangGraph
        try:
            self.graph = self._build_graph()
            self.app = self.graph.compile()
        except Exception as e:
            print(f"Warning: LangGraph compilation failed: {e}")
            print("Using simplified processing...")
            self.graph = None
            self.app = None
    
    async def _initialize_mcp(self):
        """Initialize MCP client and tools."""
        try:
            # MCP is not available in this version, using local tools only
            print("ℹ️  MCP integration not available, using local tools only")
            
            # Initialize credential vault
            try:
                from bin.credential_vault import CredentialVault
                self.credential_vault = CredentialVault()
                print("🔐 Credential vault initialized")
            except ImportError:
                self.credential_vault = None
                print("⚠️  Credential vault not available")
            
            # Initialize visualization manager
            try:
                from bin.visualization_manager import VisualizationManager
                self.visualization_manager = VisualizationManager(self.session_manager)
                print("🎨 Visualization manager initialized")
            except ImportError:
                self.visualization_manager = None
                print("⚠️  Visualization manager not available")
            
            # Initialize session viewer manager
            try:
                from bin.session_viewer_manager import get_session_viewer_manager
                self.session_viewer_manager = get_session_viewer_manager()
                print("🔗 Session viewer manager initialized")
            except ImportError:
                self.session_viewer_manager = None
                print("⚠️  Session viewer manager not available")
            
            self.mcp_tools = {
                # Core Framework Processing Tools
                'framework_processor': {
                    'description': 'Process and flatten cybersecurity frameworks (MITRE ATT&CK, D3fend, NIST)',
                    'available': True
                },
                'stix_processor': {
                    'description': 'Process STIX 2.x threat intelligence data',
                    'available': True
                },
                'ttl_processor': {
                    'description': 'Process RDF/TTL knowledge graphs',
                    'available': True
                },
                'xml_processor': {
                    'description': 'Process XML-based security frameworks',
                    'available': True
                },
                
                # Session & Output Management
                'session_manager': {
                    'description': 'Manage sessions and output files',
                    'available': True
                },
                'log_manager': {
                    'description': 'Manage comprehensive logging and audit trails',
                    'available': True
                },
                'output_formatter': {
                    'description': 'Format outputs in various formats (JSON, CSV, XML, PDF)',
                    'available': True
                },
                
                # Knowledge Graph & Memory Management
                'knowledge_manager': {
                    'description': 'Manage knowledge graph and memory',
                    'available': True
                },
                'memory_optimizer': {
                    'description': 'Optimize memory usage across dimensions',
                    'available': True
                },
                'context_rehydrator': {
                    'description': 'Rehydrate context from stored knowledge',
                    'available': True
                },
                
                # Context Memory Management Tools
                'memory_import': {
                    'description': 'Import data into context memory with automatic domain detection',
                    'available': True
                },
                'memory_retrieve': {
                    'description': 'Retrieve relevant context from memory based on queries',
                    'available': True
                },
                'memory_relationships': {
                    'description': 'Manage entity relationships in memory',
                    'available': True
                },
                'memory_stats': {
                    'description': 'Get comprehensive memory statistics and metrics',
                    'available': True
                },
                'memory_cleanup': {
                    'description': 'Clean up expired memory entries and optimize storage',
                    'available': True
                },
                'memory_export': {
                    'description': 'Export memory snapshots for backup and analysis',
                    'available': True
                },
                
                # Threat Intelligence Tools
                'threat_analyzer': {
                    'description': 'Analyze threat intelligence data and indicators',
                    'available': True
                },
                'ioc_processor': {
                    'description': 'Process Indicators of Compromise (IOCs)',
                    'available': True
                },
                'threat_hunter': {
                    'description': 'Perform threat hunting and pattern analysis',
                    'available': True
                },
                
                # Policy & Compliance Tools
                'policy_analyzer': {
                    'description': 'Analyze security policies and map to frameworks',
                    'available': True
                },
                'compliance_checker': {
                    'description': 'Check compliance against various standards',
                    'available': True
                },
                'risk_assessor': {
                    'description': 'Assess security risks and vulnerabilities',
                    'available': True
                },
                
                # Incident Response Tools
                'incident_tracker': {
                    'description': 'Track and manage security incidents',
                    'available': True
                },
                'playbook_executor': {
                    'description': 'Execute incident response playbooks',
                    'available': True
                },
                'forensics_analyzer': {
                    'description': 'Analyze forensic data and artifacts',
                    'available': True
                },
                
                # Network Security Tools
                'network_scanner': {
                    'description': 'Scan networks for vulnerabilities',
                    'available': True
                },
                'traffic_analyzer': {
                    'description': 'Analyze network traffic patterns',
                    'available': True
                },
                'firewall_manager': {
                    'description': 'Manage and configure firewalls',
                    'available': True
                },
                
                # Data Analysis Tools
                'data_analyzer': {
                    'description': 'Analyze large datasets for security insights',
                    'available': True
                },
                'pattern_detector': {
                    'description': 'Detect patterns in security data',
                    'available': True
                },
                'anomaly_detector': {
                    'description': 'Detect anomalies in security data',
                    'available': True
                },
                
                # File System & URL Processing Tools
                'file_processor': {
                    'description': 'Process files from file system locations',
                    'available': True
                },
                'csv_processor': {
                    'description': 'Process CSV files with configurable parameters',
                    'available': True
                },
                'file_summarizer': {
                    'description': 'Summarize files using LLM tasks (no iteration)',
                    'available': True
                },
                'url_processor': {
                    'description': 'Process URLs and web-based data sources',
                    'available': True
                },
                'path_validator': {
                    'description': 'Validate file system paths and URLs',
                    'available': True
                },
                'file_type_detector': {
                    'description': 'Detect file types and suggest appropriate processors',
                    'available': True
                },
                'batch_processor': {
                    'description': 'Process multiple files in batch operations',
                    'available': True
                },
                'file_metadata_extractor': {
                    'description': 'Extract metadata from files for analysis',
                    'available': True
                },
                
                # Encryption & Security Tools
                'encryption_manager': {
                    'description': 'Manage encryption keys and encrypted data',
                    'available': True
                },
                'hash_calculator': {
                    'description': 'Calculate various hash functions',
                    'available': True
                },
                'certificate_manager': {
                    'description': 'Manage SSL/TLS certificates',
                    'available': True
                },
                
                # Reporting & Visualization Tools
                'report_generator': {
                    'description': 'Generate comprehensive security reports',
                    'available': True
                },
                'dashboard_creator': {
                    'description': 'Create security dashboards',
                    'available': True
                },
                'chart_generator': {
                    'description': 'Generate charts and graphs',
                    'available': True
                },
                
                # Integration Tools
                'api_integrator': {
                    'description': 'Integrate with external security APIs',
                    'available': True
                },
                'webhook_manager': {
                    'description': 'Manage webhooks for real-time alerts',
                    'available': True
                },
                'plugin_manager': {
                    'description': 'Manage security tool plugins',
                    'available': True
                },
                
                # Workflow Management Tools
                'workflow_executor': {
                    'description': 'Execute complex security workflows',
                    'available': True
                },
                'automation_engine': {
                    'description': 'Automate security tasks and responses',
                    'available': True
                },
                'scheduler': {
                    'description': 'Schedule security tasks and scans',
                    'available': True
                },
                
                # User Interaction & Communication Tools
                'user_interaction': {
                    'description': 'Ask clarifying questions and get user input',
                    'available': True
                },
                'context_manager': {
                    'description': 'Manage conversation context and user preferences',
                    'available': True
                },
                'feedback_collector': {
                    'description': 'Collect and process user feedback',
                    'available': True
                },
                'clarification_engine': {
                    'description': 'Generate clarifying questions based on workflow needs',
                    'available': True
                },
                
                # Dynamic Workflow Adaptation Tools
                'workflow_adapter': {
                    'description': 'Dynamically adapt workflows based on user input',
                    'available': True
                },
                'conditional_branching': {
                    'description': 'Create conditional workflow branches',
                    'available': True
                },
                'workflow_validator': {
                    'description': 'Validate workflow changes and user inputs',
                    'available': True
                },
                'execution_monitor': {
                    'description': 'Monitor workflow execution and suggest adaptations',
                    'available': True
                },
                
                # spaCy NLP Processing Tools
                'nlp_processor': {
                    'description': 'Core spaCy NLP processing and text analysis',
                    'available': True
                },
                'text_categorizer': {
                    'description': 'Categorize text using spaCy NLP models',
                    'available': True
                },
                'entity_extractor': {
                    'description': 'Extract entities (people, organizations, locations, etc.)',
                    'available': True
                },
                'sentiment_analyzer': {
                    'description': 'Analyze sentiment and emotional context',
                    'available': True
                },
                'topic_modeler': {
                    'description': 'Model and extract topics from text',
                    'available': True
                },
                'keyword_extractor': {
                    'description': 'Extract key terms and concepts',
                    'available': True
                },
                'text_similarity': {
                    'description': 'Calculate text similarity and clustering',
                    'available': True
                },
                'language_detector': {
                    'description': 'Detect language and suggest appropriate models',
                    'available': True
                },
                'custom_ner': {
                    'description': 'Custom Named Entity Recognition for security domains',
                    'available': True
                },
                'text_preprocessor': {
                    'description': 'Preprocess text for analysis and modeling',
                    'available': True
                },
                
                # Workflow Navigation & Control Tools
                'workflow_navigator': {
                    'description': 'Navigate between workflow phases and restart workflows',
                    'available': True
                },
                'planner_agent': {
                    'description': 'Re-plan workflows based on new information',
                    'available': True
                },
                'template_manager': {
                    'description': 'Access and modify workflow templates dynamically',
                    'available': True
                },
                'workflow_restarter': {
                    'description': 'Restart workflows from specific phases',
                    'available': True
                },
                'phase_controller': {
                    'description': 'Control workflow phase transitions and loops',
                    'available': True
                },
                
                # Credential Management Tools
                'credential_prompter': {
                    'description': 'Safely prompt user for credentials and store in vault',
                    'available': True
                },
                'credential_retriever': {
                    'description': 'Retrieve stored credentials from vault',
                    'available': True
                },
                'credential_storer': {
                    'description': 'Store new credentials in vault',
                    'available': True
                },
                'vault_manager': {
                    'description': 'Manage credential vault operations',
                    'available': True
                },
                
                # Visualization Tools
                'dataframe_viewer': {
                    'description': 'Interactive DataFrame viewer for data validation',
                    'available': True
                },
                'workflow_diagram': {
                    'description': 'Beautiful workflow step visualization',
                    'available': True
                },
                'neo4j_graph_visualizer': {
                    'description': 'Resource relationship diagram visualization',
                    'available': True
                },
                'vega_lite_charts': {
                    'description': 'Professional data visualizations with Vega-Lite',
                    'available': True
                },
                'visualization_exporter': {
                    'description': 'Export visualizations to HTML, PNG, SVG',
                    'available': True
                },
                
                # Workflow Verification Tools
                'workflow_verifier': {
                    'description': 'Verify workflow accuracy and quality with "Check our math"',
                    'available': True
                },
                'verification_manager': {
                    'description': 'Manage workflow verification and backtracking decisions',
                    'available': True
                },
                'template_selector': {
                    'description': 'Select optimal workflow templates based on verification results',
                    'available': True
                },
                'backtrack_handler': {
                    'description': 'Handle verification failures with alternative approaches',
                    'available': True
                },
                'loop_prevention': {
                    'description': 'Prevent workflow execution loops and suggest alternatives',
                    'available': True
                }
            }
            
        except Exception as e:
            print(f"Warning: MCP initialization failed: {e}")
            print("Continuing with local tools only...")
    
    async def _register_mcp_tools(self):
        """Register MCP tools with the agent."""
        # MCP tools are already registered in _initialize_mcp
        pass
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        try:
            # Create the graph
            workflow = StateGraph(AgentState)
            
            # Add nodes
            workflow.add_node("planner", self._planner_node)
            workflow.add_node("runner", self._runner_node)
            workflow.add_node("memory_manager", self._memory_manager_node)
            workflow.add_node("workflow_executor", self._workflow_executor_node)
            workflow.add_node("workflow_verification", self._workflow_verification_node)
            
            # Define edges
            workflow.add_edge(START, "planner")
            workflow.add_edge("planner", "runner")
            workflow.add_edge("runner", "memory_manager")
            workflow.add_edge("memory_manager", "workflow_executor")
            workflow.add_edge("workflow_executor", "workflow_verification")
            workflow.add_edge("workflow_verification", END)
            
            # Add conditional edges
            workflow.add_conditional_edges(
                "planner",
                self._should_execute_workflow,
                {
                    "execute_workflow": "workflow_executor",
                    "continue": "runner"
                }
            )
            
            return workflow
            
        except Exception as e:
            print(f"Warning: LangGraph workflow creation failed: {e}")
            print("Using simplified workflow...")
            # Return a minimal graph
            return self._build_minimal_graph()
    
    def _build_minimal_graph(self) -> StateGraph:
        """Build a minimal working graph."""
        workflow = StateGraph(AgentState)
        workflow.add_node("simple_processor", self._simple_processor_node)
        workflow.add_edge(START, "simple_processor")
        workflow.add_edge("simple_processor", END)
        return workflow
    
    def _simple_processor_node(self, state: AgentState) -> AgentState:
        """Simple processing node for basic functionality."""
        # Basic processing logic
        if state.messages:
            last_message = state.messages[-1]
            if "policy" in last_message.get('content', '').lower():
                state.current_workflow = "policy_analysis"
            elif "threat" in last_message.get('content', '').lower():
                state.current_workflow = "threat_intelligence"
        
        return state
    
    def _planner_node(self, state: AgentState) -> AgentState:
        """Planner agent node - analyzes user input and creates execution plan."""
        # Implementation will be added
        return state
    
    def _runner_node(self, state: AgentState) -> AgentState:
        """Runner agent node - executes planned actions."""
        # Implementation will be added
        return state
    
    def _memory_manager_node(self, state: AgentState) -> AgentState:
        """Memory manager node - manages knowledge graph and memory."""
        # Implementation will be added
        return state
    
    def _workflow_executor_node(self, state: AgentState) -> AgentState:
        """Workflow executor node - executes workflow templates."""
        # Implementation will be added
        return state
    
    def _workflow_verification_node(self, state: AgentState) -> AgentState:
        """Workflow verification node - performs 'Check our math' verification."""
        try:
            if not self.verification_tools or not state.workflow_steps:
                return state
            
            # Generate execution ID if not present
            if not state.execution_id:
                state.execution_id = f"exec_{uuid.uuid4().hex[:8]}"
            
            # Store original question if not present
            if not state.original_question and state.messages:
                state.original_question = state.messages[0].get('content', '')
            
            # Perform workflow verification
            verification_result = self.verification_tools.check_our_math(
                execution_id=state.execution_id,
                original_question=state.original_question or "Unknown question",
                workflow_steps=state.workflow_steps,
                final_answer=state.final_answer or "No answer provided",
                question_type=self._classify_question_type(state.original_question or "")
            )
            
            if verification_result.get("success"):
                state.verification_result = verification_result["verification_result"]
                state.verification_required = False
                
                # Check if backtracking is needed
                if verification_result["verification_result"].get("needs_backtrack"):
                    state.needs_backtrack = True
                    
                    # Handle backtracking automatically
                    backtrack_result = self.verification_tools.handle_verification_failure(
                        execution_id=state.execution_id,
                        verification_result=verification_result["verification_result"],
                        original_question=state.original_question or "Unknown question"
                    )
                    
                    if backtrack_result.get("success"):
                        state.backtrack_result = backtrack_result
                        # Update workflow with alternative template
                        if backtrack_result.get("recommended_steps"):
                            state.workflow_steps = backtrack_result["recommended_steps"]
                            state.current_workflow = "alternative_approach"
                
                # Add verification summary to messages
                verification_summary = f"🔍 **Workflow Verification Complete**\n"
                verification_summary += f"Accuracy Score: {verification_result['verification_result']['accuracy_score']:.2f}/1.00\n"
                verification_summary += f"Confidence Level: {verification_result['verification_result']['confidence_level'].upper()}\n"
                
                if state.needs_backtrack:
                    verification_summary += f"⚠️ **Backtracking Required** - Alternative approach selected\n"
                else:
                    verification_summary += f"✅ **Verification Passed** - Workflow results are accurate\n"
                
                state.messages.append({
                    "role": "assistant",
                    "content": verification_summary
                })
            
            return state
            
        except Exception as e:
            print(f"Warning: Workflow verification failed: {e}")
            # Add error message to state
            state.messages.append({
                "role": "assistant",
                "content": f"⚠️ **Verification Error**: {str(e)}"
            })
            return state
    
    def _classify_question_type(self, question: str) -> str:
        """Classify the type of question for verification."""
        if not question:
            return "general_investigation"
        
        question_lower = question.lower()
        
        if any(term in question_lower for term in ["threat", "malware", "attack", "campaign"]):
            return "threat_analysis"
        elif any(term in question_lower for term in ["incident", "breach", "compromise", "response"]):
            return "incident_response"
        elif any(term in question_lower for term in ["vulnerability", "weakness", "patch"]):
            return "vulnerability_assessment"
        elif any(term in question_lower for term in ["compliance", "audit", "policy", "regulation"]):
            return "compliance_audit"
        else:
            return "general_investigation"
    
    def track_workflow_step(self, step_type: str, description: str, tools_used: List[str] = None, 
                           inputs: Dict[str, Any] = None, outputs: Dict[str, Any] = None, 
                           execution_time: float = 0.0) -> Dict[str, Any]:
        """Track a workflow step for verification purposes."""
        step = {
            "step_id": f"step_{uuid.uuid4().hex[:8]}",
            "step_type": step_type,
            "description": description,
            "tools_used": tools_used or [],
            "inputs": inputs or {},
            "outputs": outputs or {},
            "execution_time": execution_time,
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }
        
        return step
    
    def add_workflow_step(self, state: AgentState, step: Dict[str, Any]) -> None:
        """Add a workflow step to the current state."""
        if not state.workflow_steps:
            state.workflow_steps = []
        state.workflow_steps.append(step)
    
    def set_final_answer(self, state: AgentState, answer: str) -> None:
        """Set the final answer for verification."""
        state.final_answer = answer
    
    def get_verification_summary(self, execution_id: str = None) -> Dict[str, Any]:
        """Get verification summary for an execution."""
        if not self.verification_tools:
            return {"error": "Verification tools not available"}
        
        if execution_id:
            return self.verification_tools.get_verification_summary(execution_id)
        else:
            return {"error": "Execution ID required"}
    
    def get_execution_history(self, limit: int = 10) -> Dict[str, Any]:
        """Get recent execution history with verification status."""
        if not self.verification_tools:
            return {"error": "Verification tools not available"}
        
        return self.verification_tools.get_execution_history(limit)
    
    def select_workflow_template(self, question: str, question_type: str = None) -> Dict[str, Any]:
        """Select optimal workflow template for a question."""
        if not self.verification_tools:
            return {"error": "Verification tools not available"}
        
        return self.verification_tools.select_workflow_template(question, question_type)
    
    def _should_execute_workflow(self, state: AgentState) -> str:
        """Determine if a workflow should be executed."""
        if state.current_workflow:
            return "execute_workflow"
        return "continue"
    
    async def chat(self, user_input: str, session_id: str = None) -> str:
        """Main chat interface for the agent."""
        
        # Initialize session if needed
        if not session_id:
            session_id = self.session_manager.create_session()
        
        # Create initial state
        initial_state = AgentState(
            messages=[{"role": "user", "content": user_input}],
            session_id=session_id,
            knowledge_context=self.knowledge_manager.query_knowledge(user_input)
        )
        
        # Perform NLP analysis on user input
        nlp_analysis = self._analyze_text_with_spacy(user_input)
        
        # Check for file processing requests
        file_processing_request = self._detect_file_processing_request(user_input)
        if file_processing_request['detected']:
            file_response = self._handle_file_processing_request(file_processing_request, user_input)
            if file_response:
                return file_response
        
        # Check for workflow adaptation needs based on NLP analysis
        adaptation_needs = self._detect_workflow_adaptation_needs(nlp_analysis, user_input)
        if adaptation_needs['needs_adaptation'] or adaptation_needs['phase_restart'] or adaptation_needs['planner_involvement']:
            dynamic_response = self._generate_dynamic_response(nlp_analysis, adaptation_needs)
            return dynamic_response
        
        # Check for workflow adaptation signals
        if any(keyword in user_input.lower() for keyword in ['change', 'modify', 'skip', 'add', 'priority', 'detailed']):
            initial_state = self._handle_workflow_adaptation(user_input, initial_state)
            
            # Generate clarifying questions if needed
            if initial_state.pending_clarifications:
                clarification_response = "I notice you want to modify the workflow. Let me ask a few clarifying questions:\n\n"
                for i, question in enumerate(initial_state.pending_clarifications[:3], 1):  # Limit to 3 questions
                    clarification_response += f"{i}. {question}\n"
                clarification_response += "\nPlease provide your preferences so I can adapt the workflow accordingly."
                return clarification_response
        
        # Execute the graph if available
        if self.app:
            try:
                final_state = await self.app.ainvoke(initial_state)
                return self._format_response(final_state)
            except Exception as e:
                print(f"Warning: LangGraph execution failed: {e}")
                return self._process_without_langgraph(user_input, session_id)
        else:
            # Fallback to simple processing
            return self._process_without_langgraph(user_input, session_id)
    
    def _process_without_langgraph(self, user_input: str, session_id: str) -> str:
        """Process user input without LangGraph."""
        try:
            # Simple keyword-based processing
            if "policy" in user_input.lower():
                return self._handle_policy_request(user_input, session_id)
            elif "threat" in user_input.lower():
                return self._handle_threat_request(user_input, session_id)
            elif "framework" in user_input.lower():
                return self._handle_framework_request(user_input, session_id)
            else:
                return f"I understand you said: '{user_input}'. I'm currently running in simplified mode. You can ask me about policies, threats, or frameworks."
        except Exception as e:
            return f"Error processing request: {str(e)}"
    
    def _adapt_workflow_based_on_user_input(self, user_input: str, current_workflow: str) -> Dict[str, Any]:
        """Dynamically adapt workflow based on user input and feedback."""
        adaptations = {
            'workflow_changes': [],
            'new_steps': [],
            'modified_parameters': {},
            'user_clarifications_needed': []
        }
        
        # Analyze user input for workflow adaptation signals
        input_lower = user_input.lower()
        
        if "change" in input_lower or "modify" in input_lower:
            adaptations['workflow_changes'].append("User requested workflow modification")
            
        if "skip" in input_lower or "bypass" in input_lower:
            adaptations['workflow_changes'].append("User requested step bypass")
            
        if "add" in input_lower or "include" in input_lower:
            adaptations['workflow_changes'].append("User requested additional steps")
            
        if "priority" in input_lower or "urgent" in input_lower:
            adaptations['modified_parameters']['priority'] = 'high'
            
        if "detailed" in input_lower or "comprehensive" in input_lower:
            adaptations['modified_parameters']['detail_level'] = 'comprehensive'
            
        return adaptations
    
    def _generate_clarifying_questions(self, workflow_step: str, context: Dict[str, Any]) -> List[str]:
        """Generate clarifying questions based on workflow step and context."""
        questions = []
        
        if "policy_analysis" in workflow_step:
            questions.extend([
                "Which specific compliance frameworks should I focus on?",
                "What is the scope of the policy analysis?",
                "Are there any specific risk areas you'd like me to prioritize?"
            ])
            
        elif "threat_intelligence" in workflow_step:
            questions.extend([
                "What types of threats are you most concerned about?",
                "What time range should I analyze?",
                "Do you have specific threat indicators to investigate?"
            ])
            
        elif "incident_response" in workflow_step:
            questions.extend([
                "What is the severity level of this incident?",
                "Which systems or assets are affected?",
                "Do you have any containment actions already in place?"
            ])
            
        return questions
    
    def _handle_workflow_adaptation(self, user_input: str, current_state: AgentState) -> AgentState:
        """Handle workflow adaptation based on user input."""
        if current_state.current_workflow:
            # Generate adaptations
            adaptations = self._adapt_workflow_based_on_user_input(user_input, current_state.current_workflow)
            
            # Store adaptations in state
            current_state.workflow_adaptations.append(adaptations)
            
            # Generate clarifying questions if needed
            if adaptations['user_clarifications_needed']:
                current_state.pending_clarifications.extend(adaptations['user_clarifications_needed'])
            
            # Update workflow state with adaptations
            current_state.workflow_state.update(adaptations['modified_parameters'])
            
        return current_state
    
    def _detect_file_processing_request(self, user_input: str) -> Dict[str, Any]:
        """Detect if user input contains file processing requests."""
        processing_request = {
            'type': None,
            'paths': [],
            'parameters': {},
            'detected': False
        }
        
        input_lower = user_input.lower()
        
        # Detect file paths (basic pattern matching)
        import re
        file_path_patterns = [
            r'[\/\\][^\s]+\.(csv|txt|json|xml|pdf|log|pcap|pcapng)',  # File extensions
            r'[A-Z]:[\/\\][^\s]+',  # Windows paths
            r'\/[^\s]+',  # Unix paths
        ]
        
        for pattern in file_path_patterns:
            matches = re.findall(pattern, input_lower)
            if matches:
                processing_request['paths'].extend(matches)
                processing_request['detected'] = True
        
        # Detect URL patterns
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, input_lower)
        if urls:
            processing_request['paths'].extend(urls)
            processing_request['detected'] = True
        
        # Detect processing type
        if 'csv' in input_lower or 'spreadsheet' in input_lower:
            processing_request['type'] = 'csv_processing'
        elif 'summarize' in input_lower or 'summary' in input_lower:
            processing_request['type'] = 'file_summarization'
        elif 'url' in input_lower or 'web' in input_lower or 'http' in input_lower:
            processing_request['type'] = 'url_web_processing'
        elif 'file' in input_lower and ('process' in input_lower or 'analyze' in input_lower):
            processing_request['type'] = 'file_processing_analysis'
        
        # Detect parameters
        if 'detailed' in input_lower or 'comprehensive' in input_lower:
            processing_request['parameters']['detail_level'] = 'comprehensive'
        if 'quick' in input_lower or 'fast' in input_lower:
            processing_request['parameters']['detail_level'] = 'quick'
        if 'security' in input_lower or 'threat' in input_lower:
            processing_request['parameters']['focus'] = 'security'
        if 'compliance' in input_lower or 'policy' in input_lower:
            processing_request['parameters']['focus'] = 'compliance'
        
        return processing_request
    
    def _handle_file_processing_request(self, processing_request: Dict[str, Any], user_input: str) -> str:
        """Handle file processing requests and suggest appropriate workflows."""
        if not processing_request['detected']:
            return None
        
        response = "🔍 I detected a file processing request! Here's what I can help you with:\n\n"
        
        if processing_request['type'] == 'csv_processing':
            response += "📊 **CSV Processing & Analysis Workflow**\n"
            response += "   • Process CSV files with configurable parameters\n"
            response += "   • Security analysis and pattern detection\n"
            response += "   • Configurable output formats\n\n"
            
        elif processing_request['type'] == 'file_summarization':
            response += "📝 **File Summarization Workflow**\n"
            response += "   • Generate comprehensive summaries using LLM\n"
            response += "   • No iteration - complete summary in one pass\n"
            response += "   • Multiple output formats supported\n\n"
            
        elif processing_request['type'] == 'url_web_processing':
            response += "🌐 **URL & Web Data Processing Workflow**\n"
            response += "   • Process web-based data sources\n"
            response += "   • Security assessment and threat analysis\n"
            response += "   • Content analysis and validation\n\n"
            
        elif processing_request['type'] == 'file_processing_analysis':
            response += "📁 **File Processing & Analysis Workflow**\n"
            response += "   • Comprehensive file analysis\n"
            response += "   • Metadata extraction and pattern detection\n"
            response += "   • Security insights and reporting\n\n"
        
        response += "**Detected Paths:**\n"
        for path in processing_request['paths'][:3]:  # Limit to 3 paths
            response += f"   • {path}\n"
        
        if processing_request['parameters']:
            response += "\n**Suggested Parameters:**\n"
            for key, value in processing_request['parameters'].items():
                response += f"   • {key}: {value}\n"
        
        response += "\n💡 **To proceed, you can:**\n"
        response += "   1. Run the specific workflow directly\n"
        response += "   2. Modify parameters as needed\n"
        response += "   3. Ask for clarification on any aspect\n"
        
        return response
    
    def _analyze_text_with_spacy(self, text: str) -> Dict[str, Any]:
        """Analyze text using spaCy NLP capabilities with Apple Silicon optimization."""
        try:
            # Import spaCy (will be installed via requirements.txt)
            import spacy
            import platform
            
            # Check for Apple Silicon and spaCy version compatibility
            is_apple_silicon = platform.machine() == 'arm64' and platform.system() == 'Darwin'
            spacy_version = spacy.__version__
            
            # Version compatibility check for Apple Silicon
            if is_apple_silicon:
                if not spacy_version.startswith('3.7'):
                    print(f"⚠️  Warning: Apple Silicon detected but spaCy version {spacy_version} may not be optimal")
                    print("   Recommended: spaCy 3.7.2 for Apple Silicon compatibility")
            
            # Load English model with Apple Silicon optimization
            try:
                # Try to load the optimized model for Apple Silicon
                if is_apple_silicon:
                    try:
                        nlp = spacy.load("en_core_web_sm")
                        print("✅ spaCy model loaded with Apple Silicon optimization")
                    except OSError:
                        # Fallback to basic model
                        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
                        print("⚠️  Basic spaCy model loaded (some features disabled)")
                else:
                    # Standard loading for other architectures
                    nlp = spacy.load("en_core_web_sm")
                    
            except OSError as e:
                if "No module named 'en_core_web_sm'" in str(e):
                    return {
                        'error': 'spaCy English model not installed',
                        'suggestion': 'Install model: python -m spacy download en_core_web_sm',
                        'apple_silicon': is_apple_silicon,
                        'spacy_version': spacy_version
                    }
                else:
                    # Other loading errors
                    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
                    print("⚠️  Basic spaCy model loaded due to loading error")
            
            # Process the text
            doc = nlp(text)
            
            # Extract entities
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            
            # Extract key phrases and tokens
            tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
            key_phrases = [chunk.text for chunk in doc.noun_chunks]
            
            # Basic sentiment analysis (positive/negative words)
            positive_words = ['good', 'great', 'excellent', 'secure', 'safe', 'protected']
            negative_words = ['bad', 'poor', 'vulnerable', 'threat', 'attack', 'breach']
            
            positive_count = sum(1 for token in doc if token.text.lower() in positive_words)
            negative_count = sum(1 for token in doc if token.text.lower() in negative_words)
            
            sentiment = {
                'positive_score': positive_count,
                'negative_score': negative_count,
                'overall': 'neutral'
            }
            
            if positive_count > negative_count:
                sentiment['overall'] = 'positive'
            elif negative_count > positive_count:
                sentiment['overall'] = 'negative'
            
            # Text categorization based on content
            categories = []
            text_lower = text.lower()
            
            if any(word in text_lower for word in ['threat', 'attack', 'breach', 'vulnerability']):
                categories.append('security_threat')
            if any(word in text_lower for word in ['policy', 'compliance', 'regulation']):
                categories.append('policy_compliance')
            if any(word in text_lower for word in ['incident', 'response', 'alert']):
                categories.append('incident_response')
            if any(word in text_lower for word in ['analysis', 'investigation', 'forensics']):
                categories.append('analysis_investigation')
            if any(word in text_lower for word in ['network', 'system', 'infrastructure']):
                categories.append('infrastructure')
            
            return {
                'entities': entities,
                'tokens': tokens,
                'key_phrases': key_phrases,
                'sentiment': sentiment,
                'categories': categories,
                'text_length': len(text),
                'processed_tokens': len(doc),
                'apple_silicon': is_apple_silicon,
                'spacy_version': spacy_version,
                'optimization_status': 'optimized' if is_apple_silicon and spacy_version.startswith('3.7') else 'standard'
            }
            
        except ImportError:
            return {
                'error': 'spaCy not available',
                'suggestion': 'Install spaCy: pip install spacy==3.7.2 && python -m spacy download en_core_web_sm',
                'apple_silicon': platform.machine() == 'arm64' and platform.system() == 'Darwin' if 'platform' in globals() else None
            }
        except Exception as e:
            return {
                'error': f'spaCy processing error: {str(e)}',
                'fallback_analysis': True,
                'apple_silicon': platform.machine() == 'arm64' and platform.system() == 'Darwin' if 'platform' in globals() else None
            }
    
    def _detect_workflow_adaptation_needs(self, nlp_analysis: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """Detect if workflow adaptation is needed based on NLP analysis."""
        adaptation_needs = {
            'needs_adaptation': False,
            'adaptation_type': None,
            'suggested_workflow': None,
            'phase_restart': False,
            'planner_involvement': False
        }
        
        # Check sentiment for urgency
        if nlp_analysis.get('sentiment', {}).get('overall') == 'negative':
            adaptation_needs['needs_adaptation'] = True
            adaptation_needs['adaptation_type'] = 'urgent_response'
            adaptation_needs['suggested_workflow'] = 'adaptive_security_analysis'
        
        # Check categories for workflow selection
        categories = nlp_analysis.get('categories', [])
        if 'security_threat' in categories:
            adaptation_needs['needs_adaptation'] = True
            adaptation_needs['adaptation_type'] = 'threat_focused'
            adaptation_needs['suggested_workflow'] = 'intelligent_threat_detection'
        
        # Check for complex analysis needs
        if len(nlp_analysis.get('entities', [])) > 5 or len(nlp_analysis.get('key_phrases', [])) > 3:
            adaptation_needs['needs_adaptation'] = True
            adaptation_needs['adaptation_type'] = 'comprehensive_analysis'
            adaptation_needs['suggested_workflow'] = 'nlp_text_analysis'
        
        # Check for workflow restart signals
        restart_keywords = ['restart', 'start over', 'begin again', 'new approach', 'different direction']
        if any(keyword in user_input.lower() for keyword in restart_keywords):
            adaptation_needs['phase_restart'] = True
            adaptation_needs['planner_involvement'] = True
        
        # Check for planning signals
        planning_keywords = ['plan', 'strategy', 'approach', 'method', 'how to']
        if any(keyword in user_input.lower() for keyword in planning_keywords):
            adaptation_needs['planner_involvement'] = True
        
        return adaptation_needs
    
    def _generate_dynamic_response(self, nlp_analysis: Dict[str, Any], adaptation_needs: Dict[str, Any]) -> str:
        """Generate dynamic response based on NLP analysis and adaptation needs."""
        response = "🧠 **NLP Analysis Complete!** Here's what I discovered:\n\n"
        
        # Add entity information
        if nlp_analysis.get('entities'):
            response += "🔍 **Key Entities Detected:**\n"
            for entity in nlp_analysis['entities'][:5]:  # Limit to 5 entities
                response += f"   • {entity['text']} ({entity['label']})\n"
            response += "\n"
        
        # Add categorization
        if nlp_analysis.get('categories'):
            response += "📋 **Content Categories:**\n"
            for category in nlp_analysis['categories']:
                response += f"   • {category.replace('_', ' ').title()}\n"
            response += "\n"
        
        # Add sentiment analysis
        sentiment = nlp_analysis.get('sentiment', {})
        if sentiment.get('overall') != 'neutral':
            response += f"😊 **Sentiment Analysis:** {sentiment['overall'].title()}\n"
            response += f"   • Positive indicators: {sentiment.get('positive_score', 0)}\n"
            response += f"   • Negative indicators: {sentiment.get('negative_score', 0)}\n\n"
        
        # Add adaptation suggestions
        if adaptation_needs['needs_adaptation']:
            response += "🔄 **Workflow Adaptation Recommended:**\n"
            response += f"   • Type: {adaptation_needs['adaptation_type'].replace('_', ' ').title()}\n"
            response += f"   • Suggested Workflow: {adaptation_needs['suggested_workflow'].replace('_', ' ').title()}\n\n"
        
        if adaptation_needs['phase_restart']:
            response += "🔄 **Workflow Restart Recommended:**\n"
            response += "   • Current approach may not be optimal\n"
            response += "   • Suggesting new workflow direction\n\n"
        
        if adaptation_needs['planner_involvement']:
            response += "🧭 **Planner Agent Involvement:**\n"
            response += "   • Complex analysis detected\n"
            response += "   • Recommending workflow replanning\n\n"
        
        response += "💡 **Next Steps:**\n"
        if adaptation_needs['needs_adaptation']:
            response += "   1. Run suggested workflow\n"
            response += "   2. Adapt current workflow\n"
            response += "   3. Request planner assistance\n"
        else:
            response += "   1. Continue with current workflow\n"
            response += "   2. Request additional analysis\n"
            response += "   3. Modify analysis parameters\n"
        
        return response
    
    def _handle_policy_request(self, user_input: str, session_id: str) -> str:
        """Handle policy-related requests."""
        return f"Policy analysis requested: '{user_input}'. I can help you analyze security policies and map them to frameworks like MITRE ATT&CK."
    
    def _handle_threat_request(self, user_input: str, session_id: str) -> str:
        """Handle threat-related requests."""
        return f"Threat intelligence requested: '{user_input}'. I can help you process and analyze threat data."
    
    def _handle_framework_request(self, user_input: str, session_id: str) -> str:
        """Handle framework-related requests."""
        return f"Framework request: '{user_input}'. I can help you add and query cybersecurity frameworks like MITRE ATT&CK, D3fend, and NIST."
    
    def _format_response(self, state: AgentState) -> str:
        """Format the agent's response."""
        # Implementation will be added
        return "Response formatted here"
    
    async def start(self):
        """Start the agent."""
        # Host verification before initialization
        if ENCRYPTION_ENABLED:
            self._verify_host_compatibility()
        
        await self._initialize_mcp()
        print("🚀 LangGraph Cybersecurity Agent initialized!")
        print("🔧 Local tools registered:", len(self.mcp_tools))
        print("📚 Available workflows:", len(self.workflow_manager.list_templates()))
        print("🔐 Encryption:", "Enabled" if ENCRYPTION_ENABLED else "Disabled")
        
        if self.app:
            print("✅ LangGraph workflow engine: Active")
        else:
            print("⚠️  LangGraph workflow engine: Simplified mode (using fallback processing)")
        
        # Check spaCy installation and Apple Silicon compatibility
        self._check_spacy_installation()
    
    def _verify_host_compatibility(self):
        """Verify host compatibility for encrypted objects."""
        try:
            from bin.host_verification import HostVerification
            verifier = HostVerification()
            
            print("🔍 Verifying host compatibility...")
            results = verifier.verify_host_compatibility()
            
            if not results:
                print("⚠️  Host compatibility issues detected")
                verifier.prompt_for_reset()
            else:
                print("✅ Host compatibility verified")
                
        except Exception as e:
            print(f"⚠️  Host verification error: {e}")
            print("   Continuing with limited functionality...")
    
    def prompt_for_credentials(self, credential_type: str, description: str = "") -> Optional[Dict[str, str]]:
        """Safely prompt user for credentials and store in vault."""
        if not self.credential_vault:
            print("⚠️  Credential vault not available")
            return None
        
        try:
            print(f"🔐 Credential required: {credential_type}")
            if description:
                print(f"   Description: {description}")
            
            # Check if credentials already exist in vault
            existing_credential = self._get_credential_from_vault(credential_type)
            if existing_credential:
                print(f"✅ Found existing credentials for {credential_type}")
                use_existing = input("Use existing credentials? (Y/n): ").strip().lower()
                if use_existing != 'n':
                    return existing_credential
            
            # Prompt for new credentials
            print(f"\n📝 Please provide credentials for {credential_type}:")
            
            if credential_type in ['web_login', 'api_access', 'database']:
                username = input("Username: ").strip()
                password = input("Password: ").strip()
                
                if username and password:
                    # Store in vault
                    if credential_type == 'web_login':
                        url = input("URL: ").strip()
                        self.credential_vault.add_web_credential(
                            credential_type, url, username, password, description
                        )
                    elif credential_type == 'api_access':
                        self.credential_vault.add_api_key(
                            credential_type, password, description
                        )
                    else:
                        self.credential_vault.add_credential(
                            credential_type, username, password, description
                        )
                    
                    print(f"✅ Credentials stored securely in vault")
                    return {'username': username, 'password': password}
            
            elif credential_type == 'secret':
                secret_value = input("Secret value: ").strip()
                if secret_value:
                    self.credential_vault.add_secret(credential_type, secret_value, description)
                    print(f"✅ Secret stored securely in vault")
                    return {'value': secret_value}
            
            else:
                print(f"⚠️  Unknown credential type: {credential_type}")
                return None
            
        except Exception as e:
            print(f"❌ Error prompting for credentials: {e}")
            return None
        
        return None
    
    def _get_credential_from_vault(self, credential_type: str) -> Optional[Dict[str, str]]:
        """Get credentials from vault by type."""
        if not self.credential_vault:
            return None
        
        try:
            if credential_type in ['web_login', 'api_access', 'database']:
                # Try to find credentials by type
                credentials = self.credential_vault.search_credentials(credential_type)
                if credentials.get('credentials'):
                    return credentials['credentials'][0]
                elif credentials.get('web_credentials'):
                    return credentials['web_credentials'][0]
                elif credentials.get('api_keys'):
                    return credentials['api_keys'][0]
            
            elif credential_type == 'secret':
                secrets = self.credential_vault.search_credentials(credential_type)
                if secrets.get('secrets'):
                    return secrets['secrets'][0]
            
            return None
            
        except Exception as e:
            print(f"⚠️  Error retrieving credentials from vault: {e}")
            return None
    
    def get_stored_credentials(self, credential_type: str) -> Optional[Dict[str, str]]:
        """Get stored credentials from vault."""
        if not self.credential_vault:
            print("⚠️  Credential vault not available")
            return None
        
        try:
            if credential_type in ['web_login', 'api_access', 'database']:
                credentials = self.credential_vault.search_credentials(credential_type)
                if credentials.get('credentials'):
                    return self.credential_vault.get_credential(credentials['credentials'][0]['name'])
                elif credentials.get('web_credentials'):
                    return self.credential_vault.get_web_credential(credentials['web_credentials'][0]['name'])
                elif credentials.get('api_keys'):
                    return self.credential_vault.get_api_key(credentials['api_keys'][0]['name'])
            
            elif credential_type == 'secret':
                secrets = self.credential_vault.search_credentials(credential_type)
                if secrets.get('secrets'):
                    return self.credential_vault.get_secret(secrets['secrets'][0]['name'])
            
            return None
            
        except Exception as e:
            print(f"⚠️  Error retrieving stored credentials: {e}")
            return None
    
    # ============================================================================
    # VISUALIZATION METHODS
    # ============================================================================
    
    def show_dataframe_viewer(self, df, title: str = "Data Validation", description: str = "") -> Optional[str]:
        """Show interactive DataFrame viewer for data validation."""
        if not self.visualization_manager:
            print("⚠️  Visualization manager not available")
            return None
        
        try:
            return self.visualization_manager.create_dataframe_viewer(df, title, description)
        except Exception as e:
            print(f"❌ Error showing DataFrame viewer: {e}")
            return None
    
    def create_workflow_visualization(self, workflow_steps: List[Dict[str, Any]], 
                                    title: str = "Workflow Steps") -> Optional[str]:
        """Create beautiful workflow diagram visualization."""
        if not self.visualization_manager:
            print("⚠️  Visualization manager not available")
            return None
        
        try:
            return self.visualization_manager.create_workflow_diagram(workflow_steps, title)
        except Exception as e:
            print(f"❌ Error creating workflow visualization: {e}")
            return None
    
    def create_neo4j_graph_visualization(self, graph_data: Dict[str, Any], 
                                        title: str = "Resource Relationships") -> Optional[str]:
        """Create Neo4j graph visualization."""
        if not self.visualization_manager:
            print("⚠️  Visualization manager not available")
            return None
        
        try:
            return self.visualization_manager.create_neo4j_graph_visualization(graph_data, title)
        except Exception as e:
            print(f"❌ Error creating Neo4j graph visualization: {e}")
            return None
    
    def create_vega_lite_chart(self, data: pd.DataFrame, chart_spec: Dict[str, Any], 
                              title: str = "Data Visualization") -> Optional[str]:
        """Create Vega-Lite chart visualization."""
        if not self.visualization_manager:
            print("⚠️  Visualization manager not available")
            return None
        
        try:
            return self.visualization_manager.create_vega_lite_visualization(data, chart_spec, title)
        except Exception as e:
            print(f"❌ Error creating Vega-Lite chart: {e}")
            return None
    
    def export_visualization(self, visualization_type: str, data: Any, 
                           title: str, **kwargs) -> Optional[str]:
        """Export visualization to multiple formats."""
        if not self.visualization_manager:
            print("⚠️  Visualization manager not available")
            return None
        
        try:
            if visualization_type == 'dataframe':
                return self.visualization_manager._export_dataframe_html(data, title)
            elif visualization_type == 'workflow':
                return self.create_workflow_visualization(data, title)
            elif visualization_type == 'neo4j':
                return self.create_neo4j_graph_visualization(data, title)
            elif visualization_type == 'vega_lite':
                return self.create_vega_lite_chart(data, kwargs.get('chart_spec', {}), title)
            else:
                print(f"❌ Unknown visualization type: {visualization_type}")
                return None
                
        except Exception as e:
            print(f"❌ Error exporting visualization: {e}")
            return None
    
    def _check_spacy_installation(self):
        """Check spaCy installation and provide Apple Silicon compatibility info."""
        try:
            import spacy
            import platform
            
            is_apple_silicon = platform.machine() == 'arm64' and platform.system() == 'Darwin'
            spacy_version = spacy.__version__
            
            print(f"🧠 spaCy NLP Engine: {spacy_version}")
            
            if is_apple_silicon:
                if spacy_version.startswith('3.7'):
                    print("✅ Apple Silicon: Optimized (spaCy 3.7.x)")
                else:
                    print("⚠️  Apple Silicon: Suboptimal version detected")
                    print("   💡 For best performance on Apple Silicon, use: spaCy 3.7.2")
                    print("   📦 Install: pip install spacy==3.7.2")
            else:
                print("💻 Architecture: Standard (non-Apple Silicon)")
            
            # Check if English model is available
            try:
                spacy.load("en_core_web_sm")
                print("✅ English Language Model: Available")
            except OSError:
                print("⚠️  English Language Model: Not installed")
                print("   📦 Install: python -m spacy download en_core_web_sm")
                
        except ImportError:
            print("❌ spaCy: Not installed")
            print("   📦 Install: pip install spacy==3.7.2")
            print("   🌐 Model: python -m spacy download en_core_web_sm")

    # ============================================================================
    # VISUALIZATION TOOL EXECUTION METHODS
    # ============================================================================
    
    def execute_visualization_tool(self, tool_id: str, **kwargs) -> Dict[str, Any]:
        """Execute visualization tools based on MCP tool ID."""
        try:
            if tool_id == 'dataframe_viewer':
                if 'data' not in kwargs:
                    return {'error': 'Data required for DataFrame viewer'}
                df = kwargs['data']
                title = kwargs.get('title', 'Data Validation')
                description = kwargs.get('description', '')
                result = self.show_dataframe_viewer(df, title, description)
                return {
                    'tool': 'dataframe_viewer',
                    'result': result,
                    'success': result is not None
                }
                
            elif tool_id == 'workflow_diagram':
                if 'workflow_steps' not in kwargs:
                    return {'error': 'Workflow steps required for workflow diagram'}
                workflow_steps = kwargs['workflow_steps']
                title = kwargs.get('title', 'Workflow Steps')
                result = self.create_workflow_visualization(workflow_steps, title)
                return {
                    'tool': 'workflow_diagram',
                    'result': result,
                    'success': result is not None
                }
                
            elif tool_id == 'neo4j_graph_visualizer':
                if 'graph_data' not in kwargs:
                    return {'error': 'Graph data required for Neo4j visualization'}
                graph_data = kwargs['graph_data']
                title = kwargs.get('title', 'Resource Relationships')
                result = self.create_neo4j_graph_visualization(graph_data, title)
                return {
                    'tool': 'neo4j_graph_visualizer',
                    'result': result,
                    'success': result is not None
                }
                
            elif tool_id == 'vega_lite_charts':
                if 'data' not in kwargs or 'chart_spec' not in kwargs:
                    return {'error': 'Data and chart specification required for Vega-Lite charts'}
                data = kwargs['data']
                chart_spec = kwargs['chart_spec']
                title = kwargs.get('title', 'Data Visualization')
                result = self.create_vega_lite_chart(data, chart_spec, title)
                return {
                    'tool': 'vega_lite_charts',
                    'result': result,
                    'success': result is not None
                }
                
            elif tool_id == 'visualization_exporter':
                if 'visualization_type' not in kwargs or 'data' not in kwargs:
                    return {'error': 'Visualization type and data required for export'}
                visualization_type = kwargs['visualization_type']
                data = kwargs['data']
                title = kwargs.get('title', 'Exported Visualization')
                result = self.export_visualization(visualization_type, data, title, **kwargs)
                return {
                    'tool': 'visualization_exporter',
                    'result': result,
                    'success': result is not None
                }
                
            else:
                return {'error': f'Unknown visualization tool: {tool_id}'}
                
        except Exception as e:
            return {
                'error': f'Error executing visualization tool {tool_id}: {e}',
                'success': False
            }
    
    def get_visualization_tool_info(self, tool_id: str = None) -> Dict[str, Any]:
        """Get information about available visualization tools."""
        if tool_id:
            if tool_id in self.mcp_tools:
                return {
                    'tool_id': tool_id,
                    'info': self.mcp_tools[tool_id],
                    'available': True
                }
            else:
                return {
                    'tool_id': tool_id,
                    'available': False,
                    'error': 'Tool not found'
                }
        else:
            # Return all visualization tools
            viz_tools = {k: v for k, v in self.mcp_tools.items() 
                        if 'visualization' in k.lower() or 'viewer' in k.lower() or 'chart' in k.lower()}
            return {
                'available_tools': viz_tools,
                'total_count': len(viz_tools)
            }
    
    def integrate_visualization_in_workflow(self, workflow_step: str, data: Any, 
                                          visualization_type: str = 'auto', **kwargs) -> Dict[str, Any]:
        """Integrate visualization into workflow steps for data validation and analysis."""
        try:
            # Determine visualization type if auto
            if visualization_type == 'auto':
                if isinstance(data, pd.DataFrame):
                    visualization_type = 'dataframe_viewer'
                elif isinstance(data, dict) and 'nodes' in data and 'edges' in data:
                    visualization_type = 'neo4j_graph_visualizer'
                elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                    visualization_type = 'workflow_diagram'
                else:
                    visualization_type = 'vega_lite_charts'
            
            # Execute visualization
            result = self.execute_visualization_tool(visualization_type, data=data, **kwargs)
            
            # Add workflow context
            result['workflow_step'] = workflow_step
            result['timestamp'] = datetime.now().isoformat()
            result['session_id'] = getattr(self, 'current_session_id', 'unknown')
            
            # Log visualization creation
            if result.get('success'):
                print(f"🎨 Visualization created for {workflow_step}: {visualization_type}")
                if self.session_manager:
                    self.session_manager.log_activity(
                        f"Visualization created: {visualization_type} for {workflow_step}",
                        "visualization"
                    )
            
            return result
            
        except Exception as e:
            return {
                'error': f'Error integrating visualization in workflow: {e}',
                'workflow_step': workflow_step,
                'success': False
            }
    
    def suggest_workflow_visualizations(self, workflow_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Suggest appropriate visualizations for workflow data."""
        suggestions = []
        
        try:
            # Analyze workflow data and suggest visualizations
            if 'dataframes' in workflow_data:
                for df_name, df_data in workflow_data['dataframes'].items():
                    suggestions.append({
                        'type': 'dataframe_viewer',
                        'title': f'Data Validation: {df_name}',
                        'description': f'Interactive validation of {df_name} data',
                        'data_key': df_name
                    })
            
            if 'workflow_steps' in workflow_data:
                suggestions.append({
                    'type': 'workflow_diagram',
                    'title': 'Workflow Execution Steps',
                    'description': 'Visual representation of workflow progress',
                    'data_key': 'workflow_steps'
                })
            
            if 'graph_data' in workflow_data:
                suggestions.append({
                    'type': 'neo4j_graph_visualizer',
                    'title': 'Resource Relationships',
                    'description': 'Visualization of resource connections and dependencies',
                    'data_key': 'graph_data'
                })
            
            if 'analysis_results' in workflow_data:
                suggestions.append({
                    'type': 'vega_lite_charts',
                    'title': 'Analysis Results',
                    'description': 'Interactive charts of analysis outcomes',
                    'data_key': 'analysis_results'
                })
            
            return suggestions
            
        except Exception as e:
            print(f"⚠️  Error suggesting visualizations: {e}")
            return []

    # ============================================================================
    # MEMORY MANAGEMENT METHODS
    # ============================================================================
    
    def import_data_to_memory(self, data_type: str, data: Any, source: str, 
                             description: str = None, tags: List[str] = None, 
                             ttl_days: int = None, priority: int = 5) -> Dict[str, Any]:
        """Import data into context memory."""
        try:
            result = self.memory_integration.memory_tools.execute_tool(
                'import_data',
                data_type=data_type,
                data=data,
                source=source,
                description=description,
                tags=tags,
                ttl_days=ttl_days,
                priority=priority
            )
            
            if result['success']:
                print(f"🧠 Successfully imported {data_type} data into memory")
                return result
            else:
                print(f"❌ Failed to import data: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            return {'error': f'Import error: {e}', 'success': False}
    
    def retrieve_memory_context(self, query: str, domains: List[str] = None, 
                               tiers: List[str] = None, max_results: int = 10,
                               include_relationships: bool = False) -> Dict[str, Any]:
        """Retrieve relevant context from memory."""
        try:
            result = self.memory_integration.memory_tools.execute_tool(
                'retrieve_context',
                query=query,
                domains=domains,
                tiers=tiers,
                max_results=max_results,
                include_relationships=include_relationships
            )
            
            if result['success']:
                print(f"🔍 Retrieved {result.get('results_count', 0)} memory entries")
                return result
            else:
                print(f"❌ Failed to retrieve context: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            return {'error': f'Retrieve error: {e}', 'success': False}
    
    def add_memory_relationship(self, source_entity: str, target_entity: str, 
                               relationship_type: str, strength: float = 1.0,
                               metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add relationship between entities in memory."""
        try:
            result = self.memory_integration.memory_tools.execute_tool(
                'add_relationship',
                source_entity=source_entity,
                target_entity=target_entity,
                relationship_type=relationship_type,
                strength=strength,
                metadata=metadata or {}
            )
            
            if result['success']:
                print(f"🔗 Successfully added relationship: {source_entity} -> {target_entity}")
                return result
            else:
                print(f"❌ Failed to add relationship: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            return {'error': f'Relationship error: {e}', 'success': False}
    
    def get_memory_statistics(self, include_performance: bool = True, 
                             include_relationships: bool = True) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        try:
            result = self.memory_integration.memory_tools.execute_tool(
                'get_memory_stats',
                include_performance=include_performance,
                include_relationships=include_relationships
            )
            
            if result['success']:
                print("📊 Memory statistics retrieved successfully")
                return result
            else:
                print(f"❌ Failed to get memory stats: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            return {'error': f'Stats error: {e}', 'success': False}
    
    def cleanup_expired_memory(self, dry_run: bool = False, force: bool = False) -> Dict[str, Any]:
        """Clean up expired memory entries."""
        try:
            result = self.memory_integration.memory_tools.execute_tool(
                'cleanup_expired_memory',
                dry_run=dry_run,
                force=force
            )
            
            if result['success']:
                if dry_run:
                    print("🧹 Memory cleanup dry run completed")
                else:
                    print(f"🧹 Cleaned up {result.get('expired_entries_removed', 0)} expired entries")
                return result
            else:
                print(f"❌ Failed to cleanup memory: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            return {'error': f'Cleanup error: {e}', 'success': False}
    
    def export_memory_snapshot(self, include_data: bool = True, format_type: str = 'json',
                               compression: bool = True) -> Dict[str, Any]:
        """Export memory snapshot for backup or analysis."""
        try:
            result = self.memory_integration.memory_tools.execute_tool(
                'export_memory_snapshot',
                include_data=include_data,
                format=format_type,
                compression=compression
            )
            
            if result['success']:
                print(f"💾 Memory snapshot exported to {result.get('snapshot_path', 'unknown')}")
                return result
            else:
                print(f"❌ Failed to export snapshot: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            return {'error': f'Export error: {e}', 'success': False}
    
    def suggest_memory_actions(self, workflow_data: Dict[str, Any], context: str = '') -> Dict[str, Any]:
        """Get suggestions for memory management actions based on workflow data."""
        try:
            result = self.memory_integration.memory_tools.execute_tool(
                'suggest_memory_actions',
                workflow_data=workflow_data,
                context=context
            )
            
            if result['success']:
                suggestions_count = result.get('suggestions_count', 0)
                print(f"💡 Generated {suggestions_count} memory action suggestions")
                return result
            else:
                print(f"❌ Failed to get suggestions: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            return {'error': f'Suggestions error: {e}', 'success': False}
    
    def auto_import_workflow_data(self, workflow_data: Dict[str, Any], workflow_context: str = '') -> List[str]:
        """Automatically import relevant workflow data into memory."""
        try:
            imported_ids = self.memory_integration.auto_import_workflow_data(workflow_data, workflow_context)
            if imported_ids:
                print(f"🧠 Auto-imported {len(imported_ids)} data sets into memory")
            return imported_ids
        except Exception as e:
            print(f"❌ Auto-import error: {e}")
            return []
    
    def get_memory_context_for_workflow(self, workflow_query: str, max_results: int = 20) -> List[Any]:
        """Get relevant memory context for a workflow."""
        try:
            context = self.memory_integration.get_memory_context(workflow_query, max_results=max_results)
            if context:
                print(f"🧠 Retrieved {len(context)} relevant memory entries for workflow")
            return context
        except Exception as e:
            print(f"❌ Memory context error: {e}")
            return []
    
    async def _handle_memory_command(self, command: str):
        """Handle memory management commands from CLI."""
        try:
            parts = command.split(':', 1)
            if len(parts) != 2:
                print("❌ Invalid memory command format. Use: memory:command")
                return
            
            subcommand = parts[1].strip()
            
            if subcommand == 'stats':
                print("\n📊 Memory Statistics")
                print("-" * 40)
                stats = self.get_memory_statistics()
                if stats.get('success'):
                    self._display_memory_stats(stats['stats'])
                else:
                    print(f"❌ Error: {stats.get('error', 'Unknown error')}")
            
            elif subcommand == 'cleanup':
                print("\n🧹 Memory Cleanup")
                print("-" * 40)
                print("This will remove expired memory entries.")
                confirm = input("Continue? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    result = self.cleanup_expired_memory()
                    if result.get('success'):
                        print(f"✅ Cleanup completed: {result.get('expired_entries_removed', 0)} entries removed")
                    else:
                        print(f"❌ Cleanup failed: {result.get('error', 'Unknown error')}")
                else:
                    print("❌ Cleanup cancelled")
            
            elif subcommand == 'export':
                print("\n💾 Memory Export")
                print("-" * 40)
                result = self.export_memory_snapshot()
                if result.get('success'):
                    print(f"✅ Export completed: {result.get('snapshot_path', 'unknown')}")
                else:
                    print(f"❌ Export failed: {result.get('error', 'Unknown error')}")
            
            elif subcommand.startswith('import '):
                print("\n📥 Memory Import")
                print("-" * 40)
                import_parts = subcommand[7:].strip().split()
                if len(import_parts) < 2:
                    print("❌ Use: memory:import <type> <data_description>")
                    return
                
                data_type = import_parts[0]
                data_description = ' '.join(import_parts[1:])
                
                print(f"Importing {data_type} data: {data_description}")
                print("Note: This is a simplified import. For complex data, use the interactive CLI.")
                
                # Create sample data for demonstration
                sample_data = {
                    'description': data_description,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'cli_import'
                }
                
                result = self.import_data_to_memory(
                    data_type=data_type,
                    data=sample_data,
                    source='cli_import',
                    description=f"CLI imported {data_type} data",
                    tags=[data_type, 'cli_import']
                )
                
                if result.get('success'):
                    print(f"✅ Import completed: {result.get('memory_id', 'unknown')}")
                else:
                    print(f"❌ Import failed: {result.get('error', 'Unknown error')}")
            
            elif subcommand.startswith('query '):
                print("\n🔍 Memory Query")
                print("-" * 40)
                query = subcommand[6:].strip()
                if not query:
                    print("❌ Use: memory:query <search_terms>")
                    return
                
                print(f"Searching memory for: {query}")
                result = self.retrieve_memory_context(query, max_results=10)
                
                if result.get('success'):
                    results = result.get('results', [])
                    if results:
                        print(f"\n✅ Found {len(results)} results:")
                        for i, entry in enumerate(results[:5], 1):  # Show first 5
                            print(f"  {i}. {entry.get('domain', 'Unknown')}: {entry.get('description', 'No description')}")
                        if len(results) > 5:
                            print(f"  ... and {len(results) - 5} more results")
                    else:
                        print("❌ No results found")
                else:
                    print(f"❌ Query failed: {result.get('error', 'Unknown error')}")
            
            else:
                print(f"❌ Unknown memory command: {subcommand}")
                print("Available commands: stats, cleanup, export, import, query")
        
        except Exception as e:
            print(f"❌ Error handling memory command: {e}")
    
    def _display_memory_stats(self, stats: Dict[str, Any]):
        """Display memory statistics in a formatted way."""
        try:
            print(f"📈 Overview:")
            print(f"   Total entries: {stats.get('total_entries', 0):,}")
            print(f"   Total size: {stats.get('total_size_bytes', 0):,} bytes")
            print(f"   Average size: {stats.get('average_size_bytes', 0):,.0f} bytes")
            
            print(f"\n🏷️  By Domain:")
            domain_counts = stats.get('domain_counts', {})
            for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    print(f"   {domain}: {count:,}")
            
            print(f"\n⏰ By Tier:")
            tier_counts = stats.get('tier_counts', {})
            for tier, count in sorted(tier_counts.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    print(f"   {tier}: {count:,}")
            
            print(f"\n💾 Cache Status:")
            cache_stats = stats.get('cache_stats', {})
            for tier, count in cache_stats.items():
                print(f"   {tier}: {count:,}")
            
            print(f"\n🔗 Relationships:")
            rel_stats = stats.get('relationship_stats', {})
            print(f"   Total entities: {rel_stats.get('total_entities', 0):,}")
            print(f"   Total relationships: {rel_stats.get('total_relationships', 0):,}")
            
        except Exception as e:
            print(f"❌ Error displaying stats: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution function."""
    agent = LangGraphCybersecurityAgent()
    await agent.start()
    
    # Interactive chat mode
    print("\n" + "="*60)
    print("🔒 LangGraph Cybersecurity Agent")
    print("="*60)
    print("Welcome! I'm your cybersecurity analysis assistant.")
    print("I can help you with policy analysis, threat intelligence, incident response, and more.")
    print("\nAvailable workflows:")
    
    for template_name in agent.workflow_manager.list_templates():
        template = agent.workflow_manager.get_template(template_name)
        print(f"  • {template.name}: {template.description}")
    
    print("\nMemory Management Commands:")
    print("  • memory:stats - Show memory statistics")
    print("  • memory:cleanup - Clean up expired memory")
    print("  • memory:export - Export memory snapshot")
    print("  • memory:import <type> <data> - Import data to memory")
    print("  • memory:query <search> - Search memory for context")
    
    print("\nHow can I help you today? (Type 'quit' to exit)")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n🤖 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye! Stay secure!")
                break
            
            if not user_input:
                continue
            
            # Handle memory management commands
            if user_input.startswith('memory:'):
                await agent._handle_memory_command(user_input)
                continue
            
            print("\n🔄 Processing...")
            response = await agent.chat(user_input)
            print(f"\n🤖 Agent: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Stay secure!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
