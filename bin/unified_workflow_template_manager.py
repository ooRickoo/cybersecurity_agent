#!/usr/bin/env python3
"""
Unified Workflow Template Manager
Consolidates all workflow template management into a single, standardized system.

This replaces the scattered template management across:
- WorkflowTemplateManager in langgraph_cybersecurity_agent.py
- EnhancedWorkflowTemplateManager in enhanced_workflow_template_manager.py
- workflow_templates/templates.json

Features:
- Single source of truth (JSON file)
- Dynamic template loading and caching
- Template versioning and migration
- Performance optimization
- Backward compatibility
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import uuid
from collections import defaultdict

logger = logging.getLogger(__name__)

class WorkflowComplexity(Enum):
    """Workflow complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"

class WorkflowCategory(Enum):
    """Workflow categories."""
    ANALYSIS = "analysis"
    INVESTIGATION = "investigation"
    RESPONSE = "response"
    PREVENTION = "prevention"
    COMPLIANCE = "compliance"
    RESEARCH = "research"
    AUTOMATION = "automation"

class ExecutionMode(Enum):
    """Execution modes for workflows."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    INTERACTIVE = "interactive"

@dataclass
class WorkflowStep:
    """Individual step in a workflow."""
    step_id: str
    name: str
    description: str
    tool_name: str
    tool_category: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: float = 0.0
    retry_count: int = 0
    timeout: float = 300.0
    parallel_execution: bool = False
    conditional_execution: Optional[Dict[str, Any]] = None
    output_mapping: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowTemplate:
    """Complete workflow template definition."""
    template_id: str
    name: str
    description: str
    category: str
    complexity: str
    steps: List[WorkflowStep]
    execution_mode: str = "sequential"
    required_tools: List[str] = field(default_factory=list)
    required_inputs: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    version: str = "1.0"
    created_at: float = field(default_factory=time.time)
    last_modified: float = field(default_factory=time.time)
    usage_count: int = 0
    success_rate: float = 0.0
    average_execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class UnifiedWorkflowTemplateManager:
    """
    Unified workflow template manager that consolidates all template management.
    
    This is the single source of truth for all workflow templates in the system.
    """
    
    def __init__(self, templates_file: str = "workflow_templates/templates.json"):
        self.templates_file = Path(templates_file)
        self.templates: Dict[str, WorkflowTemplate] = {}
        self.template_cache: Dict[str, Any] = {}
        self.cache_timestamp: float = 0
        self.cache_ttl: float = 300  # 5 minutes cache TTL
        
        # Load templates on initialization
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load templates from JSON file."""
        try:
            if not self.templates_file.exists():
                logger.warning(f"Templates file not found: {self.templates_file}")
                self._create_default_templates()
                return
            
            with open(self.templates_file, 'r', encoding='utf-8') as f:
                templates_data = json.load(f)
            
            self.templates = {}
            for template_id, template_data in templates_data.items():
                try:
                    template = self._parse_template(template_id, template_data)
                    self.templates[template_id] = template
                except Exception as e:
                    logger.error(f"Error parsing template {template_id}: {e}")
                    continue
            
            self.cache_timestamp = time.time()
            logger.info(f"Loaded {len(self.templates)} workflow templates")
            
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            self._create_default_templates()
    
    def _parse_template(self, template_id: str, template_data: Dict[str, Any]) -> WorkflowTemplate:
        """Parse template data into WorkflowTemplate object."""
        # Parse steps
        steps = []
        for step_data in template_data.get("steps", []):
            step = WorkflowStep(
                step_id=step_data.get("step_id", str(uuid.uuid4())),
                name=step_data.get("name", ""),
                description=step_data.get("description", ""),
                tool_name=step_data.get("tool_name", ""),
                tool_category=step_data.get("tool_category", ""),
                parameters=step_data.get("parameters", {}),
                dependencies=step_data.get("dependencies", []),
                estimated_duration=step_data.get("estimated_duration", 0.0),
                retry_count=step_data.get("retry_count", 0),
                timeout=step_data.get("timeout", 300.0),
                parallel_execution=step_data.get("parallel_execution", False),
                conditional_execution=step_data.get("conditional_execution"),
                output_mapping=step_data.get("output_mapping", {})
            )
            steps.append(step)
        
        # Create template
        template = WorkflowTemplate(
            template_id=template_id,
            name=template_data.get("name", ""),
            description=template_data.get("description", ""),
            category=template_data.get("category", "analysis"),
            complexity=template_data.get("complexity", "moderate"),
            steps=steps,
            execution_mode=template_data.get("execution_mode", "sequential"),
            required_tools=template_data.get("required_tools", []),
            required_inputs=template_data.get("required_inputs", []),
            expected_outputs=template_data.get("expected_outputs", []),
            tags=template_data.get("tags", []),
            version=template_data.get("version", "1.0"),
            created_at=template_data.get("created_at", time.time()),
            last_modified=template_data.get("last_modified", time.time()),
            usage_count=template_data.get("usage_count", 0),
            success_rate=template_data.get("success_rate", 0.0),
            average_execution_time=template_data.get("average_execution_time", 0.0),
            metadata=template_data.get("metadata", {})
        )
        
        return template
    
    def _create_default_templates(self) -> None:
        """Create default templates if none exist."""
        logger.info("Creating default workflow templates")
        
        # Create basic templates that were previously hardcoded
        default_templates = {
            "network_analysis": {
                "template_id": "network_analysis",
                "name": "Network Analysis",
                "description": "Analyze network traffic and PCAP files for security insights",
                "category": "analysis",
                "complexity": "moderate",
                "steps": [
                    {
                        "step_id": "analyze_pcap",
                        "name": "Analyze PCAP File",
                        "description": "Analyze network traffic from PCAP file",
                        "tool_name": "pcap_analyzer",
                        "tool_category": "network",
                        "parameters": {},
                        "dependencies": [],
                        "estimated_duration": 30.0
                    }
                ],
                "execution_mode": "sequential",
                "required_tools": ["pcap_analyzer"],
                "required_inputs": ["pcap_file"],
                "expected_outputs": ["analysis_report"],
                "tags": ["network", "analysis", "pcap"]
            },
            "vulnerability_assessment": {
                "template_id": "vulnerability_assessment",
                "name": "Vulnerability Assessment",
                "description": "Comprehensive vulnerability scanning and assessment",
                "category": "assessment",
                "complexity": "moderate",
                "steps": [
                    {
                        "step_id": "scan_target",
                        "name": "Scan Target",
                        "description": "Perform vulnerability scan on target",
                        "tool_name": "vulnerability_scanner",
                        "tool_category": "security",
                        "parameters": {},
                        "dependencies": [],
                        "estimated_duration": 45.0
                    }
                ],
                "execution_mode": "sequential",
                "required_tools": ["vulnerability_scanner"],
                "required_inputs": ["target"],
                "expected_outputs": ["scan_report"],
                "tags": ["vulnerability", "scanning", "assessment"]
            },
            "threat_hunting": {
                "template_id": "threat_hunting",
                "name": "Threat Hunting",
                "description": "Proactive threat hunting and IOC analysis",
                "category": "investigation",
                "complexity": "complex",
                "steps": [
                    {
                        "step_id": "hunt_threats",
                        "name": "Hunt Threats",
                        "description": "Perform threat hunting analysis",
                        "tool_name": "threat_hunter",
                        "tool_category": "security",
                        "parameters": {},
                        "dependencies": [],
                        "estimated_duration": 60.0
                    }
                ],
                "execution_mode": "adaptive",
                "required_tools": ["threat_hunter"],
                "required_inputs": ["data_source"],
                "expected_outputs": ["threat_report"],
                "tags": ["threat", "hunting", "investigation"]
            }
        }
        
        # Save default templates
        self._save_templates(default_templates)
        self._load_templates()
    
    def _save_templates(self, templates_data: Dict[str, Any]) -> None:
        """Save templates to JSON file."""
        try:
            # Ensure directory exists
            self.templates_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.templates_file, 'w', encoding='utf-8') as f:
                json.dump(templates_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(templates_data)} templates to {self.templates_file}")
            
        except Exception as e:
            logger.error(f"Error saving templates: {e}")
    
    def _refresh_cache_if_needed(self) -> None:
        """Refresh cache if it's expired."""
        if time.time() - self.cache_timestamp > self.cache_ttl:
            self._load_templates()
    
    def get_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get a workflow template by ID."""
        self._refresh_cache_if_needed()
        
        # Try exact match first
        if template_id in self.templates:
            return self.templates[template_id]
        
        # Try base name match (e.g., "network_analysis" matches "network_analysis_v1")
        base_name = template_id
        for template_key in self.templates.keys():
            if template_key.startswith(base_name + "_") or template_key == base_name:
                return self.templates[template_key]
        
        return None
    
    def list_templates(self) -> List[str]:
        """List all available template IDs."""
        self._refresh_cache_if_needed()
        return list(self.templates.keys())
    
    def get_templates_by_category(self, category: str) -> List[WorkflowTemplate]:
        """Get templates by category."""
        self._refresh_cache_if_needed()
        return [
            template for template in self.templates.values()
            if template.category.lower() == category.lower()
        ]
    
    def get_templates_by_complexity(self, complexity: str) -> List[WorkflowTemplate]:
        """Get templates by complexity level."""
        self._refresh_cache_if_needed()
        return [
            template for template in self.templates.values()
            if template.complexity.lower() == complexity.lower()
        ]
    
    def search_templates(self, query: str) -> List[WorkflowTemplate]:
        """Search templates by name, description, or tags."""
        self._refresh_cache_if_needed()
        query_lower = query.lower()
        
        results = []
        for template in self.templates.values():
            # Search in name, description, and tags
            if (query_lower in template.name.lower() or
                query_lower in template.description.lower() or
                any(query_lower in tag.lower() for tag in template.tags)):
                results.append(template)
        
        return results
    
    def suggest_workflow(self, user_input: str) -> Optional[str]:
        """Suggest the best workflow template based on user input."""
        self._refresh_cache_if_needed()
        input_lower = user_input.lower()
        
        # Score each template based on keyword matches
        template_scores = {}
        for template_id, template in self.templates.items():
            score = 0
            
            # Check template name
            if any(keyword in template.name.lower() for keyword in input_lower.split()):
                score += 3
            
            # Check description
            if any(keyword in template.description.lower() for keyword in input_lower.split()):
                score += 2
            
            # Check tags
            for tag in template.tags:
                if any(keyword in tag.lower() for keyword in input_lower.split()):
                    score += 1
            
            if score > 0:
                template_scores[template_id] = score
        
        # Return template with highest score
        if template_scores:
            best_template = max(template_scores.items(), key=lambda x: x[1])
            return best_template[0]
        
        return None
    
    def create_template(self, template_data: Dict[str, Any]) -> str:
        """Create a new workflow template."""
        template_id = template_data.get("template_id", str(uuid.uuid4()))
        
        # Parse and validate template
        try:
            template = self._parse_template(template_id, template_data)
            self.templates[template_id] = template
            
            # Save to file
            self._save_all_templates()
            
            logger.info(f"Created new template: {template_id}")
            return template_id
            
        except Exception as e:
            logger.error(f"Error creating template: {e}")
            raise
    
    def update_template(self, template_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing workflow template."""
        if template_id not in self.templates:
            return False
        
        try:
            # Update template
            template = self.templates[template_id]
            for key, value in updates.items():
                if hasattr(template, key):
                    setattr(template, key, value)
            
            template.last_modified = time.time()
            
            # Save to file
            self._save_all_templates()
            
            logger.info(f"Updated template: {template_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating template: {e}")
            return False
    
    def delete_template(self, template_id: str) -> bool:
        """Delete a workflow template."""
        if template_id not in self.templates:
            return False
        
        try:
            del self.templates[template_id]
            self._save_all_templates()
            
            logger.info(f"Deleted template: {template_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting template: {e}")
            return False
    
    def _save_all_templates(self) -> None:
        """Save all templates to JSON file."""
        templates_data = {}
        for template_id, template in self.templates.items():
            templates_data[template_id] = {
                "template_id": template.template_id,
                "name": template.name,
                "description": template.description,
                "category": template.category,
                "complexity": template.complexity,
                "steps": [
                    {
                        "step_id": step.step_id,
                        "name": step.name,
                        "description": step.description,
                        "tool_name": step.tool_name,
                        "tool_category": step.tool_category,
                        "parameters": step.parameters,
                        "dependencies": step.dependencies,
                        "estimated_duration": step.estimated_duration,
                        "retry_count": step.retry_count,
                        "timeout": step.timeout,
                        "parallel_execution": step.parallel_execution,
                        "conditional_execution": step.conditional_execution,
                        "output_mapping": step.output_mapping
                    }
                    for step in template.steps
                ],
                "execution_mode": template.execution_mode,
                "required_tools": template.required_tools,
                "required_inputs": template.required_inputs,
                "expected_outputs": template.expected_outputs,
                "tags": template.tags,
                "version": template.version,
                "created_at": template.created_at,
                "last_modified": template.last_modified,
                "usage_count": template.usage_count,
                "success_rate": template.success_rate,
                "average_execution_time": template.average_execution_time,
                "metadata": template.metadata
            }
        
        self._save_templates(templates_data)
    
    def record_template_usage(self, template_id: str, execution_time: float, success: bool) -> None:
        """Record template usage statistics."""
        if template_id not in self.templates:
            return
        
        template = self.templates[template_id]
        template.usage_count += 1
        
        # Update success rate
        if success:
            current_successes = template.success_rate * (template.usage_count - 1)
            template.success_rate = (current_successes + 1) / template.usage_count
        else:
            current_successes = template.success_rate * (template.usage_count - 1)
            template.success_rate = current_successes / template.usage_count
        
        # Update average execution time
        current_total_time = template.average_execution_time * (template.usage_count - 1)
        template.average_execution_time = (current_total_time + execution_time) / template.usage_count
        
        template.last_modified = time.time()
        
        # Save updated statistics
        self._save_all_templates()
    
    def get_template_statistics(self) -> Dict[str, Any]:
        """Get overall template statistics."""
        self._refresh_cache_if_needed()
        
        total_templates = len(self.templates)
        total_usage = sum(template.usage_count for template in self.templates.values())
        avg_success_rate = (
            sum(template.success_rate for template in self.templates.values()) / total_templates
            if total_templates > 0 else 0
        )
        
        categories = defaultdict(int)
        complexities = defaultdict(int)
        
        for template in self.templates.values():
            categories[template.category] += 1
            complexities[template.complexity] += 1
        
        return {
            "total_templates": total_templates,
            "total_usage": total_usage,
            "average_success_rate": avg_success_rate,
            "categories": dict(categories),
            "complexities": dict(complexities),
            "most_used": max(self.templates.values(), key=lambda t: t.usage_count).template_id if self.templates else None
        }


# Global instance for backward compatibility
_unified_manager = None

def get_unified_workflow_template_manager() -> UnifiedWorkflowTemplateManager:
    """Get the global unified workflow template manager instance."""
    global _unified_manager
    if _unified_manager is None:
        _unified_manager = UnifiedWorkflowTemplateManager()
    return _unified_manager

# Backward compatibility aliases
def get_workflow_template_manager() -> UnifiedWorkflowTemplateManager:
    """Backward compatibility alias."""
    return get_unified_workflow_template_manager()

def get_enhanced_workflow_template_manager() -> UnifiedWorkflowTemplateManager:
    """Backward compatibility alias."""
    return get_unified_workflow_template_manager()


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the unified manager
    manager = UnifiedWorkflowTemplateManager()
    
    print("🧪 Testing Unified Workflow Template Manager")
    print("=" * 60)
    
    # List all templates
    templates = manager.list_templates()
    print(f"📋 Available templates: {len(templates)}")
    for template_id in templates:
        print(f"   - {template_id}")
    
    # Test template retrieval
    if templates:
        template_id = templates[0]
        template = manager.get_template(template_id)
        if template:
            print(f"\n🔍 Template: {template.name}")
            print(f"   Description: {template.description}")
            print(f"   Category: {template.category}")
            print(f"   Complexity: {template.complexity}")
            print(f"   Steps: {len(template.steps)}")
    
    # Test search
    search_results = manager.search_templates("analysis")
    print(f"\n🔍 Search results for 'analysis': {len(search_results)}")
    for template in search_results:
        print(f"   - {template.name}")
    
    # Test suggestion
    suggestion = manager.suggest_workflow("analyze network traffic")
    print(f"\n💡 Workflow suggestion for 'analyze network traffic': {suggestion}")
    
    # Get statistics
    stats = manager.get_template_statistics()
    print(f"\n📊 Template Statistics:")
    print(f"   Total templates: {stats['total_templates']}")
    print(f"   Total usage: {stats['total_usage']}")
    print(f"   Average success rate: {stats['average_success_rate']:.2%}")
    print(f"   Categories: {stats['categories']}")
    
    print("\n✅ Unified Workflow Template Manager test completed")
