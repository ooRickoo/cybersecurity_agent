#!/usr/bin/env python3
"""
Memory Management MCP Tools
MCP-compatible tools for the Runner Agent to use memory management dynamically in workflows.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from bin.context_memory_manager import (
    ContextMemoryManager, MemoryDomain, MemoryTier, DataType
)

class MemoryMCPTools:
    """MCP-compatible memory management tools for dynamic workflow integration."""
    
    def __init__(self, session_manager=None):
        self.memory_manager = ContextMemoryManager()
        self.session_manager = session_manager
        
        # Import patterns for common data types
        self.import_patterns = {
            'hosts': {
                'domain': MemoryDomain.HOST_INVENTORY,
                'tier': MemoryTier.LONG_TERM,
                'ttl_days': 365,
                'tags': ['hosts', 'inventory', 'assets'],
                'description_template': 'Host inventory data from {source}'
            },
            'applications': {
                'domain': MemoryDomain.APPLICATION_INVENTORY,
                'tier': MemoryTier.LONG_TERM,
                'ttl_days': 365,
                'tags': ['applications', 'inventory', 'assets'],
                'description_template': 'Application inventory data from {source}'
            },
            'users': {
                'domain': MemoryDomain.USER_INVENTORY,
                'tier': MemoryTier.LONG_TERM,
                'ttl_days': 365,
                'tags': ['users', 'inventory', 'assets'],
                'description_template': 'User inventory data from {source}'
            },
            'networks': {
                'domain': MemoryDomain.NETWORK_INVENTORY,
                'tier': MemoryTier.LONG_TERM,
                'ttl_days': 365,
                'tags': ['networks', 'inventory', 'assets'],
                'description_template': 'Network inventory data from {source}'
            },
            'iocs': {
                'domain': MemoryDomain.IOC_COLLECTION,
                'tier': MemoryTier.SHORT_TERM,
                'ttl_days': 7,
                'tags': ['iocs', 'threats', 'hunting'],
                'description_template': 'IOC collection from {source}'
            },
            'threat_actors': {
                'domain': MemoryDomain.THREAT_ACTORS,
                'tier': MemoryTier.MEDIUM_TERM,
                'ttl_days': 30,
                'tags': ['threat_actors', 'intelligence', 'threats'],
                'description_template': 'Threat actor intelligence from {source}'
            },
            'investigation': {
                'domain': MemoryDomain.INVESTIGATION_ENTITIES,
                'tier': MemoryTier.SHORT_TERM,
                'ttl_days': 7,
                'tags': ['investigation', 'entities', 'active'],
                'description_template': 'Investigation entities from {source}'
            },
            'splunk_schemas': {
                'domain': MemoryDomain.SPLUNK_SCHEMAS,
                'tier': MemoryTier.LONG_TERM,
                'ttl_days': 365,
                'tags': ['splunk', 'schemas', 'logging'],
                'description_template': 'Splunk schema from {source}'
            },
            'mitre_attack': {
                'domain': MemoryDomain.MITRE_ATTACK,
                'tier': MemoryTier.LONG_TERM,
                'ttl_days': 365,
                'tags': ['mitre', 'attack', 'framework'],
                'description_template': 'MITRE ATT&CK framework from {source}'
            },
            'mitre_d3fend': {
                'domain': MemoryDomain.MITRE_D3FEND,
                'tier': MemoryTier.LONG_TERM,
                'ttl_days': 365,
                'tags': ['mitre', 'd3fend', 'defense'],
                'description_template': 'MITRE D3fend framework from {source}'
            },
            'nist': {
                'domain': MemoryDomain.NIST_FRAMEWORKS,
                'tier': MemoryTier.LONG_TERM,
                'ttl_days': 365,
                'tags': ['nist', 'framework', 'compliance'],
                'description_template': 'NIST framework from {source}'
            },
            'grc_policies': {
                'domain': MemoryDomain.GRC_POLICIES,
                'tier': MemoryTier.LONG_TERM,
                'ttl_days': 365,
                'tags': ['grc', 'policies', 'compliance'],
                'description_template': 'GRC policies from {source}'
            }
        }
    
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get all available memory management MCP tools."""
        return {
            'import_data': {
                'name': 'Import Data to Memory',
                'description': 'Import data into context memory with automatic domain detection and TTL management',
                'category': 'memory_management',
                'parameters': {
                    'data_type': {'type': 'string', 'description': 'Type of data (hosts, applications, users, etc.)'},
                    'data': {'type': 'any', 'description': 'Data to import (DataFrame, dict, or text)'},
                    'source': {'type': 'string', 'description': 'Source identifier for the data'},
                    'description': {'type': 'string', 'description': 'Custom description (optional)'},
                    'tags': {'type': 'list', 'description': 'Additional tags (optional)'},
                    'ttl_days': {'type': 'integer', 'description': 'Custom TTL in days (optional)'},
                    'priority': {'type': 'integer', 'description': 'Priority 1-10 (optional)'}
                },
                'returns': {'type': 'dict', 'description': 'Import result with memory ID and metadata'},
                'available': True
            },
            'retrieve_context': {
                'name': 'Retrieve Context from Memory',
                'description': 'Search and retrieve relevant context from memory based on query',
                'category': 'memory_management',
                'parameters': {
                    'query': {'type': 'string', 'description': 'Search query for context retrieval'},
                    'domains': {'type': 'list', 'description': 'Specific domains to search (optional)'},
                    'tiers': {'type': 'list', 'description': 'Specific memory tiers to search (optional)'},
                    'max_results': {'type': 'integer', 'description': 'Maximum number of results to return'},
                    'include_relationships': {'type': 'boolean', 'description': 'Include entity relationships in results'}
                },
                'returns': {'type': 'dict', 'description': 'Retrieved context entries with metadata'},
                'available': True
            },
            'add_relationship': {
                'name': 'Add Entity Relationship',
                'description': 'Create or update relationships between entities in memory',
                'category': 'memory_management',
                'parameters': {
                    'source_entity': {'type': 'string', 'description': 'Source entity identifier'},
                    'target_entity': {'type': 'string', 'description': 'Target entity identifier'},
                    'relationship_type': {'type': 'string', 'description': 'Type of relationship'},
                    'strength': {'type': 'float', 'description': 'Relationship strength (0.0-1.0)'},
                    'metadata': {'type': 'dict', 'description': 'Additional relationship metadata (optional)'}
                },
                'returns': {'type': 'dict', 'description': 'Relationship creation result'},
                'available': True
            },
            'get_related_entities': {
                'name': 'Get Related Entities',
                'description': 'Retrieve entities related to a specific entity',
                'category': 'memory_management',
                'parameters': {
                    'entity': {'type': 'string', 'description': 'Entity identifier to find relationships for'},
                    'max_depth': {'type': 'integer', 'description': 'Maximum relationship depth to explore'},
                    'relationship_types': {'type': 'list', 'description': 'Filter by specific relationship types (optional)'}
                },
                'returns': {'type': 'dict', 'description': 'Related entities with relationship details'},
                'available': True
            },
            'get_memory_stats': {
                'name': 'Get Memory Statistics',
                'description': 'Retrieve comprehensive memory system statistics',
                'category': 'memory_management',
                'parameters': {
                    'include_performance': {'type': 'boolean', 'description': 'Include performance metrics (optional)'},
                    'include_relationships': {'type': 'boolean', 'description': 'Include relationship statistics (optional)'}
                },
                'returns': {'type': 'dict', 'description': 'Memory statistics and metrics'},
                'available': True
            },
            'cleanup_expired_memory': {
                'name': 'Cleanup Expired Memory',
                'description': 'Remove expired memory entries and optimize storage',
                'category': 'memory_management',
                'parameters': {
                    'dry_run': {'type': 'boolean', 'description': 'Show what would be cleaned without actually removing (optional)'},
                    'force': {'type': 'boolean', 'description': 'Force cleanup even if recently accessed (optional)'}
                },
                'returns': {'type': 'dict', 'description': 'Cleanup results with counts and freed space'},
                'available': True
            },
            'export_memory_snapshot': {
                'name': 'Export Memory Snapshot',
                'description': 'Export a complete snapshot of memory for backup or analysis',
                'category': 'memory_management',
                'parameters': {
                    'include_data': {'type': 'boolean', 'description': 'Include actual data in export (optional)'},
                    'format': {'type': 'string', 'description': 'Export format (json, csv, pickle) (optional)'},
                    'compression': {'type': 'boolean', 'description': 'Compress export file (optional)'}
                },
                'returns': {'type': 'dict', 'description': 'Export result with file path and metadata'},
                'available': True
            },
            'suggest_memory_actions': {
                'name': 'Suggest Memory Actions',
                'description': 'Analyze workflow data and suggest appropriate memory management actions',
                'category': 'memory_management',
                'parameters': {
                    'workflow_data': {'type': 'dict', 'description': 'Workflow data to analyze for memory suggestions'},
                    'context': {'type': 'string', 'description': 'Workflow context for better suggestions'}
                },
                'returns': {'type': 'dict', 'description': 'Suggested memory actions with reasoning'},
                'available': True
            }
        }
    
    def execute_tool(self, tool_id: str, **kwargs) -> Dict[str, Any]:
        """Execute a memory management tool based on MCP tool ID."""
        try:
            if tool_id == 'import_data':
                return self._execute_import_data(**kwargs)
            elif tool_id == 'retrieve_context':
                return self._execute_retrieve_context(**kwargs)
            elif tool_id == 'add_relationship':
                return self._execute_add_relationship(**kwargs)
            elif tool_id == 'get_related_entities':
                return self._execute_get_related_entities(**kwargs)
            elif tool_id == 'get_memory_stats':
                return self._execute_get_memory_stats(**kwargs)
            elif tool_id == 'cleanup_expired_memory':
                return self._execute_cleanup_expired_memory(**kwargs)
            elif tool_id == 'export_memory_snapshot':
                return self._execute_export_memory_snapshot(**kwargs)
            elif tool_id == 'suggest_memory_actions':
                return self._execute_suggest_memory_actions(**kwargs)
            else:
                return {'error': f'Unknown tool: {tool_id}', 'success': False}
                
        except Exception as e:
            return {'error': f'Tool execution error: {e}', 'success': False}
    
    def _execute_import_data(self, **kwargs) -> Dict[str, Any]:
        """Execute data import tool."""
        try:
            data_type = kwargs.get('data_type')
            data = kwargs.get('data')
            source = kwargs.get('source')
            description = kwargs.get('description')
            tags = kwargs.get('tags', [])
            ttl_days = kwargs.get('ttl_days')
            priority = kwargs.get('priority', 5)
            
            if not all([data_type, data, source]):
                return {'error': 'data_type, data, and source are required', 'success': False}
            
            # Get import pattern
            if data_type not in self.import_patterns:
                return {'error': f'Unknown data type: {data_type}', 'success': False}
            
            pattern = self.import_patterns[data_type]
            
            # Use provided values or defaults
            final_description = description or pattern['description_template'].format(source=source)
            final_tags = pattern['tags'] + (tags or [])
            final_ttl = ttl_days or pattern['ttl_days']
            
            # Import data
            memory_id = self.memory_manager.import_data(
                domain=pattern['domain'],
                data=data,
                source=source,
                tier=pattern['tier'],
                ttl_days=final_ttl,
                tags=final_tags,
                description=final_description,
                priority=priority
            )
            
            return {
                'tool': 'import_data',
                'success': True,
                'memory_id': memory_id,
                'domain': pattern['domain'].value,
                'tier': pattern['tier'].value,
                'ttl_days': final_ttl,
                'tags': final_tags,
                'description': final_description,
                'priority': priority,
                'data_type': data_type,
                'source': source
            }
            
        except Exception as e:
            return {'error': f'Import data error: {e}', 'success': False}
    
    def _execute_retrieve_context(self, **kwargs) -> Dict[str, Any]:
        """Execute context retrieval tool."""
        try:
            query = kwargs.get('query')
            domains = kwargs.get('domains')
            tiers = kwargs.get('tiers')
            max_results = kwargs.get('max_results', 10)
            include_relationships = kwargs.get('include_relationships', False)
            
            if not query:
                return {'error': 'query is required', 'success': False}
            
            # Convert domain/tier strings to enums if provided
            if domains:
                try:
                    domain_enums = [MemoryDomain(domain) for domain in domains]
                except ValueError as e:
                    return {'error': f'Invalid domain: {e}', 'success': False}
            else:
                domain_enums = None
            
            if tiers:
                try:
                    tier_enums = [MemoryTier(tier) for tier in tiers]
                except ValueError as e:
                    return {'error': f'Invalid tier: {e}', 'success': False}
            else:
                tier_enums = None
            
            # Retrieve context
            results = self.memory_manager.retrieve_context(
                query=query,
                domains=domain_enums,
                tiers=tier_enums,
                max_results=max_results
            )
            
            # Process results
            processed_results = []
            for entry in results:
                result_data = {
                    'id': entry.id,
                    'domain': entry.metadata.domain.value,
                    'tier': entry.metadata.tier.value,
                    'description': entry.metadata.description,
                    'source': entry.metadata.source,
                    'tags': entry.metadata.tags,
                    'priority': entry.metadata.priority,
                    'ttl_days': entry.metadata.ttl_days,
                    'last_accessed': entry.metadata.last_accessed.isoformat(),
                    'access_count': entry.metadata.access_count,
                    'data_type': type(entry.data).__name__,
                    'data_size': len(str(entry.data)) if entry.data else 0
                }
                
                # Include relationships if requested
                if include_relationships:
                    related = self.memory_manager.get_related_entities(str(entry.id))
                    result_data['related_entities'] = related
                
                processed_results.append(result_data)
            
            return {
                'tool': 'retrieve_context',
                'success': True,
                'query': query,
                'results_count': len(processed_results),
                'results': processed_results,
                'domains_searched': [d.value for d in (domain_enums or [])],
                'tiers_searched': [t.value for t in (tier_enums or [])]
            }
            
        except Exception as e:
            return {'error': f'Retrieve context error: {e}', 'success': False}
    
    def _execute_add_relationship(self, **kwargs) -> Dict[str, Any]:
        """Execute add relationship tool."""
        try:
            source_entity = kwargs.get('source_entity')
            target_entity = kwargs.get('target_entity')
            relationship_type = kwargs.get('relationship_type')
            strength = kwargs.get('strength', 1.0)
            metadata = kwargs.get('metadata', {})
            
            if not all([source_entity, target_entity, relationship_type]):
                return {'error': 'source_entity, target_entity, and relationship_type are required', 'success': False}
            
            # Validate strength
            if not 0.0 <= strength <= 1.0:
                return {'error': 'strength must be between 0.0 and 1.0', 'success': False}
            
            # Add relationship
            self.memory_manager.add_relationship(
                source_entity=source_entity,
                target_entity=target_entity,
                relationship_type=relationship_type,
                strength=strength,
                metadata=metadata
            )
            
            return {
                'tool': 'add_relationship',
                'success': True,
                'source_entity': source_entity,
                'target_entity': target_entity,
                'relationship_type': relationship_type,
                'strength': strength,
                'metadata': metadata
            }
            
        except Exception as e:
            return {'error': f'Add relationship error: {e}', 'success': False}
    
    def _execute_get_related_entities(self, **kwargs) -> Dict[str, Any]:
        """Execute get related entities tool."""
        try:
            entity = kwargs.get('entity')
            max_depth = kwargs.get('max_depth', 2)
            relationship_types = kwargs.get('relationship_types')
            
            if not entity:
                return {'error': 'entity is required', 'success': False}
            
            # Get related entities
            related = self.memory_manager.get_related_entities(
                entity=entity,
                max_depth=max_depth,
                relationship_types=relationship_types
            )
            
            return {
                'tool': 'get_related_entities',
                'success': True,
                'entity': entity,
                'max_depth': max_depth,
                'relationship_types': relationship_types,
                'related_count': len(related),
                'related_entities': related
            }
            
        except Exception as e:
            return {'error': f'Get related entities error: {e}', 'success': False}
    
    def _execute_get_memory_stats(self, **kwargs) -> Dict[str, Any]:
        """Execute get memory stats tool."""
        try:
            include_performance = kwargs.get('include_performance', True)
            include_relationships = kwargs.get('include_relationships', True)
            
            # Get memory statistics
            stats = self.memory_manager.get_memory_stats()
            
            # Filter based on parameters
            if not include_performance:
                stats.pop('performance_stats', None)
            
            if not include_relationships:
                stats.pop('relationship_stats', None)
            
            return {
                'tool': 'get_memory_stats',
                'success': True,
                'stats': stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': f'Get memory stats error: {e}', 'success': False}
    
    def _execute_cleanup_expired_memory(self, **kwargs) -> Dict[str, Any]:
        """Execute cleanup expired memory tool."""
        try:
            dry_run = kwargs.get('dry_run', False)
            force = kwargs.get('force', False)
            
            if dry_run:
                # For dry run, we'd need to implement a method that shows what would be cleaned
                # For now, just return current stats
                stats = self.memory_manager.get_memory_stats()
                return {
                    'tool': 'cleanup_expired_memory',
                    'success': True,
                    'dry_run': True,
                    'message': 'Dry run mode - no cleanup performed',
                    'current_stats': stats
                }
            
            # Perform actual cleanup
            expired_count = self.memory_manager.cleanup_expired_memory()
            
            return {
                'tool': 'cleanup_expired_memory',
                'success': True,
                'dry_run': False,
                'expired_entries_removed': expired_count,
                'message': f'Cleaned up {expired_count} expired entries'
            }
            
        except Exception as e:
            return {'error': f'Cleanup expired memory error: {e}', 'success': False}
    
    def _execute_export_memory_snapshot(self, **kwargs) -> Dict[str, Any]:
        """Execute export memory snapshot tool."""
        try:
            include_data = kwargs.get('include_data', True)
            format_type = kwargs.get('format', 'json')
            compression = kwargs.get('compression', True)
            
            # Export memory snapshot
            snapshot_path = self.memory_manager.export_memory_snapshot()
            
            return {
                'tool': 'export_memory_snapshot',
                'success': True,
                'snapshot_path': snapshot_path,
                'include_data': include_data,
                'format': format_type,
                'compression': compression,
                'message': f'Memory snapshot exported to {snapshot_path}'
            }
            
        except Exception as e:
            return {'error': f'Export memory snapshot error: {e}', 'success': False}
    
    def _execute_suggest_memory_actions(self, **kwargs) -> Dict[str, Any]:
        """Execute suggest memory actions tool."""
        try:
            workflow_data = kwargs.get('workflow_data')
            context = kwargs.get('context', '')
            
            if not workflow_data:
                return {'error': 'workflow_data is required', 'success': False}
            
            suggestions = []
            
            # Analyze workflow data for memory suggestions
            if isinstance(workflow_data, dict):
                # Check for data that should be imported
                if 'hosts' in workflow_data or 'host_list' in workflow_data:
                    suggestions.append({
                        'action': 'import_data',
                        'type': 'hosts',
                        'reasoning': 'Workflow contains host data that should be stored in long-term memory',
                        'priority': 'high',
                        'ttl_days': 365
                    })
                
                if 'applications' in workflow_data or 'app_list' in workflow_data:
                    suggestions.append({
                        'action': 'import_data',
                        'type': 'applications',
                        'reasoning': 'Workflow contains application data that should be stored in long-term memory',
                        'priority': 'high',
                        'ttl_days': 365
                    })
                
                if 'iocs' in workflow_data or 'threat_indicators' in workflow_data:
                    suggestions.append({
                        'action': 'import_data',
                        'type': 'iocs',
                        'reasoning': 'Workflow contains IOC data that should be stored in short-term memory for hunting',
                        'priority': 'medium',
                        'ttl_days': 7
                    })
                
                if 'mitre_techniques' in workflow_data or 'attack_patterns' in workflow_data:
                    suggestions.append({
                        'action': 'import_data',
                        'type': 'mitre_attack',
                        'reasoning': 'Workflow contains MITRE ATT&CK data that should be stored in long-term memory',
                        'priority': 'high',
                        'ttl_days': 365
                    })
                
                # Check for relationship opportunities
                if 'entities' in workflow_data and 'relationships' in workflow_data:
                    suggestions.append({
                        'action': 'add_relationships',
                        'reasoning': 'Workflow contains entity relationship data that should be stored',
                        'priority': 'medium'
                    })
            
            # Add context-specific suggestions
            if 'investigation' in context.lower():
                suggestions.append({
                    'action': 'retrieve_context',
                    'reasoning': 'Investigation context suggests retrieving relevant historical data',
                    'priority': 'high'
                })
            
            if 'threat_hunting' in context.lower():
                suggestions.append({
                    'action': 'retrieve_context',
                    'reasoning': 'Threat hunting context suggests retrieving IOC and threat actor data',
                    'priority': 'high'
                })
            
            return {
                'tool': 'suggest_memory_actions',
                'success': True,
                'workflow_context': context,
                'suggestions_count': len(suggestions),
                'suggestions': suggestions,
                'message': f'Generated {len(suggestions)} memory action suggestions'
            }
            
        except Exception as e:
            return {'error': f'Suggest memory actions error: {e}', 'success': False}

# Example usage and testing
if __name__ == "__main__":
    # Initialize memory tools
    memory_tools = MemoryMCPTools()
    
    print("\n🧪 Testing Memory MCP Tools")
    print("=" * 50)
    
    # Test tool availability
    tools = memory_tools.get_available_tools()
    print(f"📋 Available tools: {len(tools)}")
    for tool_id, tool_info in tools.items():
        print(f"   • {tool_id}: {tool_info['name']}")
    
    # Test memory statistics
    print("\n📊 Testing memory statistics...")
    stats_result = memory_tools.execute_tool('get_memory_stats')
    if stats_result['success']:
        print(f"✅ Memory stats retrieved successfully")
        stats = stats_result['stats']
        print(f"   Total entries: {stats.get('total_entries', 0)}")
        print(f"   Total size: {stats.get('total_size_bytes', 0)} bytes")
    else:
        print(f"❌ Memory stats failed: {stats_result['error']}")
    
    # Test relationship creation
    print("\n🔗 Testing relationship creation...")
    rel_result = memory_tools.execute_tool('add_relationship',
                                         source_entity='test_host_1',
                                         target_entity='test_network_1',
                                         relationship_type='belongs_to',
                                         strength=0.8)
    if rel_result['success']:
        print(f"✅ Relationship created successfully")
    else:
        print(f"❌ Relationship creation failed: {rel_result['error']}")
    
    print("\n✅ Memory MCP Tools testing completed!")
