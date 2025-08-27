#!/usr/bin/env python3
"""
Memory Workflow CLI
Interactive CLI for managing context memory through natural language commands.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from bin.context_memory_manager import (
    ContextMemoryManager, MemoryDomain, MemoryTier, DataType
)

class MemoryWorkflowCLI:
    """Interactive CLI for memory management workflows."""
    
    def __init__(self):
        self.memory_manager = ContextMemoryManager()
        self.session_id = None
        
        # Common import patterns
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
    
    def start_interactive_session(self):
        """Start interactive memory management session."""
        print("\n🧠 Context Memory Management Workflow")
        print("=" * 60)
        print("Welcome to the Context Memory Management System!")
        print("I can help you import, manage, and query cybersecurity data.")
        print("\nAvailable commands:")
        print("  • import <type> <source> - Import data into memory")
        print("  • query <search> - Search memory for relevant context")
        print("  • stats - Show memory statistics")
        print("  • relationships <entity> - Show entity relationships")
        print("  • cleanup - Clean up expired memory")
        print("  • export - Export memory snapshot")
        print("  • help - Show this help message")
        print("  • quit - Exit the session")
        print("\nExamples:")
        print("  • import hosts /path/to/hosts.csv")
        print("  • import iocs /path/to/iocs.txt")
        print("  • query 'threat actor APT29'")
        print("  • relationships host1.example.com")
        print("-" * 60)
        
        while True:
            try:
                command = input("\n🔒 Memory Manager: ").strip()
                
                if not command:
                    continue
                
                if command.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! Memory session ended.")
                    break
                
                if command.lower() == 'help':
                    self._show_help()
                    continue
                
                if command.lower() == 'stats':
                    self._show_memory_stats()
                    continue
                
                if command.lower() == 'cleanup':
                    self._cleanup_memory()
                    continue
                
                if command.lower() == 'export':
                    self._export_memory()
                    continue
                
                if command.startswith('import '):
                    self._handle_import(command[7:])
                    continue
                
                if command.startswith('query '):
                    self._handle_query(command[6:])
                    continue
                
                if command.startswith('relationships '):
                    self._handle_relationships(command[13:])
                    continue
                
                print(f"❓ Unknown command: {command}")
                print("   Type 'help' for available commands")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Memory session ended.")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _show_help(self):
        """Show detailed help information."""
        print("\n📚 Memory Management Help")
        print("=" * 40)
        
        print("\n🔧 Import Commands:")
        print("  import <type> <source>")
        print("  Types: hosts, applications, users, networks, iocs, threat_actors,")
        print("         investigation, splunk_schemas, mitre_attack, mitre_d3fend,")
        print("         nist, grc_policies")
        print("  Examples:")
        print("    import hosts /path/to/hosts.csv")
        print("    import iocs /path/to/iocs.txt")
        print("    import mitre_attack /path/to/attack.json")
        
        print("\n🔍 Query Commands:")
        print("  query <search_terms>")
        print("  Examples:")
        print("    query 'threat actor APT29'")
        print("    query 'host inventory'")
        print("    query 'splunk schema'")
        
        print("\n🔗 Relationship Commands:")
        print("  relationships <entity>")
        print("  Examples:")
        print("    relationships host1.example.com")
        print("    relationships 192.168.1.1")
        
        print("\n📊 Management Commands:")
        print("  stats - Show memory statistics")
        print("  cleanup - Clean up expired memory")
        print("  export - Export memory snapshot")
        print("  help - Show this help")
        print("  quit - Exit session")
    
    def _handle_import(self, import_command: str):
        """Handle import commands."""
        try:
            parts = import_command.strip().split()
            if len(parts) < 2:
                print("❌ Import command requires type and source")
                print("   Example: import hosts /path/to/file.csv")
                return
            
            import_type = parts[0].lower()
            source_path = parts[1]
            
            if import_type not in self.import_patterns:
                print(f"❌ Unknown import type: {import_type}")
                print(f"   Available types: {', '.join(self.import_patterns.keys())}")
                return
            
            # Load and process data
            data = self._load_data_from_source(source_path)
            if data is None:
                return
            
            # Get import pattern
            pattern = self.import_patterns[import_type]
            
            # Ask for additional details
            description = self._prompt_for_description(pattern['description_template'], source_path)
            tags = self._prompt_for_tags(pattern['tags'])
            ttl_days = self._prompt_for_ttl(pattern['ttl_days'])
            priority = self._prompt_for_priority()
            
            # Import data
            memory_id = self.memory_manager.import_data(
                domain=pattern['domain'],
                data=data,
                source=source_path,
                tier=pattern['tier'],
                ttl_days=ttl_days,
                tags=tags,
                description=description,
                priority=priority
            )
            
            print(f"✅ Successfully imported {import_type} data")
            print(f"   Memory ID: {memory_id}")
            print(f"   Domain: {pattern['domain'].value}")
            print(f"   Tier: {pattern['tier'].value}")
            print(f"   TTL: {ttl_days} days")
            
        except Exception as e:
            print(f"❌ Import failed: {e}")
    
    def _load_data_from_source(self, source_path: str) -> Any:
        """Load data from various source formats."""
        try:
            path = Path(source_path)
            
            if not path.exists():
                print(f"❌ Source file not found: {source_path}")
                return None
            
            # Determine file type and load accordingly
            if path.suffix.lower() == '.csv':
                return pd.read_csv(path)
            elif path.suffix.lower() == '.json':
                with open(path, 'r') as f:
                    return json.load(f)
            elif path.suffix.lower() == '.txt':
                with open(path, 'r') as f:
                    return f.read()
            elif path.suffix.lower() in ['.xml', '.yaml', '.yml']:
                # For now, read as text - could add proper parsers
                with open(path, 'r') as f:
                    return f.read()
            else:
                # Try to read as text
                try:
                    with open(path, 'r') as f:
                        return f.read()
                except:
                    print(f"❌ Unsupported file type: {path.suffix}")
                    return None
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def _prompt_for_description(self, template: str, source: str) -> str:
        """Prompt user for description."""
        default_description = template.format(source=source)
        print(f"📝 Description (default: {default_description}):")
        description = input("   ").strip()
        return description if description else default_description
    
    def _prompt_for_tags(self, default_tags: List[str]) -> List[str]:
        """Prompt user for additional tags."""
        print(f"🏷️  Tags (default: {', '.join(default_tags)}):")
        print("   Enter additional tags separated by commas (or press Enter for defaults):")
        additional_tags = input("   ").strip()
        
        if additional_tags:
            additional_list = [tag.strip() for tag in additional_tags.split(',')]
            return default_tags + additional_list
        else:
            return default_tags
    
    def _prompt_for_ttl(self, default_ttl: int) -> int:
        """Prompt user for TTL in days."""
        print(f"⏰ TTL in days (default: {default_ttl}):")
        ttl_input = input("   ").strip()
        
        try:
            if ttl_input:
                return int(ttl_input)
            else:
                return default_ttl
        except ValueError:
            print(f"   Invalid TTL, using default: {default_ttl}")
            return default_ttl
    
    def _prompt_for_priority(self) -> int:
        """Prompt user for priority (1-10)."""
        print("⭐ Priority (1-10, default: 5):")
        priority_input = input("   ").strip()
        
        try:
            if priority_input:
                priority = int(priority_input)
                if 1 <= priority <= 10:
                    return priority
                else:
                    print("   Priority must be 1-10, using default: 5")
                    return 5
            else:
                return 5
        except ValueError:
            print("   Invalid priority, using default: 5")
            return 5
    
    def _handle_query(self, query: str):
        """Handle query commands."""
        try:
            print(f"🔍 Searching memory for: {query}")
            
            # Ask for search scope
            domains = self._prompt_for_domains()
            tiers = self._prompt_for_tiers()
            max_results = self._prompt_for_max_results()
            
            # Search memory
            results = self.memory_manager.retrieve_context(
                query=query,
                domains=domains,
                tiers=tiers,
                max_results=max_results
            )
            
            if not results:
                print("❌ No relevant results found")
                return
            
            print(f"\n✅ Found {len(results)} relevant memory entries:")
            print("-" * 80)
            
            for i, entry in enumerate(results, 1):
                print(f"{i}. {entry.metadata.domain.value.upper()}")
                print(f"   Description: {entry.metadata.description}")
                print(f"   Source: {entry.metadata.source}")
                print(f"   Tier: {entry.metadata.tier.value}")
                print(f"   TTL: {entry.metadata.ttl_days} days")
                print(f"   Priority: {entry.metadata.priority}")
                print(f"   Tags: {', '.join(entry.metadata.tags)}")
                print(f"   Last accessed: {entry.metadata.last_accessed.strftime('%Y-%m-%d %H:%M')}")
                print(f"   Access count: {entry.metadata.access_count}")
                
                # Show data preview
                if isinstance(entry.data, pd.DataFrame):
                    print(f"   Data: DataFrame with {len(entry.data)} rows, {len(entry.data.columns)} columns")
                    print(f"   Preview: {entry.data.head(2).to_string()}")
                elif isinstance(entry.data, dict):
                    print(f"   Data: Dictionary with {len(entry.data)} keys")
                    print(f"   Preview: {dict(list(entry.data.items())[:3])}")
                elif isinstance(entry.data, str):
                    preview = entry.data[:200] + "..." if len(entry.data) > 200 else entry.data
                    print(f"   Data: Text ({len(entry.data)} characters)")
                    print(f"   Preview: {preview}")
                else:
                    print(f"   Data: {type(entry.data).__name__}")
                
                print()
            
            # Ask if user wants to see relationships
            if results:
                self._prompt_for_relationships(results)
                
        except Exception as e:
            print(f"❌ Query failed: {e}")
    
    def _prompt_for_domains(self) -> List[MemoryDomain]:
        """Prompt user for search domains."""
        print("🔍 Search domains (press Enter for all):")
        print("   Available domains:")
        for i, domain in enumerate(MemoryDomain, 1):
            print(f"   {i}. {domain.value}")
        
        domain_input = input("   Enter domain numbers separated by commas: ").strip()
        
        if not domain_input:
            return list(MemoryDomain)
        
        try:
            domain_numbers = [int(x.strip()) - 1 for x in domain_input.split(',')]
            domains = [list(MemoryDomain)[i] for i in domain_numbers if 0 <= i < len(MemoryDomain)]
            return domains if domains else list(MemoryDomain)
        except ValueError:
            print("   Invalid domain selection, searching all domains")
            return list(MemoryDomain)
    
    def _prompt_for_tiers(self) -> List[MemoryTier]:
        """Prompt user for search tiers."""
        print("🔍 Search tiers (press Enter for all):")
        print("   1. Short-term (1-7 days)")
        print("   2. Medium-term (7-30 days)")
        print("   3. Long-term (30+ days)")
        
        tier_input = input("   Enter tier numbers separated by commas: ").strip()
        
        if not tier_input:
            return list(MemoryTier)
        
        try:
            tier_numbers = [int(x.strip()) - 1 for x in tier_input.split(',')]
            tiers = [list(MemoryTier)[i] for i in tier_numbers if 0 <= i < len(MemoryTier)]
            return tiers if tiers else list(MemoryTier)
        except ValueError:
            print("   Invalid tier selection, searching all tiers")
            return list(MemoryTier)
    
    def _prompt_for_max_results(self) -> int:
        """Prompt user for maximum results."""
        print("🔍 Maximum results (default: 10):")
        max_input = input("   ").strip()
        
        try:
            if max_input:
                return int(max_input)
            else:
                return 10
        except ValueError:
            print("   Invalid number, using default: 10")
            return 10
    
    def _prompt_for_relationships(self, results: List):
        """Prompt user to explore relationships."""
        print("🔗 Explore relationships? (y/n):")
        explore = input("   ").strip().lower()
        
        if explore in ['y', 'yes']:
            print("   Enter entity name to explore relationships:")
            entity = input("   ").strip()
            
            if entity:
                related = self.memory_manager.get_related_entities(entity)
                if related:
                    print(f"   Related entities for '{entity}':")
                    for related_entity in related[:10]:  # Limit to 10
                        print(f"     • {related_entity}")
                else:
                    print(f"   No relationships found for '{entity}'")
    
    def _handle_relationships(self, entity: str):
        """Handle relationship queries."""
        try:
            print(f"🔗 Exploring relationships for: {entity}")
            
            related = self.memory_manager.get_related_entities(entity, max_depth=2)
            
            if not related:
                print(f"❌ No relationships found for '{entity}'")
                return
            
            print(f"\n✅ Found {len(related)} related entities:")
            print("-" * 60)
            
            for i, related_entity in enumerate(related, 1):
                print(f"{i}. {related_entity}")
            
            # Ask for deeper exploration
            print(f"\n🔍 Explore deeper relationships? (y/n):")
            explore_deeper = input("   ").strip().lower()
            
            if explore_deeper in ['y', 'yes']:
                print("   Enter entity name for deeper exploration:")
                deeper_entity = input("   ").strip()
                
                if deeper_entity:
                    deeper_related = self.memory_manager.get_related_entities(deeper_entity, max_depth=3)
                    if deeper_related:
                        print(f"   Deeper relationships for '{deeper_entity}':")
                        for related_entity in deeper_related[:15]:  # Limit to 15
                            print(f"     • {related_entity}")
                    else:
                        print(f"   No deeper relationships found for '{deeper_entity}'")
                        
        except Exception as e:
            print(f"❌ Relationship query failed: {e}")
    
    def _show_memory_stats(self):
        """Show comprehensive memory statistics."""
        try:
            stats = self.memory_manager.get_memory_stats()
            
            print("\n📊 Memory Statistics")
            print("=" * 50)
            
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
            
            print(f"\n⚡ Performance:")
            perf_stats = stats.get('performance_stats', {})
            print(f"   Cache hits: {perf_stats.get('cache_hits', 0):,}")
            print(f"   Cache misses: {perf_stats.get('cache_misses', 0):,}")
            print(f"   Compression savings: {perf_stats.get('compression_savings', 0):,} bytes")
            
        except Exception as e:
            print(f"❌ Error getting memory stats: {e}")
    
    def _cleanup_memory(self):
        """Clean up expired memory."""
        try:
            print("🧹 Cleaning up expired memory...")
            
            expired_count = self.memory_manager.cleanup_expired_memory()
            
            if expired_count > 0:
                print(f"✅ Cleaned up {expired_count} expired entries")
            else:
                print("✅ No expired entries to clean up")
                
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
    
    def _export_memory(self):
        """Export memory snapshot."""
        try:
            print("💾 Exporting memory snapshot...")
            
            snapshot_path = self.memory_manager.export_memory_snapshot()
            
            print(f"✅ Memory snapshot exported to: {snapshot_path}")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Context Memory Management CLI")
    parser.add_argument('--import', dest='import_cmd', metavar='TYPE:SOURCE',
                       help='Import data directly (e.g., hosts:/path/to/file.csv)')
    parser.add_argument('--query', metavar='SEARCH_TERMS',
                       help='Search memory for specific terms')
    parser.add_argument('--stats', action='store_true',
                       help='Show memory statistics')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up expired memory')
    parser.add_argument('--export', action='store_true',
                       help='Export memory snapshot')
    
    args = parser.parse_args()
    
    cli = MemoryWorkflowCLI()
    
    # Handle direct commands
    if args.import_cmd:
        # Parse import command format: TYPE:SOURCE
        import_parts = args.import_cmd.split(':', 1)
        if len(import_parts) == 2:
            data_type = import_parts[0]
            source = import_parts[1]
            
            # Create sample data for CLI import
            sample_data = {
                'description': f'Sample {data_type} data from CLI',
                'timestamp': datetime.now().isoformat(),
                'source': source,
                'sample': True
            }
            
            # Import using the pattern
            if data_type in cli.import_patterns:
                pattern = cli.import_patterns[data_type]
                memory_id = cli.memory_manager.import_data(
                    domain=pattern['domain'],
                    data=sample_data,
                    source=source,
                    tier=pattern['tier'],
                    ttl_days=pattern['ttl_days'],
                    tags=pattern['tags'] + ['cli_import'],
                    description=pattern['description_template'].format(source=source),
                    priority=5
                )
                print(f"✅ Successfully imported {data_type} data with ID: {memory_id}")
            else:
                print(f"❌ Unknown data type: {data_type}")
                print(f"   Available types: {', '.join(cli.import_patterns.keys())}")
        else:
            print("❌ Invalid import format. Use: TYPE:SOURCE (e.g., hosts:test_data)")
    elif args.query:
        cli._handle_query(args.query)
    elif args.stats:
        cli._show_memory_stats()
    elif args.cleanup:
        cli._cleanup_memory()
    elif args.export:
        cli._export_memory()
    else:
        # Start interactive session
        cli.start_interactive_session()

if __name__ == "__main__":
    main()
