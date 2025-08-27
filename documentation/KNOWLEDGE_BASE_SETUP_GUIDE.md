# 🏢 Organizational Knowledge Base Setup Guide

## Overview

This guide provides comprehensive instructions for setting up a Knowledge Base Contextual Memory system using organizational data extracted from spreadsheets and JSON files. The system is designed to support LangChain agentic workflows for complex tasks like converting Content Catalogs from Google Chronicle to Splunk ES on Splunk Cloud SPLs.

## 📊 **Data Sources Overview**

### **Governance, Risk, & Compliance (GRC) Platform**
- **Policy Domains (CSV)** - Security domain sub-domains
- **Policy Groups (CSV)** - Policy requirements & technology solutions grouping
- **Policy Requirements (CSV)** - Individual policy requirements (e.g., encryption at rest)
- **Technology Solutions (CSV)** - Solutions tied to policy domains, groups, and requirements

### **Organizational Data**
- **Organizational & Technical Terms (CSV)** - Institutional technical terms, names, acronyms, definitions

### **Enterprise Central Logging**
- **Splunk ES On-Prem** - Indexes, sourcetypes, same_messages, last_seen, security monitoring content
- **Splunk ES on Splunk Cloud** - Same as on-prem plus GRC technology solution cross-referencing
- **Google Chronicle SIEM** - Security monitoring content (active and scheduled)
- **Data Models** - Setup, fields, queries for both Splunk environments

### **Configuration Management Database (CMDB)**
- **Business Applications** - Associated devices, IPs, networks, databases, controls (JSON)
- **Devices** - Business applications, IPs, listening ports, privileged accounts, controls (JSON)
- **IP Addresses** - Business applications and associated devices (JSON)
- **Domains** - IPs, business applications, hosting devices (JSON)
- **Users** - Enterprise account IDs, privileged accounts (JSON)

## 🚀 **Quick Start: 80/20 Approach**

### **Phase 1: Data Assessment & Transformation (Day 1)**

#### **1.1 Data Quality Assessment**
```python
#!/usr/bin/env python3
"""
Data Quality Assessment Script
Quick analysis of your CSV and JSON files
"""

import pandas as pd
import json
from pathlib import Path
import sys

def assess_data_quality():
    """Assess the quality and structure of your data files."""
    
    # Define expected file structure
    expected_files = {
        'grc': [
            'policy_domains.csv',
            'policy_groups.csv', 
            'policy_requirements.csv',
            'technology_solutions.csv'
        ],
        'organizational': [
            'organizational_terms.csv'
        ],
        'logging': [
            'splunk_onprem.csv',
            'splunk_cloud.csv',
            'google_chronicle.csv'
        ],
        'cmdb': [
            'business_applications.json',
            'devices.json',
            'ip_addresses.json',
            'domains.json',
            'users.json'
        ]
    }
    
    print("🔍 Data Quality Assessment")
    print("=" * 50)
    
    for category, files in expected_files.items():
        print(f"\n📁 {category.upper()}:")
        for file in files:
            file_path = Path(f"data/{file}")
            if file_path.exists():
                if file.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    print(f"  ✅ {file}: {len(df)} rows, {len(df.columns)} columns")
                    print(f"     Columns: {', '.join(df.columns)}")
                elif file.endswith('.json'):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        print(f"  ✅ {file}: {len(data)} items")
                    else:
                        print(f"  ✅ {file}: {len(data)} keys")
            else:
                print(f"  ❌ {file}: Not found")
    
    print("\n📊 Data Transformation Recommendations:")
    print("  • Standardize column names across CSV files")
    print("  • Ensure consistent data types")
    print("  • Add unique identifiers for relationships")
    print("  • Normalize categorical values")

if __name__ == "__main__":
    assess_data_quality()
```

#### **1.2 Data Transformation Options**

**Option A: Transform Data to Standard Format (Recommended for PoC)**
```python
#!/usr/bin/env python3
"""
Data Transformation Script
Convert your CSV/JSON files to standardized format
"""

import pandas as pd
import json
from pathlib import Path
import hashlib
from typing import Dict, List, Any

class DataTransformer:
    def __init__(self, input_dir: str = "data", output_dir: str = "transformed_data"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def transform_grc_data(self):
        """Transform GRC CSV files to standardized format."""
        print("🔄 Transforming GRC Data...")
        
        # Transform Policy Domains
        if (self.input_dir / "policy_domains.csv").exists():
            df = pd.read_csv(self.input_dir / "policy_domains.csv")
            df['entity_type'] = 'policy_domain'
            df['entity_id'] = df['name'].apply(lambda x: f"domain_{hashlib.md5(x.encode()).hexdigest()[:8]}")
            df['tags'] = df['name'].apply(lambda x: f"policy,domain,{x.lower().replace(' ', '_')}")
            df.to_csv(self.output_dir / "transformed_policy_domains.csv", index=False)
            print(f"  ✅ Policy Domains: {len(df)} transformed")
        
        # Transform Policy Groups
        if (self.input_dir / "policy_groups.csv").exists():
            df = pd.read_csv(self.input_dir / "policy_groups.csv")
            df['entity_type'] = 'policy_group'
            df['entity_id'] = df['name'].apply(lambda x: f"group_{hashlib.md5(x.encode()).hexdigest()[:8]}")
            df['tags'] = df.apply(lambda row: f"policy,group,{row.get('policy_domain', '').lower().replace(' ', '_')}", axis=1)
            df.to_csv(self.output_dir / "transformed_policy_groups.csv", index=False)
            print(f"  ✅ Policy Groups: {len(df)} transformed")
        
        # Transform Policy Requirements
        if (self.input_dir / "policy_requirements.csv").exists():
            df = pd.read_csv(self.input_dir / "policy_requirements.csv")
            df['entity_type'] = 'policy_requirement'
            df['entity_id'] = df['name'].apply(lambda x: f"req_{hashlib.md5(x.encode()).hexdigest()[:8]}")
            df['tags'] = df.apply(lambda row: f"policy,requirement,{row.get('policy_group', '').lower().replace(' ', '_')}", axis=1)
            df.to_csv(self.output_dir / "transformed_policy_requirements.csv", index=False)
            print(f"  ✅ Policy Requirements: {len(df)} transformed")
        
        # Transform Technology Solutions
        if (self.input_dir / "technology_solutions.csv").exists():
            df = pd.read_csv(self.input_dir / "technology_solutions.csv")
            df['entity_type'] = 'technology_solution'
            df['entity_id'] = df['name'].apply(lambda x: f"tech_{hashlib.md5(x.encode()).hexdigest()[:8]}")
            df['tags'] = df.apply(lambda row: f"technology,solution,{row.get('policy_domain', '').lower().replace(' ', '_')}", axis=1)
            df.to_csv(self.output_dir / "transformed_technology_solutions.csv", index=False)
            print(f"  ✅ Technology Solutions: {len(df)} transformed")
    
    def transform_organizational_data(self):
        """Transform organizational terms CSV."""
        print("🏢 Transforming Organizational Data...")
        
        if (self.input_dir / "organizational_terms.csv").exists():
            df = pd.read_csv(self.input_dir / "organizational_terms.csv")
            df['entity_type'] = 'organizational_term'
            df['entity_id'] = df['term'].apply(lambda x: f"term_{hashlib.md5(x.encode()).hexdigest()[:8]}")
            df['tags'] = df.apply(lambda row: f"organization,term,{row.get('category', '').lower().replace(' ', '_')}", axis=1)
            df.to_csv(self.output_dir / "transformed_organizational_terms.csv", index=False)
            print(f"  ✅ Organizational Terms: {len(df)} transformed")
    
    def transform_logging_data(self):
        """Transform logging platform CSV files."""
        print("📊 Transforming Logging Data...")
        
        platforms = ['splunk_onprem', 'splunk_cloud', 'google_chronicle']
        
        for platform in platforms:
            file_path = self.input_dir / f"{platform}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['entity_type'] = 'logging_component'
                df['platform'] = platform
                df['entity_id'] = df.apply(lambda row: f"{platform}_{hashlib.md5(str(row).encode()).hexdigest()[:8]}", axis=1)
                df['tags'] = df.apply(lambda row: f"logging,{platform},{row.get('type', 'component')}", axis=1)
                df.to_csv(self.output_dir / f"transformed_{platform}.csv", index=False)
                print(f"  ✅ {platform}: {len(df)} transformed")
    
    def transform_cmdb_data(self):
        """Transform CMDB JSON files."""
        print("🏗️ Transforming CMDB Data...")
        
        cmdb_files = ['business_applications.json', 'devices.json', 'ip_addresses.json', 'domains.json', 'users.json']
        
        for file in cmdb_files:
            file_path = self.input_dir / file
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                    df['entity_type'] = 'cmdb_entity'
                    df['data_source'] = file.replace('.json', '')
                    df['entity_id'] = df.apply(lambda row: f"cmdb_{file.replace('.json', '')}_{hashlib.md5(str(row).encode()).hexdigest()[:8]}", axis=1)
                    df['tags'] = f"cmdb,{file.replace('.json', '')}"
                    df.to_csv(self.output_dir / f"transformed_{file.replace('.json', '')}.csv", index=False)
                    print(f"  ✅ {file}: {len(df)} transformed")
                else:
                    # Handle nested JSON structures
                    flattened_data = self._flatten_json(data)
                    df = pd.DataFrame([flattened_data])
                    df['entity_type'] = 'cmdb_entity'
                    df['data_source'] = file.replace('.json', '')
                    df['entity_id'] = f"cmdb_{file.replace('.json', '')}_{hashlib.md5(str(data).encode()).hexdigest()[:8]}"
                    df['tags'] = f"cmdb,{file.replace('.json', '')}"
                    df.to_csv(self.output_dir / f"transformed_{file.replace('.json', '')}.csv", index=False)
                    print(f"  ✅ {file}: 1 transformed")
    
    def _flatten_json(self, data: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested JSON structure."""
        items = []
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_json(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(self._flatten_json(item, f"{new_key}_{i}", sep=sep).items())
                    else:
                        items.append((f"{new_key}_{i}", item))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def transform_all(self):
        """Transform all data files."""
        print("🚀 Starting Data Transformation...")
        self.transform_grc_data()
        self.transform_organizational_data()
        self.transform_logging_data()
        self.transform_cmdb_data()
        print("\n✅ All data transformation complete!")
        print(f"📁 Transformed files saved to: {self.output_dir}")

if __name__ == "__main__":
    transformer = DataTransformer()
    transformer.transform_all()
```

**Option B: Use cs_util_lg.py Utility (Minimal Transformation)**
```python
#!/usr/bin/env python3
"""
Direct Data Import Script
Use existing cs_util_lg.py utility with minimal data preparation
"""

import pandas as pd
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from bin.context_memory_manager import ContextMemoryManager
from bin.master_catalog import MasterCatalog

class DirectDataImporter:
    def __init__(self):
        self.memory_manager = ContextMemoryManager()
        self.master_catalog = MasterCatalog()
    
    def import_csv_directly(self, csv_file: str, entity_type: str, domain: str = "organizational"):
        """Import CSV file directly using existing utilities."""
        try:
            df = pd.read_csv(csv_file)
            
            # Create memory entries for each row
            for index, row in df.iterrows():
                # Create unique identifier
                entity_id = f"{entity_type}_{index}_{hashlib.md5(str(row).encode()).hexdigest()[:8]}"
                
                # Store in memory manager
                self.memory_manager.store_memory(
                    entity_id=entity_id,
                    domain=domain,
                    tier="long_term",
                    data_type=entity_type,
                    source=csv_file,
                    content=row.to_dict(),
                    tags=[entity_type, domain, "csv_import"]
                )
            
            print(f"✅ Imported {len(df)} {entity_type} entities from {csv_file}")
            
        except Exception as e:
            print(f"❌ Error importing {csv_file}: {e}")
    
    def import_json_directly(self, json_file: str, entity_type: str, domain: str = "organizational"):
        """Import JSON file directly using existing utilities."""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for index, item in enumerate(data):
                    entity_id = f"{entity_type}_{index}_{hashlib.md5(str(item).encode()).hexdigest()[:8]}"
                    
                    self.memory_manager.store_memory(
                        entity_id=entity_id,
                        domain=domain,
                        tier="long_term",
                        data_type=entity_type,
                        source=json_file,
                        content=item,
                        tags=[entity_type, domain, "json_import"]
                    )
            else:
                entity_id = f"{entity_type}_{hashlib.md5(str(data).encode()).hexdigest()[:8]}"
                
                self.memory_manager.store_memory(
                    entity_id=entity_id,
                    domain=domain,
                    tier="long_term",
                    data_type=entity_type,
                    source=json_file,
                    content=data,
                    tags=[entity_type, domain, "json_import"]
                )
            
            print(f"✅ Imported {entity_type} entities from {json_file}")
            
        except Exception as e:
            print(f"❌ Error importing {json_file}: {e}")
    
    def import_all_data(self, data_dir: str = "data"):
        """Import all data files in the data directory."""
        data_path = Path(data_dir)
        
        if not data_path.exists():
            print(f"❌ Data directory {data_dir} not found")
            return
        
        print("🚀 Starting Direct Data Import...")
        
        # Import CSV files
        for csv_file in data_path.glob("*.csv"):
            entity_type = csv_file.stem
            self.import_csv_directly(str(csv_file), entity_type)
        
        # Import JSON files
        for json_file in data_path.glob("*.json"):
            entity_type = json_file.stem
            self.import_json_directly(str(json_file), entity_type)
        
        print("✅ Direct data import complete!")

if __name__ == "__main__":
    importer = DirectDataImporter()
    importer.import_all_data()
```

## 🧠 **Medium/Long-Term Memory Setup**

### **Memory Tier Configuration**

```python
#!/usr/bin/env python3
"""
Memory Configuration Script
Set up medium and long-term memory tiers for organizational knowledge
"""

from bin.context_memory_manager import ContextMemoryManager
from bin.master_catalog import MasterCatalog
import json

class MemoryConfiguration:
    def __init__(self):
        self.memory_manager = ContextMemoryManager()
        self.master_catalog = MasterCatalog()
    
    def configure_memory_tiers(self):
        """Configure memory tiers for organizational knowledge."""
        print("🧠 Configuring Memory Tiers...")
        
        # Configure long-term memory for core organizational knowledge
        self.memory_manager.configure_tier(
            tier="long_term",
            ttl_days=365,  # 1 year retention
            compression_ratio=0.8,
            priority=1
        )
        
        # Configure medium-term memory for operational knowledge
        self.memory_manager.configure_tier(
            tier="medium_term", 
            ttl_days=90,   # 3 months retention
            compression_ratio=0.9,
            priority=2
        )
        
        # Configure short-term memory for active workflows
        self.memory_manager.configure_tier(
            tier="short_term",
            ttl_days=30,   # 1 month retention
            compression_ratio=1.0,
            priority=3
        )
        
        print("✅ Memory tiers configured")
    
    def create_knowledge_domains(self):
        """Create knowledge domains for different data types."""
        print("🏗️ Creating Knowledge Domains...")
        
        domains = [
            {
                "domain_id": "grc",
                "name": "Governance, Risk & Compliance",
                "description": "Policy domains, groups, requirements, and technology solutions",
                "tags": ["policy", "compliance", "risk", "governance"]
            },
            {
                "domain_id": "organizational",
                "name": "Organizational Knowledge",
                "description": "Technical terms, acronyms, definitions, and institutional knowledge",
                "tags": ["organization", "terms", "definitions", "acronyms"]
            },
            {
                "domain_id": "logging",
                "name": "Enterprise Logging Platforms",
                "description": "Splunk ES, Google Chronicle, data models, and monitoring content",
                "tags": ["logging", "splunk", "chronicle", "monitoring", "siem"]
            },
            {
                "domain_id": "cmdb",
                "name": "Configuration Management",
                "description": "Business applications, devices, IPs, networks, and users",
                "tags": ["cmdb", "infrastructure", "applications", "devices", "networks"]
            }
        ]
        
        for domain in domains:
            self.master_catalog.register_domain(
                domain_id=domain["domain_id"],
                name=domain["name"],
                description=domain["description"],
                tags=domain["tags"]
            )
            print(f"  ✅ Created domain: {domain['name']}")
    
    def setup_memory_cleanup(self):
        """Set up automatic memory cleanup and maintenance."""
        print("🧹 Setting up Memory Cleanup...")
        
        # Configure cleanup schedules
        self.memory_manager.set_cleanup_schedule(
            short_term_cleanup_days=7,
            medium_term_cleanup_days=30,
            long_term_cleanup_days=180
        )
        
        # Set memory limits
        self.memory_manager.set_memory_limits(
            short_term_max_size_mb=100,
            medium_term_max_size_mb=500,
            long_term_max_size_mb=2000
        )
        
        print("✅ Memory cleanup configured")
    
    def configure_all(self):
        """Configure all memory settings."""
        self.configure_memory_tiers()
        self.create_knowledge_domains()
        self.setup_memory_cleanup()
        print("\n🎉 Memory configuration complete!")

if __name__ == "__main__":
    config = MemoryConfiguration()
    config.configure_all()
```

## 🔄 **Workflow Integration**

### **Workflow Problem Breakdown**

```python
#!/usr/bin/env python3
"""
Workflow Problem Breakdown Script
Demonstrates how the agent breaks down complex problems using knowledge base
"""

from bin.workflow_verification_system import WorkflowVerifier
from bin.workflow_template_manager import WorkflowTemplateManager
import json

class WorkflowProblemBreakdown:
    def __init__(self):
        self.verifier = WorkflowVerifier()
        self.template_manager = WorkflowTemplateManager()
    
    def breakdown_chronicle_to_splunk_conversion(self, content_catalog: Dict[str, Any]):
        """Break down Chronicle to Splunk conversion problem."""
        print("🔄 Breaking down Chronicle to Splunk conversion problem...")
        
        # Step 1: Analyze content catalog structure
        print("\n📋 Step 1: Content Catalog Analysis")
        catalog_analysis = self._analyze_content_catalog(content_catalog)
        print(f"  • Content types: {len(catalog_analysis['content_types'])}")
        print(f"  • Monitoring rules: {catalog_analysis['monitoring_rules']}")
        print(f"  • Data sources: {len(catalog_analysis['data_sources'])}")
        
        # Step 2: Map to policy requirements
        print("\n📋 Step 2: Policy Requirement Mapping")
        policy_mapping = self._map_to_policy_requirements(catalog_analysis)
        print(f"  • Mapped policies: {len(policy_mapping)}")
        
        # Step 3: Identify technology solutions
        print("\n📋 Step 3: Technology Solution Identification")
        tech_solutions = self._identify_technology_solutions(policy_mapping)
        print(f"  • Technology solutions: {len(tech_solutions)}")
        
        # Step 4: Generate Splunk SPL equivalents
        print("\n📋 Step 4: Splunk SPL Generation")
        splunk_equivalents = self._generate_splunk_equivalents(content_catalog, tech_solutions)
        print(f"  • Generated SPL queries: {len(splunk_equivalents)}")
        
        # Step 5: Validate against existing Splunk environment
        print("\n📋 Step 5: Splunk Environment Validation")
        validation_results = self._validate_splunk_environment(splunk_equivalents)
        print(f"  • Valid queries: {validation_results['valid']}")
        print(f"  • Issues found: {validation_results['issues']}")
        
        return {
            'catalog_analysis': catalog_analysis,
            'policy_mapping': policy_mapping,
            'tech_solutions': tech_solutions,
            'splunk_equivalents': splunk_equivalents,
            'validation_results': validation_results
        }
    
    def _analyze_content_catalog(self, catalog: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the structure of the content catalog."""
        analysis = {
            'content_types': set(),
            'monitoring_rules': 0,
            'data_sources': set(),
            'complexity_score': 0
        }
        
        # Analyze content structure
        if 'content' in catalog:
            for item in catalog['content']:
                if 'type' in item:
                    analysis['content_types'].add(item['type'])
                
                if 'monitoring_rule' in item:
                    analysis['monitoring_rules'] += 1
                
                if 'data_source' in item:
                    analysis['data_sources'].add(item['data_source'])
        
        # Calculate complexity score
        analysis['complexity_score'] = (
            len(analysis['content_types']) * 0.3 +
            analysis['monitoring_rules'] * 0.4 +
            len(analysis['data_sources']) * 0.3
        )
        
        return analysis
    
    def _map_to_policy_requirements(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Map content catalog to policy requirements."""
        # This would query the knowledge base for relevant policies
        # For now, return mock data
        return [
            {'policy': 'Data Encryption', 'requirement': 'Encryption at rest', 'relevance': 0.9},
            {'policy': 'Access Control', 'requirement': 'Privileged access monitoring', 'relevance': 0.8},
            {'policy': 'Network Security', 'requirement': 'Network traffic monitoring', 'relevance': 0.7}
        ]
    
    def _identify_technology_solutions(self, policy_mapping: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify technology solutions for policy requirements."""
        # This would query the knowledge base for technology solutions
        return [
            {'solution': 'Bitlocker', 'policy': 'Data Encryption', 'splunk_integration': True},
            {'solution': 'Privileged Access Management', 'policy': 'Access Control', 'splunk_integration': True},
            {'solution': 'Network Monitoring Tools', 'policy': 'Network Security', 'splunk_integration': True}
        ]
    
    def _generate_splunk_equivalents(self, catalog: Dict[str, Any], tech_solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate Splunk SPL equivalents for Chronicle content."""
        splunk_queries = []
        
        # This would use the knowledge base to generate appropriate SPL
        # For now, return mock data
        for solution in tech_solutions:
            if solution['splunk_integration']:
                splunk_queries.append({
                    'original_content': f"Chronicle content for {solution['solution']}",
                    'splunk_query': f"index=* sourcetype={solution['solution'].lower()} | stats count by host",
                    'policy_compliance': solution['policy'],
                    'confidence': 0.85
                })
        
        return splunk_queries
    
    def _validate_splunk_environment(self, splunk_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate generated SPL against existing Splunk environment."""
        # This would query the knowledge base for Splunk environment details
        return {
            'valid': len(splunk_queries),
            'issues': 0,
            'recommendations': ['Ensure proper index permissions', 'Validate sourcetype availability']
        }

# Usage example
if __name__ == "__main__":
    breakdown = WorkflowProblemBreakdown()
    
    # Mock content catalog
    mock_catalog = {
        'name': 'Security Monitoring Content',
        'content': [
            {'type': 'detection_rule', 'monitoring_rule': True, 'data_source': 'windows_events'},
            {'type': 'correlation_rule', 'monitoring_rule': True, 'data_source': 'network_traffic'}
        ]
    }
    
    results = breakdown.breakdown_chronicle_to_splunk_conversion(mock_catalog)
    print(f"\n🎯 Problem breakdown complete! Generated {len(results['splunk_equivalents'])} Splunk queries.")
```

## 🔍 **Search and Retrieval**

### **Knowledge Base Search Interface**

```python
#!/usr/bin/env python3
"""
Knowledge Base Search Interface
Provides search capabilities across all organizational knowledge
"""

from bin.context_memory_manager import ContextMemoryManager
from bin.master_catalog import MasterCatalog
import json
from typing import List, Dict, Any

class KnowledgeBaseSearch:
    def __init__(self):
        self.memory_manager = ContextMemoryManager()
        self.master_catalog = MasterCatalog()
    
    def search_knowledge(self, query: str, entity_types: List[str] = None, 
                        domains: List[str] = None, tags: List[str] = None) -> List[Dict[str, Any]]:
        """Search across all knowledge domains."""
        results = []
        
        # Search in memory manager
        memory_results = self.memory_manager.search_memory(
            query=query,
            domains=domains,
            entity_types=entity_types,
            tags=tags
        )
        results.extend(memory_results)
        
        # Search in master catalog
        catalog_results = self.master_catalog.search_global_index(
            query=query,
            node_types=entity_types,
            domain_ids=domains
        )
        results.extend(catalog_results)
        
        # Remove duplicates and sort by relevance
        unique_results = self._deduplicate_results(results)
        sorted_results = sorted(unique_results, key=lambda x: x.get('relevance', 0), reverse=True)
        
        return sorted_results
    
    def search_by_policy(self, policy_name: str) -> Dict[str, Any]:
        """Search for policy-related knowledge."""
        results = self.search_knowledge(
            query=policy_name,
            entity_types=['policy_domain', 'policy_group', 'policy_requirement', 'technology_solution']
        )
        
        # Group results by type
        grouped_results = {
            'policy_domains': [],
            'policy_groups': [],
            'policy_requirements': [],
            'technology_solutions': []
        }
        
        for result in results:
            entity_type = result.get('entity_type', '')
            if 'domain' in entity_type:
                grouped_results['policy_domains'].append(result)
            elif 'group' in entity_type:
                grouped_results['policy_groups'].append(result)
            elif 'requirement' in entity_type:
                grouped_results['policy_requirements'].append(result)
            elif 'solution' in entity_type:
                grouped_results['technology_solutions'].append(result)
        
        return grouped_results
    
    def search_by_technology(self, technology_name: str) -> Dict[str, Any]:
        """Search for technology-related knowledge."""
        results = self.search_knowledge(
            query=technology_name,
            entity_types=['technology_solution', 'logging_component', 'cmdb_entity']
        )
        
        return {
            'technology_solutions': [r for r in results if 'solution' in r.get('entity_type', '')],
            'logging_integration': [r for r in results if 'logging' in r.get('entity_type', '')],
            'cmdb_entities': [r for r in results if 'cmdb' in r.get('entity_type', '')]
        }
    
    def search_by_compliance(self, compliance_framework: str) -> List[Dict[str, Any]]:
        """Search for compliance-related knowledge."""
        return self.search_knowledge(
            query=compliance_framework,
            entity_types=['policy_requirement', 'technology_solution'],
            tags=['compliance', 'policy']
        )
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on entity ID."""
        seen_ids = set()
        unique_results = []
        
        for result in results:
            entity_id = result.get('entity_id', '')
            if entity_id and entity_id not in seen_ids:
                seen_ids.add(entity_id)
                unique_results.append(result)
        
        return unique_results

# Usage example
if __name__ == "__main__":
    search = KnowledgeBaseSearch()
    
    # Search for encryption-related knowledge
    encryption_results = search.search_by_policy("encryption")
    print(f"Found {len(encryption_results['technology_solutions'])} encryption technology solutions")
    
    # Search for Splunk-related knowledge
    splunk_results = search.search_by_technology("Splunk")
    print(f"Found {len(splunk_results['logging_integration'])} Splunk logging components")
```

## 🚀 **Implementation Steps**

### **Step 1: Data Preparation (Day 1-2)**
1. **Assess data quality** using the assessment script
2. **Choose transformation approach** (Option A or B)
3. **Run transformation/import scripts**
4. **Validate data integrity**

### **Step 2: Memory Configuration (Day 2-3)**
1. **Configure memory tiers** for different data types
2. **Create knowledge domains** for GRC, organizational, logging, and CMDB
3. **Set up memory cleanup** and maintenance schedules
4. **Test memory operations**

### **Step 3: Workflow Integration (Day 3-4)**
1. **Integrate with existing workflow system**
2. **Test problem breakdown** for Chronicle to Splunk conversion
3. **Validate knowledge retrieval** in workflows
4. **Optimize search performance**

### **Step 4: Testing and Validation (Day 4-5)**
1. **Test end-to-end workflows**
2. **Validate knowledge accuracy**
3. **Performance testing**
4. **Documentation and training**

## 🎯 **Key Benefits**

1. **Centralized Knowledge** - All organizational knowledge in one searchable system
2. **Workflow Integration** - Seamless integration with existing agentic workflows
3. **Policy Compliance** - Direct mapping between policies and technology solutions
4. **Cross-Platform Context** - Understanding of Chronicle vs Splunk differences
5. **Scalable Architecture** - Easy to add new data sources and knowledge types

## 🔧 **Troubleshooting**

### **Common Issues and Solutions**

1. **Data Import Errors**
   - Check file encoding (use UTF-8)
   - Validate CSV/JSON format
   - Ensure required columns exist

2. **Memory Performance Issues**
   - Adjust memory tier configurations
   - Implement data compression
   - Use appropriate TTL settings

3. **Search Accuracy Issues**
   - Refine search algorithms
   - Add more comprehensive tagging
   - Implement fuzzy matching

4. **Workflow Integration Issues**
   - Verify tool registration
   - Check memory access permissions
   - Validate data flow between components

This comprehensive setup provides a robust foundation for organizational knowledge management while maintaining the flexibility to grow and adapt to your specific needs.
