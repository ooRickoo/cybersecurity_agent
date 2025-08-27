# Browser MCP Tools & Framework Knowledge Graph Integration Guide

## 🚀 **Overview**

This guide covers the new browser MCP tools and framework knowledge graph integration that enables the Cybersecurity Agent to:

- **Search online** for cybersecurity information and frameworks
- **Download frameworks** like MITRE ATT&CK, D3FEND, NIST SP 800-53
- **Flatten frameworks** for easy querying and analysis
- **Store frameworks** in knowledge graph with TTL-based memory management
- **Create workflow task contexts** for framework operations

## 🏗️ **Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    User Request                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              ADK Integration Layer                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Browser       │  │   Framework     │  │   Smart     │ │
│  │   MCP Tools     │  │   Knowledge     │  │   Data      │ │
│  │                 │  │   Graph         │  │   Manager   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Execution Layer                               │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │   Online        │  │   Local         │                  │
│  │   Resources     │  │   Processing    │                  │
│  └─────────────────┘  └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ **Core Components**

### **1. Browser MCP Tools (`bin/browser_mcp_tools.py`)**
- **Online Search**: Search for cybersecurity information and frameworks
- **Framework Download**: Download frameworks from official sources
- **Framework Flattening**: Convert complex frameworks to flat, queryable formats
- **Format Support**: STIX, JSON, XML, CSV frameworks

### **2. Framework Knowledge Graph (`bin/framework_knowledge_graph.py`)**
- **Memory Management**: TTL-based framework storage (30-60 days)
- **Query Engine**: Fast search and retrieval of framework data
- **Task Context**: Workflow-specific context management
- **Background Cleanup**: Automatic expiration of old frameworks

### **3. ADK Integration (`bin/adk_integration.py`)**
- **15 ADK Tools**: Including 7 new browser and framework tools
- **Tool Schemas**: Proper ADK-compatible tool definitions
- **Execution Engine**: Unified tool execution interface

## 🔄 **Available Frameworks**

### **MITRE ATT&CK**
- **Type**: STIX format
- **Source**: https://attack.mitre.org/
- **Content**: Enterprise attack patterns and techniques
- **Use Case**: Threat intelligence and attack analysis

### **MITRE D3FEND**
- **Type**: JSON format
- **Source**: https://d3fend.mitre.org/
- **Content**: Defensive techniques and countermeasures
- **Use Case**: Security control selection and defense planning

### **NIST SP 800-53**
- **Type**: JSON format (OSCAL)
- **Source**: NIST Cybersecurity Framework
- **Content**: Security and privacy controls
- **Use Case**: Compliance and security control mapping

### **CVE Database**
- **Type**: CSV format
- **Source**: https://cve.mitre.org/
- **Content**: Common vulnerabilities and exposures
- **Use Case**: Vulnerability assessment and management

### **CAPEC Database**
- **Type**: XML format
- **Source**: https://capec.mitre.org/
- **Content**: Attack pattern enumeration and classification
- **Use Case**: Threat modeling and attack pattern analysis

## 🚀 **Available ADK Tools**

### **Browser Tools (4 tools)**
1. **`search_online`** - Search for cybersecurity information
2. **`download_framework`** - Download cybersecurity frameworks
3. **`flatten_framework`** - Flatten frameworks for analysis
4. **`get_available_frameworks`** - List available frameworks

### **Knowledge Graph Tools (4 tools)**
1. **`store_framework`** - Store framework in knowledge graph
2. **`query_framework`** - Query stored framework data
3. **`create_task_context`** - Create workflow task context
4. **`get_knowledge_graph_stats`** - Get knowledge graph statistics

### **Smart Data Management Tools (3 tools)**
1. **`register_data`** - Register data for local management
2. **`get_llm_context`** - Generate intelligent LLM context
3. **`process_locally`** - Process data using local tools

### **Core Workflow Tools (4 tools)**
1. **`generate_dynamic_workflow`** - Create dynamic workflows
2. **`analyze_threats`** - Threat analysis and enrichment
3. **`enrich_data`** - Data enrichment operations
4. **`get_system_status`** - Comprehensive system status

## 🔄 **Workflow Process**

### **Step 1: Framework Discovery**
```python
# Search for available frameworks
search_result = await adk.execute_tool('search_online', 
                                      query='threat intelligence frameworks')

# Get list of available frameworks
frameworks = await adk.execute_tool('get_available_frameworks')
```

### **Step 2: Framework Download**
```python
# Download MITRE ATT&CK framework
download_result = await adk.execute_tool('download_framework',
                                        framework_id='mitre_attack')

# Download NIST SP 800-53
nist_result = await adk.execute_tool('download_framework',
                                    framework_id='nist_sp800_53')
```

### **Step 3: Framework Flattening**
```python
# Flatten MITRE ATT&CK for analysis
flatten_result = await adk.execute_tool('flatten_framework',
                                       framework_id='mitre_attack',
                                       flatten_strategy='auto')

# Flatten NIST SP 800-53
nist_flatten = await adk.execute_tool('flatten_framework',
                                     framework_id='nist_sp800_53')
```

### **Step 4: Knowledge Graph Storage**
```python
# Store flattened framework with 30-day TTL
store_result = await adk.execute_tool('store_framework',
                                     framework_data=flatten_result['flattened_framework'],
                                     ttl_days=30)
```

### **Step 5: Task Context Creation**
```python
# Create workflow task context
task_context = await adk.execute_tool('create_task_context',
                                     task_name='threat_analysis',
                                     task_type='security_analysis',
                                     frameworks_required=['mitre_attack', 'nist_sp800_53'],
                                     priority=1)
```

### **Step 6: Framework Querying**
```python
# Query stored framework
query_result = await adk.execute_tool('query_framework',
                                     framework_id='mitre_attack',
                                     query='persistence techniques',
                                     max_results=50)
```

## 📊 **Framework Flattening Strategies**

### **STIX Framework Flattening**
```python
# MITRE ATT&CK STIX data
flattened_data = {
    "total_items": 1000,
    "items": [
        {
            "id": "attack-pattern--1234",
            "type": "attack-pattern",
            "name": "T1234: Account Manipulation",
            "description": "Adversaries may manipulate accounts...",
            "properties": {
                "kill_chain_phases": ["persistence"],
                "external_references": [...]
            }
        }
    ],
    "query_index": {
        "persistence": ["attack-pattern--1234"],
        "account": ["attack-pattern--1234"],
        "manipulation": ["attack-pattern--1234"]
    }
}
```

### **JSON Framework Flattening**
```python
# NIST SP 800-53 OSCAL data
flattened_data = {
    "total_items": 500,
    "items": [
        {
            "id": "AC-1",
            "type": "control",
            "title": "Access Control Policy and Procedures",
            "description": "The organization develops...",
            "properties": {
                "family": "Access Control",
                "priority": "P1"
            }
        }
    ],
    "query_index": {
        "access": ["AC-1"],
        "control": ["AC-1"],
        "policy": ["AC-1"]
    }
}
```

### **XML Framework Flattening**
```python
# CAPEC XML data
flattened_data = {
    "total_items": 200,
    "items": [
        {
            "id": "CAPEC-1",
            "type": "attack_pattern",
            "name": "Accessing Functionality Not Properly Constrained",
            "description": "An adversary engages in activities...",
            "properties": {
                "likelihood": "High",
                "severity": "High"
            }
        }
    ],
    "query_index": {
        "accessing": ["CAPEC-1"],
        "functionality": ["CAPEC-1"],
        "constrained": ["CAPEC-1"]
    }
}
```

## 🧠 **Knowledge Graph Memory Management**

### **TTL-Based Expiration**
```python
# Framework storage with TTL
framework_memory = FrameworkMemory(
    framework_id="mitre_attack_2024",
    framework_name="MITRE ATT&CK 2024",
    ttl_days=30,  # Expires in 30 days
    created_at=datetime.now(),
    last_accessed=datetime.now()
)
```

### **Automatic Cleanup**
- **Background Thread**: Runs every hour
- **Expiration Check**: Validates TTL for all frameworks
- **Memory Cleanup**: Removes expired frameworks
- **Cache Management**: Maintains optimal memory usage

### **Access Logging**
```python
# Track framework access patterns
access_log = {
    "framework_id": "mitre_attack",
    "access_type": "query",
    "timestamp": "2024-01-01T10:00:00",
    "task_id": "threat_analysis_001",
    "access_details": {"query": "persistence techniques"}
}
```

## 📋 **Workflow Task Context Management**

### **Task Context Creation**
```python
task_context = WorkflowTaskContext(
    task_id="threat_analysis_001",
    task_name="Advanced Persistent Threat Analysis",
    task_type="security_analysis",
    frameworks_required=["mitre_attack", "capec_database"],
    priority=1,
    status="active"
)
```

### **Context Data Management**
```python
# Update task context with analysis results
await adk.execute_tool('update_task_context',
                       task_id="threat_analysis_001",
                       context_data={
                           "threat_actors": ["APT28", "APT29"],
                           "techniques_found": ["T1234", "T5678"],
                           "risk_score": 8.5
                       },
                       status="completed")
```

## 🎯 **Use Cases and Examples**

### **Use Case 1: Threat Intelligence Analysis**
```python
# 1. Download and flatten MITRE ATT&CK
attack_framework = await adk.execute_tool('download_framework', 
                                         framework_id='mitre_attack')
flattened_attack = await adk.execute_tool('flatten_framework', 
                                         framework_id='mitre_attack')

# 2. Store in knowledge graph
await adk.execute_tool('store_framework',
                       framework_data=flattened_attack['flattened_framework'],
                       ttl_days=30)

# 3. Create analysis task context
task_context = await adk.execute_tool('create_task_context',
                                     task_name='threat_actor_analysis',
                                     task_type='threat_intelligence',
                                     frameworks_required=['mitre_attack'])

# 4. Query for specific techniques
persistence_techniques = await adk.execute_tool('query_framework',
                                               framework_id='mitre_attack',
                                               query='persistence',
                                               max_results=100)
```

### **Use Case 2: Compliance Mapping**
```python
# 1. Download NIST SP 800-53
nist_framework = await adk.execute_tool('download_framework',
                                       framework_id='nist_sp800_53')

# 2. Flatten for analysis
flattened_nist = await adk.execute_tool('flatten_framework',
                                       framework_id='nist_sp800_53')

# 3. Store in knowledge graph
await adk.execute_tool('store_framework',
                       framework_data=flattened_nist['flattened_framework'],
                       ttl_days=60)  # Longer TTL for compliance

# 4. Map controls to threats
access_controls = await adk.execute_tool('query_framework',
                                        framework_id='nist_sp800_53',
                                        query='access control',
                                        max_results=50)
```

### **Use Case 3: Vulnerability Assessment**
```python
# 1. Download CVE database
cve_data = await adk.execute_tool('download_framework',
                                  framework_id='cve_database')

# 2. Flatten CVE data
flattened_cve = await adk.execute_tool('flatten_framework',
                                       framework_id='cve_database')

# 3. Store in knowledge graph
await adk.execute_tool('store_framework',
                       framework_data=flattened_cve['flattened_framework'],
                       ttl_days=7)  # Short TTL for vulnerability data

# 4. Query for specific vulnerabilities
critical_vulns = await adk.execute_tool('query_framework',
                                       framework_id='cve_database',
                                       query='critical severity',
                                       max_results=100)
```

## 🔧 **Performance Optimization**

### **Caching Strategy**
- **Download Cache**: 1-day cache for framework downloads
- **Memory Cache**: In-memory framework storage
- **Query Index**: Fast text search capabilities
- **Background Cleanup**: Automatic memory management

### **Memory Efficiency**
- **TTL Management**: Automatic expiration of old data
- **Size Monitoring**: Track memory usage and optimize
- **Lazy Loading**: Load frameworks only when needed
- **Compression**: Efficient storage of large frameworks

### **Query Optimization**
- **Indexed Search**: Fast text-based queries
- **Result Limiting**: Configurable result counts
- **Context Awareness**: Task-specific query optimization
- **Caching**: Cache frequently accessed results

## 🧪 **Testing and Validation**

### **Test Browser Tools**
```bash
python3 bin/browser_mcp_tools.py
```

### **Test Framework Knowledge Graph**
```bash
python3 bin/framework_knowledge_graph.py
```

### **Test ADK Integration**
```bash
python3 bin/adk_integration.py
```

### **Test Individual Tools**
```python
import asyncio
from bin.adk_integration import ADKIntegration

async def test_framework_workflow():
    adk = ADKIntegration()
    
    # Test complete workflow
    print("🔍 Testing framework workflow...")
    
    # Download and flatten
    download = await adk.execute_tool('download_framework', framework_id='mitre_attack')
    flatten = await adk.execute_tool('flatten_framework', framework_id='mitre_attack')
    
    # Store and query
    store = await adk.execute_tool('store_framework', framework_data=flatten['flattened_framework'])
    query = await adk.execute_tool('query_framework', framework_id='mitre_attack', query='persistence')
    
    print(f"Workflow completed: {all([download['success'], flatten['success'], store['success'], query['success']])}")

asyncio.run(test_framework_workflow())
```

## 🔮 **Future Enhancements**

### **1. Advanced Framework Processing**
- **Machine Learning**: Intelligent framework analysis
- **Cross-Reference**: Link related frameworks
- **Version Management**: Track framework updates
- **Custom Frameworks**: Support for user-defined frameworks

### **2. Enhanced Search Capabilities**
- **Semantic Search**: AI-powered content understanding
- **Fuzzy Matching**: Handle typos and variations
- **Context Awareness**: Task-specific search optimization
- **Multi-Language**: Support for international frameworks

### **3. Performance Improvements**
- **Distributed Storage**: Scale across multiple nodes
- **Real-time Updates**: Live framework synchronization
- **Advanced Caching**: Multi-tier caching strategies
- **Query Optimization**: Intelligent query planning

## 📚 **Best Practices**

### **1. Framework Management**
- **Regular Updates**: Keep frameworks current
- **TTL Optimization**: Set appropriate expiration times
- **Storage Monitoring**: Track memory usage
- **Cleanup Scheduling**: Regular maintenance windows

### **2. Task Context Usage**
- **Clear Naming**: Use descriptive task names
- **Framework Selection**: Only load required frameworks
- **Context Updates**: Keep task context current
- **Priority Management**: Set appropriate task priorities

### **3. Query Optimization**
- **Specific Queries**: Use targeted search terms
- **Result Limits**: Set reasonable result counts
- **Index Usage**: Leverage query indexes
- **Caching**: Utilize built-in caching

## 🎯 **Summary**

Our Browser MCP Tools and Framework Knowledge Graph integration provides:

✅ **Online Framework Access** - Download from official sources  
✅ **Intelligent Flattening** - Convert complex frameworks to flat formats  
✅ **TTL-Based Memory** - Automatic expiration and cleanup  
✅ **Fast Querying** - Indexed search capabilities  
✅ **Task Context** - Workflow-specific framework management  
✅ **ADK Integration** - 15 tools for comprehensive cybersecurity operations  

The system automatically manages framework lifecycle, from download to expiration, while providing fast, intelligent access to cybersecurity knowledge. This enables dynamic workflows that can leverage the latest threat intelligence and compliance frameworks without manual intervention.

---

*For additional support or questions, refer to the main documentation or contact the development team.*
