# Quick Reference Guide - Enhanced Cybersecurity Agent

## 🚀 Quick Start Commands

### Start the Agent
```bash
python langgraph_cybersecurity_agent.py
```

### Memory Management
```bash
# Interactive memory management
python bin/memory_workflow_cli.py

# Import data
python bin/memory_workflow_cli.py --import "hosts:host_inventory.csv"

# Query memory
python bin/memory_workflow_cli.py --query "threat actor APT29"

# View statistics
python bin/memory_workflow_cli.py --stats
```

### Salt Management
```bash
# Generate device-bound salt
python bin/salt_manager.py --device-bound

# View salt information
python bin/salt_manager.py --info

# Generate session salt
python bin/salt_manager.py --session
```

### Host Verification
```bash
# Check host compatibility
python bin/host_verification.py
```

## 🛠️ Core Components

### 1. Enhanced Session Manager
- **Location**: `bin/enhanced_session_manager.py`
- **Purpose**: Comprehensive logging and output organization
- **Key Features**:
  - Session-specific folders in `session-outputs/`
  - Detailed logs in `session-logs/`
  - Automatic file organization by type

### 2. Enhanced Web Search
- **Location**: `bin/enhanced_web_search.py`
- **Purpose**: Advanced web research with Selenium
- **Key Features**:
  - Multi-engine search (Google, Bing, DuckDuckGo)
  - JavaScript rendering and content extraction
  - Session integration for logging

### 3. Splunk Integration
- **Location**: `bin/splunk_integration.py`
- **Purpose**: Full-featured Splunk instance management
- **Key Features**:
  - Multi-instance support (on-prem + cloud)
  - Scheduled job management
  - Data discovery and analysis
  - Instance comparison

### 4. Context Memory Manager
- **Location**: `bin/context_memory_manager.py`
- **Purpose**: Multi-dimensional memory management
- **Key Features**:
  - Short/medium/long-term memory tiers
  - Domain-specific organization
  - TTL management and cleanup

## 📁 Directory Structure

```
Cybersecurity-Agent/
├── session-logs/           # Detailed activity logs
├── session-outputs/        # Organized output files
│   └── {session_id}/      # Unique session folders
│       ├── data/          # DataFrames and structured data
│       ├── visualizations/ # Charts and graphs
│       └── text/          # Text outputs and reports
├── etc/                   # System configuration
│   └── credential_vault.db # Encrypted credential storage
├── knowledge-objects/     # Encrypted knowledge graph
└── bin/                   # Core system components
    ├── enhanced_session_manager.py
    ├── enhanced_web_search.py
    ├── splunk_integration.py
    ├── context_memory_manager.py
    ├── memory_mcp_tools.py
    ├── memory_workflow_cli.py
    ├── visualization_mcp_tools.py
    ├── credential_vault.py
    ├── salt_manager.py
    └── host_verification.py
```

## 🔧 Common Operations

### Import Data to Memory
```python
# Via CLI
python bin/memory_workflow_cli.py --import "hosts:host_inventory.csv"

# Via Python
from bin.context_memory_manager import ContextMemoryManager
memory = ContextMemoryManager()
memory.import_data("hosts", data, domain="infrastructure", tier="long_term")
```

### Web Research
```python
from bin.enhanced_web_search import EnhancedWebSearch
from bin.enhanced_session_manager import EnhancedSessionManager

session_mgr = EnhancedSessionManager()
web_search = EnhancedWebSearch(session_mgr)

# Search with content extraction
results = web_search.search_web("APT29 techniques", extract_content=True)
```

### Splunk Operations
```python
from bin.splunk_integration import SplunkIntegration

splunk = SplunkIntegration()

# Add instance
splunk.add_instance("prod", "splunk.company.com", username="admin", password="pass")

# Discover data sources
discovery = splunk.discover_data_sources("prod")

# Execute search
results = splunk.execute_search("index=security | stats count by sourcetype", "prod")
```

### Session Management
```python
from bin.enhanced_session_manager import EnhancedSessionManager

session_mgr = EnhancedSessionManager()

# Save DataFrame
session_mgr.save_dataframe(df, "host_analysis", "Host inventory analysis results")

# Save visualization
session_mgr.save_visualization(chart, "threat_landscape", "Threat landscape visualization")

# Save text output
session_mgr.save_text_output("Analysis complete", "final_report", "Final analysis report")
```

## 🎯 Memory Management Patterns

### Data Import Patterns
```python
# Host inventory (long-term)
memory.import_data("hosts", host_data, domain="infrastructure", tier="long_term", ttl_days=365)

# Threat intelligence (medium-term)
memory.import_data("threats", threat_data, domain="threat_intel", tier="medium_term", ttl_days=30)

# Investigation data (short-term)
memory.import_data("investigation", ioc_data, domain="incident_response", tier="short_term", ttl_days=7)
```

### Memory Retrieval
```python
# Get context for specific domain
context = memory.get_context(domain="infrastructure", max_entries=100)

# Search for specific entities
results = memory.search("APT29", domain="threat_intel")

# Get related entities
related = memory.get_related_entities("host_001", max_depth=2)
```

## 🔐 Security Features

### Encryption Management
- **Device-bound encryption**: Tied to hardware fingerprint
- **Session encryption**: Portable across devices
- **Credential vault**: Secure storage of secrets
- **Host verification**: Automatic detection of environment changes

### Environment Variables
```bash
# Enable encryption (default: true)
export ENCRYPTION_ENABLED=true

# Set encryption password hash
export ENCRYPTION_PASSWORD_HASH="your_hashed_password"

# Disable encryption (not recommended)
export ENCRYPTION_ENABLED=false
```

## 📊 Visualization Tools

### Available Visualizations
1. **DataFrame Viewer**: Interactive table visualization
2. **Workflow Diagrams**: Process and workflow visualization
3. **Neo4j Graph**: Relationship and network visualization
4. **Vega-Lite Charts**: Statistical and analytical charts

### Export Formats
- **HTML**: Interactive web-based visualizations
- **PNG**: High-resolution static images
- **SVG**: Scalable vector graphics

### Usage in Workflows
```python
# Via MCP tools (automatically available to Runner Agent)
visualization_tools = {
    "dataframe_viewer": "View and analyze tabular data",
    "workflow_diagram": "Visualize workflow processes",
    "neo4j_graph": "Explore entity relationships",
    "vega_lite_charts": "Create statistical visualizations"
}
```

## 🚨 Troubleshooting

### Common Issues

#### Host Verification Mismatch
```bash
# Reset encrypted data
python bin/host_verification.py
# Choose option 1 to reset when prompted
```

#### Memory Import Errors
```bash
# Check memory status
python bin/memory_workflow_cli.py --stats

# Verify data format
python bin/memory_workflow_cli.py --import "test:test_data.csv"
```

#### Session Output Not Found
```bash
# Check session logs
ls -la session-logs/

# Verify session manager initialization
python -c "from bin.enhanced_session_manager import EnhancedSessionManager; print('OK')"
```

### Debug Commands
```bash
# Test salt manager
python -c "from bin.salt_manager import SaltManager; sm = SaltManager(); print(sm.get_salt_info())"

# Test memory manager
python -c "from bin.context_memory_manager import ContextMemoryManager; mm = ContextMemoryManager(); print('OK')"

# Test Splunk integration
python -c "from bin.splunk_integration import SplunkIntegration; print('OK')"
```

## 📚 Additional Resources

### Documentation Files
- `COMPREHENSIVE_ENHANCEMENTS_SUMMARY.md`: Detailed feature overview
- `README.md`: Main project documentation
- Component-specific docstrings in Python files

### Example Workflows
- Policy-to-MITRE mapping
- Host inventory analysis
- Threat intelligence correlation
- Splunk instance comparison

### Support and Updates
- Check component docstrings for latest API information
- Review session logs for detailed execution information
- Use memory statistics for system health monitoring

---

*This guide provides quick access to the most commonly used features. For detailed information, refer to the comprehensive documentation and source code.*
