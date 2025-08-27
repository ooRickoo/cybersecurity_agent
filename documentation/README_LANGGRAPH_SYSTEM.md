# 🔒 LangGraph Cybersecurity Agent

A dynamic, multi-agent cybersecurity analysis system built with LangGraph and MCP (Model Context Protocol) tools.

## 🚀 **Key Features**

### **Multi-Agent Architecture**
- **Planner Agent**: Analyzes user input and creates execution plans
- **Runner Agent**: Executes planned actions using available tools
- **Memory Manager**: Manages knowledge graph and multi-dimensional memory
- **Workflow Executor**: Executes predefined workflow templates

### **Dynamic Workflow Management**
- **Workflow Templates**: Predefined workflows for common cybersecurity tasks
- **Custom Workflows**: Create and execute custom workflow templates
- **Tool Discovery**: Automatic discovery and registration of MCP tools
- **Context-Aware Execution**: Workflows adapt based on available context

### **Multi-Dimensional Memory System**
- **Short-Term Memory**: Current session context (< 1KB)
- **Running-Term Memory**: Active workflows (1KB - 10KB)
- **Long-Term Memory**: Persistent knowledge (> 10KB)
- **Context Rehydration**: Load relevant context when needed

### **Framework Processing**
- **Multiple Formats**: JSON, CSV, XML, TTL, STIX
- **Automatic Flattening**: Convert complex frameworks to queryable format
- **Structure Analysis**: Analyze framework complexity and relationships
- **Intelligent Storage**: Store frameworks in appropriate memory dimension

### **Encryption & Security**
- **File Encryption**: Encrypt sensitive data files
- **Password Management**: Easy password change workflow
- **Environment Control**: Enable/disable encryption via environment variables
- **Secure Storage**: Encrypted SQLite databases and output files

## 🛠️ **Installation**

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Environment Configuration**
Copy `env_config.txt` to `.env` and configure:
```bash
cp env_config.txt .env
# Edit .env with your settings
```

### **3. Enable Encryption (Optional)**
```bash
export ENCRYPTION_ENABLED=true
export ENCRYPTION_PASSWORD=YourSecurePassword
```

## 🎯 **Usage**

### **Interactive Mode**
```bash
python cybersecurity_cli.py
```

### **CSV Processing Mode**
```bash
python cybersecurity_cli.py -csv policies.csv -output analysis_report.csv
```

### **Custom Prompt Mode**
```bash
python cybersecurity_cli.py -csv threats.csv -prompt "Analyze these threats and map to MITRE ATT&CK" -output threat_analysis.json
```

## 🔧 **Available Workflows**

### **1. Policy Analysis**
- Data ingestion and validation
- MITRE ATT&CK framework mapping
- Risk assessment and scoring
- Report generation

### **2. Threat Intelligence**
- Threat data collection
- Context enrichment
- Threat correlation
- Intelligence reporting

### **3. Incident Response**
- Incident assessment
- Containment management
- Root cause investigation
- Remediation planning
- Documentation

## 📚 **Knowledge Management**

### **Adding Frameworks**
```python
# Add MITRE ATT&CK framework
framework_data = {...}  # Your framework data
framework_id = knowledge_manager.add_framework("MITRE_ATTACK", framework_data)
```

### **Querying Knowledge**
```python
# Query across all memory dimensions
results = knowledge_manager.query_knowledge("authentication policies")
```

### **Context Rehydration**
```python
# Load relevant context
context = knowledge_manager.rehydrate_context(["policy_context", "threat_context"])
```

## 🔐 **Encryption Management**

### **Enable Encryption**
```bash
export ENCRYPTION_ENABLED=true
export ENCRYPTION_PASSWORD=Vosteen2025
```

### **Change Encryption Password**
```bash
python -c "
from bin.mcp_tools import EncryptionManager
old_manager = EncryptionManager('OldPassword')
new_manager = EncryptionManager('NewPassword')
# Re-encrypt files with new password
"
```

### **File Encryption**
- SQLite databases are automatically encrypted
- Output files can be encrypted on demand
- Session logs are encrypted when encryption is enabled

## 🏗️ **Architecture**

### **LangGraph Workflow**
```
START → Planner → Runner → Memory Manager → Workflow Executor → END
                ↓
            Conditional Edge
                ↓
        Execute Workflow or Continue
```

### **Memory Dimensions**
```
Short-Term Memory (Session Context)
├── Current conversation
├── Active tools
└── Temporary data

Running-Term Memory (Active Workflows)
├── Workflow state
├── Intermediate results
└── Tool outputs

Long-Term Memory (Persistent Knowledge)
├── Framework data
├── Historical analysis
└── Persistent context
```

### **MCP Tool Integration**
```
Agent → MCP Client → MCP Server → Tools
  ↓         ↓           ↓         ↓
State   Discovery   Execution   Results
```

## 📁 **File Structure**

```
cybersecurity-agent/
├── langgraph_cybersecurity_agent.py    # Main agent
├── cybersecurity_cli.py                # CLI interface
├── bin/
│   └── mcp_tools.py                   # MCP tools and utilities
├── session-logs/                       # Session logs
├── session-outputs/                    # Session outputs
├── knowledge-objects/                  # Knowledge base
├── checkpoints/                        # LangGraph checkpoints
├── requirements.txt                    # Dependencies
└── env_config.txt                     # Environment config
```

## 🔄 **Workflow Examples**

### **Policy Analysis Workflow**
1. **Data Ingestion**: Load CSV policy data
2. **Framework Mapping**: Map to MITRE ATT&CK
3. **Risk Assessment**: Calculate risk scores
4. **Report Generation**: Create analysis report

### **Threat Intelligence Workflow**
1. **Data Collection**: Gather threat data
2. **Enrichment**: Add context and metadata
3. **Correlation**: Identify patterns and relationships
4. **Reporting**: Generate intelligence report

### **Incident Response Workflow**
1. **Assessment**: Evaluate incident severity
2. **Containment**: Implement containment measures
3. **Investigation**: Analyze root cause
4. **Remediation**: Plan and execute fixes
5. **Documentation**: Record incident details

## 🚀 **Getting Started**

### **1. Quick Start**
```bash
# Install dependencies
pip install -r requirements.txt

# Start interactive mode
python cybersecurity_cli.py
```

### **2. Add Your First Framework**
```bash
# In interactive mode
🤖 You: Add MITRE ATT&CK framework to knowledge base
```

### **3. Run Policy Analysis**
```bash
# Process a CSV file
python cybersecurity_cli.py -csv policies.csv -output analysis.csv
```

### **4. Custom Workflow**
```bash
# Create custom workflow template
🤖 You: Create workflow template for vulnerability assessment
```

## 🔧 **Development**

### **Adding New Tools**
1. Create tool function in `bin/mcp_tools.py`
2. Register tool in `register_mcp_tools()`
3. Update workflow templates as needed

### **Adding New Workflows**
1. Define workflow in `WorkflowTemplateManager`
2. Add required tools
3. Define input/output schemas

### **Custom Memory Dimensions**
1. Extend `KnowledgeGraphManager`
2. Add new dimension logic
3. Update relevance calculations

## 🎉 **Benefits Over Previous System**

- **Clean Architecture**: LangGraph provides better workflow management
- **Dynamic Tool Discovery**: MCP tools are automatically discovered
- **Multi-Dimensional Memory**: Intelligent memory management
- **Workflow Templates**: Reusable, configurable workflows
- **Better Error Handling**: Robust error recovery and logging
- **Encryption Support**: Built-in security features
- **Session Management**: Comprehensive logging and output organization

## 🔮 **Future Enhancements**

- **Real-time Streaming**: Live workflow progress updates
- **Advanced Analytics**: Machine learning-powered analysis
- **Integration APIs**: REST and GraphQL endpoints
- **Distributed Memory**: Multi-node knowledge sharing
- **Advanced Encryption**: Hardware security modules
- **Workflow Orchestration**: Complex multi-workflow coordination

---

**🎯 Ready to revolutionize your cybersecurity workflows with LangGraph!** 🚀
