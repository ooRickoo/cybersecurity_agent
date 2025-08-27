# Cloud Integration & Enhanced Workflow Guide

## Overview
This guide explains how to integrate the new cloud integrations (Azure Resource Graph, Google Resource Manager) and enhanced workflow logic into your main cybersecurity agent system.

## 🚀 New Components Added

### 1. Azure Resource Graph Integration
- **File**: `bin/azure_resource_graph.py`
- **Purpose**: Full Azure resource querying and management
- **Features**: KQL queries, credential management, local data export

### 2. Google Resource Manager Integration
- **File**: `bin/google_resource_manager.py`
- **Purpose**: Comprehensive GCP resource management
- **Features**: Asset inventory, compute/storage analysis, local data export

### 3. Enhanced Local Scratch Tools
- **File**: `bin/local_scratch_tools.py`
- **Purpose**: Powerful local data processing and analysis
- **Features**: Data cleaning, anomaly detection, pattern extraction, reporting

### 4. Enhanced Workflow Logic
- **File**: `bin/enhanced_workflow_logic.py`
- **Purpose**: Intelligent workflow adaptation with local-first preference
- **Features**: Dynamic strategy selection, phase execution, error handling

## 🔧 Integration Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Update Main Agent File
Add the following imports to your `langgraph_cybersecurity_agent.py`:

```python
# Cloud integrations
from bin.azure_resource_graph import AzureResourceGraphIntegration, AzureResourceGraphMCPTools
from bin.google_resource_manager import GoogleResourceManagerIntegration, GoogleResourceManagerMCPTools

# Enhanced tools
from bin.local_scratch_tools import LocalScratchTools, LocalScratchMCPTools
from bin.enhanced_workflow_logic import EnhancedWorkflowLogic, WorkflowStrategy

# Enhanced session manager (if not already imported)
from bin.enhanced_session_manager import EnhancedSessionManager
```

### Step 3: Initialize Components in Main Agent
Add to your `LangGraphCybersecurityAgent.__init__` method:

```python
def __init__(self, encryption_manager=None):
    # ... existing initialization code ...
    
    # Initialize enhanced session manager
    self.session_manager = EnhancedSessionManager()
    
    # Initialize local scratch tools
    self.scratch_tools = LocalScratchTools(self.session_manager)
    
    # Initialize enhanced workflow logic
    self.workflow_logic = EnhancedWorkflowLogic(
        self.session_manager,
        self.memory_manager,
        self.scratch_tools
    )
    
    # Initialize cloud integrations (optional - only if credentials available)
    try:
        self.azure_integration = AzureResourceGraphIntegration(
            self.session_manager,
            self.credential_vault,
            self.memory_manager
        )
        self.azure_mcp_tools = AzureResourceGraphMCPTools(self.azure_integration)
    except Exception as e:
        logger.warning(f"Azure integration not available: {e}")
        self.azure_integration = None
        self.azure_mcp_tools = None
    
    try:
        self.gcp_integration = GoogleResourceManagerIntegration(
            self.session_manager,
            self.credential_vault,
            self.memory_manager
        )
        self.gcp_mcp_tools = GoogleResourceManagerMCPTools(self.gcp_integration)
    except Exception as e:
        logger.warning(f"Google Cloud integration not available: {e}")
        self.gcp_integration = None
        self.gcp_mcp_tools = None
```

### Step 4: Add MCP Tools to Tool Registry
Update your MCP tools dictionary:

```python
def _register_mcp_tools(self):
    """Register all MCP tools."""
    self.mcp_tools = {
        # ... existing tools ...
        
        # Local scratch tools
        "scratch_load_data": {
            "name": "scratch_load_data",
            "description": "Load data from various file formats into a DataFrame",
            "parameters": {
                "file_path": {"type": "string", "description": "Path to the data file"},
                "data_type": {"type": "string", "description": "Type of file (auto, csv, json, parquet, excel)"}
            },
            "returns": {"type": "object", "description": "Loaded DataFrame information"}
        },
        "scratch_analyze_data": {
            "name": "scratch_analyze_data",
            "description": "Analyze the structure and characteristics of a DataFrame",
            "parameters": {
                "df": {"type": "object", "description": "DataFrame to analyze"}
            },
            "returns": {"type": "object", "description": "Analysis results"}
        },
        "scratch_clean_data": {
            "name": "scratch_clean_data",
            "description": "Clean and preprocess DataFrame",
            "parameters": {
                "df": {"type": "object", "description": "DataFrame to clean"},
                "remove_duplicates": {"type": "boolean", "description": "Whether to remove duplicate rows"},
                "handle_missing": {"type": "string", "description": "How to handle missing values (drop, fill, interpolate)"},
                "fill_value": {"type": "any", "description": "Value to fill missing values with"}
            },
            "returns": {"type": "object", "description": "Cleaned DataFrame"}
        },
        "scratch_filter_data": {
            "name": "scratch_filter_data",
            "description": "Filter DataFrame based on multiple conditions",
            "parameters": {
                "df": {"type": "object", "description": "DataFrame to filter"},
                "filters": {"type": "object", "description": "Dictionary of column: value pairs for filtering"},
                "operator": {"type": "string", "description": "Logical operator (AND, OR)"}
            },
            "returns": {"type": "object", "description": "Filtered DataFrame"}
        },
        "scratch_generate_report": {
            "name": "scratch_generate_report",
            "description": "Generate a comprehensive summary report for a DataFrame",
            "parameters": {
                "df": {"type": "object", "description": "DataFrame to summarize"},
                "include_analysis": {"type": "boolean", "description": "Whether to include detailed analysis"}
            },
            "returns": {"type": "object", "description": "Summary report"}
        },
        
        # Azure Resource Graph tools (if available)
        **({"azure_query_resources": {
            "name": "azure_query_resources",
            "description": "Execute KQL query against Azure Resource Graph",
            "parameters": {
                "query": {"type": "string", "description": "KQL query string"},
                "subscriptions": {"type": "array", "items": {"type": "string"}, "description": "Subscription IDs (optional)"},
                "max_results": {"type": "integer", "description": "Maximum results to return"}
            },
            "returns": {"type": "object", "description": "Query results with data and metadata"}
        }} if self.azure_mcp_tools else {}),
        
        # Google Cloud tools (if available)
        **({"gcp_list_projects": {
            "name": "gcp_list_projects",
            "description": "List all accessible Google Cloud projects",
            "parameters": {
                "parent": {"type": "string", "description": "Parent resource (organization or folder)"}
            },
            "returns": {"type": "object", "description": "Project list with metadata"}
        }} if self.gcp_mcp_tools else {}),
        
        # ... other existing tools ...
    }
```

### Step 5: Add Workflow Planning Method
Add this method to your main agent class:

```python
def plan_workflow(self, user_request: str) -> Dict[str, Any]:
    """
    Plan a workflow based on user request.
    
    Args:
        user_request: User's request or question
        
    Returns:
        Workflow plan with phases and tool recommendations
    """
    try:
        # Get available tools
        available_tools = {name: tool for name, tool in self.mcp_tools.items()}
        
        # Plan workflow
        plan = self.workflow_logic.plan_workflow(user_request, available_tools)
        
        # Log workflow planning
        self.session_manager.log_activity(
            "workflow_planning",
            f"Planned workflow for: {user_request}",
            metadata={"plan": plan}
        )
        
        return plan
        
    except Exception as e:
        logger.error(f"Failed to plan workflow: {e}")
        return {"status": "error", "error": str(e)}
```

### Step 6: Add Workflow Execution Method
Add this method to your main agent class:

```python
def execute_workflow_phase(self, phase: Dict[str, Any], context_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute a specific workflow phase.
    
    Args:
        phase: Phase definition from workflow plan
        context_data: Context data from previous phases
        
    Returns:
        Phase execution results
    """
    try:
        # Get available tools
        available_tools = {name: tool for name, tool in self.mcp_tools.items()}
        
        # Execute phase
        results = self.workflow_logic.execute_workflow_phase(phase, available_tools, context_data)
        
        # Log phase execution
        self.session_manager.log_activity(
            "workflow_execution",
            f"Executed phase: {phase.get('phase', 'unknown')}",
            metadata={"phase": phase, "results": results}
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to execute workflow phase: {e}")
        return {"status": "error", "error": str(e)}
```

## 🎯 Usage Examples

### Example 1: Azure Resource Analysis
```python
# User request: "Analyze our Azure resources for security vulnerabilities"

# 1. Plan workflow
plan = agent.plan_workflow("Analyze our Azure resources for security vulnerabilities")

# 2. Execute phases
for phase in plan["phases"]:
    results = agent.execute_workflow_phase(phase)
    
    # Check if we need to adapt
    if results.get("status") == "error":
        adaptation = agent.workflow_logic.adapt_workflow(results)
        print(f"Workflow adapted: {adaptation}")
    
    # Continue with next phase...
```

### Example 2: Google Cloud Resource Comparison
```python
# User request: "Compare our GCP projects for resource usage patterns"

# 1. Plan workflow
plan = agent.plan_workflow("Compare our GCP projects for resource usage patterns")

# 2. Execute workflow
# The workflow logic will automatically:
# - Check for local data first
# - Use local scratch tools for analysis
# - Only query GCP if necessary
# - Export results to local scratch for further processing
```

### Example 3: Local Data Analysis
```python
# User request: "Analyze the security logs we collected yesterday"

# 1. Plan workflow (will automatically choose LOCAL_FIRST strategy)
plan = agent.plan_workflow("Analyze the security logs we collected yesterday")

# 2. Execute workflow
# The workflow will:
# - Assess available local data
# - Use local scratch tools for cleaning and analysis
# - Generate comprehensive reports
# - Save everything to session outputs
```

## 🔐 Credential Management

### Azure Credentials
The Azure integration will automatically prompt for credentials on first use:
1. **Service Principal**: Tenant ID, Client ID, Client Secret, Subscription ID
2. **Default Credentials**: Uses your local Azure CLI configuration

### Google Cloud Credentials
The GCP integration will automatically prompt for credentials on first use:
1. **Service Account Key**: Path to JSON key file
2. **Default Credentials**: Uses your local gcloud configuration

### Credential Storage
- Credentials are stored in the encrypted credential vault
- Connection information is stored in long-term memory
- No need to re-enter credentials on subsequent runs

## 📊 Local Scratch Tools Features

### Data Loading
- **Auto-detection**: Automatically detects file type (CSV, JSON, Parquet, Excel)
- **Caching**: Loaded data is cached for performance
- **Error handling**: Robust error handling with detailed logging

### Data Analysis
- **Structure analysis**: Column types, missing values, data quality issues
- **Anomaly detection**: IQR and Z-score based anomaly detection
- **Pattern extraction**: Email, IP, URL, date, phone number patterns

### Data Processing
- **Cleaning**: Remove duplicates, handle missing values, clean column names
- **Filtering**: Complex filtering with AND/OR logic
- **Aggregation**: Group by operations with multiple aggregation functions

### Reporting
- **Comprehensive reports**: Basic info, missing data, data quality metrics
- **Multiple formats**: JSON, CSV, HTML export options
- **Session integration**: All outputs automatically saved to session folders

## 🧠 Enhanced Workflow Logic Features

### Strategy Selection
- **LOCAL_FIRST**: Prefer local scratch tools when good local data exists
- **HYBRID**: Balance local and external tools
- **EXTERNAL_FIRST**: Prefer external tools for data collection
- **ADAPTIVE**: Dynamically adapt based on available data and tools

### Workflow Phases
1. **Data Assessment**: Evaluate available local data
2. **Local Analysis**: Use local scratch tools for analysis
3. **Data Collection**: Collect additional data if needed
4. **Synthesis**: Combine results from multiple sources
5. **Result Generation**: Generate final reports and outputs

### Adaptive Behavior
- **Error handling**: Automatically adapts to connection/authentication errors
- **Data sufficiency**: Checks if enough data exists for analysis
- **User feedback**: Adapts based on user input and preferences
- **Performance optimization**: Prioritizes local processing for speed

## 🚨 Troubleshooting

### Common Issues

#### Azure Integration Errors
```bash
# Check if Azure credentials are properly configured
python -c "from bin.azure_resource_graph import AzureResourceGraphIntegration; print('Azure integration OK')"

# Verify Azure CLI is configured
az account show
```

#### Google Cloud Integration Errors
```bash
# Check if gcloud is configured
gcloud auth list

# Verify service account key file
python -c "from bin.google_resource_manager import GoogleResourceManagerIntegration; print('GCP integration OK')"
```

#### Local Scratch Tools Errors
```bash
# Test local scratch tools
python -c "from bin.local_scratch_tools import LocalScratchTools; print('Local scratch tools OK')"

# Check pandas installation
python -c "import pandas; print(f'Pandas version: {pandas.__version__}')"
```

### Debug Commands
```bash
# Test workflow logic
python -c "from bin.enhanced_workflow_logic import EnhancedWorkflowLogic; print('Workflow logic OK')"

# Check session manager
python -c "from bin.enhanced_session_manager import EnhancedSessionManager; print('Session manager OK')"

# Verify memory manager
python -c "from bin.context_memory_manager import ContextMemoryManager; print('Memory manager OK')"
```

## 📈 Performance Benefits

### Local-First Approach
- **Faster execution**: No network latency for local data
- **Reduced costs**: Minimize external API calls
- **Better reliability**: No dependency on external service availability
- **Enhanced privacy**: Sensitive data stays local

### Intelligent Caching
- **Data caching**: Loaded data cached in memory
- **Analysis caching**: Analysis results cached for reuse
- **Session persistence**: All outputs saved to session folders

### Adaptive Workflows
- **Dynamic strategy selection**: Choose optimal approach based on data availability
- **Error recovery**: Automatically adapt to failures
- **User preference learning**: Adapt based on user feedback

## 🔮 Future Enhancements

### Planned Features
- **Real-time monitoring**: Live data streaming from cloud sources
- **Advanced analytics**: Machine learning integration for pattern detection
- **Workflow templates**: Pre-built workflows for common security tasks
- **Collaborative analysis**: Multi-user workflow execution

### Integration Opportunities
- **Additional cloud providers**: AWS, Oracle Cloud, IBM Cloud
- **Security tools**: SIEM integration, vulnerability scanners
- **Compliance frameworks**: Automated compliance checking
- **DevOps integration**: CI/CD pipeline integration

## 📚 Additional Resources

### Documentation Files
- `COMPREHENSIVE_ENHANCEMENTS_SUMMARY.md`: Complete feature overview
- `QUICK_REFERENCE_GUIDE.md`: Fast access to common operations
- Component-specific docstrings in Python files

### Example Workflows
- Azure security analysis
- GCP resource optimization
- Multi-cloud comparison
- Local data investigation

### Support and Updates
- Check component docstrings for latest API information
- Review session logs for detailed execution information
- Use workflow status for system health monitoring

---

*This guide provides comprehensive integration instructions for the new cloud tools and enhanced workflow logic. For detailed information, refer to the individual component documentation and source code.*
