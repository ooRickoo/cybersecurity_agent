# 📖 Usage Examples Guide

This guide provides comprehensive, real-world usage examples for the Cybersecurity Agent, demonstrating how to use all available workflows and features with synthetic test data.

## 📁 Test Data Files

All examples use synthetic test data located in the `usage-example-files/` directory:

- **`sample_malware_indicators.csv`** - Malware file indicators with hashes, types, and threat levels
- **`sample_network_assets.csv`** - Network assets with IP addresses, services, and vulnerabilities
- **`sample_incident_data.csv`** - Security incident data with descriptions and status
- **`sample_threat_indicators.csv`** - Threat intelligence indicators (IOCs) with correlation data
- **`sample_file_metadata.csv`** - File metadata with security assessments
- **`sample_vulnerability_data.csv`** - CVE vulnerability data with severity scores
- **`verified_cybersecurity_patents.csv`** - Real cybersecurity patents for analysis

## 🔍 Malware Analysis Examples

### Basic Malware Analysis

```bash
# Analyze malware indicators from CSV data
python cs_util_lg.py -workflow malware_analysis \
  -problem "Analyze malware indicators and classify threat levels" \
  -input-file usage-example-files/sample_malware_indicators.csv \
  --output malware_analysis_results.csv

# Analyze a single suspicious file (if you have one)
python cs_util_lg.py -workflow malware_analysis \
  -problem "analyze: /path/to/suspicious/file.exe" \
  --output single_file_analysis.json

# Scan directory for malware
python cs_util_lg.py -workflow malware_analysis \
  -problem "scan directory: /suspicious/folder" \
  --output directory_scan_results.csv
```

### Advanced Malware Analysis with Custom Enrichment

```python
from bin.dynamic_enrichment_workflows import DynamicEnrichmentWorkflows, EnrichmentType, DataType

# Initialize enrichment system
dew = DynamicEnrichmentWorkflows()

# Create malware analysis enrichment workflow
workflow = dew.create_workflow(
    name="Malware Indicator Enrichment",
    description="Enrich malware indicators with additional analysis",
    input_file="usage-example-files/sample_malware_indicators.csv",
    output_file="enriched_malware_indicators.csv",
    enrichment_type=EnrichmentType.BATCH
)

# Add threat level normalization
dew.add_enrichment_column(
    workflow_id=workflow.workflow_id,
    column_name="threat_level_normalized",
    source_column="threat_level",
    enrichment_type="classification",
    enrichment_function="classify_threat_level",
    data_type=DataType.STRING
)

# Add file type categorization
dew.add_enrichment_column(
    workflow_id=workflow.workflow_id,
    column_name="file_type_category",
    source_column="file_type",
    enrichment_type="categorization",
    enrichment_function="to_lowercase",
    data_type=DataType.STRING
)

# Add hash extraction for additional analysis
dew.add_enrichment_column(
    workflow_id=workflow.workflow_id,
    column_name="hash_type",
    source_column="file_hash",
    enrichment_type="analysis",
    enrichment_function="extract_hash",
    data_type=DataType.STRING
)

# Execute enrichment
result = dew.execute_workflow(workflow.workflow_id)
print(f"Enrichment completed: {result['status']}")
print(f"Output file: {result['output_file']}")
```

## 🌐 Vulnerability Scanning Examples

### Network Asset Scanning

```bash
# Scan network assets from CSV data
python cs_util_lg.py -workflow vulnerability_scanning \
  -problem "Scan network assets and assess security posture" \
  -input-file usage-example-files/sample_network_assets.csv \
  --output vulnerability_scan_results.csv

# Single IP scan
python cs_util_lg.py -workflow vulnerability_scan \
  -problem "scan: 192.168.1.1" \
  --output ip_scan_results.json

# Port scan specific service
python cs_util_lg.py -workflow vulnerability_scan \
  -problem "port scan: example.com" \
  --output port_scan_results.json

# Web vulnerability scan
python cs_util_lg.py -workflow vulnerability_scan \
  -problem "web scan: https://example.com" \
  --output web_scan_results.json
```

### CVE Analysis and Risk Assessment

```bash
# Analyze CVE data for risk assessment
python cs_util_lg.py -workflow vulnerability_scanning \
  -problem "Perform comprehensive vulnerability assessment with CVE analysis" \
  -input-file usage-example-files/sample_vulnerability_data.csv \
  --output comprehensive_vuln_assessment.csv
```

### Custom Vulnerability Enrichment

```python
from bin.dynamic_enrichment_workflows import DynamicEnrichmentWorkflows, EnrichmentType, DataType

# Create vulnerability enrichment workflow
workflow = dew.create_workflow(
    name="Vulnerability Data Enrichment",
    description="Enrich CVE data with risk assessments",
    input_file="usage-example-files/sample_vulnerability_data.csv",
    output_file="enriched_vulnerability_data.csv",
    enrichment_type=EnrichmentType.BATCH
)

# Add severity normalization
dew.add_enrichment_column(
    workflow_id=workflow.workflow_id,
    column_name="severity_normalized",
    source_column="severity",
    enrichment_type="classification",
    enrichment_function="classify_threat_level",
    data_type=DataType.STRING
)

# Add exploit availability flag
dew.add_enrichment_column(
    workflow_id=workflow.workflow_id,
    column_name="has_exploit",
    source_column="exploit_available",
    enrichment_type="validation",
    enrichment_function="is_valid_boolean",
    data_type=DataType.BOOLEAN
)

# Execute enrichment
result = dew.execute_workflow(workflow.workflow_id)
```

## 📊 Patent Analysis Examples

### Basic Patent Analysis

```bash
# Analyze cybersecurity patents with AI insights
python cs_util_lg.py -workflow patent_analysis \
  -problem "Analyze cybersecurity patents and generate value propositions" \
  -input-file usage-example-files/verified_cybersecurity_patents.csv \
  --output enriched_patents.csv

# Detailed patent analysis with custom enrichment
python cs_util_lg.py -workflow patent_analysis \
  -problem "Take the input file list of cybersecurity patents, import into a local dataframe, iterate through the list of US patents # and associated publication #, add additional patent details into the dataframe from patent public APIs, adding a new column summarizing the value add for the patent in 1-3 lines and a new column categorizing the patents in a logical way. Then export the results to a csv file." \
  -input-file usage-example-files/real_cybersecurity_patents.csv \
  --output session_patent_analysis.csv
```

### Patent Data Enrichment

```python
# Create patent enrichment workflow
workflow = dew.create_workflow(
    name="Patent Data Enrichment",
    description="Enrich patent data with additional analysis",
    input_file="usage-example-files/verified_cybersecurity_patents.csv",
    output_file="enriched_patent_data.csv",
    enrichment_type=EnrichmentType.BATCH
)

# Add title normalization
dew.add_enrichment_column(
    workflow_id=workflow.workflow_id,
    column_name="title_normalized",
    source_column="title",
    enrichment_type="processing",
    enrichment_function="to_lowercase",
    data_type=DataType.STRING
)

# Add word count analysis
dew.add_enrichment_column(
    workflow_id=workflow.workflow_id,
    column_name="title_word_count",
    source_column="title",
    enrichment_type="analysis",
    enrichment_function="count_words",
    data_type=DataType.INTEGER
)

# Execute enrichment
result = dew.execute_workflow(workflow.workflow_id)
```

## 🚨 Incident Response Examples

### Incident Data Analysis

```bash
# Process incident data and generate reports
python cs_util_lg.py -workflow incident_response \
  -problem "Analyze security incidents and generate response recommendations" \
  -input-file usage-example-files/sample_incident_data.csv \
  --output incident_analysis_results.csv

# Create new incident
python cs_util_lg.py -workflow incident_response \
  -problem "Create incident for suspicious network activity" \
  --output incident_created.json
```

### Incident Response with Custom Workflow

```python
from bin.incident_response_tools import IncidentResponseTools, IncidentSeverity, IncidentStatus

# Initialize incident response tools
irt = IncidentResponseTools()

# Create incident
incident = irt.create_incident(
    title="Suspicious Network Activity Detected",
    description="Unusual network traffic patterns detected from internal workstation",
    severity=IncidentSeverity.HIGH,
    affected_systems=["workstation-01", "workstation-02"],
    indicators=["192.168.1.100", "malicious-domain.com"],
    tags=["network", "suspicious"]
)

print(f"Created incident: {incident.incident_id}")

# Add timeline event
irt.add_timeline_event(
    incident_id=incident.incident_id,
    event_type="detection",
    description="Anomalous network traffic detected",
    source="network_monitor",
    confidence=0.8
)

# Collect evidence
evidence = irt.collect_evidence(
    incident_id=incident.incident_id,
    evidence_type="network",
    source="network_capture.pcap",
    collected_by="analyst_john",
    metadata={"capture_duration": "1 hour", "packet_count": 10000}
)

# Generate incident report
report = irt.generate_incident_report(incident.incident_id)
print(f"Generated report with {report['evidence_count']} evidence items")
```

## 🕵️ Threat Intelligence Examples

### IOC Analysis

```bash
# Analyze threat indicators and correlate IOCs
python cs_util_lg.py -workflow threat_intelligence \
  -problem "Analyze threat indicators and identify attack patterns" \
  -input-file usage-example-files/sample_threat_indicators.csv \
  --output threat_intelligence_results.csv

# Correlate IOCs across multiple sources
python cs_util_lg.py -workflow threat_intelligence \
  -problem "Correlate IOCs and identify threat campaigns" \
  --output ioc_correlation_results.csv
```

### Threat Intelligence with Custom Analysis

```python
from bin.threat_intelligence_tools import ThreatIntelligenceTools, IOCType, ThreatLevel, ConfidenceLevel

# Initialize threat intelligence tools
tit = ThreatIntelligenceTools()

# Add IOCs
ioc1 = tit.add_ioc(
    ioc_type=IOCType.IP_ADDRESS,
    value="192.168.1.100",
    threat_level=ThreatLevel.HIGH,
    confidence=ConfidenceLevel.HIGH,
    source="threat_feed",
    description="Malicious IP address",
    tags=["malware", "c2"],
    campaigns=["campaign_1"],
    threat_actors=["actor_1"]
)

ioc2 = tit.add_ioc(
    ioc_type=IOCType.DOMAIN,
    value="malicious-domain.com",
    threat_level=ThreatLevel.HIGH,
    confidence=ConfidenceLevel.MEDIUM,
    source="dns_analysis",
    description="Malicious domain",
    tags=["malware", "c2"],
    campaigns=["campaign_1"],
    threat_actors=["actor_1"]
)

# Correlate IOCs
correlations = tit.correlate_iocs(["192.168.1.100", "malicious-domain.com"])
print(f"IOC correlation confidence: {correlations['confidence_score']:.2f}")

# Add threat actor
actor = tit.add_threat_actor(
    name="APT Group Alpha",
    aliases=["Alpha Group", "Group A"],
    description="Advanced persistent threat group",
    country="Unknown",
    motivation=["espionage", "financial"],
    capabilities=["spear_phishing", "lateral_movement", "data_exfiltration"]
)

# Add campaign
campaign = tit.add_campaign(
    name="Operation Silent Strike",
    description="Long-term espionage campaign",
    threat_actors=[actor.actor_id],
    targets=["government", "defense"],
    techniques=["spear_phishing", "lateral_movement"]
)

# Get threat landscape
landscape = tit.get_threat_landscape(days=30)
print(f"Threat landscape: {landscape['total_iocs']} IOCs in last 30 days")
```

## 📁 File Forensics Examples

### File Metadata Analysis

```bash
# Analyze file metadata and detect suspicious files
python cs_util_lg.py -workflow file_forensics \
  -problem "Analyze file metadata and identify suspicious files" \
  -input-file usage-example-files/sample_file_metadata.csv \
  --output file_forensics_results.csv

# Analyze single file
python cs_util_lg.py -workflow file_forensics \
  -problem "analyze: /path/to/suspicious/file.exe" \
  --output file_analysis_results.json
```

### File Forensics with Custom Enrichment

```python
# Create file forensics enrichment workflow
workflow = dew.create_workflow(
    name="File Forensics Enrichment",
    description="Enrich file metadata with security analysis",
    input_file="usage-example-files/sample_file_metadata.csv",
    output_file="enriched_file_forensics.csv",
    enrichment_type=EnrichmentType.BATCH
)

# Add entropy analysis
dew.add_enrichment_column(
    workflow_id=workflow.workflow_id,
    column_name="entropy_category",
    source_column="entropy",
    enrichment_type="classification",
    enrichment_function="classify_threat_level",
    data_type=DataType.STRING
)

# Add file size category
dew.add_enrichment_column(
    workflow_id=workflow.workflow_id,
    column_name="size_category",
    source_column="file_size",
    enrichment_type="categorization",
    enrichment_function="classify_threat_level",
    data_type=DataType.STRING
)

# Add suspicious file flag
dew.add_enrichment_column(
    workflow_id=workflow.workflow_id,
    column_name="is_suspicious_flag",
    source_column="is_suspicious",
    enrichment_type="validation",
    enrichment_function="is_valid_boolean",
    data_type=DataType.BOOLEAN
)

# Execute enrichment
result = dew.execute_workflow(workflow.workflow_id)
```

## 🧠 Memory-Enhanced Workflows

### Using Memory Integration

```python
from bin.enhanced_memory_integration import EnhancedMemoryIntegration, MemoryType

# Initialize memory system
emi = EnhancedMemoryIntegration()

# Store workflow experience
memory_id = emi.store_memory(
    memory_type=MemoryType.WORKFLOW_EXPERIENCE,
    content={
        "workflow": "malware_analysis",
        "problem": "analyze suspicious files",
        "solution": "use static analysis first, then dynamic if needed",
        "success": True
    },
    tags=["malware", "analysis", "static", "dynamic"],
    relevance_score=0.9
)

# Recall relevant memories for new problem
context = {"workflow_type": "malware_analysis", "problem": "analyze new malware sample"}
memories = emi.recall_contextual_memories(context, max_results=5)

for memory in memories:
    print(f"Memory: {memory.memory_id}")
    print(f"Relevance: {memory.relevance_score:.2f}")
    print(f"Reason: {memory.match_reason}")

# Get memory insights
insights = emi.get_memory_insights(context)
print(f"Memory insights: {len(insights['success_recommendations'])} recommendations")

# Enrich context with memory
enriched_context = emi.enrich_context_with_memory(context)
print(f"Enriched context with {len(enriched_context.get('memory_insights', {}).get('success_recommendations', []))} recommendations")
```

## 📊 Comprehensive Session Logging

### Using Enhanced Session Logger

```python
from bin.enhanced_session_logger import EnhancedSessionLogger, SessionStatus

# Initialize session logger
logger = EnhancedSessionLogger()

# Start session
session_id = logger.start_session(user_id="analyst_001", workflow_type="malware_analysis")

# Log user question
question_id = logger.log_user_question(
    question_text="Analyze this malware sample for threat indicators",
    context={"file_path": "/tmp/suspicious.exe", "priority": "high"}
)

# Log tool execution
tool_id = logger.log_tool_execution(
    tool_name="malware_analysis",
    function_name="analyze_file",
    input_parameters={"file_path": "/tmp/suspicious.exe"},
    output_result={"threat_level": "high", "malware_detected": True},
    execution_time_ms=2500.5,
    success=True
)

# Log memory recall
memory_id = logger.log_memory_recall(
    query="malware analysis patterns",
    context_used="file_analysis",
    memories_found=3,
    recall_time_ms=150.2
)

# Log file operation
file_id = logger.log_file_operation("create", "malware_analysis_results.csv", file_size=1024, success=True)

# End session
logger.end_session(SessionStatus.COMPLETED)

# Get session summary
summary = logger.get_session_summary(session_id)
print(f"Session completed: {summary['total_questions']} questions, {summary['total_tool_executions']} tool executions")

# Get analytics
analytics = logger.get_performance_analytics()
print(f"Analytics: {analytics['total_sessions']} sessions, {analytics['success_rate']:.2f} success rate")
```

## 🎯 Enhanced Tool Selection

### Using Intelligent Tool Selection

```python
from bin.enhanced_tool_selection import EnhancedToolSelection, ToolSelectionCriteria, ProblemComplexity, ToolExecutionType

# Initialize tool selection system
ets = EnhancedToolSelection()

# Create selection criteria
criteria = ToolSelectionCriteria(
    problem_type="malware_analysis",
    complexity=ProblemComplexity.MODERATE,
    required_capabilities=["static_analysis", "yara_scanning"],
    preferred_execution_type=ToolExecutionType.LOCAL,
    context_tags=["malware", "analysis"]
)

# Get tool recommendations
recommendations = ets.select_tools(criteria, max_tools=3)

for rec in recommendations:
    print(f"Tool: {rec.tool_id}")
    print(f"Score: {rec.score:.2f}")
    print(f"Reasoning: {', '.join(rec.reasoning)}")
    print(f"Estimated time: {rec.estimated_execution_time:.1f}s")
    print(f"Resource usage: {rec.resource_usage}")

# Record tool execution for learning
ets.record_tool_execution("malware_analyzer", success=True, execution_time=45.2)

# Get tool statistics
stats = ets.get_tool_statistics()
print(f"Tool statistics: {stats['total_tools']} tools, {stats['performance_metrics']['avg_success_rate']:.2f} avg success rate")
```

## 🚀 Enhanced Cybersecurity Agent

### Using the Unified Agent

```python
from bin.enhanced_cybersecurity_agent import EnhancedCybersecurityAgent, AgentConfiguration, WorkflowRequest
import uuid
import asyncio

# Initialize agent with enhanced configuration
config = AgentConfiguration(
    enable_enhanced_tool_selection=True,
    enable_dynamic_enrichment=True,
    enable_comprehensive_logging=True,
    enable_memory_integration=True,
    enable_incident_response=True,
    enable_threat_intelligence=True,
    local_tool_preference=True,
    auto_memory_recall=True
)

agent = EnhancedCybersecurityAgent(config)

# Start session
session_id = agent.start_session(user_id="analyst_001", workflow_type="malware_analysis")

# Create workflow request
request = WorkflowRequest(
    request_id=f"REQ-{uuid.uuid4().hex[:8].upper()}",
    workflow_type="malware_analysis",
    problem_description="Analyze suspicious file for malware",
    input_data={"file_path": "/tmp/suspicious.exe"},
    context={"priority": "high", "tags": ["malware", "analysis"]},
    user_id="analyst_001"
)

# Execute workflow
async def execute_workflow():
    result = await agent.execute_workflow(request)
    print(f"Workflow result: {result.success}")
    print(f"Tools used: {result.tools_used}")
    print(f"Execution time: {result.execution_time_ms:.2f}ms")
    print(f"Memory insights: {len(result.memory_insights.get('success_recommendations', []))} recommendations")
    return result

# Run workflow
result = asyncio.run(execute_workflow())

# Get agent status
status = agent.get_agent_status()
print(f"Agent status: {status['available_tools']} tools available")

# Get available workflows
workflows = agent.get_available_workflows()
print(f"Available workflows: {len(workflows)}")

# End session
agent.end_session(SessionStatus.COMPLETED)

# Cleanup
agent.cleanup()
```

## 🌐 Interactive Session Viewer

### Starting the Session Viewer

```bash
# Start the session viewer
python session-viewer/start-viewer.py

# Access at http://localhost:3001
# Features:
# - View session results and outputs
# - Download analysis files
# - Browse workflow execution history
# - Access comprehensive session logs
# - Interactive data visualization
```

### Session Viewer Features

1. **Dashboard**: Overview of all sessions and recent activity
2. **Session Details**: Detailed view of individual session results
3. **File Downloads**: Download analysis results and reports
4. **Log Viewer**: Browse comprehensive session logs
5. **Performance Analytics**: View performance metrics and statistics

## 📈 Performance Optimization Tips

### 1. Use Local Tools When Possible
```python
# Configure agent to prefer local tools
config = AgentConfiguration(
    local_tool_preference=True,
    enable_enhanced_tool_selection=True
)
```

### 2. Enable Memory Integration
```python
# Use memory to avoid redundant processing
config = AgentConfiguration(
    enable_memory_integration=True,
    auto_memory_recall=True
)
```

### 3. Batch Processing for Large Datasets
```python
# Use batch processing for large CSV files
workflow = dew.create_workflow(
    name="Large Dataset Processing",
    input_file="large_dataset.csv",
    output_file="processed_dataset.csv",
    batch_size=1000,  # Process 1000 rows at a time
    max_workers=4     # Use 4 parallel workers
)
```

### 4. Enable Comprehensive Logging for Debugging
```python
# Enable detailed logging for troubleshooting
config = AgentConfiguration(
    enable_comprehensive_logging=True,
    session_logging_level=LogLevel.DEBUG
)
```

## 🔧 Troubleshooting Common Issues

### 1. File Not Found Errors
```bash
# Ensure test files exist
ls -la usage-example-files/

# Check file permissions
chmod 644 usage-example-files/*.csv
```

### 2. Memory Issues with Large Datasets
```python
# Reduce batch size for memory-constrained systems
workflow = dew.create_workflow(
    batch_size=100,  # Smaller batches
    max_workers=2    # Fewer workers
)
```

### 3. Tool Selection Issues
```python
# Check available tools
status = agent.get_agent_status()
print(f"Available tools: {status['available_tools']}")

# Verify tool registration
stats = ets.get_tool_statistics()
print(f"Registered tools: {stats['total_tools']}")
```

### 4. Session Viewer Not Loading
```bash
# Check if session viewer is running
ps aux | grep "start-viewer.py"

# Restart session viewer
python session-viewer/start-viewer.py
```

## 📚 Additional Resources

- **[Malware Analysis Guide](MALWARE_ANALYSIS_GUIDE.md)** - Detailed malware analysis documentation
- **[Vulnerability Scanning Guide](VULNERABILITY_SCANNING_GUIDE.md)** - Network security assessment guide
- **[File Forensics Guide](FILE_FORENSICS_GUIDE.md)** - Digital forensics documentation
- **[AI Tools Guide](AI_TOOLS_GUIDE.md)** - AI-powered analysis features
- **[Network Analysis Guide](NETWORK_ANALYSIS_GUIDE.md)** - Network security analysis

## 🤝 Contributing

To add new usage examples:

1. Create synthetic test data in `usage-example-files/`
2. Add example commands to this guide
3. Test examples with the provided test data
4. Update documentation as needed

## 📞 Support

For questions or issues with usage examples:

1. Check the troubleshooting section above
2. Review the detailed documentation guides
3. Test with the provided synthetic data files
4. Verify environment setup and dependencies
