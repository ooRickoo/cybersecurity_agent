# 📁 Usage Example Files

This directory contains synthetic test data files for testing and demonstrating the Cybersecurity Agent's capabilities. All files contain **synthetic, non-malicious data** designed for safe testing and learning.

## 📋 Available Test Files

### **Core Security Analysis Files**

#### `sample_malware_indicators.csv`
- **Purpose**: Malware analysis and threat detection
- **Content**: File hashes, names, types, threat levels, and metadata
- **Use Case**: Test malware analysis workflows and threat classification
- **Example Usage**:
  ```bash
  python cs_util_lg.py -workflow malware_analysis \
    -problem "Analyze malware indicators and classify threat levels" \
    -input-file usage-example-files/sample_malware_indicators.csv \
    --output malware_analysis_results.csv
  ```

#### `sample_network_assets.csv`
- **Purpose**: Network vulnerability scanning and asset management
- **Content**: IP addresses, hostnames, services, open ports, vulnerabilities
- **Use Case**: Test network scanning and vulnerability assessment workflows
- **Example Usage**:
  ```bash
  python cs_util_lg.py -workflow vulnerability_scanning \
    -problem "Scan network assets and assess security posture" \
    -input-file usage-example-files/sample_network_assets.csv \
    --output vulnerability_scan_results.csv
  ```

#### `sample_incident_data.csv`
- **Purpose**: Incident response and security incident management
- **Content**: Incident IDs, descriptions, severity levels, affected systems
- **Use Case**: Test incident response workflows and incident analysis
- **Example Usage**:
  ```bash
  python cs_util_lg.py -workflow incident_response \
    -problem "Analyze security incidents and generate response recommendations" \
    -input-file usage-example-files/sample_incident_data.csv \
    --output incident_analysis_results.csv
  ```

#### `sample_threat_indicators.csv`
- **Purpose**: Threat intelligence and IOC analysis
- **Content**: IOCs (IPs, domains, URLs, hashes), threat levels, confidence scores
- **Use Case**: Test threat intelligence workflows and IOC correlation
- **Example Usage**:
  ```bash
  python cs_util_lg.py -workflow threat_intelligence \
    -problem "Analyze threat indicators and identify attack patterns" \
    -input-file usage-example-files/sample_threat_indicators.csv \
    --output threat_intelligence_results.csv
  ```

#### `sample_file_metadata.csv`
- **Purpose**: File forensics and digital evidence analysis
- **Content**: File metadata, hashes, timestamps, security assessments
- **Use Case**: Test file forensics workflows and metadata analysis
- **Example Usage**:
  ```bash
  python cs_util_lg.py -workflow file_forensics \
    -problem "Analyze file metadata and identify suspicious files" \
    -input-file usage-example-files/sample_file_metadata.csv \
    --output file_forensics_results.csv
  ```

#### `sample_vulnerability_data.csv`
- **Purpose**: Vulnerability management and CVE analysis
- **Content**: CVE IDs, descriptions, severity scores, affected software
- **Use Case**: Test vulnerability assessment and CVE correlation workflows
- **Example Usage**:
  ```bash
  python cs_util_lg.py -workflow vulnerability_scanning \
    -problem "Perform comprehensive vulnerability assessment with CVE analysis" \
    -input-file usage-example-files/sample_vulnerability_data.csv \
    --output comprehensive_vuln_assessment.csv
  ```

### **Patent Analysis Files**

#### `verified_cybersecurity_patents.csv`
- **Purpose**: AI-powered patent analysis and value proposition generation
- **Content**: Real cybersecurity patents with publication numbers
- **Use Case**: Test patent analysis workflows with AI insights
- **Example Usage**:
  ```bash
  python cs_util_lg.py -workflow patent_analysis \
    -problem "Analyze cybersecurity patents and generate value propositions" \
    -input-file usage-example-files/verified_cybersecurity_patents.csv \
    --output enriched_patents.csv
  ```

#### `real_cybersecurity_patents.csv`
- **Purpose**: Comprehensive patent analysis with API enrichment
- **Content**: Cybersecurity patents with detailed metadata
- **Use Case**: Test advanced patent analysis with API data enrichment
- **Example Usage**:
  ```bash
  python cs_util_lg.py -workflow patent_analysis \
    -problem "Take the input file list of cybersecurity patents, import into a local dataframe, iterate through the list of US patents # and associated publication #, add additional patent details into the dataframe from patent public APIs, adding a new column summarizing the value add for the patent in 1-3 lines and a new column categorizing the patents in a logical way. Then export the results to a csv file." \
    -input-file usage-example-files/real_cybersecurity_patents.csv \
    --output session_patent_analysis.csv
  ```

#### `real_patent_test.csv`
- **Purpose**: Basic patent analysis testing
- **Content**: Simple patent data for basic workflow testing
- **Use Case**: Test basic patent analysis functionality

#### `cybersecurity_patents_v2.csv`
- **Purpose**: Extended patent analysis dataset
- **Content**: Additional cybersecurity patents for comprehensive testing
- **Use Case**: Test patent analysis with larger datasets

## 🚀 Quick Start Examples

### **1. Malware Analysis**
```bash
# Analyze malware indicators
python cs_util_lg.py -workflow malware_analysis \
  -problem "Analyze malware indicators and classify threat levels" \
  -input-file usage-example-files/sample_malware_indicators.csv \
  --output malware_results.csv
```

### **2. Network Vulnerability Scanning**
```bash
# Scan network assets
python cs_util_lg.py -workflow vulnerability_scanning \
  -problem "Scan network assets and assess security posture" \
  -input-file usage-example-files/sample_network_assets.csv \
  --output network_scan_results.csv
```

### **3. Incident Response**
```bash
# Analyze security incidents
python cs_util_lg.py -workflow incident_response \
  -problem "Analyze security incidents and generate response recommendations" \
  -input-file usage-example-files/sample_incident_data.csv \
  --output incident_results.csv
```

### **4. Threat Intelligence**
```bash
# Analyze threat indicators
python cs_util_lg.py -workflow threat_intelligence \
  -problem "Analyze threat indicators and identify attack patterns" \
  -input-file usage-example-files/sample_threat_indicators.csv \
  --output threat_intel_results.csv
```

### **5. File Forensics**
```bash
# Analyze file metadata
python cs_util_lg.py -workflow file_forensics \
  -problem "Analyze file metadata and identify suspicious files" \
  -input-file usage-example-files/sample_file_metadata.csv \
  --output forensics_results.csv
```

### **6. Patent Analysis**
```bash
# Analyze cybersecurity patents
python cs_util_lg.py -workflow patent_analysis \
  -problem "Analyze cybersecurity patents and generate value propositions" \
  -input-file usage-example-files/verified_cybersecurity_patents.csv \
  --output patent_results.csv
```

## 🔧 Advanced Usage

### **Dynamic Data Enrichment**
```python
from bin.dynamic_enrichment_workflows import DynamicEnrichmentWorkflows, EnrichmentType, DataType

# Create enrichment workflow
dew = DynamicEnrichmentWorkflows()
workflow = dew.create_workflow(
    name="Malware Indicator Enrichment",
    input_file="usage-example-files/sample_malware_indicators.csv",
    output_file="enriched_malware_indicators.csv",
    enrichment_type=EnrichmentType.BATCH
)

# Add custom enrichment columns
dew.add_enrichment_column(
    workflow_id=workflow.workflow_id,
    column_name="threat_level_normalized",
    source_column="threat_level",
    enrichment_type="classification",
    enrichment_function="classify_threat_level",
    data_type=DataType.STRING
)

# Execute enrichment
result = dew.execute_workflow(workflow.workflow_id)
```

### **Enhanced Tool Selection**
```python
from bin.enhanced_tool_selection import EnhancedToolSelection, ToolSelectionCriteria, ProblemComplexity, ToolExecutionType

# Initialize tool selection
ets = EnhancedToolSelection()

# Create selection criteria
criteria = ToolSelectionCriteria(
    problem_type="malware_analysis",
    complexity=ProblemComplexity.MODERATE,
    required_capabilities=["static_analysis", "yara_scanning"],
    preferred_execution_type=ToolExecutionType.LOCAL
)

# Get tool recommendations
recommendations = ets.select_tools(criteria, max_tools=3)
```

### **Memory-Enhanced Workflows**
```python
from bin.enhanced_memory_integration import EnhancedMemoryIntegration, MemoryType

# Initialize memory system
emi = EnhancedMemoryIntegration()

# Store workflow experience
memory_id = emi.store_memory(
    memory_type=MemoryType.WORKFLOW_EXPERIENCE,
    content={"workflow": "malware_analysis", "success": True},
    tags=["malware", "analysis"],
    relevance_score=0.9
)

# Recall relevant memories
context = {"workflow_type": "malware_analysis"}
memories = emi.recall_contextual_memories(context, max_results=5)
```

## 📊 Data File Structure

### **CSV File Format**
All CSV files follow a consistent structure:
- **Headers**: Descriptive column names
- **Data Types**: Mixed data types (strings, numbers, dates, booleans)
- **Encoding**: UTF-8 encoding
- **Separator**: Comma-separated values
- **Quotes**: Fields with special characters are quoted

### **Sample Data Characteristics**
- **Synthetic Data**: All data is artificially generated for testing
- **Realistic Format**: Data follows real-world patterns and structures
- **Safe Content**: No actual malicious content or sensitive information
- **Comprehensive Coverage**: Covers various cybersecurity scenarios

## 🔍 File Validation

### **Data Integrity**
- All files contain valid CSV data
- No missing required fields
- Consistent data formats
- Proper encoding and escaping

### **Security Considerations**
- No actual malicious content
- No real IP addresses or domains
- No sensitive personal information
- Safe for testing in any environment

## 📚 Additional Resources

- **[Usage Examples Guide](../documentation/USAGE_EXAMPLES_GUIDE.md)** - Comprehensive usage examples
- **[Malware Analysis Guide](../documentation/MALWARE_ANALYSIS_GUIDE.md)** - Detailed malware analysis
- **[Vulnerability Scanning Guide](../documentation/VULNERABILITY_SCANNING_GUIDE.md)** - Network security assessment
- **[File Forensics Guide](../documentation/FILE_FORENSICS_GUIDE.md)** - Digital forensics
- **[AI Tools Guide](../documentation/AI_TOOLS_GUIDE.md)** - AI-powered analysis

## 🤝 Contributing

To add new test files:

1. Create synthetic data following the established patterns
2. Ensure data is safe and non-malicious
3. Include comprehensive metadata and examples
4. Update this README with file descriptions
5. Test with the provided workflows

## ⚠️ Important Notes

- **Synthetic Data Only**: All files contain artificially generated data
- **Safe for Testing**: No actual malicious content or sensitive information
- **Educational Purpose**: Designed for learning and testing the agent's capabilities
- **No Real Threats**: All threat indicators and malware samples are fictional
- **Environment Safe**: Safe to use in any environment without security concerns
