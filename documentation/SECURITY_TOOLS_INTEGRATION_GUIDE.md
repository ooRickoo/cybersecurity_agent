# Security Tools Integration Guide

## Overview

This guide covers the integration of comprehensive security tools into the Cybersecurity Agent's MCP framework and agentic workflow system. The new tools include:

- **Host Scanning Tools**: nmap-based network discovery and security assessment
- **Hashing Tools**: Cryptographic hashing for forensics and data integrity
- **MCP Integration**: Dynamic tool discovery and execution
- **Query Path Integration**: Context-aware tool selection

## New Security Tools

### 1. Host Scanning Tools (`host_scanning_tools.py`)

#### Core Components
- **NmapScanner**: Low-level nmap integration with XML parsing
- **HostScanningManager**: High-level management with scan templates
- **Scan Types**: Quick, stealth, comprehensive, vulnerability, OS detection, service detection, topology
- **Scan Intensity**: Configurable timing templates (T0-T5)

#### Available Scan Templates
- `quick_audit`: Fast port scan for basic network audit
- `security_assessment`: Comprehensive security assessment with vulnerability detection
- `network_discovery`: Network topology discovery and mapping
- `service_inventory`: Detailed service and version detection
- `stealth_scan`: Stealthy scan for sensitive environments

#### Key Features
- Automatic nmap detection and path resolution
- XML output parsing for structured results
- Host information extraction (IP, hostname, OS, ports, services)
- Security risk analysis and recommendations
- Performance tracking and scan history

### 2. Hashing Tools (`hashing_tools.py`)

#### Core Components
- **HashCalculator**: Core hashing functionality
- **HashingManager**: High-level management with hash templates
- **Supported Algorithms**: MD5, SHA1, SHA224, SHA256, SHA384, SHA512, SHA3 variants, BLAKE2B, BLAKE2S, RIPEMD160, WHIRLPOOL

#### Available Hash Templates
- `quick_verification`: MD5 for basic verification
- `secure_verification`: SHA256 for critical data
- `maximum_security`: SHA512 for sensitive data
- `legacy_compatibility`: SHA1 for older systems
- `fast_processing`: BLAKE2B for high-performance

#### Key Features
- File and string hashing with progress tracking
- Directory batch hashing with parallel processing
- HMAC generation for data authentication
- Hash verification and integrity checking
- Performance analytics and pattern analysis

### 3. Security MCP Integration (`security_mcp_integration.py`)

#### Core Components
- **SecurityMCPIntegrationLayer**: Unified MCP interface for all security tools
- **SecurityToolsQueryPathIntegration**: Query Path integration for dynamic tool selection
- **Tool Registry**: Comprehensive tool discovery and metadata

#### MCP Tools Available
- `quick_host_scan`: Quick port scanning
- `security_assessment_scan`: Comprehensive security assessment
- `network_discovery_scan`: Network topology mapping
- `hash_string`: String hashing
- `hash_file`: File hashing
- `verify_hash`: Hash verification
- `create_hmac`: HMAC generation
- `batch_hash_files`: Batch file hashing

## Integration with Agentic Workflow System

### 1. MCP Server Integration

The security tools are automatically discovered and registered by the MCP server in `cs_ai_tools.py`:

```python
# Security tools are discovered automatically
if (hasattr(self.tool_manager, 'security_tools') and 
    self.tool_manager.security_tools and 
    ('security' not in self._registered_categories or force)):
    self._register_security_tools(self.tool_manager.security_tools)
    self._registered_categories.add('security')
```

### 2. Query Path Integration

The Query Path can now discover and select security tools based on problem descriptions:

```python
from knowledge_objects.security_mcp_integration import SecurityToolsQueryPathIntegration

# Initialize integration
query_integration = SecurityToolsQueryPathIntegration(security_mcp_integration)

# Discover relevant tools for a problem
problem = "Scan network for vulnerabilities and hash suspicious files"
context = {"targets": ["192.168.1.1", "192.168.1.100"]}

relevant_tools = query_integration.discover_relevant_tools(problem, context)
# Returns: ["security_assessment_scan", "hash_file", "batch_hash_files"]
```

### 3. Runner Agent Integration

The Runner Agent can execute security tools with context-aware parameters:

```python
# Execute security assessment scan
scan_result = await security_mcp_integration.execute_tool(
    "security_assessment_scan",
    {
        "targets": ["192.168.1.1"],
        "intensity": "normal"
    }
)

# Execute file hashing
hash_result = await security_mcp_integration.execute_tool(
    "hash_file",
    {
        "file_path": "/path/to/suspicious/file",
        "algorithm": "sha256"
    }
)
```

## Usage Examples

### 1. Network Security Assessment Workflow

```python
async def network_security_workflow():
    """Complete network security assessment workflow."""
    
    # Initialize security tools
    security_tools = SecurityMCPIntegrationLayer()
    
    # 1. Network discovery
    discovery_result = await security_tools.execute_tool(
        "network_discovery_scan",
        {"targets": ["192.168.1.0/24"], "include_ports": True}
    )
    
    # 2. Security assessment of discovered hosts
    if discovery_result["success"]:
        hosts = discovery_result["hosts_found"]
        for host in hosts:
            assessment_result = await security_tools.execute_tool(
                "security_assessment_scan",
                {"targets": [host], "intensity": "aggressive"}
            )
            
            # 3. Hash suspicious files if found
            if assessment_result.get("security_risks"):
                for risk in assessment_result["security_risks"]:
                    if risk["severity"] == "high":
                        hash_result = await security_tools.execute_tool(
                            "hash_file",
                            {"file_path": risk["file_path"], "algorithm": "sha256"}
                        )
    
    return {
        "discovery": discovery_result,
        "assessments": assessment_results,
        "file_hashes": hash_results
    }
```

### 2. Forensics Analysis Workflow

```python
async def forensics_analysis_workflow():
    """Forensics analysis workflow using hashing tools."""
    
    security_tools = SecurityMCPIntegrationLayer()
    
    # 1. Hash multiple files for integrity checking
    file_paths = [
        "/evidence/file1.exe",
        "/evidence/file2.dll",
        "/evidence/file3.txt"
    ]
    
    batch_hash_result = await security_tools.execute_tool(
        "batch_hash_files",
        {"file_paths": file_paths, "algorithm": "sha256"}
    )
    
    # 2. Create HMAC for authentication
    hmac_result = await security_tools.execute_tool(
        "create_hmac",
        {
            "data": "Evidence collection report",
            "key": "secret_forensics_key",
            "algorithm": "sha256"
        }
    )
    
    # 3. Verify specific file hash
    verification_result = await security_tools.execute_tool(
        "verify_hash",
        {
            "original_hash": "abc123...",
            "computed_hash": batch_hash_result["results"][0]["hash_value"],
            "algorithm": "sha256"
        }
    )
    
    return {
        "batch_hashes": batch_hash_result,
        "hmac": hmac_result,
        "verification": verification_result
    }
```

### 3. Dynamic Tool Selection

```python
async def dynamic_security_workflow(problem_description: str, context: Dict[str, Any]):
    """Dynamic security workflow based on problem description."""
    
    # Initialize integrations
    security_tools = SecurityMCPIntegrationLayer()
    query_integration = SecurityToolsQueryPathIntegration(security_tools)
    
    # 1. Discover relevant tools
    relevant_tools = query_integration.discover_relevant_tools(problem_description, context)
    
    # 2. Score tools by relevance
    tool_scores = {}
    for tool_id in relevant_tools:
        score = query_integration.get_tool_relevance_score(
            tool_id, problem_description, context
        )
        tool_scores[tool_id] = score
    
    # 3. Execute tools in order of relevance
    results = {}
    for tool_id, score in sorted(tool_scores.items(), key=lambda x: x[1], reverse=True):
        if score > 0.5:  # Only execute highly relevant tools
            try:
                result = await security_tools.execute_tool(tool_id, context)
                results[tool_id] = result
            except Exception as e:
                results[tool_id] = {"error": str(e)}
    
    return {
        "problem": problem_description,
        "tool_scores": tool_scores,
        "execution_results": results
    }
```

## Configuration and Setup

### 1. Dependencies

Ensure the following dependencies are installed:

```bash
# For host scanning
sudo apt-get install nmap  # Ubuntu/Debian
brew install nmap          # macOS
# Windows: Download from nmap.org

# Python dependencies (already in requirements.txt)
pip install asyncio
pip install pathlib
```

### 2. Environment Variables

```bash
# Optional: Customize scan behavior
export NMAP_TIMING_TEMPLATE="T3"  # Normal timing
export NMAP_SCRIPT_PATH="/usr/share/nmap/scripts"  # Custom scripts
```

### 3. Tool Manager Integration

Add security tools to your tool manager:

```python
from knowledge_objects.security_mcp_integration import SecurityMCPIntegrationLayer

class ToolManager:
    def __init__(self):
        # ... other managers ...
        self.security_tools = SecurityMCPIntegrationLayer()
    
    def get_security_tools(self):
        return self.security_tools
```

## Performance and Monitoring

### 1. Performance Metrics

The security tools provide comprehensive performance tracking:

```python
# Get tool statistics
stats = security_tools.get_tool_statistics()
print(f"Total executions: {stats['performance']['total_executions']}")
print(f"Success rate: {stats['performance']['successful_executions'] / stats['performance']['total_executions']}")
print(f"Average execution time: {stats['performance']['average_execution_time']:.3f}s")
```

### 2. Tool Usage Analytics

```python
# Analyze tool usage patterns
usage_counts = stats['performance']['tool_usage_counts']
for tool_id, count in sorted(usage_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{tool_id}: {count} executions")
```

### 3. Hash Pattern Analysis

```python
# Analyze hashing patterns
hash_analysis = hashing_manager.analyze_hash_patterns()
print(f"Most used algorithm: {max(hash_analysis['algorithm_usage'].items(), key=lambda x: x[1])[0]}")
print(f"Error rate: {hash_analysis['error_patterns']['error_rate']:.2%}")
```

## Security Considerations

### 1. Network Scanning

- **Permission**: Ensure you have authorization to scan target networks
- **Stealth**: Use appropriate timing templates to avoid detection
- **Logging**: All scan activities are logged for audit purposes
- **Rate Limiting**: Respect network policies and rate limits

### 2. File Hashing

- **Integrity**: Use strong algorithms (SHA256+) for critical files
- **Verification**: Always verify hashes against known good values
- **Storage**: Store hashes securely and separately from files
- **Compliance**: Follow organizational policies for evidence handling

### 3. HMAC Usage

- **Key Management**: Use strong, randomly generated keys
- **Key Rotation**: Implement regular key rotation policies
- **Algorithm Selection**: Choose algorithms based on security requirements

## Troubleshooting

### 1. Common Issues

#### Nmap Not Found
```bash
# Check nmap installation
which nmap
nmap --version

# Install if missing
sudo apt-get install nmap  # Ubuntu/Debian
brew install nmap          # macOS
```

#### Permission Denied
```bash
# For network scanning, may need elevated privileges
sudo python3 your_script.py

# Or use unprivileged scans
nmap -sS -Pn target  # SYN scan without ping
```

#### Hash Algorithm Not Supported
```python
# Check available algorithms
from hashlib import algorithms_available
print(algorithms_available)

# Use fallback algorithm
algorithm = HashAlgorithm.SHA256  # Always available
```

### 2. Performance Optimization

- **Parallel Processing**: Use batch operations for multiple files
- **Chunk Size**: Adjust file hashing chunk size for large files
- **Scan Intensity**: Choose appropriate nmap timing templates
- **Resource Limits**: Monitor system resources during intensive operations

## Future Enhancements

### 1. Planned Features

- **Vulnerability Database Integration**: CVE lookup and scoring
- **Threat Intelligence**: Integration with threat feeds
- **Compliance Reporting**: Automated compliance checking
- **Machine Learning**: Anomaly detection in scan results

### 2. Integration Opportunities

- **SIEM Integration**: Send results to security information systems
- **Ticketing Systems**: Create tickets for discovered issues
- **Asset Management**: Update asset inventory with scan results
- **Risk Management**: Automated risk scoring and prioritization

## Conclusion

The new security tools provide a comprehensive foundation for cybersecurity analysis within the agentic workflow system. They enable:

- **Dynamic Discovery**: Tools are automatically discovered and registered
- **Context-Aware Selection**: Tools are chosen based on problem and context
- **Comprehensive Coverage**: Host scanning, hashing, and forensics capabilities
- **Performance Monitoring**: Built-in analytics and optimization
- **Integration Ready**: Seamless integration with existing MCP framework

These tools transform the Cybersecurity Agent into a powerful platform for automated security assessment, forensics analysis, and incident response workflows.

---

**Next Steps**: 
1. Test the tools with your specific use cases
2. Integrate with your existing security workflows
3. Customize scan templates and hash algorithms as needed
4. Monitor performance and optimize based on usage patterns
