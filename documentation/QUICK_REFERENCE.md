# Cybersecurity Agent - Quick Reference Guide

## 🚀 Quick Start

### 1. Start the System
```bash
# Option 1: Use startup script (recommended)
./start.sh

# Option 2: Direct Python execution
python3 cs_util_lg.py
```

### 2. Basic Commands
```bash
# List all available tools
python3 cs_util_lg.py -list-workflows

# Execute a specific tool
python3 cs_util_lg.py execute-tool --tool "quick_host_scan" --params '{"targets": ["192.168.1.1"]}'

# Start workflow
python3 cs_util_lg.py -workflow --mode automated --input data.csv --prompt "Analyze security"
```

## 🔧 Essential MCP Tools

### Security Assessment
```bash
# Host scanning
python3 cs_util_lg.py execute-tool --tool "quick_host_scan" --params '{"targets": ["192.168.1.0/24"]}'

# Cryptography evaluation
python3 cs_util_lg.py execute-tool --tool "evaluate_algorithm_security" --params '{"algorithm": "AES-256", "key_length": 256, "mode": "GCM"}'

# File hashing
python3 cs_util_lg.py execute-tool --tool "hash_file" --params '{"file_path": "suspicious_file.exe", "algorithm": "sha256"}'
```

### Network Analysis
```bash
# Ping host
python3 cs_util_lg.py execute-tool --tool "ping_host" --params '{"target": "192.168.1.1"}'

# DNS lookup
python3 cs_util_lg.py execute-tool --tool "dns_lookup" --params '{"domain": "example.com"}'

# Port scan
python3 cs_util_lg.py execute-tool --tool "port_scan" --params '{"target": "192.168.1.1", "ports": [80, 443, 22]}'
```

### Data Management
```bash
# Convert file format
python3 cs_util_lg.py execute-tool --tool "convert_file" --params '{"input_file": "data.csv", "output_file": "report.html", "output_format": "html"}'

# SQLite query
python3 cs_util_lg.py execute-tool --tool "sqlite_query" --params '{"query": "SELECT * FROM users WHERE role = \"admin\""}'
```

## 🧠 Context Memory Operations

### Store Information
```python
from bin.enhanced_context_memory import EnhancedContextMemoryManager

memory = EnhancedContextMemoryManager()

# Short-term (session-specific)
memory.store_short_term("session_123", "scan_results", {"targets": ["192.168.1.1"], "findings": ["open_port_22"]})

# Medium-term (important session data)
memory.store_medium_term("session_123", "vulnerabilities", {"critical": 2, "high": 5, "medium": 8})

# Long-term (persistent knowledge)
memory.store_long_term("networks", "subnet_192_168_1", {"subnet": "192.168.1.0/24", "devices": 15})
```

### Retrieve Information
```python
# Get session data
session_data = memory.get_session_memories("session_123")

# Get domain data
network_data = memory.get_domain_data("networks", "subnet_192_168_1")

# Search across domains
results = memory.search_all_domains("vulnerability")
```

### Memory Management
```python
# Clear session
memory.clear_session_memory("session_123")

# Backup
memory.backup_knowledge_base("backup_2024_01")

# Restore
memory.restore_knowledge_base("backup_2024_01")
```

## 🔄 Workflow Modes

### Automated Mode
```bash
python3 cs_util_lg.py -workflow \
  --mode automated \
  --input threat_data.csv \
  --prompt "Analyze and categorize security threats by severity" \
  --output enriched_threats.csv
```

### Manual Mode
```bash
python3 cs_util_lg.py -workflow --mode manual
```

### Hybrid Mode
```bash
python3 cs_util_lg.py -workflow \
  --mode hybrid \
  --input network_data.csv \
  --prompt "Identify potential security issues"
```

## 📊 Tool Categories

### Security Tools
- **Host Scanning**: `quick_host_scan`, `security_assessment_scan`, `network_discovery_scan`
- **Hashing**: `hash_string`, `hash_file`, `verify_hash`, `create_hmac`, `batch_hash_files`
- **Cryptography**: `evaluate_algorithm_security`, `evaluate_implementation_security`, `evaluate_key_quality`

### Network Tools
- **Analysis**: `ping_host`, `dns_lookup`, `traceroute`, `port_scan`
- **Statistics**: `get_netstat`, `get_arp_table`

### Data Tools
- **File Operations**: `convert_file`, `write_html_report`, `write_markdown_report`
- **Compression**: `extract_archive`, `create_archive`, `list_archive_contents`
- **Database**: `sqlite_query`, `sqlite_execute`, `sqlite_backup`

### AI/ML Tools
- **OpenAI**: `openai_reasoning`, `openai_categorize`, `openai_summarize`
- **Local ML**: `classify_text`, `extract_entities`, `sentiment_analysis`

## 🎯 Common Use Cases

### 1. Security Assessment
```bash
# Scan network for vulnerabilities
python3 cs_util_lg.py execute-tool --tool "security_assessment_scan" --params '{"targets": ["192.168.1.0/24"], "intensity": "normal"}'

# Evaluate cryptographic implementation
python3 cs_util_lg.py execute-tool --tool "evaluate_implementation_security" --params '{"implementation_data": {"padding": "OAEP", "iv_generation": "random"}}'
```

### 2. Incident Response
```bash
# Hash suspicious files
python3 cs_util_lg.py execute-tool --tool "batch_hash_files" --params '{"file_paths": ["file1.exe", "file2.dll"], "algorithm": "sha256"}'

# Analyze network traffic
python3 cs_util_lg.py execute-tool --tool "port_scan" --params '{"target": "192.168.1.100", "ports": [21, 22, 23, 25, 80, 443]}'
```

### 3. Compliance and Reporting
```bash
# Generate HTML report
python3 cs_util_lg.py execute-tool --tool "write_html_report" --params '{"data": {"findings": ["vuln1", "vuln2"]}, "output_file": "security_report.html", "title": "Security Assessment Report"}'

# Export data to CSV
python3 cs_util_lg.py execute-tool --tool "convert_file" --params '{"input_file": "security_data.json", "output_file": "compliance_report.csv", "output_format": "csv"}'
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure bin directory is in Python path
   export PYTHONPATH="${PYTHONPATH}:${PWD}/bin"
   
   # Or use startup script
   ./start.sh
   ```

2. **Tool Execution Failures**
   ```bash
   # Check tool parameters
   python3 cs_util_lg.py -list-workflows --detailed
   
   # Verify dependencies
   pip3 install -r requirements.txt
   ```

3. **Memory Issues**
   ```bash
   # Check disk space
   df -h
   
   # Verify database permissions
   ls -la knowledge-objects/
   ```

### Debug Mode
```bash
# Enable detailed logging
export PYTHONPATH="${PYTHONPATH}:${PWD}/bin"
python3 -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from bin.cs_ai_tools import tool_manager
print('Tools loaded successfully')
"
```

## 📚 Additional Resources

- **Full Documentation**: `documentation/` folder
- **Integration Guides**: See documentation for detailed tool integration
- **Examples**: Check `session-outputs/` for workflow examples
- **Logs**: Review `session-logs/` for detailed execution logs

## 🆘 Getting Help

1. Check this quick reference
2. Review the main README.md
3. Check session logs for error details
4. Verify all dependencies are installed
5. Ensure proper file permissions

---

**Quick Start Command**: `./start.sh`

**Main CLI**: `python3 cs_util_lg.py`

**Status**: ✅ **Ready for Production Use**
