# Enhanced Features Integration Guide

This guide explains how to integrate and use the new enhanced features in the Cybersecurity Agent system.

## Table of Contents

1. [Enhanced Splunk Tools](#enhanced-splunk-tools)
2. [Enhanced Memory Management](#enhanced-memory-management)
3. [Output Distribution System](#output-distribution-system)
4. [Enhanced User Interaction](#enhanced-user-interaction)
5. [Integration Examples](#integration-examples)
6. [Configuration and Setup](#configuration-and-setup)

## Enhanced Splunk Tools

### Overview
The enhanced Splunk tools provide comprehensive data discovery, enrichment, and analysis capabilities using the specific query patterns you requested.

### Key Features

#### Data Discovery
```python
# Discover what data sources are feeding into Splunk indexes
# Uses the pattern: index!=* splunk!=* earliest=-4h | stats latest(_time) as last_seen, latest(_raw) as sample_message by index, sourcetype, app

from bin.enhanced_splunk_tools import EnhancedSplunkTools

# Initialize with your existing components
enhanced_splunk = EnhancedSplunkTools(
    splunk_integration=your_splunk_integration,
    session_manager=your_session_manager,
    memory_manager=your_memory_manager
)

# Discover data sources
result = enhanced_splunk.discover_data_sources(
    time_range="-4h",
    exclude_indexes=["*", "splunk*"],
    summary_fields=["index", "sourcetype", "app"]
)
```

#### Index Performance Analysis
```python
# Analyze Splunk index performance and usage patterns
result = enhanced_splunk.analyze_index_performance(time_range="-24h")

# This provides:
# - Performance metrics by index, sourcetype, and app
# - Data freshness assessment
# - Volume categorization
# - Recommendations for optimization
```

#### Custom Query Execution
```python
# Execute custom Splunk queries
custom_query = "index!=* splunk!=* earliest=-4h | stats latest(_time) as last_seen, latest(_raw) as sample_message, count by index, sourcetype, app, host, source"

result = enhanced_splunk.execute_custom_query(
    query=custom_query,
    max_results=1000
)
```

### Query Templates
The system includes predefined templates for common data discovery patterns:

- **Data Discovery**: Basic index and sourcetype discovery
- **Index Analysis**: Usage and performance analysis
- **Data Flow Analysis**: Volume and flow pattern analysis
- **Source Technology Discovery**: Technology identification
- **Data Quality Assessment**: Quality and completeness evaluation

### Data Enrichment
The system automatically identifies enrichment opportunities:
- IP addresses → Geolocation, threat intelligence, network info
- Domain names → DNS info, reputation, SSL certificates
- Email addresses → Domain info, breach data, reputation
- URLs → Domain info, category, reputation
- File paths → File type, hash info, reputation

## Enhanced Memory Management

### Overview
Enhanced memory management provides deletion capabilities, backup/restore functionality with encrypted zip files, and comprehensive lifecycle management.

### Key Features

#### Memory Deletion
```python
from bin.enhanced_memory_manager import EnhancedMemoryManager

enhanced_memory = EnhancedMemoryManager(
    memory_manager=your_memory_manager,
    session_manager=your_session_manager,
    credential_vault=your_credential_vault
)

# Delete single memory
result = enhanced_memory.delete_memory(
    memory_id="memory_123",
    domain="threat_intel",
    tier="medium_term"
)

# Delete memories by criteria
result = enhanced_memory.delete_memories_by_criteria({
    'domain': 'threat_intel',
    'tier': 'short_term',
    'data_type': 'ioc'
})

# Delete all memories (dangerous operation)
result = enhanced_memory.delete_all_memories(confirmation="DELETE_ALL_MEMORIES")
```

#### Encrypted Backup and Restore
```python
# Create encrypted backup
result = enhanced_memory.backup_memories(
    backup_name="threat_intel_backup",
    include_deleted=True  # Include deletion history
)

# The system will prompt for a password to encrypt the backup
# Backup is stored as an encrypted ZIP file in knowledge-objects/backup/

# Restore from backup
result = enhanced_memory.restore_memories(
    backup_file="knowledge-objects/backup/threat_intel_backup_20241201_143022.zip",
    password="your_backup_password"
)
```

#### Backup Management
```python
# List available backups
backups = enhanced_memory.list_backups()

# Delete backup file
result = enhanced_memory.delete_backup("backup_filename.zip")

# Get deletion history
history = enhanced_memory.get_deletion_history()
```

### Backup Features
- **AES Encryption**: All backups are encrypted with AES-256
- **Compression**: Uses LZMA compression for efficient storage
- **Metadata**: Includes backup metadata and deletion history
- **Organized Structure**: Groups memories by domain and tier
- **Password Protection**: User-defined encryption passwords

## Output Distribution System

### Overview
The output distribution system can send files to various destinations including CIFS/SMB, NFS, object storage, SSH, FTP, SCP, and streaming to TCP/UDP ports in multiple formats.

### Key Features

#### File Distribution
```python
from bin.output_distribution import OutputDistributor

distributor = OutputDistributor(
    session_manager=your_session_manager,
    credential_vault=your_credential_vault
)

# Distribute to CIFS/SMB share
result = distributor.distribute_to_cifs(
    file_path="session-outputs/analysis_report.pdf",
    share_path="//server/share",
    destination_path="reports/security_analysis.pdf"
)

# Distribute to NFS mount
result = distributor.distribute_to_nfs(
    file_path="session-outputs/threat_report.csv",
    mount_point="/mnt/nfs_share",
    destination_path="threat_intel/reports/"
)

# Distribute to S3
result = distributor.distribute_to_object_storage(
    file_path="session-outputs/compliance_report.pdf",
    bucket_name="security-reports",
    object_key="compliance/2024/dec/report.pdf",
    storage_type="s3"
)
```

#### Network Transfer
```python
# Transfer via SSH/SCP
result = distributor.distribute_via_ssh(
    file_path="session-outputs/incident_report.pdf",
    host="security-server.example.com",
    port=22,
    remote_path="/home/analyst/reports/",
    credentials={'username': 'analyst', 'password': 'password'},
    use_scp=True
)

# Transfer via FTP
result = distributor.distribute_via_ftp(
    file_path="session-outputs/threat_analysis.pdf",
    host="ftp.example.com",
    port=21,
    remote_path="security/reports/",
    credentials={'username': 'user', 'password': 'pass'}
)
```

#### Real-time Streaming
```python
# Stream data to network endpoint
result = distributor.stream_to_network(
    data={'threat_level': 'high', 'ioc_count': 15, 'affected_hosts': 3},
    host="192.168.1.100",
    port=514,
    protocol="udp",
    format_type="cef"  # Common Event Format
)

# Available formats: json, cef, csv, xml, syslog, leef, raw
```

### Supported Output Formats
- **JSON**: Structured data exchange
- **CEF**: Common Event Format for SIEM integration
- **CSV**: Tabular data for analysis tools
- **XML**: Structured markup for enterprise systems
- **Syslog**: Standard logging format
- **LEEF**: Log Event Extended Format
- **Raw**: Plain text output

### Credential Management
The system automatically retrieves credentials from the credential vault:
- CIFS credentials: `cifs_{server}`
- SSH credentials: `ssh_{host}`
- Storage credentials: `{storage_type}_{bucket_name}`

## Enhanced User Interaction

### Overview
The enhanced user interaction system provides fluid, natural conversation capabilities with intelligent context understanding and dynamic workflow adaptation.

### Key Features

#### Natural Language Processing
```python
from bin.enhanced_user_interaction import EnhancedUserInteraction

user_interaction = EnhancedUserInteraction(
    session_manager=your_session_manager,
    memory_manager=your_memory_manager,
    workflow_manager=your_workflow_manager
)

# Process user input
result = user_interaction.process_user_input(
    "I need to hunt for threats in my Splunk data from the last 4 hours"
)

# Result includes:
# - Intent analysis (threat_hunting)
# - Context-aware response
# - Workflow suggestions
# - Follow-up questions
# - Suggested actions
```

#### Intent Recognition
The system recognizes various intents:
- **threat_hunting**: Threat detection and hunting
- **policy_analysis**: Security policy analysis
- **incident_response**: Incident handling
- **vulnerability_assessment**: Vulnerability scanning
- **data_analysis**: Data processing and analysis
- **compliance_assessment**: Compliance evaluation
- **memory_management**: Knowledge base management
- **output_distribution**: File and data distribution

#### Context-Aware Responses
```python
# The system maintains conversation context
summary = user_interaction.get_conversation_summary()

# Context includes:
# - Current topic
# - Recent questions
# - User preferences
# - Workflow history
# - Suggested actions
```

### Conversation Flow
1. **Input Processing**: Analyze user input for intent
2. **Context Update**: Update conversation context
3. **Response Generation**: Generate context-aware response
4. **Workflow Suggestions**: Suggest relevant workflows
5. **Follow-up Questions**: Generate relevant questions
6. **Action Suggestions**: Suggest specific actions

## Integration Examples

### Complete Threat Hunting Workflow
```python
# 1. User asks about threat hunting
user_input = "I need to hunt for threats in my Splunk environment"

# 2. Process user input
interaction_result = user_interaction.process_user_input(user_input)

# 3. Based on intent, suggest Splunk data discovery
if interaction_result['intent'] == 'threat_hunting':
    # Discover data sources
    discovery_result = enhanced_splunk.discover_data_sources(
        time_range="-4h",
        exclude_indexes=["*", "splunk*"]
    )
    
    # Analyze discovered data
    if discovery_result['success']:
        # Store results in memory
        memory_manager.import_data(
            "splunk_discovery",
            discovery_result,
            domain="splunk_integration",
            tier="medium_term"
        )
        
        # Create visualizations
        # ... visualization code ...
        
        # Suggest next steps
        print("I've discovered your data sources. Would you like me to:")
        print("1. Analyze specific indexes for threats?")
        print("2. Search for specific IOCs?")
        print("3. Create threat hunting dashboards?")
```

### Memory Management Workflow
```python
# 1. User wants to backup memories
user_input = "I need to backup my threat intelligence data"

# 2. Process intent
interaction_result = user_interaction.process_user_input(user_input)

# 3. Execute backup
if interaction_result['intent'] == 'memory_management':
    backup_result = enhanced_memory.backup_memories(
        backup_name="threat_intel_backup",
        include_deleted=True
    )
    
    if backup_result['success']:
        print(f"Backup created successfully: {backup_result['backup_file']}")
        print(f"Size: {backup_result['backup_size']} bytes")
        
        # Optionally distribute backup
        distributor.distribute_to_cifs(
            file_path=backup_result['backup_file'],
            share_path="//backup-server/security",
            destination_path="memory_backups/"
        )
```

### Output Distribution Workflow
```python
# 1. User wants to send results
user_input = "Send my threat analysis report to the SIEM"

# 2. Process intent
interaction_result = user_interaction.process_user_input(user_input)

# 3. Distribute in appropriate format
if interaction_result['intent'] == 'output_distribution':
    # Stream to SIEM in CEF format
    result = distributor.stream_to_network(
        data=threat_analysis_data,
        host="siem.example.com",
        port=514,
        protocol="udp",
        format_type="cef"
    )
    
    if result['success']:
        print(f"Data streamed to SIEM: {result['bytes_sent']} bytes sent")
```

## Configuration and Setup

### Dependencies Installation
```bash
# Install new dependencies
pip install pyzipper paramiko smbclient smbprotocol

# Install spaCy model (if not already installed)
python -m spacy download en_core_web_sm
```

### Credential Vault Setup
```python
# Add credentials for various services
credential_vault.add_credential(
    "cifs_backup-server",
    {
        "username": "backup_user",
        "password": "secure_password",
        "domain": "EXAMPLE"
    }
)

credential_vault.add_credential(
    "ssh_security-server",
    {
        "username": "analyst",
        "password": "secure_password"
    }
)

credential_vault.add_credential(
    "s3_security-reports",
    {
        "access_key": "your_access_key",
        "secret_key": "your_secret_key",
        "region": "us-east-1"
    }
)
```

### Directory Structure
```
knowledge-objects/
├── backup/                    # Encrypted memory backups
├── context_memory/           # Context memory database
└── deletion_history.json     # Deletion tracking

session-outputs/              # Session outputs
session-logs/                 # Session logs
```

### Environment Variables
```bash
# Enable encryption (default: true)
ENCRYPTION_ENABLED=true

# Encryption password hash
ENCRYPTION_PASSWORD_HASH=your_hashed_password

# Backup directory
MEMORY_BACKUP_DIR=knowledge-objects/backup
```

## Best Practices

### Splunk Integration
1. **Use the specific query pattern** for data discovery
2. **Leverage enrichment opportunities** for better threat intelligence
3. **Store discovery results** in memory for future reference
4. **Use performance analysis** to optimize index usage

### Memory Management
1. **Regular backups** of important knowledge
2. **Use descriptive backup names** for easy identification
3. **Include deletion history** for audit purposes
4. **Secure backup passwords** and store them safely

### Output Distribution
1. **Choose appropriate formats** for destination systems
2. **Use credential vault** for secure authentication
3. **Test connections** before production use
4. **Monitor distribution history** for troubleshooting

### User Interaction
1. **Use natural language** - the system understands various phrasings
2. **Provide context** for better workflow suggestions
3. **Follow suggested actions** for optimal results
4. **Use follow-up questions** to refine requirements

## Troubleshooting

### Common Issues

#### Splunk Connection Issues
- Verify credentials in vault
- Check network connectivity
- Ensure proper permissions

#### Memory Backup Issues
- Verify backup directory permissions
- Check available disk space
- Ensure encryption password is correct

#### Distribution Issues
- Verify destination credentials
- Check network connectivity
- Ensure proper file permissions

#### User Interaction Issues
- Verify spaCy model installation
- Check conversation context
- Reset context if needed

### Debug Mode
Enable debug logging for troubleshooting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Conclusion

The enhanced features provide a comprehensive cybersecurity analysis platform with:
- **Intelligent data discovery** in Splunk environments
- **Secure memory management** with encrypted backups
- **Flexible output distribution** to various systems
- **Natural user interaction** for seamless operation

These features work together to create a powerful, user-friendly cybersecurity analysis environment that focuses on solving problems rather than managing tools.
