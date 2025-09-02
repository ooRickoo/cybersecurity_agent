# 🧠 Knowledge Graph Context Memory System

## Overview

The Knowledge Graph Context Memory system is the core intelligence engine of the Cybersecurity Agent, providing persistent, searchable, and contextually-aware storage for organizational knowledge, threat intelligence, compliance data, and operational insights.

## 🎯 What It Is

The Knowledge Graph Context Memory is a sophisticated, encrypted storage system that:

- **Stores Multi-Dimensional Data**: Handles CSV, JSON, YAML, and XML formats with automatic field normalization
- **Creates Intelligent Relationships**: Automatically establishes connections between related data points
- **Provides Context-Aware Retrieval**: Delivers relevant information based on query context
- **Ensures Data Security**: Uses enterprise-grade encryption (same as credential vault)
- **Supports Real-Time Adaptation**: Dynamically updates and adapts based on new information

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Knowledge Graph Context Memory            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Data Import   │  │  Memory Nodes   │  │Relationships│ │
│  │   Engine        │  │   (Encrypted)   │  │ (Graph DB)  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Field Normalize │  │  Search Engine  │  │ LLM Context │ │
│  │ & Standardize   │  │  (TF-IDF + ML)  │  │  Generator  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input Processing** → File format detection and validation
2. **Field Normalization** → Standardized naming and data types
3. **Entity Extraction** → Automatic identification of IPs, hashes, domains, CVEs
4. **Relationship Creation** → Intelligent connection mapping
5. **Encrypted Storage** → Secure persistence in SQLite database
6. **Context Retrieval** → Relevant information delivery for LLM calls

## 🔧 How It Works

### 1. Data Import Process

#### CSV Import
```bash
# Import organizational data
python cs_util_lg.py -memory organizational_data.csv -memory-config '{
  "create_relationships": true,
  "extract_entities": true,
  "normalize_fields": true
}'
```

**What Happens:**
- Field names are normalized (e.g., `IP_Address` → `ip_address`)
- Data types are standardized (strings converted to appropriate types)
- Entities are extracted (IPs, domains, hashes)
- Relationships are created based on common patterns

#### JSON Import
```bash
# Import application data with nested structure explosion
python cs_util_lg.py -memory applications.json -memory-config '{
  "flatten_nested": true,
  "create_relationships": true,
  "extract_entities": true,
  "normalize_fields": true,
  "max_depth": 5
}'
```

**What Happens:**
- Nested JSON structures are flattened into individual nodes
- Each flattened item becomes a memory node
- Relationships are created between related items
- Field normalization ensures consistency

### 2. Field Normalization

The system automatically standardizes field names and data types:

| Original Field | Normalized Field | Data Type |
|----------------|------------------|-----------|
| `IP_Address` | `ip_address` | String |
| `Created_Date` | `created_at` | ISO Date |
| `Is_Active` | `is_active` | Boolean |
| `Priority_Level` | `priority` | Integer |

### 3. Entity Extraction

Automatic identification of security-relevant entities:

- **IP Addresses**: `192.168.1.100`
- **Hashes**: `a1b2c3d4e5f6...`
- **Domains**: `example.com`
- **CVEs**: `CVE-2023-1234`
- **Ports**: `443`, `8080`

### 4. Relationship Creation

The system creates intelligent connections:

- **Entity-based**: Items sharing the same IP address
- **Semantic**: Items with similar content patterns
- **Temporal**: Items created around the same time
- **Categorical**: Items belonging to the same security domain

## 📊 Usage Examples

### Bulk Data Import

#### Import Multiple CSV Files
```bash
# Import organizational structure
python cs_util_lg.py -memory org_structure.csv

# Import threat intelligence
python cs_util_lg.py -memory threat_indicators.csv

# Import compliance policies
python cs_util_lg.py -memory compliance_policies.csv
```

#### Import Complex JSON Applications
```bash
# Import application portfolio with full relationship mapping
python cs_util_lg.py -memory applications.json -memory-config '{
  "flatten_nested": true,
  "create_relationships": true,
  "extract_entities": true,
  "normalize_fields": true,
  "max_depth": 10,
  "relationship_strength_threshold": 0.3
}'
```

### Memory Management

#### View Memory Statistics
```bash
# In interactive mode
memory stats

# Shows:
# - Total nodes and relationships
# - Memory by type and category
# - Encryption status
# - Import history
```

#### Search Memories
```bash
# Search for specific content
memory search "SQL injection"

# Search with filters
memory search "threat" -filters '{"category": "threat_intelligence"}'
```

#### Get LLM Context
```bash
# Get relevant context for LLM calls
memory context "analyze network security posture"

# Returns:
# - Relevant memories
# - Relationship context
# - Confidence scores
```

### Advanced Operations

#### Memory Encryption Management
```bash
# Check encryption status
memory encryption

# Re-encrypt existing data
memory re-encrypt
```

#### Cache Optimization
```bash
# View cache performance
memory cache

# Shows:
# - Cache hit rates
# - LLM calls saved
# - Optimization statistics
```

## 🗑️ Data Management

### Deleting Specific Objects

#### Delete by Content
```python
# Using the enhanced knowledge memory system
from bin.enhanced_knowledge_memory import enhanced_knowledge_memory

# Delete specific memory nodes
await enhanced_knowledge_memory.delete_memory_by_content("obsolete_threat_indicator")

# Delete by category
await enhanced_knowledge_memory.delete_memory_by_category("expired_policies")
```

#### Delete by Domain
```python
# Delete entire domains while preserving dependencies
await enhanced_knowledge_memory.delete_domain("legacy_systems", preserve_dependencies=True)

# This will:
# 1. Identify dependent relationships
# 2. Preserve critical connections
# 3. Update relationship indices
# 4. Maintain referential integrity
```

### Relationship Maintenance

The system automatically maintains relationship integrity:

- **Cascade Updates**: When a node is deleted, dependent relationships are updated
- **Index Rebuilding**: Relationship indices are automatically rebuilt
- **Dependency Tracking**: Critical dependencies are preserved
- **Referential Integrity**: Database constraints ensure data consistency

## 🔐 Security Features

### Encryption

- **Same Encryption as Credential Vault**: Uses identical encryption settings
- **Device-Bound Salt**: Unique salt per device for enhanced security
- **PBKDF2 Key Derivation**: 100,000 iterations for strong key generation
- **Fernet Encryption**: AES-128-CBC with HMAC for data integrity

### Access Control

- **Password-Based Encryption**: User-provided password required
- **Host Verification**: Device fingerprint validation
- **Session Isolation**: Memory contexts isolated between sessions
- **Audit Logging**: All operations logged for compliance

## 📈 Performance Optimization

### Caching Strategy

- **Decision Caching**: Planning decisions cached for 2 hours
- **Pattern Caching**: Analysis patterns cached for reuse
- **Workflow Caching**: Workflow configurations cached
- **Smart TTL**: Automatic cache expiration based on usage

### Local ML Processing

- **Text Classification**: TF-IDF + rule-based categorization
- **Pattern Detection**: Isolation Forest for anomaly detection
- **Similarity Analysis**: Cosine similarity with TF-IDF
- **Clustering**: K-means for data grouping
- **Feature Extraction**: Automatic feature importance analysis

### Batch Processing

- **Optimal Batch Size**: 10 items per batch for efficiency
- **Parallel Processing**: Multiple batches processed simultaneously
- **LLM Call Reduction**: 60-90% reduction in LLM calls
- **Performance Monitoring**: Real-time efficiency tracking

## 🚀 Advanced Features

### Real-Time Context Adaptation

The system continuously adapts based on:

- **New Data**: Automatically integrates new information
- **Usage Patterns**: Learns from query patterns
- **Relationship Evolution**: Updates connections based on new insights
- **Context Enrichment**: Enhances existing memories with new context

### Workflow Integration

The Knowledge Graph Context Memory integrates with:

- **Threat Hunting**: Provides historical context and patterns
- **Incident Response**: Delivers relevant incident history
- **Compliance Assessment**: Offers policy and regulation context
- **Risk Assessment**: Provides risk history and patterns
- **Bulk Data Import**: Handles large-scale data ingestion

### Export and Backup

```python
# Export memory data
await enhanced_knowledge_memory.export_memory(
    format="json",
    include_relationships=True,
    include_metadata=True
)

# Backup entire knowledge base
await enhanced_knowledge_memory.backup_knowledge_base(
    backup_path="backups/knowledge_base_backup.db",
    include_encryption_keys=False
)
```

## 🔍 Troubleshooting

### Common Issues

#### Import Failures
```bash
# Check file format
file organizational_data.csv

# Validate CSV structure
head -5 organizational_data.csv

# Check file permissions
ls -la organizational_data.csv
```

#### Memory Search Issues
```bash
# Check database status
memory stats

# Verify encryption
memory encryption

# Clear cache if needed
memory cache --clear
```

#### Performance Issues
```bash
# Check cache performance
memory cache

# Monitor memory usage
memory stats --detailed

# Optimize database
memory optimize
```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
# Set debug environment variable
export KNOWLEDGE_MEMORY_DEBUG=1

# Run with verbose output
python cs_util_lg.py -memory file.csv --verbose
```

## 📚 Best Practices

### Data Import

1. **Standardize Formats**: Use consistent field naming conventions
2. **Validate Data**: Ensure data quality before import
3. **Batch Processing**: Import related data together
4. **Relationship Planning**: Plan how data should connect
5. **Regular Updates**: Keep knowledge base current

### Memory Management

1. **Regular Cleanup**: Remove obsolete or duplicate data
2. **Relationship Review**: Periodically review and optimize relationships
3. **Performance Monitoring**: Track search and retrieval performance
4. **Backup Strategy**: Regular backups of knowledge base
5. **Access Control**: Limit access to sensitive information

### Security

1. **Strong Passwords**: Use complex encryption passwords
2. **Regular Updates**: Keep encryption keys current
3. **Access Logging**: Monitor all access to knowledge base
4. **Encryption Verification**: Regularly verify encryption status
5. **Secure Storage**: Protect backup files and encryption keys

## 🔮 Future Enhancements

### Planned Features

- **GraphQL API**: Modern API for external integrations
- **Advanced Analytics**: Machine learning-based insights
- **Real-Time Sync**: Multi-device synchronization
- **Advanced Search**: Semantic search with embeddings
- **Workflow Automation**: Automated knowledge processing

### Integration Roadmap

- **SIEM Integration**: Real-time threat intelligence feeds
- **Cloud Platforms**: Multi-cloud knowledge sharing
- **API Ecosystem**: Third-party tool integrations
- **Mobile Support**: Mobile knowledge access
- **Collaboration Tools**: Team knowledge sharing features

## 📞 Support

For technical support or feature requests:

- **Documentation**: Check this guide and related documentation
- **CLI Help**: Use `python cs_util_lg.py --help`
- **Memory Commands**: Use `memory` command in interactive mode
- **Debug Mode**: Enable debug logging for detailed troubleshooting

---

*This documentation covers the Knowledge Graph Context Memory system v2.0. For updates and additional features, check the project repository.*
