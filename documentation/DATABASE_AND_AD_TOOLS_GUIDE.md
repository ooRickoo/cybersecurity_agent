# Database and Active Directory Tools Guide

This guide covers the comprehensive database connectivity and Active Directory querying tools integrated with the Cybersecurity Agent. These tools provide read-only access to various data sources with intelligent credential management and NLP-friendly interfaces.

## 🗄️ Database Tools Overview

### Supported Database Types

- **Microsoft SQL Server** (MSSQL, Azure SQL)
- **MongoDB** (including Azure Cosmos DB)
- **PostgreSQL** (planned)
- **MySQL** (planned)
- **SQLite** (local data storage)

### Key Features

- 🔐 **Intelligent Credential Management**: Credentials stored securely in vault
- 🧠 **Memory Integration**: Connection info and schemas stored in searchable memory
- 🔍 **Schema Discovery**: Automatic database structure analysis
- 📊 **Data Export**: Export to DataFrames or SQLite for local analysis
- 🔗 **Relationship Analysis**: Automatic foreign key and join suggestions
- 🚫 **Read-Only Access**: Safe, non-destructive operations only

## 🔗 Active Directory Tools Overview

### Supported Operations

- **User Enumeration**: List all users with comprehensive properties
- **Group Management**: List groups and analyze memberships
- **Permission Analysis**: Security assessment of user accounts
- **Flexible Search**: Multi-field user search capabilities
- **Data Export**: CSV export for external analysis

### Key Features

- 🔐 **LDAP Integration**: Support for both ldap3 and python-ldap
- 🛡️ **Security Analysis**: Automatic detection of security misconfigurations
- 📋 **Comprehensive Properties**: Full user and group attribute retrieval
- 🔍 **Flexible Queries**: Search across multiple user attributes
- 💾 **Export Capabilities**: CSV export for external tools

## 🚀 Getting Started

### 1. Database Connections

#### Connect to MSSQL/Azure SQL

```python
# The agent will automatically handle credential collection
connect_to_database(
    database_type="mssql",
    host="your-server.database.windows.net",
    port=1433,
    database_name="your_database",
    username="your_username",
    password="your_password",
    reason="Security audit of user access patterns"
)
```

#### Connect to MongoDB

```python
connect_to_database(
    database_type="mongodb",
    host="your-mongodb-server",
    port=27017,
    database_name="your_database",
    username="your_username",
    password="your_password",
    reason="Analysis of document collections for security insights"
)
```

### 2. Active Directory Connections

#### Connect to On-Premises AD

```python
connect_to_active_directory(
    domain="yourdomain.com",
    server="dc01.yourdomain.com",
    port=389,  # 636 for LDAPS
    username="admin@yourdomain.com",
    password="your_password",
    use_ssl=True,
    reason="Security assessment of user permissions and group memberships"
)
```

## 📊 Database Operations

### Schema Discovery

```python
# Automatically discover database structure
discover_database_schema(
    connection_id="your_connection_id",
    reason="Understanding database structure for security analysis"
)
```

### Query Execution

```python
# Execute custom queries
query_database(
    connection_id="your_connection_id",
    query="SELECT * FROM users WHERE last_login > '2024-01-01'",
    limit=1000,
    reason="Identifying recently active users for access review"
)
```

### Table Exploration

```python
# Explore specific tables in detail
explore_table(
    connection_id="your_connection_id",
    table_name="users",
    include_sample_data=True,
    sample_limit=50,
    reason="Analyzing user table structure and sample data"
)
```

### Data Export

```python
# Export to pandas DataFrame
export_to_dataframe(
    connection_id="your_connection_id",
    query="SELECT username, email, last_login FROM users",
    dataframe_name="user_activity",
    reason="Local analysis of user activity patterns"
)

# Export to SQLite
export_to_sqlite(
    connection_id="your_connection_id",
    query="SELECT * FROM audit_logs WHERE timestamp > '2024-01-01'",
    table_name="audit_data",
    reason="Creating local copy for detailed analysis"
)
```

### Relationship Analysis

```python
# Analyze table relationships
analyze_database_relationships(
    connection_id="your_connection_id",
    tables=["users", "roles", "permissions"],
    reason="Understanding data relationships for security modeling"
)
```

## 👥 Active Directory Operations

### User Enumeration

```python
# Get all users
get_all_ad_users(
    connection_id="your_connection_id",
    reason="Comprehensive user inventory for security assessment"
)
```

### Group Analysis

```python
# Get all groups
get_all_ad_groups(
    connection_id="your_connection_id",
    include_members=True,
    reason="Analyzing group memberships for access control review"
)

# Get specific group members
get_ad_group_members(
    connection_id="your_connection_id",
    group_name="Domain Admins",
    include_user_details=True,
    reason="Identifying privileged group members for security review"
)
```

### User Search

```python
# Search for specific users
search_ad_users(
    connection_id="your_connection_id",
    search_term="admin",
    search_fields=["displayName", "sAMAccountName", "mail"],
    reason="Finding administrative accounts for security review"
)
```

### Permission Analysis

```python
# Analyze user permissions
analyze_ad_user_permissions(
    connection_id="your_connection_id",
    username="admin_user",
    include_group_details=True,
    reason="Comprehensive security assessment of administrative account"
)
```

### Data Export

```python
# Export users to CSV
export_ad_users_to_csv(
    connection_id="your_connection_id",
    file_path="ad_users_export.csv",
    reason="Creating external report for compliance documentation"
)
```

## 🔐 Security Features

### Credential Management

- **Secure Storage**: Credentials stored in encrypted vault
- **Environment Variables**: Fallback to secure environment storage
- **Automatic Cleanup**: Credentials removed after session
- **Access Logging**: All credential access logged for audit

### Access Control

- **Read-Only Operations**: No destructive operations allowed
- **Connection Limits**: Maximum concurrent connections enforced
- **Session Timeouts**: Automatic disconnection after inactivity
- **Audit Logging**: All operations logged with reasons

### Data Protection

- **No Sensitive Data Storage**: Raw credentials never stored in memory
- **Encrypted Communication**: SSL/TLS for all database connections
- **Secure Disposal**: Memory cleared after operations
- **Access Validation**: All connections validated before use

## 🧠 Memory Integration

### Stored Information

- **Connection Configurations**: Reusable connection settings
- **Database Schemas**: Cached schema information
- **Query Patterns**: Frequently used query templates
- **Relationship Maps**: Table and field relationships

### Memory Categories

- `DATABASE_CONNECTIONS`: Connection configurations and status
- `DATABASE_SCHEMAS`: Discovered database structures
- `ACTIVE_DIRECTORY_CONNECTIONS`: AD connection information
- `QUERY_PATTERNS`: Common query templates and results

## 📋 Best Practices

### 1. Connection Management

- **Reuse Connections**: Don't create new connections for each operation
- **Monitor Status**: Regularly check connection health
- **Clean Disconnect**: Always disconnect when finished
- **Resource Limits**: Be mindful of connection pool limits

### 2. Query Optimization

- **Use Limits**: Always limit large result sets
- **Specific Fields**: Select only needed columns
- **Index Awareness**: Understand table indexing for performance
- **Batch Operations**: Group related queries together

### 3. Security Considerations

- **Principle of Least Privilege**: Use minimal required permissions
- **Audit Logging**: Log all operations with business reasons
- **Data Classification**: Be aware of data sensitivity levels
- **Compliance Requirements**: Follow organizational data handling policies

### 4. Error Handling

- **Graceful Degradation**: Handle connection failures gracefully
- **Retry Logic**: Implement appropriate retry mechanisms
- **Fallback Options**: Have alternative data sources when possible
- **User Communication**: Clear error messages for troubleshooting

## 🚨 Troubleshooting

### Common Issues

#### Connection Failures

```python
# Check connection status
get_connection_status(connection_id="your_connection_id")

# List all connections
list_database_connections()
list_ad_connections()
```

#### Schema Discovery Issues

```python
# Force schema refresh
discover_database_schema(
    connection_id="your_connection_id",
    reason="Refreshing schema after database changes"
)
```

#### Performance Issues

```python
# Limit result sets
query_database(
    connection_id="your_connection_id",
    query="SELECT * FROM large_table",
    limit=100,  # Reduce from default 1000
    reason="Performance testing with limited results"
)
```

### Debug Information

- **Connection Logs**: Check connection establishment logs
- **Query Performance**: Monitor query execution times
- **Memory Usage**: Track memory consumption during operations
- **Network Latency**: Consider network performance for remote databases

## 🔄 Integration with Agent Workflows

### Planner Agent Usage

The Planner Agent can use these tools to:

- **Assess Data Sources**: Understand available databases and AD domains
- **Plan Investigations**: Design efficient query strategies
- **Resource Allocation**: Determine which connections to establish
- **Workflow Design**: Create optimal analysis sequences

### Runner Agent Usage

The Runner Agent can use these tools to:

- **Execute Queries**: Run planned database and AD operations
- **Data Collection**: Gather information from multiple sources
- **Analysis Support**: Provide data for security assessments
- **Report Generation**: Export data for external analysis

### Example Workflow

```python
# 1. Planner identifies need for user access review
# 2. Runner connects to AD and database
# 3. Runner enumerates users and groups
# 4. Runner queries database for access logs
# 5. Runner exports data for analysis
# 6. Runner disconnects and cleans up
```

## 📚 Advanced Features

### Custom Query Templates

```python
# Store common query patterns
QUERY_TEMPLATES = {
    "recent_logins": "SELECT * FROM login_logs WHERE timestamp > ?",
    "privileged_users": "SELECT * FROM users WHERE role IN ('admin', 'superuser')",
    "failed_attempts": "SELECT * FROM auth_logs WHERE status = 'failed'"
}
```

### Batch Operations

```python
# Execute multiple related queries
batch_queries = [
    "SELECT COUNT(*) FROM users",
    "SELECT COUNT(*) FROM groups", 
    "SELECT COUNT(*) FROM computers"
]

for query in batch_queries:
    result = query_database(connection_id, query)
    # Process results
```

### Automated Analysis

```python
# Automated security assessment
def security_assessment(connection_id):
    # Get all users
    users = get_all_ad_users(connection_id)
    
    # Analyze each user
    for user in users:
        permissions = analyze_ad_user_permissions(connection_id, user.sam_account_name)
        # Process security findings
    
    # Export results
    export_ad_users_to_csv(connection_id)
```

## 🔮 Future Enhancements

### Planned Features

- **Real-time Monitoring**: Live database and AD monitoring
- **Advanced Analytics**: Machine learning-based anomaly detection
- **Compliance Reporting**: Automated compliance assessment reports
- **Integration APIs**: REST APIs for external tool integration
- **Visualization**: Built-in data visualization capabilities

### Extension Points

- **Custom Connectors**: Framework for adding new data sources
- **Plugin System**: Modular tool architecture
- **Custom Exporters**: Support for additional export formats
- **Advanced Filters**: Complex query building capabilities

## 📞 Support and Resources

### Documentation

- **API Reference**: Complete tool documentation
- **Examples**: Sample workflows and use cases
- **Troubleshooting**: Common issues and solutions
- **Best Practices**: Security and performance guidelines

### Community

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Community support and ideas
- **Contributions**: Code contributions and improvements
- **Feedback**: User experience and usability feedback

---

**Note**: These tools are designed for security professionals and cybersecurity analysts. Always ensure you have proper authorization before accessing any data sources, and follow your organization's security policies and procedures.
