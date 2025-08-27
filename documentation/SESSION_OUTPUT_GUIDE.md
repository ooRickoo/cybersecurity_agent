# Session Output Management Guide

## 🚀 **Overview**

The Session Output Management system automatically creates session folders and saves output files for chat sessions, workflows, and tool executions. This ensures all generated content is properly organized and persisted in the `session-output` folder.

## 🏗️ **Key Features**

- **Automatic Session Creation**: Creates unique session IDs with descriptive names
- **Smart Folder Management**: Only creates session folders when there are output files
- **Multiple Content Types**: Supports text, JSON, CSV, HTML, markdown, and binary files
- **Session Metadata**: Tracks session information and file details
- **TTL-Based Cleanup**: Automatic cleanup of old session folders
- **ADK Integration**: Full integration with Google ADK tools

## 📁 **Directory Structure**

```
session-output/
├── session_name_sessionid/
│   ├── output_file1.txt
│   ├── output_file2.json
│   ├── output_file3.md
│   └── session_metadata.json
└── another_session_sessionid/
    ├── workflow_results.csv
    ├── analysis_report.html
    └── session_metadata.json
```

## 🛠️ **Core Components**

### **1. SessionOutputManager**
- **Location**: `bin/session_output_manager.py`
- **Purpose**: Manages session lifecycle and file operations
- **Features**: Session creation, file management, automatic saving

### **2. ADK Integration Tools**
- **Location**: `bin/adk_integration.py`
- **Tools**: 4 new session output tools integrated with ADK
- **Access**: Available through ADK tool execution

### **3. Convenience Functions**
- **Global Access**: Simple functions for easy integration
- **Auto-Initialization**: Automatic manager instance creation
- **Error Handling**: Robust error handling and logging

## 🚀 **Available ADK Tools**

### **1. `create_session`**
Creates a new session for output management.

**Parameters:**
- `session_name` (optional): Descriptive name for the session
- `session_metadata` (optional): Additional metadata dictionary

**Returns:**
```json
{
  "success": true,
  "session_id": "uuid-string",
  "session_name": "session_name",
  "message": "Session created successfully"
}
```

**Example:**
```python
result = await adk.execute_tool('create_session', 
                               session_name='threat_analysis',
                               session_metadata={'purpose': 'security_analysis'})
```

### **2. `add_output_file`**
Adds an output file to a session.

**Parameters:**
- `session_id`: ID of the session
- `filename`: Name of the output file
- `content`: Content to save (string, dict, or bytes)
- `content_type`: Type of content ('text', 'json', 'csv', 'html', 'markdown', 'binary')
- `metadata` (optional): Additional file metadata

**Returns:**
```json
{
  "success": true,
  "filename": "filename",
  "session_id": "session_id",
  "message": "Output file added successfully"
}
```

**Example:**
```python
result = await adk.execute_tool('add_output_file',
                               session_id='session_id',
                               filename='analysis_report.md',
                               content='# Analysis Report\n\nContent here...',
                               content_type='markdown',
                               metadata={'author': 'agent', 'version': '1.0'})
```

### **3. `save_session`**
Saves all output files for a session to disk.

**Parameters:**
- `session_id`: ID of the session to save
- `force_save` (optional): Force save even if no output files

**Returns:**
```json
{
  "success": true,
  "session_folder": "session-output/session_name_sessionid",
  "saved_files": ["file1.txt", "file2.json"],
  "total_files": 2,
  "metadata_file": "session_metadata.json"
}
```

**Example:**
```python
result = await adk.execute_tool('save_session', 
                               session_id='session_id',
                               force_save=False)
```

### **4. `end_session`**
Ends a session and optionally saves outputs.

**Parameters:**
- `session_id`: ID of the session to end
- `save_outputs` (optional): Whether to save outputs before ending

**Returns:**
```json
{
  "success": true,
  "session_name": "session_name",
  "session_id": "session_id",
  "total_files": 3,
  "save_result": {...}
}
```

**Example:**
```python
result = await adk.execute_tool('end_session', 
                               session_id='session_id',
                               save_outputs=True)
```

## 🔄 **Workflow Integration**

### **Step 1: Session Creation**
```python
# Create session for workflow execution
session_result = await adk.execute_tool('create_session',
                                       session_name='workflow_execution',
                                       session_metadata={
                                           'workflow_type': 'threat_analysis',
                                           'input_files': ['network_traffic.pcap'],
                                           'priority': 'high'
                                       })

session_id = session_result['session_id']
```

### **Step 2: Add Output Files During Execution**
```python
# Add workflow results
await adk.execute_tool('add_output_file',
                       session_id=session_id,
                       filename='workflow_results.json',
                       content=workflow_results,
                       content_type='json')

# Add analysis report
await adk.execute_tool('add_output_file',
                       session_id=session_id,
                       filename='analysis_report.md',
                       content=markdown_report,
                       content_type='markdown')

# Add raw data
await adk.execute_tool('add_output_file',
                       session_id=session_id,
                       filename='raw_data.csv',
                       content=csv_data,
                       content_type='csv')
```

### **Step 3: Save and End Session**
```python
# Save all outputs
save_result = await adk.execute_tool('save_session', session_id=session_id)

# End session
end_result = await adk.execute_tool('end_session', 
                                   session_id=session_id,
                                   save_outputs=True)
```

## 📊 **Content Type Support**

### **Text Files**
```python
await adk.execute_tool('add_output_file',
                       session_id=session_id,
                       filename='summary.txt',
                       content='This is a text summary.',
                       content_type='text')
```

### **JSON Files**
```python
await adk.execute_tool('add_output_file',
                       session_id=session_id,
                       filename='data.json',
                       content={'key': 'value', 'array': [1, 2, 3]},
                       content_type='json')
```

### **Markdown Files**
```python
await adk.execute_tool('add_output_file',
                       session_id=session_id,
                       filename='report.md',
                       content='# Report\n\n## Section 1\n\nContent here...',
                       content_type='markdown')
```

### **CSV Files**
```python
await adk.execute_tool('add_output_file',
                       session_id=session_id,
                       filename='results.csv',
                       content='name,value,status\nitem1,100,active\nitem2,200,inactive',
                       content_type='csv')
```

### **HTML Files**
```python
await adk.execute_tool('add_output_file',
                       session_id=session_id,
                       filename='dashboard.html',
                       content='<html><body><h1>Dashboard</h1></body></html>',
                       content_type='html')
```

### **Binary Files**
```python
await adk.execute_tool('add_output_file',
                       session_id=session_id,
                       filename='data.bin',
                       content=binary_data,
                       content_type='binary')
```

## 🎯 **Use Cases**

### **1. Workflow Execution Tracking**
```python
# Create session for workflow
session_id = await adk.execute_tool('create_session', 
                                   session_name='workflow_execution')

# Track workflow steps
await adk.execute_tool('add_output_file',
                       session_id=session_id,
                       filename='workflow_log.json',
                       content={'step': 'initialization', 'status': 'completed'},
                       content_type='json')

# Save workflow results
await adk.execute_tool('add_output_file',
                       session_id=session_id,
                       filename='workflow_results.json',
                       content=workflow_output,
                       content_type='json')

# End session
await adk.execute_tool('end_session', session_id=session_id)
```

### **2. Threat Analysis Sessions**
```python
# Create threat analysis session
session_id = await adk.execute_tool('create_session',
                                   session_name='threat_analysis',
                                   session_metadata={'threat_level': 'high'})

# Add analysis results
await adk.execute_tool('add_output_file',
                       session_id=session_id,
                       filename='threat_report.md',
                       content=threat_report,
                       content_type='markdown')

await adk.execute_tool('add_output_file',
                       session_id=session_id,
                       filename='threat_indicators.json',
                       content=indicators,
                       content_type='json')

# Save and end
await adk.execute_tool('save_session', session_id=session_id)
await adk.execute_tool('end_session', session_id=session_id)
```

### **3. Framework Processing Sessions**
```python
# Create framework processing session
session_id = await adk.execute_tool('create_session',
                                   session_name='framework_processing')

# Add downloaded frameworks
await adk.execute_tool('add_output_file',
                       session_id=session_id,
                       filename='mitre_attack_data.json',
                       content=attack_data,
                       content_type='json')

await adk.execute_tool('add_output_file',
                       session_id=session_id,
                       filename='framework_summary.md',
                       content=summary_report,
                       content_type='markdown')

# Save and end
await adk.execute_tool('save_session', session_id=session_id)
await adk.execute_tool('end_session', session_id=session_id)
```

## 🔧 **Advanced Features**

### **1. Automatic Cleanup**
```python
# Clean up old sessions (older than 30 days)
cleanup_result = manager.cleanup_old_sessions(max_age_days=30)
```

### **2. Session Status Monitoring**
```python
# Get session status
status = manager.get_session_status(session_id)

# Get all sessions status
all_status = manager.get_all_sessions_status()
```

### **3. Batch Operations**
```python
# Save all active sessions
save_all_result = manager.save_all_sessions()
```

## 📋 **Best Practices**

### **1. Session Naming**
- Use descriptive names: `threat_analysis_2024`, `workflow_execution_001`
- Include date/time if relevant: `daily_scan_20240824`
- Use consistent naming conventions

### **2. File Organization**
- Group related files in the same session
- Use meaningful filenames: `analysis_results.json`, `summary_report.md`
- Include version information in filenames if needed

### **3. Metadata Usage**
- Add relevant session metadata: purpose, priority, input files
- Include file metadata: author, version, creation date
- Use consistent metadata structure

### **4. Error Handling**
- Always check tool execution results
- Handle session creation failures gracefully
- Implement proper cleanup on errors

## 🧪 **Testing and Validation**

### **Test Session Output Manager**
```bash
python3 bin/session_output_manager.py
```

### **Test ADK Integration**
```bash
python3 bin/adk_integration.py
```

### **Test Individual Tools**
```python
import asyncio
from bin.adk_integration import ADKIntegration

async def test_session_workflow():
    adk = ADKIntegration()
    
    # Create session
    session = await adk.execute_tool('create_session', 
                                    session_name='test_workflow')
    
    if session['success']:
        session_id = session['session_id']
        
        # Add files
        await adk.execute_tool('add_output_file',
                              session_id=session_id,
                              filename='test.txt',
                              content='Test content',
                              content_type='text')
        
        # Save and end
        await adk.execute_tool('save_session', session_id=session_id)
        await adk.execute_tool('end_session', session_id=session_id)

asyncio.run(test_session_workflow())
```

## 🔮 **Future Enhancements**

### **1. Advanced File Formats**
- **Compression**: Automatic compression of large files
- **Encryption**: Optional encryption for sensitive outputs
- **Streaming**: Support for streaming large files

### **2. Enhanced Metadata**
- **Tags**: Categorization and search capabilities
- **Relationships**: Link related sessions and files
- **Versioning**: Track file versions and changes

### **3. Performance Optimization**
- **Async I/O**: Non-blocking file operations
- **Caching**: Intelligent caching of frequently accessed files
- **Compression**: Automatic compression and decompression

### **4. Integration Features**
- **Web Interface**: Web-based session management
- **API Endpoints**: RESTful API for external access
- **Notifications**: Real-time session status updates

## 🎯 **Summary**

The Session Output Management system provides:

✅ **Automatic Session Management** - Create, manage, and track sessions  
✅ **Smart File Organization** - Only create folders when needed  
✅ **Multiple Content Types** - Support for all common file formats  
✅ **ADK Integration** - 4 new tools integrated with Google ADK  
✅ **Session Metadata** - Comprehensive tracking and organization  
✅ **Automatic Cleanup** - TTL-based cleanup of old sessions  
✅ **Error Handling** - Robust error handling and logging  

The system automatically ensures that all output files are properly organized in session-specific folders within the `session-output` directory, making it easy to track and manage the results of different operations and workflows.

---

*For additional support or questions, refer to the main documentation or contact the development team.*
