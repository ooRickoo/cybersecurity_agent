# Dynamic Workflow Generation & Enhanced Session Logging Guide

## Overview

This guide covers the new **Dynamic Workflow Generation** system and **Enhanced Session Logging** that make the Cybersecurity AI Helper more adaptive, intelligent, and traceable.

## 🚀 Dynamic Workflow Generation

### What It Is

The Dynamic Workflow Generator creates workflows on-the-fly based on problem analysis, making the system more adaptive while maintaining speed through intelligent caching and pattern recognition.

### Key Features

- **Intelligent Problem Analysis**: Automatically classifies problems and determines requirements
- **Component Library**: Reusable workflow components for different tasks
- **Smart Assembly**: Automatically selects and arranges components based on problem type
- **Execution Planning**: Creates optimized execution plans with parallel/sequential steps
- **Adaptation Points**: Identifies where workflows can adapt during execution
- **Pattern Caching**: Remembers successful workflow patterns for faster generation
- **Performance Metrics**: Tracks generation time, component usage, and cache efficiency

### How It Works

1. **Problem Analysis**: Analyzes user input, file types, and desired outputs
2. **Component Selection**: Chooses appropriate components from the library
3. **Execution Planning**: Creates optimized execution plan with dependencies
4. **Adaptation Planning**: Identifies potential adaptation points
5. **Workflow Assembly**: Combines everything into a executable workflow
6. **Caching**: Stores successful patterns for future reuse

### Component Types

| Type | Description | Examples |
|------|-------------|----------|
| **DATA_IMPORT** | Import data from various sources | CSV Import, JSON Import |
| **ANALYSIS** | Analyze data for patterns and insights | Pattern Analysis, Threat Intelligence |
| **PROCESSING** | Process and transform data | Data Enrichment, ML Classification |
| **VALIDATION** | Validate data quality and integrity | Data Quality Check |
| **EXPORT** | Export processed data | CSV Export, JSON Export |
| **INTEGRATION** | Integrate with external systems | MITRE ATT&CK Mapping |

### Complexity Levels

- **SIMPLE**: Quick operations (import/export, basic validation)
- **MODERATE**: Analysis and processing tasks
- **COMPLEX**: ML operations, threat intelligence, advanced analytics

### Usage Examples

#### Command Line
```bash
# Generate workflow for threat analysis
python3 cs_util_lg.py dynamic-workflow "Analyze this CSV for security threats" \
  --input-files threat_data.csv \
  --outputs enriched_data threat_report \
  --execute

# Generate and export workflow
python3 cs_util_lg.py dynamic-workflow "Process logs for compliance" \
  --input-files access_logs.csv \
  --outputs compliance_report \
  --export
```

#### Interactive Mode
```
🔍 What would you like to do? (or type 'help' for options): 
Analyze this CSV file for security threats and export enriched data

🔍 Attempting dynamic workflow generation...
🚀 Dynamic workflow generated with 0.85 confidence!
   Components: 5
   Estimated time: 45.0s
   Execution plan: 4 steps

🔧 Execute this dynamic workflow? (y/n): y
```

### Workflow Components

#### Data Import Components
- **CSV Import**: Automatic format detection, encoding handling
- **JSON Import**: Schema validation, nested data support
- **Excel Import**: Multi-sheet support, format preservation

#### Analysis Components
- **Pattern Analysis**: Anomaly detection, trend analysis
- **Threat Intelligence**: External feed integration, IOC matching
- **Statistical Analysis**: Descriptive statistics, correlation analysis

#### Processing Components
- **Data Enrichment**: Context addition, external data integration
- **ML Classification**: Automated categorization, prediction
- **Data Transformation**: Format conversion, structure modification

#### Validation Components
- **Data Quality Check**: Completeness, accuracy, consistency
- **Schema Validation**: Format compliance, field validation
- **Business Rule Check**: Domain-specific validation rules

#### Export Components
- **CSV Export**: Configurable formatting, metadata inclusion
- **JSON Export**: Pretty printing, schema documentation
- **Report Export**: Formatted reports, visualizations

### Execution Strategies

- **Sequential**: Components run one after another
- **Parallel**: Multiple components run simultaneously
- **Conditional**: Components run based on conditions
- **Iterative**: Components run in loops with adaptation

### Adaptation Points

- **Fallback Components**: Switch to simpler alternatives if complex components fail
- **Parameter Adjustment**: Modify component parameters based on results
- **Resource Allocation**: Adjust resource usage based on performance
- **Error Recovery**: Handle failures gracefully with alternative paths

## 📝 Enhanced Session Logging

### What It Is

Comprehensive logging system that captures all session activities, user interactions, workflow executions, and system performance for complete traceability and analysis.

### Key Features

- **Real-time Logging**: Logs activities as they happen
- **Structured Data**: JSON-based logging with metadata
- **Performance Tracking**: Execution times, resource usage, success rates
- **Error Tracking**: Detailed error logging with context
- **Session Management**: Automatic session creation and cleanup
- **File Management**: Automatic log rotation and cleanup

### Log Categories

#### Session Metadata
- Session ID, start/end times, duration
- User information, system configuration
- Performance metrics, error counts

#### Agent Interactions
- User questions and system responses
- Tool usage and results
- Workflow executions and outcomes

#### Workflow Executions
- Workflow steps and status
- Component execution details
- Performance metrics and timing

#### Tool Calls
- Tool names and parameters
- Execution results and errors
- Performance data and resource usage

#### Decision Points
- System decisions and reasoning
- Alternative paths considered
- Confidence scores and thresholds

#### Data Operations
- File operations and transformations
- Data quality metrics
- Processing results and statistics

### Log Structure

```json
{
  "session_metadata": {
    "session_id": "session_12345",
    "start_time": "2024-01-15T10:30:00Z",
    "end_time": "2024-01-15T11:45:00Z",
    "duration_ms": 4500000,
    "user_agent": "CS_AI_CLI/1.0"
  },
  "performance_metrics": {
    "total_tool_calls": 25,
    "total_workflow_steps": 12,
    "total_errors": 2,
    "session_duration_ms": 4500000
  },
  "agent_interactions": [
    {
      "timestamp": "2024-01-15T10:30:15Z",
      "type": "question",
      "question": "Analyze this CSV for threats",
      "context": "interactive_mode",
      "metadata": {"input_type": "menu_selection"}
    }
  ],
  "workflow_executions": [
    {
      "timestamp": "2024-01-15T10:30:30Z",
      "workflow_name": "dynamic_12345",
      "step_name": "execution_started",
      "status": "in_progress",
      "metadata": {
        "component_count": 5,
        "estimated_duration": 45.0,
        "confidence_score": 0.85
      }
    }
  ]
}
```

### Usage

#### Automatic Logging
Session logging happens automatically in interactive mode:
- User inputs are logged as questions
- System responses are logged as answers
- Workflow executions are tracked
- Errors and performance issues are recorded

#### Manual Logging
```python
# Log information
session_logger.log_info("event_type", "description", metadata={})

# Log errors
session_logger.log_error("error_type", "error_message", metadata={})

# Log workflow execution
session_logger.log_workflow_execution(
    workflow_name="workflow_name",
    step_name="step_name",
    status="status",
    metadata={}
)
```

#### Session Management
```python
# Start session
session_logger.log_info("session_start", "Session description")

# End session with summary
summary = session_logger.get_session_summary()
session_logger.end_session(summary)
```

### Log Files

- **Location**: `session-logs/` directory
- **Format**: JSON files with timestamp
- **Naming**: `session_name_sessionid.json`
- **Rotation**: Automatic cleanup of old logs
- **Limit**: Maximum 200 log files

### Performance Impact

- **Minimal Overhead**: Asynchronous logging with buffering
- **Efficient Storage**: Compressed JSON format
- **Smart Cleanup**: Automatic removal of old logs
- **Background Processing**: Non-blocking log operations

## 🔧 Integration

### CLI Integration

The dynamic workflow generator and session logging are fully integrated into the CLI:

- **Interactive Mode**: Automatic workflow generation and comprehensive logging
- **Command Line**: Direct workflow generation and execution
- **Batch Processing**: Automated workflow generation for multiple problems

### MCP Integration

- **Tool Discovery**: Dynamic workflow components appear as MCP tools
- **Workflow Execution**: Generated workflows can be executed via MCP
- **Session Tracking**: MCP tool usage is logged in sessions

### Workflow System Integration

- **Template Library**: Dynamic workflows complement existing templates
- **Problem Orchestration**: Integrates with problem-driven orchestrator
- **Execution Engine**: Uses existing workflow execution infrastructure

## 📊 Performance & Optimization

### Caching Strategy

- **Pattern Cache**: Remembers successful workflow patterns
- **Component Cache**: Caches frequently used components
- **Result Cache**: Stores workflow execution results

### Optimization Techniques

- **Parallel Execution**: Identifies components that can run simultaneously
- **Resource Allocation**: Optimizes resource usage based on complexity
- **Adaptive Execution**: Adjusts execution based on runtime conditions

### Performance Metrics

- **Generation Time**: How long it takes to create workflows
- **Execution Time**: Actual vs. estimated execution time
- **Cache Hit Rate**: Effectiveness of pattern caching
- **Success Rate**: Percentage of successful workflow executions

## 🚀 Future Enhancements

### Planned Features

- **ML-Based Generation**: Use machine learning to improve workflow generation
- **Advanced Caching**: Intelligent cache management with LRU and predictive loading
- **Real-time Adaptation**: Dynamic workflow modification during execution
- **Performance Prediction**: Better estimation of execution times
- **Integration APIs**: REST APIs for external workflow generation

### Extension Points

- **Custom Components**: User-defined workflow components
- **Plugin System**: Extensible component library
- **Workflow Templates**: Save and reuse generated workflows
- **Collaboration**: Share workflows between users and teams

## 🛠️ Troubleshooting

### Common Issues

#### Dynamic Workflow Generation Fails
- Check component library initialization
- Verify problem description format
- Review input file types and outputs
- Check for missing dependencies

#### Session Logging Issues
- Verify session logger initialization
- Check file permissions for log directory
- Review disk space availability
- Check for import errors

#### Performance Issues
- Monitor cache hit rates
- Review component complexity levels
- Check resource allocation
- Analyze execution patterns

### Debug Mode

Enable debug logging for troubleshooting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Log Analysis

Use the session logs to analyze:
- User behavior patterns
- Workflow performance
- Error frequencies
- System bottlenecks

## 📚 Examples

### Example 1: Threat Analysis Workflow

**Input**: "Analyze this CSV for security threats and export enriched data"

**Generated Workflow**:
1. **Data Import**: CSV Import
2. **Analysis**: Pattern Analysis, Threat Intelligence
3. **Processing**: Data Enrichment
4. **Validation**: Data Quality Check
5. **Export**: CSV Export

**Execution Plan**:
- Step 1: Data Import (sequential)
- Step 2: Parallel Analysis (parallel)
- Step 3: Data Processing (sequential)
- Step 4: Validation (parallel)
- Step 5: Export (sequential)

### Example 2: Compliance Check Workflow

**Input**: "Check this data for compliance and map to MITRE ATT&CK"

**Generated Workflow**:
1. **Data Import**: CSV Import
2. **Validation**: Data Quality Check
3. **Integration**: MITRE ATT&CK Mapping
4. **Processing**: Data Enrichment
5. **Export**: JSON Export

### Example 3: ML Processing Workflow

**Input**: "Apply machine learning to classify this data"

**Generated Workflow**:
1. **Data Import**: CSV Import
2. **Validation**: Data Quality Check
3. **Processing**: ML Classification
4. **Processing**: Data Enrichment
5. **Export**: CSV Export

## 🎯 Best Practices

### Workflow Generation

- **Clear Problem Descriptions**: Be specific about what you want to achieve
- **Input File Types**: Specify file formats and structures
- **Output Requirements**: Define desired output formats and content
- **Constraints**: Specify any limitations or requirements

### Session Logging

- **Regular Review**: Periodically review session logs for insights
- **Performance Monitoring**: Track execution times and success rates
- **Error Analysis**: Analyze error patterns for system improvements
- **User Experience**: Use logs to understand user behavior and needs

### Performance Optimization

- **Cache Management**: Monitor cache hit rates and adjust strategies
- **Component Selection**: Choose appropriate complexity levels
- **Resource Allocation**: Balance performance and resource usage
- **Adaptation Points**: Use adaptation for better reliability

## 🔗 Related Documentation

- [CSV Enrichment Workflow Guide](CSV_ENRICHMENT_WORKFLOW_GUIDE.md)
- [Interactive AI Helper Guide](INTERACTIVE_AI_HELPER_GUIDE.md)
- [MITRE ATT&CK Mapping Guide](MITRE_ATTACK_MAPPING_GUIDE.md)
- [Problem-Driven Orchestration Guide](PROBLEM_DRIVEN_ORCHESTRATION_GUIDE.md)

---

*This guide covers the dynamic workflow generation and enhanced session logging capabilities. For additional support or questions, refer to the main documentation or contact the development team.*
