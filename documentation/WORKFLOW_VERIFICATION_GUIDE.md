# Workflow Verification System Guide

This guide covers the comprehensive "Check our math" verification system integrated with the Cybersecurity Agent. This system provides intelligent workflow verification, accuracy assessment, and automatic backtracking to prevent loops and improve results.

## 🔍 Overview

The Workflow Verification System is designed to:

- **Verify Accuracy**: Compare questions, steps taken, and final outputs
- **Prevent Loops**: Track execution paths to avoid repeating failed approaches
- **Enable Backtracking**: Automatically select alternative workflow templates when verification fails
- **Improve Quality**: Provide detailed analysis and recommendations for workflow improvement

## 🏗️ System Architecture

### Core Components

1. **Workflow Verification System** (`workflow_verification_system.py`)
   - Main verification engine
   - Accuracy scoring and analysis
   - Loop detection and prevention

2. **Workflow Template Manager** (`workflow_template_manager.py`)
   - Template selection and management
   - Backtracking decision handling
   - Alternative approach generation

3. **MCP Tools** (`workflow_verification_mcp_tools.py`)
   - NLP-friendly interfaces for agents
   - Comprehensive verification workflows
   - User-friendly output generation

## 🚀 Getting Started

### Basic Verification Workflow

```python
# 1. Execute your workflow and collect step information
workflow_steps = [
    {
        "step_id": "step1",
        "step_type": "data_collection",
        "description": "Gather threat intelligence data",
        "inputs": {"source": "threat_feeds"},
        "outputs": {"threat_data": "malware_indicators"},
        "tools_used": ["threat_intel_tool"],
        "execution_time": 2.5,
        "status": "completed"
    },
    # ... more steps
]

# 2. Call the verification system
verification_result = check_our_math(
    execution_id="exec_001",
    original_question="What is the current threat landscape?",
    workflow_steps=workflow_steps,
    final_answer="Based on analysis, we've identified...",
    question_type="threat_analysis"
)
```

### Automatic Backtracking

```python
# If verification fails, handle backtracking automatically
if verification_result["verification_result"]["needs_backtrack"]:
    backtrack_result = handle_verification_failure(
        execution_id="exec_001",
        verification_result=verification_result["verification_result"],
        original_question="What is the current threat landscape?"
    )
    
    # Execute alternative workflow
    if backtrack_result["success"]:
        print(f"Using alternative template: {backtrack_result['template_name']}")
        # Execute the new workflow template
```

## 📊 Verification Process

### 1. Question-Answer Alignment Check

The system analyzes how well the final answer addresses the original question:

- **Keyword Matching**: Identifies key terms from the question in the answer
- **Completeness Assessment**: Checks if the answer covers all aspects of the question
- **Relevance Scoring**: Measures the relevance of the answer to the question

### 2. Workflow Completeness Check

Evaluates whether the workflow steps are appropriate for the question type:

- **Expected Patterns**: Checks for required step types (e.g., analysis, assessment, conclusion)
- **Step Coverage**: Ensures all necessary phases are included
- **Complexity Matching**: Aligns workflow complexity with question complexity

### 3. Evidence Quality Check

Assesses the quality of evidence supporting the conclusions:

- **Data Sources**: Identifies tools and data sources used
- **Output Quality**: Evaluates the richness of step outputs
- **Specificity**: Checks for concrete data points and metrics

### 4. Logical Consistency Check

Verifies the logical flow of the workflow:

- **Step Sequence**: Ensures logical progression (e.g., data collection before analysis)
- **Reasoning Chain**: Checks for logical indicators in the final answer
- **Assumption Validation**: Identifies potential logical gaps

### 5. Tool Usage Appropriateness Check

Evaluates whether the right tools were used for the question:

- **Tool Selection**: Checks if tools match the question type
- **Tool Diversity**: Ensures appropriate variety of tools
- **Effectiveness**: Assesses tool usage patterns

## 🎯 Accuracy Scoring

### Scoring Components

Each verification check contributes to the overall accuracy score:

- **Question-Answer Alignment**: 20% weight
- **Workflow Completeness**: 20% weight
- **Evidence Quality**: 20% weight
- **Logical Consistency**: 20% weight
- **Tool Usage Appropriateness**: 20% weight

### Confidence Levels

Based on accuracy scores:

- **High Confidence** (≥0.8): Workflow passed verification
- **Medium Confidence** (0.6-0.79): Workflow needs review
- **Low Confidence** (<0.6): Workflow failed, backtracking required

### Question Type Thresholds

Different question types have different accuracy thresholds:

- **Threat Analysis**: High threshold (0.8) due to critical nature
- **Incident Response**: High threshold (0.85) for operational accuracy
- **Vulnerability Assessment**: Very high threshold (0.9) for security
- **Compliance Audit**: Very high threshold (0.9) for regulatory requirements
- **General Investigation**: Standard threshold (0.8)

## 🔄 Backtracking System

### When Backtracking is Triggered

Backtracking is automatically triggered when:

- **Accuracy Score < 0.6**: Workflow significantly underperforms
- **Multiple Issues**: More than 2 critical issues identified
- **Loop Detection**: Same execution path used multiple times
- **Verification Errors**: System errors prevent proper verification

### Backtracking Process

1. **Failure Analysis**: Analyze why the original approach failed
2. **Template Selection**: Choose alternative workflow template
3. **Confidence Assessment**: Calculate confidence in the alternative
4. **Recommendation Generation**: Provide specific next steps

### Alternative Template Selection

The system intelligently selects alternatives based on:

- **Failure Patterns**: Addresses specific failure reasons
- **Template Success Rates**: Prefers proven approaches
- **Complexity Matching**: Aligns with question requirements
- **Tool Availability**: Ensures required capabilities exist

## 🛤️ Loop Prevention

### Path Tracking

The system tracks execution paths to prevent loops:

- **Path Hashing**: Creates unique identifiers for execution sequences
- **Usage Counting**: Monitors how often each path is used
- **Threshold Detection**: Triggers alerts when paths are overused

### Loop Detection

Loops are detected when:

- **Same Path Used**: Identical step sequence executed multiple times
- **Threshold Exceeded**: Path used more than 3 times
- **No Improvement**: Repeated attempts don't improve accuracy

### Alternative Path Generation

When loops are detected, the system suggests:

- **Step Reordering**: Change the sequence of workflow steps
- **Step Combination**: Merge related steps for efficiency
- **Step Splitting**: Break complex steps into simpler ones
- **Tool Variation**: Use different tools for the same purpose

## 📋 Available MCP Tools

### Core Verification Tools

1. **`check_our_math`**
   - Main verification function
   - Comprehensive accuracy assessment
   - Detailed issue identification

2. **`handle_verification_failure`**
   - Automatic backtracking handling
   - Alternative template selection
   - Confidence assessment

3. **`get_verification_summary`**
   - Complete verification results
   - Execution history and statistics
   - Performance metrics

### Template Management Tools

4. **`select_workflow_template`**
   - Intelligent template selection
   - Question type classification
   - Complexity assessment

5. **`get_template_variations`**
   - Template complexity variations
   - Simplified and comprehensive versions
   - Customization options

6. **`get_backtrack_summary`**
   - Backtracking decision history
   - Alternative approach tracking
   - Performance analysis

### Analysis and Reporting Tools

7. **`get_execution_history`**
   - Recent workflow executions
   - Success rate tracking
   - Performance trends

8. **`get_template_performance_stats`**
   - Template success rates
   - Usage statistics
   - Performance comparisons

## 🔧 Integration Examples

### Example 1: Threat Analysis Workflow

```python
# Execute threat analysis workflow
threat_workflow_steps = [
    {"step_type": "data_collection", "description": "Gather threat intel", ...},
    {"step_type": "analysis", "description": "Analyze threat patterns", ...},
    {"step_type": "assessment", "description": "Assess risk level", ...}
]

# Verify the workflow
verification = check_our_math(
    execution_id="threat_001",
    original_question="What threats are targeting our financial systems?",
    workflow_steps=threat_workflow_steps,
    final_answer="We've identified 3 high-risk threats...",
    question_type="threat_analysis"
)

# Handle failures automatically
if verification["verification_result"]["needs_backtrack"]:
    backtrack = handle_verification_failure(
        execution_id="threat_001",
        verification_result=verification["verification_result"],
        original_question="What threats are targeting our financial systems?"
    )
```

### Example 2: Incident Response Workflow

```python
# Execute incident response
incident_steps = [
    {"step_type": "detection", "description": "Detect security incident", ...},
    {"step_type": "assessment", "description": "Assess incident scope", ...},
    {"step_type": "response", "description": "Contain and respond", ...}
]

# Verify with specialized checks
verification = check_our_math(
    execution_id="incident_001",
    original_question="How should we respond to this data breach?",
    workflow_steps=incident_steps,
    final_answer="Immediate containment actions...",
    question_type="incident_response"
)
```

### Example 3: Vulnerability Assessment Workflow

```python
# Execute vulnerability assessment
vuln_steps = [
    {"step_type": "scanning", "description": "Scan for vulnerabilities", ...},
    {"step_type": "assessment", "description": "Assess severity", ...},
    {"step_type": "planning", "description": "Plan remediation", ...}
]

# Verify with high accuracy requirements
verification = check_our_math(
    execution_id="vuln_001",
    original_question="What vulnerabilities exist in our web applications?",
    workflow_steps=vuln_steps,
    final_answer="Critical SQL injection vulnerability found...",
    question_type="vulnerability_assessment"
)
```

## 📈 Best Practices

### 1. Workflow Design

- **Clear Objectives**: Ensure each step has a clear purpose
- **Logical Flow**: Design steps in logical sequence
- **Tool Selection**: Choose appropriate tools for each step
- **Output Quality**: Ensure each step produces meaningful outputs

### 2. Verification Integration

- **Regular Verification**: Verify workflows after each execution
- **Issue Tracking**: Monitor and address verification issues
- **Template Learning**: Use successful templates as references
- **Performance Monitoring**: Track accuracy improvements over time

### 3. Backtracking Strategy

- **Early Detection**: Identify failures early in the process
- **Alternative Planning**: Have backup approaches ready
- **Learning Integration**: Incorporate lessons from failures
- **Continuous Improvement**: Refine templates based on results

### 4. Loop Prevention

- **Path Diversity**: Use different approaches for similar problems
- **Step Variation**: Vary workflow steps when possible
- **Tool Rotation**: Use different tools for the same purpose
- **Complexity Adjustment**: Match workflow complexity to problem complexity

## 🚨 Troubleshooting

### Common Issues

#### Verification Failures

```python
# Check verification details
summary = get_verification_summary("exec_001")
print(f"Issues: {summary['summary']['issues_found']}")
print(f"Recommendations: {summary['summary']['recommendations']}")

# Handle backtracking
if summary['summary']['verification_status'] == 'failed':
    backtrack = handle_verification_failure(
        execution_id="exec_001",
        verification_result=summary['summary'],
        original_question="Your question here"
    )
```

#### Template Selection Issues

```python
# Check available templates
template = select_workflow_template("Your question here")
if not template['success']:
    print(f"Error: {template['error']}")
    print(f"Message: {template['message']}")

# Get template variations
variations = get_template_variations("template_id")
print(f"Available variations: {len(variations['variations'])}")
```

#### Performance Issues

```python
# Check template performance
stats = get_template_performance_stats()
for template_type, type_stats in stats['stats'].items():
    print(f"{template_type}: {type_stats['avg_success_rate']:.1%} success rate")

# Check execution history
history = get_execution_history(limit=20)
print(f"Recent executions: {len(history['history'])}")
```

### Debug Information

- **Verification Logs**: Check verification system logs
- **Template Performance**: Monitor template success rates
- **Execution Paths**: Review execution path history
- **Backtrack Decisions**: Analyze backtracking decisions

## 🔮 Advanced Features

### Custom Verification Templates

```python
# Create custom verification criteria
custom_verification = {
    "accuracy_checks": [
        "custom_check_1",
        "custom_check_2"
    ],
    "confidence_thresholds": {
        "high": 0.85,
        "medium": 0.65,
        "low": 0.45
    }
}
```

### Template Customization

```python
# Get template variations
variations = get_template_variations("base_template_id")

# Customize template for specific needs
custom_template = {
    "steps": variations[0]["steps"],
    "prerequisites": ["custom_requirement"],
    "success_criteria": {"custom_metric": True}
}
```

### Performance Analytics

```python
# Track verification performance over time
execution_history = get_execution_history(limit=100)
accuracy_trends = [exec['accuracy'] for exec in execution_history['history']]

# Analyze template effectiveness
template_stats = get_template_performance_stats()
best_templates = [template for template in template_stats['stats'].values() 
                 if template['avg_success_rate'] > 0.8]
```

## 📞 Support and Resources

### Documentation

- **API Reference**: Complete tool documentation
- **Examples**: Sample workflows and verification scenarios
- **Troubleshooting**: Common issues and solutions
- **Best Practices**: Verification and workflow design guidelines

### Community

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Community support and ideas
- **Contributions**: Code contributions and improvements
- **Feedback**: User experience and usability feedback

---

**Note**: The Workflow Verification System is designed to improve the accuracy and reliability of cybersecurity workflows. Always review verification results and use backtracking recommendations to enhance your analysis approaches.
