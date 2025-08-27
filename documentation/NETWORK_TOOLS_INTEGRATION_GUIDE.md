# 🌐 **Network Tools Integration Guide - Query Path & Runner Agent**

## 🎯 **Overview**

This guide shows you how to integrate the comprehensive collection of MCP network tools with your existing Query Path and Runner Agent system. The network tools provide internal network analysis capabilities and are designed to be dynamically discoverable and executable by your agentic workflow system.

## 🛠️ **Available Network Tools**

### **1. Network Ping Tool (`network_ping`)**
- **Category**: Connectivity
- **Capability**: Ping
- **Description**: Comprehensive ping tool with latency analysis, packet loss, and jitter calculation
- **Input**: `host`, `count`, `timeout`, `size`
- **Output**: Ping results, statistics, analysis
- **Use Case**: Network connectivity testing, latency measurement, quality assessment

### **2. Network DNS Lookup Tool (`network_dns_lookup`)**
- **Category**: DNS
- **Capability**: DNS Lookup
- **Description**: Comprehensive DNS resolution tool with multiple record types and performance analysis
- **Input**: `hostname`, `record_type`, `nameserver`
- **Output**: DNS results, resolution time, analysis
- **Use Case**: DNS troubleshooting, record validation, performance analysis

### **3. Network Netstat Tool (`network_netstat`)**
- **Category**: Statistics
- **Capability**: Network Stats
- **Description**: Comprehensive network statistics tool showing connections, listening ports, and interface statistics
- **Input**: `protocol`, `state`, `interface`
- **Output**: Connections, listening ports, interface stats, analysis
- **Use Case**: Network monitoring, connection analysis, security assessment

### **4. Network ARP Tool (`network_arp`)**
- **Category**: Statistics
- **Capability**: ARP Management
- **Description**: Comprehensive ARP table management tool for viewing, adding, and deleting ARP entries
- **Input**: `interface`, `action`, `ip_address`, `mac_address`
- **Output**: ARP table, interface info, analysis
- **Use Case**: ARP table management, MAC address resolution, security monitoring

### **5. Network Traceroute Tool (`network_traceroute`)**
- **Category**: Connectivity
- **Capability**: Traceroute
- **Description**: Comprehensive traceroute tool for analyzing network paths and identifying bottlenecks
- **Input**: `host`, `max_hops`, `timeout`, `protocol`
- **Output**: Path analysis, hop details, bottleneck analysis
- **Use Case**: Network path analysis, bottleneck identification, routing troubleshooting

### **6. Network Port Scanner Tool (`network_port_scanner`)**
- **Category**: Security
- **Capability**: Port Scan
- **Description**: Comprehensive port scanner for identifying open ports, services, and potential security vulnerabilities
- **Input**: `host`, `port_range`, `scan_type`, `timeout`
- **Output**: Open ports, service identification, security analysis, vulnerability assessment
- **Use Case**: Security auditing, service discovery, vulnerability assessment

## 🔗 **Integration with Query Path & Runner Agent**

### **Step 1: Import Network Tools into Your System**

```python
# In your agentic_workflow_system.py or main integration file
from network_mcp_integration import NetworkMCPIntegrationLayer, NetworkToolsQueryPathIntegration

class AgenticWorkflowSystem:
    def __init__(self, base_path: str = "knowledge-objects"):
        # ... existing initialization ...
        
        # Add network tools integration
        self.network_mcp_integration = NetworkMCPIntegrationLayer()
        self.network_query_path_integration = NetworkToolsQueryPathIntegration(
            self.network_mcp_integration
        )
        
        # ... rest of initialization ...
```

### **Step 2: Enhance Query Path with Network Tools**

```python
class QueryPath:
    def __init__(self, mcp_integration: MCPIntegrationLayer, 
                 context_memory: EnhancedContextMemoryManager, 
                 enhanced_memory: EnhancedAgenticMemorySystem,
                 network_integration: NetworkToolsQueryPathIntegration):  # Add this
        # ... existing initialization ...
        self.network_integration = network_integration
    
    async def select_tools_for_problem(self, problem_description: str, context: Dict[str, Any]) -> List[str]:
        """Select optimal tools for a given problem using context-aware routing."""
        # ... existing tool selection logic ...
        
        # Add network tools to available tools
        network_tools = self.network_integration.discover_network_tools(problem_description)
        if network_tools:
            # Score network tools based on relevance
            for tool in network_tools:
                relevance_score = self.network_integration.score_tool_relevance(
                    tool["tool_id"], problem_description, context
                )
                # Add to available tools with network-specific scoring
                available_tools.append({
                    'tool_id': tool["tool_id"],
                    'metadata': tool,
                    'performance': relevance_score,  # Use relevance as performance
                    'capability': 'network_analysis'
                })
        
        # ... rest of tool selection logic ...
```

### **Step 3: Enhance Runner Agent with Network Tool Execution**

```python
class RunnerAgent:
    def __init__(self, mcp_integration: MCPIntegrationLayer, 
                 context_memory: EnhancedContextMemoryManager, 
                 enhanced_memory: EnhancedAgenticMemorySystem,
                 network_integration: NetworkMCPIntegrationLayer):  # Add this
        # ... existing initialization ...
        self.network_integration = network_integration
    
    async def _execute_single_row_workflow(self, problem_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow for a single row or sub-problem."""
        # ... existing tool selection ...
        
        # Execute workflow with selected tools
        workflow_result = await self.mcp_integration.execute_workflow_with_tools(
            WorkflowContext(
                problem_type=ProblemType.ANALYSIS,
                problem_description=problem_description,
                priority=context.get('priority', 1),
                complexity=context.get('complexity', 1)
            ),
            selected_tools
        )
        
        # Execute network tools if they were selected
        network_tools_executed = []
        for tool_id in selected_tools:
            if tool_id.startswith("network_"):
                try:
                    # Execute network tool with appropriate parameters
                    network_result = await self._execute_network_tool(tool_id, context)
                    network_tools_executed.append({
                        'tool_id': tool_id,
                        'result': network_result
                    })
                except Exception as e:
                    logger.error(f"Network tool execution failed: {e}")
        
        # Combine results
        if network_tools_executed:
            workflow_result['network_analysis'] = network_tools_executed
        
        return workflow_result
    
    async def _execute_network_tool(self, tool_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific network tool with context-aware parameters."""
        # Extract parameters from context or use defaults
        if tool_id == "network_ping":
            host = context.get('target_host', '8.8.8.8')
            return await self.network_integration.execute_tool(tool_id, host=host)
            
        elif tool_id == "network_dns_lookup":
            hostname = context.get('target_hostname', 'google.com')
            return await self.network_integration.execute_tool(tool_id, hostname=hostname)
            
        elif tool_id == "network_port_scanner":
            host = context.get('target_host', '127.0.0.1')
            port_range = context.get('port_range', '1-1024')
            return await self.network_integration.execute_tool(tool_id, host=host, port_range=port_range)
            
        elif tool_id == "network_traceroute":
            host = context.get('target_host', '8.8.8.8')
            return await self.network_integration.execute_tool(tool_id, host=host)
            
        elif tool_id == "network_netstat":
            return await self.network_integration.execute_tool(tool_id)
            
        elif tool_id == "network_arp":
            return await self.network_integration.execute_tool(tool_id, action="show")
        
        return {"success": False, "error": f"Unknown network tool: {tool_id}"}
```

## 🎯 **Dynamic Tool Discovery Examples**

### **Example 1: Network Connectivity Problem**

```python
# Problem: "Check network connectivity to server 192.168.1.100"
problem_description = "Check network connectivity to server 192.168.1.100"

# Query Path will discover these tools:
discovered_tools = network_integration.suggest_tools_for_problem(problem_description)

# Expected results:
# 1. network_ping (relevance: 2) - "connectivity" + "ping"
# 2. network_traceroute (relevance: 1) - "connectivity"
# 3. network_dns_lookup (relevance: 1) - "resolution"

# Runner Agent will execute:
# - Ping to 192.168.1.100 to check basic connectivity
# - Traceroute to identify network path and potential bottlenecks
# - DNS lookup if hostname resolution is needed
```

### **Example 2: Security Assessment Problem**

```python
# Problem: "Perform security assessment of host 10.0.0.50"
problem_description = "Perform security assessment of host 10.0.0.50"

# Query Path will discover these tools:
discovered_tools = network_integration.suggest_tools_for_problem(problem_description)

# Expected results:
# 1. network_port_scanner (relevance: 2) - "security" + "scan"
# 2. network_arp (relevance: 1) - "security"
# 3. network_netstat (relevance: 1) - "ports"

# Runner Agent will execute:
# - Port scan to identify open services and potential vulnerabilities
# - ARP table analysis to detect potential ARP spoofing
# - Netstat analysis to understand current network connections
```

### **Example 3: DNS Troubleshooting Problem**

```python
# Problem: "Resolve DNS issues with internal services"
problem_description = "Resolve DNS issues with internal services"

# Query Path will discover these tools:
discovered_tools = network_integration.suggest_tools_for_problem(problem_description)

# Expected results:
# 1. network_dns_lookup (relevance: 2) - "dns" + "resolution"
# 2. network_ping (relevance: 1) - "connectivity"

# Runner Agent will execute:
# - DNS lookups for various record types (A, AAAA, CNAME, MX)
# - Ping tests to verify connectivity to resolved IPs
```

## 🔧 **Advanced Integration Features**

### **1. Context-Aware Parameter Extraction**

```python
def extract_network_parameters(self, problem_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Extract network-specific parameters from problem description and context."""
    parameters = {}
    
    # Extract IP addresses and hostnames
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    hostname_pattern = r'\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
    
    ips = re.findall(ip_pattern, problem_description)
    hostnames = re.findall(hostname_pattern, problem_description)
    
    if ips:
        parameters['target_host'] = ips[0]
    elif hostnames:
        parameters['target_hostname'] = hostnames[0]
    
    # Extract port ranges
    port_pattern = r'port[s]?\s+(\d+(?:-\d+)?)'
    port_match = re.search(port_pattern, problem_description, re.IGNORECASE)
    if port_match:
        parameters['port_range'] = port_match.group(1)
    
    # Extract protocol information
    if 'tcp' in problem_description.lower():
        parameters['protocol'] = 'tcp'
    elif 'udp' in problem_description.lower():
        parameters['protocol'] = 'udp'
    
    return parameters
```

### **2. Intelligent Tool Chaining**

```python
async def execute_network_workflow(self, problem_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute intelligent network workflow with tool chaining."""
    workflow_results = []
    
    # Discover relevant tools
    suggested_tools = self.network_integration.suggest_tools_for_problem(problem_description)
    
    # Execute tools in logical order
    for tool in suggested_tools[:3]:  # Top 3 most relevant tools
        tool_id = tool['tool_id']
        
        # Extract parameters
        parameters = self.extract_network_parameters(problem_description, context)
        
        # Execute tool
        result = await self.network_integration.execute_tool(tool_id, **parameters)
        
        # Store result
        workflow_results.append({
            'tool_id': tool_id,
            'tool_name': tool['name'],
            'relevance_score': tool['relevance_score'],
            'result': result,
            'execution_time': result.get('execution_time', 0)
        })
        
        # Update context with results for next tools
        context[f'{tool_id}_result'] = result
    
    return {
        'workflow_type': 'network_analysis',
        'problem_description': problem_description,
        'tools_executed': len(workflow_results),
        'total_execution_time': sum(r['execution_time'] for r in workflow_results),
        'results': workflow_results,
        'summary': self._generate_network_workflow_summary(workflow_results)
    }
```

### **3. Performance-Based Tool Selection**

```python
def select_optimal_network_tools(self, problem_description: str, context: Dict[str, Any]) -> List[str]:
    """Select optimal network tools based on performance and relevance."""
    suggested_tools = self.network_integration.suggest_tools_for_problem(problem_description)
    
    # Score tools based on multiple factors
    scored_tools = []
    for tool in suggested_tools:
        score = 0.0
        
        # Relevance score (40%)
        score += tool['relevance_score'] * 0.4
        
        # Performance score (30%)
        metadata = tool['metadata']
        if metadata['usage_count'] > 0:
            performance_score = metadata['success_rate'] * 0.3
            score += performance_score
        
        # Context relevance (20%)
        if self._is_context_relevant(tool, context):
            score += 0.2
        
        # Tool complexity (10%) - prefer simpler tools for simple problems
        if len(problem_description.split()) < 10:  # Simple problem
            if tool['tool_id'] in ['network_ping', 'network_dns_lookup']:
                score += 0.1
        
        scored_tools.append((tool['tool_id'], score))
    
    # Sort by score and return top tools
    scored_tools.sort(key=lambda x: x[1], reverse=True)
    return [tool_id for tool_id, score in scored_tools[:3]]
```

## 📊 **Monitoring and Analytics**

### **1. Tool Performance Tracking**

```python
def get_network_tools_performance(self) -> Dict[str, Any]:
    """Get comprehensive performance metrics for network tools."""
    stats = self.network_integration.get_tool_statistics()
    
    # Add enhanced analytics
    enhanced_stats = {
        **stats,
        'tool_efficiency': {},
        'category_performance': {},
        'recommendations': []
    }
    
    # Calculate tool efficiency
    for tool_id, tool_info in stats['tool_performance'].items():
        if tool_info['executions'] > 0:
            efficiency = tool_info['success_rate'] / tool_info['avg_execution_time']
            enhanced_stats['tool_efficiency'][tool_id] = efficiency
    
    # Generate recommendations
    if enhanced_stats['tool_efficiency']:
        best_tool = max(enhanced_stats['tool_efficiency'].items(), key=lambda x: x[1])
        enhanced_stats['recommendations'].append(f"Best performing tool: {best_tool[0]}")
    
    return enhanced_stats
```

### **2. Workflow Success Analysis**

```python
def analyze_network_workflow_success(self, workflow_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze success patterns in network workflows."""
    analysis = {
        'total_workflows': len(workflow_results),
        'successful_workflows': 0,
        'tool_success_rates': {},
        'common_failure_patterns': [],
        'optimization_opportunities': []
    }
    
    for workflow in workflow_results:
        if workflow.get('success', False):
            analysis['successful_workflows'] += 1
        
        # Track tool success rates
        for tool_result in workflow.get('results', []):
            tool_id = tool_result['tool_id']
            if tool_id not in analysis['tool_success_rates']:
                analysis['tool_success_rates'][tool_id] = {'success': 0, 'total': 0}
            
            analysis['tool_success_rates'][tool_id]['total'] += 1
            if tool_result['result'].get('success', False):
                analysis['tool_success_rates'][tool_id]['success'] += 1
    
    # Calculate success rates
    for tool_id, counts in analysis['tool_success_rates'].items():
        if counts['total'] > 0:
            counts['success_rate'] = counts['success'] / counts['total']
    
    return analysis
```

## 🚀 **Usage Examples**

### **Example 1: Basic Integration**

```python
# Initialize system with network tools
system = AgenticWorkflowSystem(".")
system.network_mcp_integration = NetworkMCPIntegrationLayer()

# Execute network analysis workflow
result = await system.execute_workflow(
    "Check network connectivity and security of server 192.168.1.100",
    {
        'priority': 4,
        'complexity': 6,
        'mode': 'automated',
        'target_host': '192.168.1.100'
    },
    ExecutionMode.AUTOMATED
)

print(f"Network analysis completed: {result['success']}")
if result.get('network_analysis'):
    for tool_result in result['network_analysis']:
        print(f"  {tool_result['tool_id']}: {tool_result['result']['success']}")
```

### **Example 2: Advanced Workflow**

```python
# Execute complex network security assessment
security_workflow = await system.execute_workflow(
    "Perform comprehensive security assessment of network segment 10.0.0.0/24",
    {
        'priority': 5,
        'complexity': 8,
        'mode': 'hybrid',
        'network_segment': '10.0.0.0/24',
        'security_level': 'high'
    },
    ExecutionMode.HYBRID
)

# The system will automatically:
# 1. Use network_ping to check host availability
# 2. Use network_port_scanner to identify open services
# 3. Use network_arp to detect potential ARP spoofing
# 4. Use network_traceroute to map network topology
# 5. Synthesize results into comprehensive security report
```

## 🎉 **Benefits of Integration**

✅ **Dynamic Tool Discovery**: Tools are automatically discovered based on problem description
✅ **Intelligent Tool Selection**: Query Path selects optimal tools based on relevance and performance
✅ **Context-Aware Execution**: Runner Agent executes tools with appropriate parameters
✅ **Performance Optimization**: System learns from tool performance and optimizes selection
✅ **Comprehensive Analysis**: Multiple tools work together for complete network insights
✅ **Security Integration**: Network tools integrate with security workflows
✅ **Scalable Architecture**: Easy to add new network tools and capabilities

## 🔮 **Future Enhancements**

- **Network Topology Mapping**: Automatic network discovery and mapping
- **Real-time Monitoring**: Continuous network monitoring and alerting
- **Machine Learning**: AI-powered network anomaly detection
- **Integration APIs**: REST APIs for external system integration
- **Custom Tool Development**: Framework for building custom network tools
- **Cloud Integration**: Support for cloud network analysis tools

This integration provides your Query Path and Runner Agent with powerful network analysis capabilities, making it a comprehensive cybersecurity and network management platform! 🌐🚀
