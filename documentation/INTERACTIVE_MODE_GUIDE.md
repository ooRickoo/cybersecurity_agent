# Interactive Cybersecurity AI Helper Mode

## 🚀 **Overview**

The Cybersecurity Agent now features an **interactive AI helper mode** that provides a conversational interface for cybersecurity tasks. This mode intelligently blends local tools with LLM tasks, using local processing when possible for speed and only calling LLMs when complex analysis is needed.

## 🎯 **Key Features**

### **⚡ Smart Tool Selection**
- **Local tools first**: Simple queries use fast local processing
- **LLM when needed**: Complex analysis automatically uses LLM capabilities
- **Hybrid approach**: Moderate complexity uses a blend of both
- **Intelligent routing**: Automatically determines the best approach

### **🛡️ Interactive Cybersecurity Helper**
- **10 unique welcome messages**: Randomly selected for variety
- **10 unique goodbye messages**: Personalized farewells
- **Comprehensive menu system**: Easy navigation and discovery
- **Natural language queries**: Ask questions in plain English

### **🔄 Dynamic Workflow Integration**
- **Seamless workflow access**: CSV enrichment and other workflows
- **Context-aware responses**: Understands your current session
- **Progressive problem solving**: Break down complex issues step by step

## 🚀 **Getting Started**

### **Starting Interactive Mode**
```bash
# Run without parameters to enter interactive mode
python3 cs_util_lg.py

# Or explicitly start interactive mode
python3 cs_util_lg.py  # No command = interactive mode
```

### **Quick Exit**
```bash
# Type any of these to exit:
quit, exit, bye, q
```

## 📋 **Main Menu Options**

### **🔧 Tools**
```bash
tools
# or
t
```
**What it does**: Shows all available MCP tools organized by category
**Use case**: Discover what capabilities are available

### **🔄 Workflows**
```bash
workflows
# or
w
```
**What it does**: Lists available workflow templates and systems
**Use case**: See what automated workflows you can run

### **📊 Status**
```bash
status
# or
s
```
**What it does**: Shows system health, tool availability, and performance
**Use case**: Check if everything is working correctly

### **📈 CSV Enrichment**
```bash
csv-enrich
```
**What it does**: Interactive CSV enrichment workflow with guided setup
**Use case**: Enrich CSV data with LLM processing

### **🔍 Analyze**
```bash
analyze [your query]
# Example: analyze this log file for threats
```
**What it does**: Analyzes data using local tools when possible, LLM when needed
**Use case**: Get insights from data without manual processing

### **🏷️ Categorize**
```bash
categorize [your data]
# Example: categorize these security events by severity
```
**What it does**: Categorizes data using local classification tools or LLM
**Use case**: Organize and classify security information

### **📝 Summarize**
```bash
summarize [your content]
# Example: summarize the security findings
```
**What it does**: Creates summaries using local tools or LLM processing
**Use case**: Get concise overviews of long reports or data

### **❓ Help**
```bash
help
# or
h
# or
?
```
**What it does**: Shows detailed help and usage examples
**Use case**: Learn how to use the system effectively

### **🧹 Clear Screen**
```bash
clear
# or
cls
```
**What it does**: Clears the terminal screen
**Use case**: Clean up the display for better readability

## 💬 **Natural Language Queries**

### **Direct Questions**
You can ask questions directly without using specific commands:

```
🔍 What would you like to do? (or type 'help' for options): 
Show me tools for network analysis
```

### **Complex Requests**
```
🔍 What would you like to do? (or type 'help' for options): 
I need to analyze this log file for potential security threats and categorize them by severity
```

### **Workflow Requests**
```
🔍 What would you like to do? (or type 'help' for options): 
Help me enrich some threat data with additional context
```

## ⚡ **Smart Processing Logic**

### **Query Complexity Assessment**
The system automatically analyzes your queries and chooses the best approach:

#### **Simple Queries (Local Tools)**
- **Keywords**: `show`, `list`, `count`, `basic`, `simple`, `status`, `tools`
- **Length**: Short queries (< 10 words)
- **Examples**: "Show me tools", "List workflows", "System status"

#### **Moderate Queries (Hybrid Approach)**
- **Keywords**: Mixed complexity indicators
- **Length**: Medium queries (5-15 words)
- **Examples**: "Analyze this data", "Categorize events", "Summarize findings"

#### **Complex Queries (LLM Required)**
- **Keywords**: `analyze`, `understand`, `explain`, `why`, `how`, `complex`, `relationship`
- **Length**: Long queries (> 15 words)
- **Examples**: "Explain why this security event occurred and what it means for our infrastructure"

### **Intelligent Tool Selection**
```
⚡ Using local tools for fast analysis...
✅ Local analysis result: [result]

🤖 Using LLM for complex analysis...
✅ LLM analysis result: [result]

🔄 Using hybrid approach...
✅ Hybrid handling result: [result]
```

## 📊 **CSV Enrichment Workflow**

### **Interactive Setup**
When you run `csv-enrich`, the system guides you through the setup:

```
📊 CSV Enrichment Workflow
----------------------------------------
📥 Input CSV file path: sample_data.csv
📤 Output CSV file path: enriched_data.csv
🤖 Enrichment prompt: Analyze threat level and categorize each entry
📦 Batch size (default 100): 50
🔄 Max retries (default 3): 5
🎯 Quality threshold 0.0-1.0 (default 0.8): 0.9
```

### **Automatic Execution**
The system then runs the complete workflow:
1. **Import** CSV data
2. **Analyze** columns and determine new ones needed
3. **Create** new columns
4. **Process** with LLM (row by row)
5. **Validate** data quality
6. **Export** enriched results

## 🎭 **Welcome & Goodbye Messages**

### **Welcome Messages (Randomly Selected)**
- 🛡️ Welcome to the Cybersecurity AI Helper! Ready to defend the digital realm?
- 🚀 Greetings, cyber warrior! Your AI assistant is online and ready for action.
- 🔒 Hello there! Time to secure, analyze, and protect. What's on your mind?
- ⚡ Welcome to the command center! Let's make cybersecurity simple and effective.
- 🕵️ Greetings, digital detective! Ready to investigate and secure?
- 🛡️ Hello, security specialist! Your AI partner is here to help.
- 🚀 Welcome aboard! Let's navigate the cybersecurity landscape together.
- 🔒 Greetings! Time to turn complex security challenges into simple solutions.
- ⚡ Hello there! Ready to accelerate your security operations?
- 🕵️ Welcome, cyber investigator! Let's solve security mysteries together.

### **Goodbye Messages (Randomly Selected)**
- 🛡️ Stay secure out there! Until next time, cyber warrior.
- 🚀 Mission accomplished! Keep defending the digital frontier.
- 🔒 Stay vigilant and secure! Your AI assistant will be here when you return.
- ⚡ Powering down... but your security knowledge remains active!
- 🕵️ Case closed for now! Stay sharp and stay safe.
- 🛡️ Logging out... but the security never stops!
- 🚀 Shutting down systems... until our next security mission!
- 🔒 Disconnecting... but your security awareness stays connected!
- ⚡ Powering off... but your cyber defenses remain online!
- 🕵️ Signing off... but the investigation never truly ends!

## 🔧 **Advanced Usage**

### **Keyboard Shortcuts**
- **Ctrl+C**: Interrupt current operation
- **Ctrl+D**: Exit (EOF)
- **Enter**: Submit command

### **Command Chaining**
You can combine commands and queries:
```
🔍 What would you like to do? (or type 'help' for options): 
tools
🔧 Available MCP Tools:
[shows tools]

🔍 What would you like to do? (or type 'help' for options): 
analyze the network traffic data
🔍 Analyzing: the network traffic data
⚡ Using local tools for fast analysis...
✅ Local analysis result: [result]
```

### **Context Awareness**
The system remembers your session and can provide contextual responses:
```
🔍 What would you like to do? (or type 'help' for options): 
What tools did you just show me?
[system can reference previous interactions]
```

## 🚀 **Performance Optimization**

### **Local Tool Priority**
- **Fast response**: Local tools respond in milliseconds
- **No API calls**: Avoids external service delays
- **Resource efficient**: Uses local processing power

### **LLM Optimization**
- **Only when needed**: LLMs called only for complex tasks
- **Batch processing**: Multiple items processed together
- **Caching**: Results cached for similar queries

### **Hybrid Approach**
- **Best of both**: Combines local speed with LLM intelligence
- **Progressive enhancement**: Starts local, adds LLM if needed
- **Fallback handling**: Graceful degradation if tools fail

## 🔍 **Troubleshooting**

### **Common Issues**

#### **No Tools Available**
```
❌ No tools available. Try running 'discover-tools' first.
```
**Solution**: Run `discover-tools` command or restart the CLI

#### **MCP Server Not Available**
```
❌ MCP Server: Not available
```
**Solution**: Check if the tool manager is properly initialized

#### **Interactive Mode Not Starting**
```
🚀 Starting Interactive Cybersecurity AI Helper...
[stuck on loading]
```
**Solution**: Check system resources and tool availability

### **Debug Mode**
Enable detailed logging for troubleshooting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### **Error Recovery**
The system automatically handles errors and continues:
```
❌ Error: [error description]
[system continues and shows menu again]
```

## 📚 **Integration Examples**

### **With Existing Workflows**
```
🔍 What would you like to do? (or type 'help' for options): 
csv-enrich
[interactive CSV enrichment workflow]
```

### **With MCP Tools**
```
🔍 What would you like to do? (or type 'help' for options): 
tools
[shows available MCP tools]
```

### **With Agentic Workflows**
```
🔍 What would you like to do? (or type 'help' for options): 
workflows
[shows available workflow templates]
```

## 🚀 **Future Enhancements**

### **Planned Features**
- **Voice interaction**: Speech-to-text and text-to-speech
- **Visual interface**: Web-based GUI option
- **Multi-language support**: Internationalization
- **Advanced context**: Long-term memory and learning
- **Custom workflows**: User-defined workflow creation

### **Extensibility**
The interactive mode is designed to be easily extensible:
- **New commands**: Add custom interactive commands
- **Tool integration**: Integrate new tools seamlessly
- **Workflow types**: Add new workflow categories
- **Response types**: Customize output formats

## 📖 **References**

- **Main CLI**: `cs_util_lg.py`
- **Interactive Mode**: `run_interactive_mode()` method
- **Workflow Templates**: `bin/workflow_templates.py`
- **CSV Enrichment**: `bin/csv_enrichment_executor.py`
- **MCP Integration**: `bin/cs_ai_tools.py`

---

**Status**: ✅ **Fully Implemented and Tested**

The Interactive Cybersecurity AI Helper Mode is now fully functional and provides a powerful, intelligent interface for cybersecurity tasks. It automatically optimizes performance by using local tools when possible and LLMs only when needed, giving you the best of both worlds: speed and intelligence.
