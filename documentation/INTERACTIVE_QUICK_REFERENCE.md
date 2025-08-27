# 🛡️ Interactive Cybersecurity AI Helper - Quick Reference

## 🚀 **Start Interactive Mode**
```bash
python3 cs_util_lg.py  # No parameters = interactive mode
```

## 📋 **Quick Commands**

| Command | Shortcut | What It Does |
|---------|----------|--------------|
| `help` | `h`, `?` | Show detailed help |
| `menu` | `m` | Show main menu |
| `tools` | `t` | List available MCP tools |
| `workflows` | `w` | Show available workflows |
| `status` | `s` | Show system status |
| `csv-enrich` | - | Start CSV enrichment workflow |
| `clear` | `cls` | Clear screen |
| `quit` | `q`, `exit`, `bye` | Exit interactive mode |

## 🔍 **Smart Query Handling**

### **⚡ Local Tools (Fast)**
- Simple queries: `show`, `list`, `status`, `tools`
- Short queries: < 10 words
- **Response time**: Milliseconds

### **🔄 Hybrid Approach**
- Moderate complexity: 5-15 words
- Mixed indicators
- **Response time**: Seconds

### **🤖 LLM Required (Intelligent)**
- Complex queries: `analyze`, `explain`, `understand`, `why`, `how`
- Long queries: > 15 words
- **Response time**: Variable (API dependent)

## 📊 **CSV Enrichment Workflow**

```bash
csv-enrich
```
**Interactive Setup**:
1. Input CSV file path
2. Output CSV file path  
3. Enrichment prompt
4. Batch size (default: 100)
5. Max retries (default: 3)
6. Quality threshold (default: 0.8)

## 💬 **Natural Language Examples**

### **Direct Questions**
```
Show me tools for network analysis
What workflows are available?
Help me understand this log file
```

### **Workflow Requests**
```
I need to enrich some threat data
Analyze this CSV for security threats
Categorize these log entries by severity
```

### **Analysis Requests**
```
analyze this log file for threats
categorize these security events
summarize the security findings
```

## 🎭 **Welcome & Goodbye Messages**

**10 Unique Welcome Messages** - Randomly selected each time
**10 Unique Goodbye Messages** - Personalized farewells

## ⚡ **Performance Features**

- **Local tools first** for speed
- **LLM only when needed** for intelligence
- **Intelligent routing** based on query complexity
- **Batch processing** for large datasets
- **Progress tracking** for long operations

## 🔧 **Integration**

- **Seamless MCP tool access**
- **Workflow template integration**
- **Agentic workflow system**
- **Context memory management**
- **Session logging and tracking**

## 🚀 **Pro Tips**

1. **Start simple**: Use `tools` and `workflows` to discover capabilities
2. **Be specific**: Clear queries get better responses
3. **Use natural language**: Ask questions as you would to a colleague
4. **Leverage local tools**: Simple operations are lightning fast
5. **Save complex analysis**: Use LLM for truly complex problems

---

**💡 Remember**: The system automatically chooses the best approach for your query!
**⚡ Fast local tools** for simple tasks, **🤖 intelligent LLM** for complex analysis.
