# 🚀 **Quick Start Guide - LangGraph Cybersecurity Agent**

## ⚡ **Get Running in 5 Minutes**

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Start Interactive Mode**
```bash
python cs_util_lg.py
```

### **3. Try Your First Command**
```
🤖 You: Add MITRE ATT&CK framework to knowledge base
```

---

## 🔐 **Enable Encryption (Optional)**

### **Basic Setup**
```bash
# Enable encryption
export ENCRYPTION_ENABLED=true
export ENCRYPTION_PASSWORD_HASH="$(echo -n 'YourPassword' | sha256sum | cut -d' ' -f1)"

# Salt will be automatically generated and managed by the system
# For manual salt management, use the salt manager utility:
# python bin/salt_manager.py --info

# Or use the default
export ENCRYPTION_PASSWORD_HASH="$(echo -n 'Vosteen2025' | sha256sum | cut -d' ' -f1)"
export ENCRYPTION_SALT="cybersecurity_agent_salt"
```

### **Change Password Later**
```bash
python bin/encryption_manager.py --change-password "NewSecurePassword"
```

---

## 📁 **Process Your First File**

### **CSV Analysis**
```bash
# Create a sample CSV file
echo "policy_id,policy_name,description
POL001,Password Policy,Strong password requirements
POL002,Access Control,Role-based access control" > policies.csv

# Process with the agent
python cs_util_lg.py -csv policies.csv -output analysis.csv
```

### **Custom Analysis**
```bash
python cs_util_lg.py -csv policies.csv \
  -prompt "Map these policies to MITRE ATT&CK and assess risks" \
  -output policy_analysis.json
```

---

## 🧠 **Add Knowledge Frameworks**

### **In Interactive Mode**
```
🤖 You: Add MITRE ATT&CK framework to knowledge base
🤖 You: Add D3fend defense framework
🤖 You: Add NIST SP 800-53 controls
```

### **Query Knowledge**
```
🤖 You: What are the top threats for authentication policies?
🤖 You: Show me defense techniques for network attacks
🤖 You: List NIST controls for data protection
```

---

## 🔄 **Run Workflows**

### **Policy Analysis Workflow**
```
🤖 You: Run policy analysis workflow on my policies.csv file
```

### **Threat Intelligence Workflow**
```
🤖 You: Execute threat intelligence workflow for recent threats
```

### **Incident Response Workflow**
```
🤖 You: Start incident response workflow for security breach
```

---

## 🛠️ **Available Commands**

### **CLI Options**
```bash
# Interactive mode (default)
python cs_util_lg.py

# Process CSV file
python cs_util_lg.py -csv <file> -output <file>

# Custom prompt
python cs_util_lg.py -csv <file> -prompt "<your question>" -output <file>
```

### **Encryption Management**
```bash
# List encrypted files
python bin/encryption_manager.py --list-files

# Encrypt specific file
python bin/encryption_manager.py --encrypt-file sensitive.csv

# Decrypt specific file
python bin/encryption_manager.py --decrypt-file sensitive.csv.encrypted
```

---

## 📊 **Monitor Your Sessions**

### **Session Logs**
```bash
# View session logs
ls -la session-logs/

# Check specific session
cat session-logs/session_20241201_143022_abc12345.json
```

### **Session Outputs**
```bash
# View session outputs
ls -la session-outputs/

# Check specific session files
ls -la session-outputs/<session-id>/
```

---

## 🔧 **Troubleshooting**

### **Common Issues**
```bash
# Check Python version (requires 3.9+)
python --version

# Verify dependencies
pip list | grep -E "(langgraph|pandas|cryptography)"

# Test system
python documentation/test_langgraph_system.py
```

### **Reset System**
```bash
# Clear sessions (if needed)
rm -rf session-logs/* session-outputs/*

# Reset knowledge base (if needed)
rm -rf knowledge-objects/*
```

---

## 📚 **Next Steps**

### **Learn More**
- **`README.md`** - Project overview
- **`FEATURES.md`** - Complete feature list
- **`documentation/README_LANGGRAPH_SYSTEM.md`** - Detailed system guide

### **Advanced Usage**
- Create custom workflow templates
- Integrate with external MCP servers
- Develop custom tools and extensions
- Scale to multiple instances

---

## 🎯 **You're Ready!**

**Your LangGraph Cybersecurity Agent is now running and ready to:**
- 🔍 Analyze security policies and frameworks
- 🧠 Manage multi-dimensional knowledge
- 🔐 Encrypt sensitive data
- 🔄 Execute dynamic workflows
- 📊 Process and analyze security data

**Start exploring with: `python cs_util_lg.py`** 🚀
