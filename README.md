# 🛡️ Cybersecurity Agent - LangGraph Cybersecurity Agent

## Overview

Cybersecurity Agent is a comprehensive, AI-powered cybersecurity agent built with LangGraph that provides advanced threat hunting, policy analysis, incident response, and organizational knowledge management capabilities. The system integrates multiple security tools, maintains contextual memory, and supports complex workflow orchestration for cybersecurity operations.

## 🚀 Key Features

### **Core Capabilities**
- **AI-powered Analysis**: Advanced threat detection and analysis using LangChain and OpenAI
- **Workflow Orchestration**: Complex workflow management with verification and backtracking
- **Contextual Memory**: Multi-tier memory system for long-term knowledge retention
- **Multi-Platform Integration**: Support for Splunk, Google Chronicle, Azure, AWS, and more
- **Policy Compliance**: Automated policy mapping and compliance validation

### **Knowledge Management**
- **Organizational Knowledge Base**: Centralized storage for policies, technologies, and configurations
- **Cross-Platform Context**: Understanding of different SIEM and security tool capabilities
- **Policy Mapping**: Direct correlation between policies, requirements, and technology solutions
- **Searchable Memory**: Intelligent search across all organizational knowledge

### **Workflow Verification**
- **"Check Our Math" System**: Automated verification of workflow accuracy and completeness
- **Loop Prevention**: Intelligent backtracking to avoid repeated failures
- **Template Management**: Dynamic workflow template selection and optimization
- **Performance Analytics**: Continuous improvement through workflow analysis

## 🏗️ Architecture

```
Cybersecurity Agent/
├── langgraph_cybersecurity_agent.py    # Main agent implementation
├── bin/                                # Core utilities and tools
│   ├── workflow_verification_system.py # Workflow verification engine
│   ├── context_memory_manager.py       # Memory management system
│   ├── master_catalog.py               # Knowledge domain management
│   ├── active_directory_tools.py       # AD integration tools
│   ├── database_mcp_tools.py           # Database management tools
│   └── ...                            # Additional specialized tools
├── knowledge-objects/                  # Knowledge base storage
├── session-viewer/                     # Web-based session management
├── etc/                                # Configuration and credentials
├── documentation/                      # Comprehensive documentation
└── requirements.txt                    # Python dependencies
```

## 🚀 Quick Start

### **Prerequisites**
- Python 3.8+
- OpenAI API key (for AI capabilities)
- Access to security tools (Splunk, Chronicle, etc.)

### **Installation**
```bash
# Clone the repository
git clone <repository-url>
cd Cybersecurity-Agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
```

### **Basic Usage**
```bash
# Start the unified CLI interface (full features)
python cs_util_lg.py

# Process CSV files
python cs_util_lg.py -csv data.csv -output results.json

# Execute advanced workflows
python cs_util_lg.py -workflow threat_hunting -problem "Investigate APT29 activity"

# List available workflows
python cs_util_lg.py -list-workflows

# Interactive mode
python cs_util_lg.py
```

## 📚 Documentation

### **Core Guides**
- **[Knowledge Base Setup Guide](KNOWLEDGE_BASE_SETUP_GUIDE.md)** - Comprehensive guide for setting up organizational knowledge bases
- **[Quick Reference Guide](QUICK_REFERENCE_GUIDE.md)** - Fast reference for common operations
- **[Workflow Verification Guide](documentation/WORKFLOW_VERIFICATION_GUIDE.md)** - Understanding the verification system

### **Specialized Documentation**
- **[Session Viewer Usage](documentation/SESSION_VIEWER_USAGE_GUIDE.md)** - Web interface for session management
- **[Database and AD Tools](documentation/DATABASE_AND_AD_TOOLS_GUIDE.md)** - Database and Active Directory integration
- **[Implementation Summary](documentation/IMPLEMENTATION_SUMMARY.md)** - Technical implementation details

## 🔧 Configuration

### **Environment Variables**
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4

# Security Tool Credentials
SPLUNK_HOST=your_splunk_host
SPLUNK_USERNAME=your_username
SPLUNK_PASSWORD=your_password

# Database Connections
DATABASE_URL=your_database_connection_string
```

### **Knowledge Base Setup**
1. **Prepare Data**: Organize your CSV/JSON files in a `data/` directory
2. **Run Assessment**: Use the data quality assessment script
3. **Transform Data**: Choose between transformation or direct import
4. **Configure Memory**: Set up memory tiers and knowledge domains
5. **Test Integration**: Validate with sample workflows

## 🎯 Use Cases

### **Content Migration**
- **Google Chronicle to Splunk ES**: Automated conversion of security monitoring content
- **Policy Mapping**: Correlate policies with technology implementations
- **Compliance Validation**: Ensure security controls meet policy requirements

### **Threat Hunting**
- **Intelligence Correlation**: Connect threat indicators across platforms
- **Historical Analysis**: Leverage long-term memory for pattern recognition
- **Automated Response**: Trigger workflows based on threat detection

### **Incident Response**
- **Contextual Analysis**: Understand incident context using organizational knowledge
- **Workflow Orchestration**: Coordinate response across multiple tools
- **Documentation**: Automatic generation of incident reports and lessons learned

## 🔒 Security Features

- **Credential Vault**: Encrypted storage of sensitive credentials
- **Host Verification**: Device-bound encryption and verification
- **Access Control**: Role-based access to different system components
- **Audit Logging**: Comprehensive logging of all system activities

## 🧪 Testing

```bash
# Run all tests
pytest

# Test specific components
pytest bin/test_verification_integration.py
pytest bin/test_session_viewer_integration.py

# Test knowledge base
python bin/test_knowledge_base_integration.py
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### **Development Guidelines**
- Follow PEP 8 style guidelines
- Add comprehensive tests for new features
- Update documentation for any API changes
- Use type hints for function parameters

## 📊 Performance

- **Memory Management**: Efficient multi-tier memory system with automatic cleanup
- **Search Optimization**: Fast full-text search across all knowledge domains
- **Workflow Execution**: Parallel processing of independent workflow steps
- **Scalability**: Designed to handle large organizational knowledge bases

## 🔮 Roadmap

### **Short Term (1-3 months)**
- Enhanced AI model integration
- Additional SIEM platform support
- Improved workflow templates

### **Medium Term (3-6 months)**
- Machine learning for threat detection
- Advanced visualization capabilities
- API-first architecture

### **Long Term (6+ months)**
- Federated knowledge sharing
- Advanced threat intelligence integration
- Autonomous security operations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check the comprehensive guides in the `documentation/` folder
- **Issues**: Report bugs and feature requests via GitHub Issues
- **Discussions**: Join community discussions in GitHub Discussions
- **Security**: Report security vulnerabilities via security@example.com

## 🙏 Acknowledgments

- **LangGraph Team**: For the excellent workflow orchestration framework
- **OpenAI**: For providing the AI capabilities that power the agent
- **Security Community**: For feedback and contributions to the project

---

**Stay vigilant, Security Professional! 🛡️**

*Cybersecurity Agent - Your AI-powered cybersecurity companion*
