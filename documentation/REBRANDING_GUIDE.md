# 🔄 Rebranding Guide

This guide explains how to easily rebrand the Cybersecurity Agent project using the automated rebranding script.

## 🚀 Quick Rebranding

### **Default Rebranding (Recommended)**
```bash
# Run the default rebranding (CyberGuard → Cybersecurity Agent)
python bin/rebrand_agent.py
```

### **Custom Rebranding**
```bash
# Custom rebranding example
python bin/rebrand_agent.py "OldName" "NewName"

# Example: Change "Cybersecurity Agent" to "SecurityBot"
python bin/rebrand_agent.py "Cybersecurity Agent" "SecurityBot"
```

## 📋 What Gets Updated

The rebranding script automatically updates:

### **File Contents**
- ✅ Python files (`.py`)
- ✅ Documentation (`.md`)
- ✅ JavaScript/React files (`.js`, `.jsx`, `.ts`, `.tsx`)
- ✅ HTML files (`.html`)
- ✅ Configuration files (`.txt`, `.json`)

### **Branding Elements**
- ✅ Project names and titles
- ✅ User interface text
- ✅ Chat outputs and responses
- ✅ Documentation references
- ✅ Component labels and headers

### **Excluded Directories**
- ❌ `.git/` - Version control
- ❌ `__pycache__/` - Python cache
- ❌ `node_modules/` - Node.js dependencies
- ❌ `venv/`, `.venv/` - Virtual environments

## 🔧 How It Works

1. **Discovery**: Scans all project files for branding references
2. **Content Update**: Replaces old branding with new branding in file contents
3. **Filename Update**: Optionally renames files containing old branding
4. **Reporting**: Shows exactly what was changed

## 📊 Example Output

```
🔄 Using default rebranding mappings

📋 Rebranding mappings:
  • CyberGuard → Cybersecurity Agent
  • cyberguard → cybersecurity agent
  • Security Analyst → Security Professional

📁 Found 357 files to process

🔄 Updating file contents...
✅ Updated: README.md
✅ Updated: cs_util_lg.py
✅ Updated: session-viewer/client/src/components/Header.js

📊 Content updates: 15/357 files modified
🎉 Rebranding complete!
```

## 🚨 Safety Features

### **Backup Before Rebranding**
```bash
# Create a backup branch
git checkout -b backup-before-rebranding
git add .
git commit -m "Backup before rebranding"

# Run rebranding
python bin/rebrand_agent.py

# If something goes wrong, restore from backup
git checkout backup-before-rebranding
```

### **Revert Changes**
```bash
# Revert all changes
git checkout .

# Or revert specific files
git checkout README.md cs_util_lg.py
```

## 🎯 Common Rebranding Scenarios

### **1. Project Name Change**
```bash
python bin/rebrand_agent.py "Cybersecurity Agent" "SecurityBot"
```

### **2. Company Rebranding**
```bash
python bin/rebrand_agent.py "OldCompany" "NewCompany"
```

### **3. Product Line Change**
```bash
python bin/rebrand_agent.py "Threat Hunter" "Security Analyst"
```

### **4. Version Update**
```bash
python bin/rebrand_agent.py "v1.0" "v2.0"
```

## 🔍 Verification

After rebranding, verify the changes:

```bash
# Search for old branding (should return no results)
grep -r "CyberGuard" . --exclude-dir=.git --exclude-dir=__pycache__

# Search for new branding (should return many results)
grep -r "Cybersecurity Agent" . --exclude-dir=.git --exclude-dir=__pycache__
```

## 💡 Tips

1. **Test First**: Run on a copy of your project first
2. **Commit Changes**: Always commit before rebranding
3. **Review Changes**: Check the output to ensure expected files were updated
4. **Update External**: Remember to update external references (websites, documentation, etc.)

## 🆘 Troubleshooting

### **Script Not Found**
```bash
# Ensure you're in the project root
cd /path/to/Cybersecurity-Agent
python bin/rebrand_agent.py
```

### **Permission Errors**
```bash
# Make script executable
chmod +x bin/rebrand_agent.py
```

### **Encoding Issues**
```bash
# The script handles UTF-8 encoding automatically
# If issues persist, check file encodings
file -i filename.txt
```

## 📞 Support

If you encounter issues with the rebranding script:

1. Check the error messages for specific file paths
2. Ensure you have write permissions to the project directory
3. Verify Python 3.6+ is installed
4. Check that all required modules are available

---

**Remember**: The rebranding script is designed to be safe and reversible. Always test on a copy first and commit your changes before running!
