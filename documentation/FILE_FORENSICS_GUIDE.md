# 📁 File Forensics Guide

## Overview

The File Tools Manager provides comprehensive digital forensics capabilities for file system analysis, metadata extraction, archive handling, and evidence collection. It's designed for cybersecurity investigations, incident response, and digital forensics workflows.

## Features

### **File Metadata Analysis**
- **Comprehensive Metadata**: File size, timestamps, permissions, ownership
- **Hash Calculation**: MD5, SHA1, SHA256 hash computation
- **File Type Detection**: Magic byte and MIME type identification
- **Entropy Analysis**: File entropy calculation for packed/encrypted content
- **String Extraction**: Readable string extraction from binary files

### **Archive Analysis**
- **Multi-Format Support**: ZIP, TAR, GZ, BZ2, XZ archive handling
- **Content Extraction**: Safe extraction of archive contents
- **Archive Creation**: Create archives for evidence preservation
- **Nested Archive Support**: Handle archives within archives
- **Corruption Detection**: Identify corrupted or malicious archives

### **File System Analysis**
- **Directory Tree Analysis**: Complete directory structure mapping
- **File Search**: Advanced file search with multiple criteria
- **Bulk Operations**: Process multiple files efficiently
- **File Operations**: Copy, move, delete with metadata preservation
- **Security Assessment**: File security level classification

### **Digital Forensics**
- **Evidence Collection**: Systematic evidence gathering
- **Chain of Custody**: Maintain evidence integrity
- **Timeline Analysis**: File system timeline reconstruction
- **Deleted File Recovery**: Attempt to recover deleted files
- **File Carving**: Extract files from unallocated space

## Usage

### **Command Line Interface**
```bash
# Analyze single file
python cs_util_lg.py -workflow data_conversion -problem "analyze file: /path/to/file.exe"

# Extract archive
python cs_util_lg.py -workflow data_conversion -problem "extract: /path/to/archive.zip"

# Scan directory
python cs_util_lg.py -workflow data_conversion -problem "scan directory: /suspicious/folder"

# File system analysis
python cs_util_lg.py -workflow data_conversion -problem "analyze filesystem: /evidence/partition"
```

### **Programmatic Usage**
```python
from bin.file_tools_manager import FileToolsManager, FileType, SecurityLevel

# Initialize file tools
ftm = FileToolsManager()

# Get file metadata
metadata = ftm.get_file_metadata("/path/to/file.exe")
print(f"File type: {metadata.file_type}")
print(f"Entropy: {metadata.entropy}")
print(f"Security level: {metadata.security_level}")

# Analyze file
analysis = ftm.analyze_file("/path/to/file.exe")
print(f"Risk score: {analysis.risk_score}")
print(f"Findings: {analysis.findings}")

# Extract archive
result = ftm.extract_archive("/path/to/archive.zip")
print(f"Extracted {result['file_count']} files")
```

## File Analysis Methods

### **1. Metadata Extraction**

#### **Basic File Information**
```python
metadata = ftm.get_file_metadata("/path/to/file.exe")

# File properties
print(f"File name: {metadata.file_name}")
print(f"File size: {metadata.file_size} bytes")
print(f"File type: {metadata.file_type}")
print(f"MIME type: {metadata.mime_type}")

# Timestamps
print(f"Created: {metadata.created_time}")
print(f"Modified: {metadata.modified_time}")
print(f"Accessed: {metadata.accessed_time}")

# Permissions and ownership
print(f"Permissions: {metadata.permissions}")
print(f"Owner: {metadata.owner}")
print(f"Group: {metadata.group}")
```

#### **Hash Calculation**
```python
# Multiple hash algorithms
print(f"MD5: {metadata.md5_hash}")
print(f"SHA1: {metadata.sha1_hash}")
print(f"SHA256: {metadata.sha256_hash}")

# Use hashes for file identification
if metadata.sha256_hash in known_malware_hashes:
    print("⚠️ Known malware hash detected!")
```

#### **File Type Detection**
```python
# Automatic file type detection
file_types = {
    FileType.TEXT: "Text file",
    FileType.BINARY: "Binary file", 
    FileType.ARCHIVE: "Archive file",
    FileType.EXECUTABLE: "Executable file",
    FileType.IMAGE: "Image file",
    FileType.DOCUMENT: "Document file"
}

print(f"Detected type: {file_types.get(metadata.file_type, 'Unknown')}")
```

### **2. Entropy Analysis**

#### **Entropy Calculation**
```python
# Calculate file entropy
entropy = ftm._calculate_entropy("/path/to/file.exe")
print(f"File entropy: {entropy:.2f}")

# Interpret entropy values
if entropy > 7.5:
    print("High entropy - possibly packed or encrypted")
elif entropy > 6.0:
    print("Medium entropy - compressed or encoded")
else:
    print("Low entropy - normal file content")
```

#### **Packed File Detection**
```python
# Detect packed executables
if metadata.entropy > 7.5 and metadata.file_type == "executable":
    print("⚠️ Possible packed executable detected")
    print("Recommendation: Analyze with malware detection tools")
```

### **3. String Extraction**

#### **String Analysis**
```python
# Extract strings from file
strings = ftm._extract_strings("/path/to/file.exe", min_length=4)
print(f"Found {len(strings)} strings")

# Analyze suspicious strings
suspicious_patterns = [
    "cmd.exe", "powershell", "regsvr32", "rundll32",
    "download", "execute", "payload", "backdoor"
]

suspicious_strings = []
for string in strings:
    for pattern in suspicious_patterns:
        if pattern.lower() in string.lower():
            suspicious_strings.append(string)
            break

if suspicious_strings:
    print(f"⚠️ Found {len(suspicious_strings)} suspicious strings")
    for s in suspicious_strings[:10]:  # Show first 10
        print(f"  • {s}")
```

### **4. Security Assessment**

#### **Security Level Classification**
```python
# Automatic security assessment
security_levels = {
    SecurityLevel.SAFE: "✅ Safe",
    SecurityLevel.SUSPICIOUS: "⚠️ Suspicious", 
    SecurityLevel.MALICIOUS: "🚨 Malicious",
    SecurityLevel.UNKNOWN: "❓ Unknown"
}

print(f"Security level: {security_levels.get(metadata.security_level, 'Unknown')}")
```

#### **Risk Factors**
```python
# Analyze risk factors
risk_factors = []

if metadata.entropy > 7.5:
    risk_factors.append("High entropy (packed/encrypted)")

if metadata.file_type == "executable":
    risk_factors.append("Executable file")

if metadata.file_name.endswith(('.scr', '.pif', '.bat', '.cmd')):
    risk_factors.append("Suspicious file extension")

if len(risk_factors) > 0:
    print("Risk factors identified:")
    for factor in risk_factors:
        print(f"  • {factor}")
```

## Archive Analysis

### **Archive Extraction**

#### **Supported Formats**
```python
# Extract different archive types
archive_formats = {
    '.zip': 'ZIP archive',
    '.tar': 'TAR archive', 
    '.tar.gz': 'GZIP compressed TAR',
    '.tgz': 'GZIP compressed TAR',
    '.gz': 'GZIP compressed file',
    '.bz2': 'BZIP2 compressed file',
    '.xz': 'XZ compressed file'
}

# Extract archive
result = ftm.extract_archive("/path/to/archive.zip")
if result["success"]:
    print(f"Extracted to: {result['extract_path']}")
    print(f"Files extracted: {result['file_count']}")
    for file in result['extracted_files'][:10]:  # Show first 10
        print(f"  • {file}")
```

#### **Safe Extraction**
```python
# Extract with safety checks
def safe_extract(archive_path, extract_to=None):
    """Safely extract archive with validation."""
    try:
        # Validate archive first
        if not ftm._validate_archive(archive_path):
            return {"success": False, "error": "Invalid or corrupted archive"}
        
        # Extract to temporary directory
        result = ftm.extract_archive(archive_path, extract_to)
        
        # Scan extracted files for malware
        if result["success"]:
            for file_path in result["extracted_files"]:
                full_path = Path(result["extract_path"]) / file_path
                if full_path.is_file():
                    analysis = ftm.analyze_file(full_path)
                    if analysis.risk_score > 0.5:
                        print(f"⚠️ Suspicious file extracted: {file_path}")
        
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### **Archive Creation**

#### **Create Evidence Archives**
```python
# Create archive for evidence preservation
def create_evidence_archive(source_path, archive_path):
    """Create archive for evidence preservation."""
    try:
        # Create archive with metadata preservation
        success = ftm.create_archive(source_path, archive_path, "zip")
        
        if success:
            # Verify archive integrity
            verification = ftm._verify_archive(archive_path)
            if verification:
                print(f"✅ Evidence archive created: {archive_path}")
                return True
            else:
                print("❌ Archive verification failed")
                return False
        else:
            print("❌ Archive creation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error creating archive: {e}")
        return False
```

## File System Analysis

### **Directory Analysis**

#### **Directory Tree Mapping**
```python
# Get complete directory structure
tree = ftm.get_directory_tree("/evidence/partition", max_depth=5)
print(f"Directory: {tree['name']}")
print(f"Type: {tree['type']}")
print(f"Size: {tree['size']} bytes")
print(f"Modified: {tree['modified']}")

# Analyze children
if "children" in tree:
    for child in tree["children"][:10]:  # Show first 10
        print(f"  • {child['name']} ({child['type']})")
```

#### **File Search**
```python
# Search for specific file types
executables = ftm.find_files(
    "/evidence/partition",
    pattern="*.exe",
    file_type=FileType.EXECUTABLE,
    min_size=1024,  # At least 1KB
    max_size=10*1024*1024  # At most 10MB
)

print(f"Found {len(executables)} executable files")
for exe in executables:
    print(f"  • {exe}")
```

#### **Suspicious File Detection**
```python
# Find suspicious files
def find_suspicious_files(directory):
    """Find potentially suspicious files."""
    suspicious_files = []
    
    # Search for executables
    executables = ftm.find_files(directory, file_type=FileType.EXECUTABLE)
    
    for exe in executables:
        metadata = ftm.get_file_metadata(exe)
        analysis = ftm.analyze_file(exe)
        
        # Check for suspicious characteristics
        if (analysis.risk_score > 0.5 or 
            metadata.entropy > 7.5 or
            analysis.security_level == SecurityLevel.SUSPICIOUS):
            suspicious_files.append({
                "file": exe,
                "risk_score": analysis.risk_score,
                "entropy": metadata.entropy,
                "security_level": analysis.security_level
            })
    
    return suspicious_files

# Find suspicious files
suspicious = find_suspicious_files("/evidence/partition")
print(f"Found {len(suspicious)} suspicious files")
for file_info in suspicious:
    print(f"  • {file_info['file']} (Risk: {file_info['risk_score']:.2f})")
```

### **Timeline Analysis**

#### **File System Timeline**
```python
def create_file_timeline(directory):
    """Create timeline of file system activities."""
    timeline = []
    
    # Get all files
    all_files = ftm.find_files(directory)
    
    for file_path in all_files:
        metadata = ftm.get_file_metadata(file_path)
        
        # Add timeline entries
        timeline.append({
            "timestamp": metadata.created_time,
            "event": "FILE_CREATED",
            "file": file_path,
            "size": metadata.file_size
        })
        
        timeline.append({
            "timestamp": metadata.modified_time,
            "event": "FILE_MODIFIED", 
            "file": file_path,
            "size": metadata.file_size
        })
        
        timeline.append({
            "timestamp": metadata.accessed_time,
            "event": "FILE_ACCESSED",
            "file": file_path,
            "size": metadata.file_size
        })
    
    # Sort by timestamp
    timeline.sort(key=lambda x: x["timestamp"])
    
    return timeline

# Create timeline
timeline = create_file_timeline("/evidence/partition")
print(f"Timeline entries: {len(timeline)}")

# Show recent activity
recent_activity = timeline[-20:]  # Last 20 events
for event in recent_activity:
    print(f"{event['timestamp']}: {event['event']} - {event['file']}")
```

## Digital Forensics Workflows

### **Evidence Collection**

#### **Systematic Evidence Gathering**
```python
def collect_evidence(target_directory, output_directory):
    """Systematically collect evidence from target directory."""
    evidence_log = []
    
    # Create output directory
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Create directory tree
    tree = ftm.get_directory_tree(target_directory)
    tree_file = output_path / "directory_tree.json"
    with open(tree_file, 'w') as f:
        json.dump(tree, f, indent=2)
    evidence_log.append(f"Directory tree: {tree_file}")
    
    # 2. Extract all archives
    archives = ftm.find_files(target_directory, file_type=FileType.ARCHIVE)
    for archive in archives:
        result = ftm.extract_archive(archive, output_path / "extracted")
        if result["success"]:
            evidence_log.append(f"Extracted: {archive}")
    
    # 3. Analyze suspicious files
    suspicious = find_suspicious_files(target_directory)
    suspicious_file = output_path / "suspicious_files.json"
    with open(suspicious_file, 'w') as f:
        json.dump(suspicious, f, indent=2)
    evidence_log.append(f"Suspicious files: {suspicious_file}")
    
    # 4. Create timeline
    timeline = create_file_timeline(target_directory)
    timeline_file = output_path / "file_timeline.json"
    with open(timeline_file, 'w') as f:
        json.dump(timeline, f, indent=2)
    evidence_log.append(f"Timeline: {timeline_file}")
    
    # 5. Generate evidence report
    report = {
        "collection_timestamp": datetime.now().isoformat(),
        "target_directory": target_directory,
        "output_directory": output_directory,
        "evidence_files": evidence_log,
        "summary": {
            "total_files": len(ftm.find_files(target_directory)),
            "suspicious_files": len(suspicious),
            "archives_found": len(archives)
        }
    }
    
    report_file = output_path / "evidence_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

# Collect evidence
evidence_report = collect_evidence("/evidence/partition", "/output/evidence")
print("Evidence collection complete")
print(f"Report: {evidence_report['evidence_files']}")
```

### **Chain of Custody**

#### **Evidence Integrity**
```python
def maintain_chain_of_custody(file_path, action, operator):
    """Maintain chain of custody for evidence files."""
    custody_log = {
        "file_path": file_path,
        "action": action,
        "operator": operator,
        "timestamp": datetime.now().isoformat(),
        "file_hash": None,
        "notes": ""
    }
    
    # Calculate file hash for integrity verification
    if Path(file_path).exists():
        metadata = ftm.get_file_metadata(file_path)
        custody_log["file_hash"] = metadata.sha256_hash
    
    # Log the action
    log_file = Path(file_path).parent / "custody_log.json"
    
    if log_file.exists():
        with open(log_file, 'r') as f:
            log_entries = json.load(f)
    else:
        log_entries = []
    
    log_entries.append(custody_log)
    
    with open(log_file, 'w') as f:
        json.dump(log_entries, f, indent=2)
    
    return custody_log

# Example usage
custody_entry = maintain_chain_of_custody(
    "/evidence/file.exe",
    "COLLECTED",
    "Investigator Smith"
)
print(f"Custody logged: {custody_entry['timestamp']}")
```

## Integration with Other Tools

### **Malware Analysis Integration**
```python
from bin.malware_analysis_tools import MalwareAnalysisTools

def forensic_malware_analysis(file_path):
    """Combine file forensics with malware analysis."""
    # Get file metadata
    metadata = ftm.get_file_metadata(file_path)
    
    # Perform malware analysis
    mat = MalwareAnalysisTools()
    malware_result = mat.analyze_file(file_path)
    
    # Combine results
    combined_result = {
        "file_metadata": metadata,
        "malware_analysis": malware_result,
        "forensic_assessment": {
            "entropy": metadata.entropy,
            "file_type": metadata.file_type,
            "security_level": metadata.security_level,
            "timestamps": {
                "created": metadata.created_time,
                "modified": metadata.modified_time,
                "accessed": metadata.accessed_time
            }
        }
    }
    
    return combined_result

# Analyze suspicious file
result = forensic_malware_analysis("/evidence/suspicious.exe")
print(f"File type: {result['forensic_assessment']['file_type']}")
print(f"Malware detected: {result['malware_analysis'].malware_detected}")
```

### **Database Integration**
```python
from bin.sqlite_manager import SQLiteManager

def store_forensic_evidence(file_path, analysis_result):
    """Store forensic evidence in database."""
    db = SQLiteManager()
    
    # Store file metadata
    db.insert_data("forensic_artifacts", {
        "artifact_id": analysis_result["file_metadata"].sha256_hash,
        "artifact_type": "file_analysis",
        "file_path": file_path,
        "file_size": analysis_result["file_metadata"].file_size,
        "file_type": analysis_result["file_metadata"].file_type,
        "entropy": analysis_result["file_metadata"].entropy,
        "security_level": analysis_result["file_metadata"].security_level,
        "analysis_results": json.dumps(analysis_result),
        "created_time": analysis_result["file_metadata"].created_time,
        "modified_time": analysis_result["file_metadata"].modified_time
    })
    
    print("Forensic evidence stored in database")

# Store evidence
store_forensic_evidence("/evidence/file.exe", result)
```

## Performance Optimization

### **Batch Processing**
```python
def batch_file_analysis(file_list, batch_size=100):
    """Process multiple files in batches."""
    results = []
    
    for i in range(0, len(file_list), batch_size):
        batch = file_list[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(file_list)-1)//batch_size + 1}")
        
        for file_path in batch:
            try:
                metadata = ftm.get_file_metadata(file_path)
                analysis = ftm.analyze_file(file_path)
                results.append({
                    "file": file_path,
                    "metadata": metadata,
                    "analysis": analysis
                })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    return results

# Process large number of files
all_files = ftm.find_files("/evidence/partition")
results = batch_file_analysis(all_files, batch_size=50)
print(f"Processed {len(results)} files")
```

### **Memory Management**
```python
def memory_efficient_analysis(directory):
    """Memory-efficient file analysis."""
    results = []
    
    # Process files one at a time to minimize memory usage
    for file_path in ftm.find_files(directory):
        try:
            # Get metadata (lightweight)
            metadata = ftm.get_file_metadata(file_path)
            
            # Only analyze suspicious files
            if (metadata.entropy > 7.0 or 
                metadata.file_type == "executable" or
                metadata.security_level == SecurityLevel.SUSPICIOUS):
                
                analysis = ftm.analyze_file(file_path)
                results.append({
                    "file": file_path,
                    "metadata": metadata,
                    "analysis": analysis
                })
            
            # Clean up memory
            del metadata
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return results
```

## Troubleshooting

### **Common Issues**

#### **Large File Handling**
```python
# Handle large files efficiently
def analyze_large_file(file_path):
    """Analyze large files without loading entirely into memory."""
    try:
        # Get basic metadata first
        metadata = ftm.get_file_metadata(file_path)
        
        # For very large files, limit analysis
        if metadata.file_size > 100 * 1024 * 1024:  # 100MB
            print(f"Large file detected ({metadata.file_size} bytes)")
            print("Performing limited analysis...")
            
            # Only calculate hash and basic info
            return {
                "file_path": file_path,
                "file_size": metadata.file_size,
                "file_type": metadata.file_type,
                "entropy": metadata.entropy,
                "analysis_limited": True
            }
        else:
            # Full analysis for smaller files
            return ftm.analyze_file(file_path)
            
    except Exception as e:
        return {"error": str(e)}
```

#### **Permission Issues**
```python
# Handle permission issues gracefully
def safe_file_analysis(file_path):
    """Safely analyze files with permission handling."""
    try:
        # Check if file is readable
        if not os.access(file_path, os.R_OK):
            return {"error": "File not readable", "file": file_path}
        
        # Check if file is too large
        file_size = os.path.getsize(file_path)
        if file_size > 500 * 1024 * 1024:  # 500MB
            return {"error": "File too large for analysis", "file": file_path}
        
        # Proceed with analysis
        return ftm.analyze_file(file_path)
        
    except PermissionError:
        return {"error": "Permission denied", "file": file_path}
    except Exception as e:
        return {"error": str(e), "file": file_path}
```

## Best Practices

### **Evidence Handling**
1. **Never Modify Originals**: Always work with copies
2. **Maintain Chain of Custody**: Document all actions
3. **Verify Integrity**: Use hashes to verify file integrity
4. **Secure Storage**: Store evidence in encrypted containers
5. **Document Everything**: Maintain detailed logs

### **Analysis Workflow**
1. **Initial Assessment**: Quick metadata analysis
2. **Suspicious File Identification**: Focus on high-risk files
3. **Deep Analysis**: Detailed analysis of suspicious files
4. **Correlation**: Cross-reference with other evidence
5. **Reporting**: Generate comprehensive reports

### **Performance Optimization**
1. **Batch Processing**: Process files in batches
2. **Selective Analysis**: Only analyze suspicious files
3. **Memory Management**: Clean up after processing
4. **Parallel Processing**: Use multiple threads when possible
5. **Caching**: Cache frequently accessed data

This guide provides comprehensive information about using the file forensics tools effectively. For additional support or advanced use cases, refer to the main documentation or contact the development team.
