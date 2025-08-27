# 🔐 Salt Management Guide

## Overview

The Cybersecurity Agent now uses a **persistent salt management system** that ensures encryption keys remain consistent across multiple runs. This is critical for data accessibility - without persistent salts, encrypted data becomes unrecoverable.

## 🚨 **Why Persistent Salts Matter**

### ❌ **Wrong Approach (Previous):**
```bash
# This generates a NEW salt every time - BAD!
export ENCRYPTION_SALT="$(openssl rand -hex 16)"
```

**Problems:**
- Salt changes on every terminal session
- Previously encrypted data becomes inaccessible
- Encryption keys are different each time
- Data loss risk

### ✅ **Correct Approach (Current):**
```bash
# Salt is automatically managed by the system
# No manual salt setting needed
```

**Benefits:**
- Salt persists across sessions and reboots
- Same encryption key for all operations
- Previously encrypted data remains accessible
- Secure and consistent

## 🔧 **How It Works**

### **Automatic Salt Management:**
1. **First Run:** System generates a cryptographically secure random salt
2. **Storage:** Salt is stored in `.salt` file with integrity verification
3. **Subsequent Runs:** System reads the existing salt from `.salt` file
4. **Consistency:** Same salt used for all encryption/decryption operations

### **Salt File Security:**
- **Location:** `.salt` (in project root)
- **Permissions:** `600` (owner read/write only)
- **Integrity:** SHA-256 hash verification
- **Backup:** Can be backed up and restored

## 📋 **Salt Management Commands**

### **View Salt Information:**
```bash
python bin/salt_manager.py --info
```

**Output:**
```
🔐 Salt Information:
   File: .salt
   Exists: True
   Length: 256 bits
   Salt: a1b2c3d4e5f6...
   File Size: 64 bytes
   Permissions: 600
   Integrity: ✅ Verified
```

### **Verify Salt Integrity:**
```bash
python bin/salt_manager.py --verify
```

### **Backup Salt:**
```bash
python bin/salt_manager.py --backup /path/to/backup/salt.backup
```

### **Restore Salt:**
```bash
python bin/salt_manager.py --restore /path/to/backup/salt.backup
```

### **Generate New Salt (DANGEROUS):**
```bash
python bin/salt_manager.py --generate
```

**⚠️  WARNING:** This will make ALL previously encrypted data inaccessible!

## 🔄 **Migration from Old System**

### **If You Previously Set ENCRYPTION_SALT:**

1. **Remove the environment variable:**
   ```bash
   unset ENCRYPTION_SALT
   ```

2. **Remove from shell profiles:**
   ```bash
   # Remove these lines from ~/.bashrc, ~/.zshrc, etc.
   export ENCRYPTION_SALT="$(openssl rand -hex 16)"
   export ENCRYPTION_SALT="your_salt_value"
   ```

3. **Restart the agent:**
   ```bash
   python cs_util_lg.py
   ```

4. **Verify salt generation:**
   ```bash
   python bin/salt_manager.py --info
   ```

## 🛡️ **Security Features**

### **Cryptographic Strength:**
- **Salt Length:** 256 bits (32 bytes)
- **Generation:** `secrets.token_bytes()` (cryptographically secure)
- **Storage:** Binary format with integrity verification
- **Permissions:** Owner-only access (600)

### **Integrity Protection:**
- **Hash Verification:** SHA-256 hash stored with salt
- **Corruption Detection:** Automatic detection of file corruption
- **Auto-Recovery:** Generates new salt if corruption detected
- **Fallback Protection:** Deterministic fallback if generation fails

## 📁 **File Structure**

```
Cybersecurity-Agent/
├── .salt                    # Persistent salt file (auto-generated)
├── bin/
│   └── salt_manager.py     # Salt management utility
├── knowledge-objects/       # Encrypted data (uses persistent salt)
├── session-logs/           # Encrypted logs (uses persistent salt)
└── session-outputs/        # Encrypted outputs (uses persistent salt)
```

## 🔍 **Troubleshooting**

### **Salt File Not Found:**
```bash
# Check if salt file exists
ls -la .salt

# Generate new salt if needed
python bin/salt_manager.py
```

### **Permission Denied:**
```bash
# Fix permissions
chmod 600 .salt

# Check ownership
ls -la .salt
```

### **Salt Corruption:**
```bash
# Verify integrity
python bin/salt_manager.py --verify

# If corrupted, system will auto-generate new salt
python bin/salt_manager.py --info
```

### **Backup/Restore Issues:**
```bash
# Check backup file integrity
python bin/salt_manager.py --verify

# Restore from backup
python bin/salt_manager.py --restore /path/to/backup
```

## 📚 **Best Practices**

### **✅ Do:**
- Let the system manage salts automatically
- Backup the `.salt` file securely
- Use the salt manager for verification
- Keep `.salt` file in version control (if needed)

### **❌ Don't:**
- Set `ENCRYPTION_SALT` environment variable
- Generate random salts manually
- Share `.salt` files between installations
- Delete `.salt` file without backup

## 🔐 **Encryption Workflow**

### **Data Encryption:**
1. System reads persistent salt from `.salt` file
2. Derives encryption key from password hash + salt
3. Encrypts data using derived key
4. Stores encrypted data with `.encrypted` extension

### **Data Decryption:**
1. System reads persistent salt from `.salt` file
2. Derives same encryption key from password hash + salt
3. Decrypts data using derived key
4. Provides decrypted data for processing

## 🎯 **Key Benefits**

1. **Data Persistence:** Encrypted data remains accessible across sessions
2. **Security:** Cryptographically strong, randomly generated salts
3. **Simplicity:** No manual salt management required
4. **Reliability:** Automatic corruption detection and recovery
5. **Backup Support:** Easy backup and restore of salt files
6. **Cross-Platform:** Works on all supported operating systems

## 🚀 **Quick Start**

1. **Install the agent** (salt will be auto-generated)
2. **Enable encryption:** `export ENCRYPTION_ENABLED=true`
3. **Set password hash:** `export ENCRYPTION_PASSWORD_HASH="your_hash"`
4. **Run the agent:** `python cs_util_lg.py`
5. **Verify salt:** `python bin/salt_manager.py --info`

The system will automatically handle all salt management, ensuring your encrypted data remains secure and accessible! 🔐✨
