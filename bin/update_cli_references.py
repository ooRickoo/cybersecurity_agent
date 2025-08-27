#!/usr/bin/env python3
"""
Update CLI references script
Updates all documentation references from cs_ai_cli.py to cs_util_lg.py
"""

import os
import re
from pathlib import Path

def update_file_references(file_path: Path) -> bool:
    """Update CLI references in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Update CLI references
        content = re.sub(r'cs_ai_cli\.py', 'cs_util_lg.py', content)
        content = re.sub(r'cs_ai_cli', 'cs_util_lg', content)
        
        # Update command examples to match new CLI structure
        content = re.sub(r'python3? cs_util_lg\.py list-tools', 'python3 cs_util_lg.py -list-workflows', content)
        content = re.sub(r'python3? cs_util_lg\.py workflow', 'python3 cs_util_lg.py -workflow', content)
        content = re.sub(r'python3? cs_util_lg\.py automated', 'python3 cs_util_lg.py -workflow automated', content)
        content = re.sub(r'python3? cs_util_lg\.py manual', 'python3 cs_util_lg.py -workflow manual', content)
        content = re.sub(r'python3? cs_util_lg\.py hybrid', 'python3 cs_util_lg.py -workflow hybrid', content)
        
        # If content changed, write it back
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False

def main():
    """Main function to update all CLI references."""
    project_root = Path(__file__).parent.parent
    
    # Find all documentation files
    doc_extensions = {'.md', '.txt'}
    exclude_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'env'}
    
    files_to_update = []
    for file_path in project_root.rglob('*'):
        if file_path.is_file() and file_path.suffix in doc_extensions:
            if not any(exclude_dir in file_path.parts for exclude_dir in exclude_dirs):
                files_to_update.append(file_path)
    
    print(f"🔍 Found {len(files_to_update)} documentation files to process")
    
    # Update file references
    updated_files = 0
    for file_path in files_to_update:
        if update_file_references(file_path):
            print(f"✅ Updated: {file_path.relative_to(project_root)}")
            updated_files += 1
    
    print(f"\n📊 CLI reference updates: {updated_files}/{len(files_to_update)} files modified")
    print("\n🎉 CLI reference update complete!")
    print("\n💡 The unified CLI is now: cs_util_lg.py")
    print("💡 All documentation has been updated to reflect the new CLI structure")

if __name__ == "__main__":
    main()
