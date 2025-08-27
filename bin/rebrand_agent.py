#!/usr/bin/env python3
"""
Simple rebranding script for Cybersecurity Agent
Usage: python bin/rebrand_agent.py [old_name] [new_name]
Example: python bin/rebrand_agent.py "Cybersecurity Agent" "Cybersecurity Agent"
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Tuple

# Default rebranding mappings
DEFAULT_MAPPINGS = [
    ("Cybersecurity Agent", "Cybersecurity Agent"),
    ("Cybersecurity Agent", "cybersecurity agent"),
    ("Cybersecurity Agent", "Cybersecurity Agent"),
    ("Security Professional", "Security Professional"),
    ("Security Professional", "security professional"),
    ("AI-powered", "AI-powered"),
    ("AI-powered", "AI-powered"),
    ("AI-powered", "AI-powered"),
    ("AI-powered", "AI-powered"),
]

def find_files_to_update() -> List[Path]:
    """Find all files that need rebranding updates."""
    project_root = Path(__file__).parent.parent
    files_to_update = []
    
    # File extensions to process
    extensions = {'.py', '.md', '.js', '.jsx', '.ts', '.tsx', '.html', '.txt', '.json'}
    
    # Directories to exclude
    exclude_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'env'}
    
    for file_path in project_root.rglob('*'):
        if file_path.is_file() and file_path.suffix in extensions:
            # Skip files in excluded directories
            if not any(exclude_dir in file_path.parts for exclude_dir in exclude_dirs):
                files_to_update.append(file_path)
    
    return files_to_update

def update_file_content(file_path: Path, mappings: List[Tuple[str, str]]) -> bool:
    """Update branding in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply all mappings
        for old_text, new_text in mappings:
            content = re.sub(r'\b' + re.escape(old_text) + r'\b', new_text, content, flags=re.IGNORECASE)
        
        # If content changed, write it back
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False

def update_filenames(project_root: Path, mappings: List[Tuple[str, str]]) -> List[Tuple[Path, Path]]:
    """Update filenames that contain old branding."""
    renamed_files = []
    
    for file_path in project_root.rglob('*'):
        if file_path.is_file():
            old_name = file_path.name
            new_name = old_name
            
            # Apply mappings to filename
            for old_text, new_text in mappings:
                new_name = new_name.replace(old_text, new_text)
                new_name = new_name.replace(old_text.lower(), new_text.lower())
                new_name = new_name.replace(old_text.upper(), new_text.upper())
            
            if new_name != old_name:
                new_path = file_path.parent / new_name
                try:
                    file_path.rename(new_path)
                    renamed_files.append((file_path, new_path))
                    print(f"📁 Renamed: {old_name} → {new_name}")
                except Exception as e:
                    print(f"❌ Error renaming {old_name}: {e}")
    
    return renamed_files

def main():
    """Main rebranding function."""
    if len(sys.argv) == 3:
        old_name = sys.argv[1]
        new_name = sys.argv[2]
        custom_mappings = [(old_name, new_name)]
        print(f"🔄 Custom rebranding: {old_name} → {new_name}")
    else:
        custom_mappings = []
        print("🔄 Using default rebranding mappings")
    
    # Combine custom and default mappings
    all_mappings = custom_mappings + DEFAULT_MAPPINGS
    
    print("\n📋 Rebranding mappings:")
    for old_text, new_text in all_mappings:
        print(f"  • {old_text} → {new_text}")
    
    project_root = Path(__file__).parent.parent
    print(f"\n🔍 Project root: {project_root}")
    
    # Find files to update
    files_to_update = find_files_to_update()
    print(f"\n📁 Found {len(files_to_update)} files to process")
    
    # Update file contents
    print("\n🔄 Updating file contents...")
    updated_files = 0
    for file_path in files_to_update:
        if update_file_content(file_path, all_mappings):
            print(f"✅ Updated: {file_path.relative_to(project_root)}")
            updated_files += 1
    
    print(f"\n📊 Content updates: {updated_files}/{len(files_to_update)} files modified")
    
    # Update filenames (optional)
    print("\n🔄 Updating filenames...")
    renamed_files = update_filenames(project_root, all_mappings)
    print(f"📊 Filename updates: {len(renamed_files)} files renamed")
    
    print("\n🎉 Rebranding complete!")
    print("\n💡 To revert changes, run:")
    print("   git checkout .")
    print("\n💡 To apply custom rebranding:")
    print(f"   python bin/rebrand_agent.py \"{old_name}\" \"{new_name}\"" if custom_mappings else "   python bin/rebrand_agent.py \"OldName\" \"NewName\"")

if __name__ == "__main__":
    main()
