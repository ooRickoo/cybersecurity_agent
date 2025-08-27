#!/usr/bin/env python3
"""
Test script to verify session viewer setup
"""

import os
import sys
from pathlib import Path

def test_setup():
    """Test the session viewer setup."""
    print("🧪 Testing Session Viewer Setup...")
    print("=" * 50)
    
    # Check project structure
    print("\n📁 Checking project structure...")
    
    required_files = [
        "package.json",
        "server.js", 
        "client/package.json",
        "client/tailwind.config.js",
        "client/src/App.js",
        "client/src/index.css",
        "client/src/components/Header.js",
        "client/src/components/Dashboard.js",
        "README.md",
        "start-viewer.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = Path(__file__).parent / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} required files")
        return False
    
    print("\n✅ All required files present")
    
    # Check Node.js
    print("\n🔧 Checking Node.js...")
    try:
        import subprocess
        result = subprocess.run(['node', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Node.js found: {version}")
        else:
            print("❌ Node.js not working properly")
            return False
    except Exception as e:
        print(f"❌ Error checking Node.js: {e}")
        return False
    
    # Check npm
    print("\n📦 Checking npm...")
    try:
        result = subprocess.run(['npm', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ npm found: {version}")
        else:
            print("❌ npm not working properly")
            return False
    except Exception as e:
        print(f"❌ Error checking npm: {e}")
        return False
    
    # Check session-outputs directory
    print("\n📂 Checking session-outputs directory...")
    session_outputs = Path(__file__).parent.parent / "session-outputs"
    if session_outputs.exists():
        print(f"✅ session-outputs directory found: {session_outputs}")
        
        # List some files
        files = list(session_outputs.rglob("*"))[:5]
        if files:
            print("   Sample files:")
            for file in files:
                if file.is_file():
                    size = file.stat().st_size
                    print(f"   - {file.name} ({size} bytes)")
        else:
            print("   (No files found)")
    else:
        print(f"⚠️  session-outputs directory not found: {session_outputs}")
        print("   This is expected if no workflows have run yet")
    
    # Check knowledge-objects directory
    print("\n🧠 Checking knowledge-objects directory...")
    knowledge_objects = Path(__file__).parent.parent / "knowledge-objects"
    if knowledge_objects.exists():
        print(f"✅ knowledge-objects directory found: {knowledge_objects}")
    else:
        print(f"⚠️  knowledge-objects directory not found: {knowledge_objects}")
    
    print("\n" + "=" * 50)
    print("🎉 Setup verification complete!")
    print("\n📋 Next steps:")
    print("1. Install dependencies: cd session-viewer && npm install")
    print("2. Install client dependencies: cd client && npm install")
    print("3. Build client: cd .. && npm run build-client")
    print("4. Start viewer: python start-viewer.py")
    print("\n🔗 Or use the launcher: python start-viewer.py")
    
    return True

if __name__ == "__main__":
    success = test_setup()
    sys.exit(0 if success else 1)
