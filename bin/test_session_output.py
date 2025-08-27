#!/usr/bin/env python3
"""
Test script to verify session output management is working correctly.
"""

import sys
from pathlib import Path

# Add bin directory to path
sys.path.append(str(Path(__file__).parent / 'bin'))

from session_output_manager import create_session, add_output_file, save_session
import pandas as pd

def test_session_output():
    """Test the session output management system."""
    
    print("🧪 Testing Session Output Management...")
    
    # Create a test session
    session_id = create_session("test_policy_mapping")
    print(f"✅ Created session: {session_id}")
    
    # Create some test data
    test_data = {
        'policy_id': ['POL001', 'POL002'],
        'policy_name': ['Test Policy 1', 'Test Policy 2'],
        'description': ['Test description 1', 'Test description 2']
    }
    
    df = pd.DataFrame(test_data)
    
    # Create output directory
    output_dir = Path(f"session-output/{session_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save test CSV
    output_file = output_dir / "test_results.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✅ Saved test CSV to: {output_file}")
    
    # Add file to session
    add_output_file(session_id, str(output_file), "Test Policy Results")
    print(f"✅ Added file to session: {output_file}")
    
    # Save session
    save_session(session_id)
    print(f"✅ Saved session: {session_id}")
    
    # Check if session folder was created
    if output_dir.exists():
        print(f"✅ Session folder created: {output_dir}")
        print(f"📁 Contents: {list(output_dir.iterdir())}")
    else:
        print(f"❌ Session folder not created: {output_dir}")
    
    print("\n🎯 Test completed!")

if __name__ == "__main__":
    test_session_output()
