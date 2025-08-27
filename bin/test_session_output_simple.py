#!/usr/bin/env python3
"""
Simple test script for session output manager
"""

import sys
from pathlib import Path

# Add bin directory to path
sys.path.append(str(Path(__file__).parent / 'bin'))

from session_output_manager import create_session, add_output_file, save_session

def test_session_output():
    """Test the session output system."""
    print("🧪 Testing Session Output Manager...")
    
    try:
        # Create a session
        session_id = create_session("test_session")
        print(f"✅ Created session: {session_id}")
        
        # Add a test file
        test_content = "This is a test file content for session output testing."
        result = add_output_file(session_id, "test_output.txt", test_content, "text")
        print(f"✅ Added output file: {result}")
        
        # Save the session
        save_result = save_session(session_id)
        print(f"✅ Save session result: {save_result}")
        
        print("🎉 Session output test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Session output test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_session_output()
