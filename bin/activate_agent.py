#!/usr/bin/env python3
"""
Simple Activation Utility for Cybersecurity Agent
Easy-to-use script for activating the cybersecurity agent on this host.
"""

import sys
import getpass
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from bin.activation_manager import ActivationManager

def main():
    """Simple activation interface."""
    print("🛡️  Cybersecurity Agent Activation")
    print("=" * 40)
    
    # Initialize activation manager
    activation_manager = ActivationManager()
    
    # Check current status
    status = activation_manager.get_activation_status()
    if status['activated']:
        print("✅ Agent is already activated on this host")
        print(f"   Device ID: {status['device_fingerprint'][:16]}...")
        
        # Ask if user wants to re-activate
        response = input("\nDo you want to re-activate? (y/N): ").strip().lower()
        if response != 'y':
            print("Activation cancelled.")
            return
    
    print("\n🔐 Activation Process:")
    print("   This will bind the agent to this specific host")
    print("   You'll need your master password")
    print("   The activation cannot be transferred to another machine")
    
    # Get password
    password = getpass.getpass("\nEnter your master password: ")
    if not password:
        print("❌ No password provided. Activation cancelled.")
        return
    
    # Confirm password
    password_confirm = getpass.getpass("Confirm password: ")
    if password != password_confirm:
        print("❌ Passwords don't match. Activation cancelled.")
        return
    
    # Create activation
    print("\n🔄 Creating activation...")
    success, message = activation_manager.create_activation(password)
    
    if success:
        print(f"\n{message}")
        print("\n🎉 Activation complete!")
        print("   The cybersecurity agent is now bound to this host")
        print("   You can now use cs_util_lg.py securely")
    else:
        print(f"\n{message}")
        print("\n❌ Activation failed. Please try again.")

if __name__ == '__main__':
    main()

