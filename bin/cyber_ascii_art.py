#!/usr/bin/env python3
"""
Cyber ASCII Art Collections
Compact, classic hacker-style ASCII art for startup and closing messages.
"""

import random
from typing import List

class CyberASCIIArt:
    """Collection of cyber-themed ASCII art for startup and closing messages."""
    
    # Opening messages - compact and cyber-themed
    OPENING_ART = [
        # 1. Matrix-style terminal
        """
    ╔══════════════╗
    ║ CYBER AGENT  ║
    ║   ONLINE     ║
    ╚══════════════╝
        """,
        
        # 2. Circuit board
        """
    ┌─┐ ┌─┐ ┌─┐
    │█│ │█│ │█│
    └─┘ └─┘ └─┘
    ────███────
        """,
        
        # 3. Binary flow
        """
    10101010
    01010101
    ██░░██░░
        """,
        
        # 4. Network nodes
        """
      ○──●──○
      │  │  │
      ●──○──●
        """,
        
        # 5. Digital lock
        """
    ████████
    █░░░░░░█
    █░░██░░█
    ████████
        """,
        
        # 6. Code matrix
        """
    ░█░█░█░
    █░█░█░█
    ░█░█░█░
        """,
        
        # 7. Security shield
        """
      ████
    ████████
    █░░░░░░█
    ████████
        """,
        
        # 8. Data stream
        """
    ░░░███░░░
    ███░░░███
    ░░░███░░░
        """,
        
        # 9. Terminal prompt
        """
    ┌─[root@cyber]─[~]
    └──╼ $
        """,
        
        # 10. Encryption key
        """
    ██░░██░░
    ░██░░██░
    ██░░██░░
        """
    ]
    
    # Closing messages - farewell style
    CLOSING_ART = [
        # 1. Shutdown sequence
        """
    ╔══════════════╗
    ║ SHUTTING     ║
    ║   DOWN       ║
    ╚══════════════╝
        """,
        
        # 2. Power off
        """
      ████████
      █░░░░░░█
      █░░██░░█
      ████████
        """,
        
        # 3. Goodbye matrix
        """
    01010101
    10101010
    ░█░░█░░█
        """,
        
        # 4. Disconnect
        """
      ○──●──○
      │  │  │
      ●──○──●
        """,
        
        # 5. Lock up
        """
    ████████
    █░░░░░░█
    █░░██░░█
    ████████
        """,
        
        # 6. Fade out
        """
    ██░░██░░
    ░██░░██░
    ██░░██░░
        """,
        
        # 7. Sign off
        """
      ████
    ████████
    █░░░░░░█
    ████████
        """,
        
        # 8. End session
        """
    ░░░███░░░
    ███░░░███
    ░░░███░░░
        """,
        
        # 9. Logout
        """
    ┌─[root@cyber]─[~]
    └──╼ logout
        """,
        
        # 10. Secure exit
        """
    ██░░██░░
    ░██░░██░
    ██░░██░░
        """
    ]
    
    @classmethod
    def get_random_opening(cls) -> str:
        """Get a random opening ASCII art."""
        return random.choice(cls.OPENING_ART).strip()
    
    @classmethod
    def get_random_closing(cls) -> str:
        """Get a random closing ASCII art."""
        return random.choice(cls.CLOSING_ART).strip()
    
    @classmethod
    def get_all_openings(cls) -> List[str]:
        """Get all opening ASCII art pieces."""
        return [art.strip() for art in cls.OPENING_ART]
    
    @classmethod
    def get_all_closings(cls) -> List[str]:
        """Get all closing ASCII art pieces."""
        return [art.strip() for art in cls.CLOSING_ART]

# Quick test function
if __name__ == "__main__":
    print("🎯 Cyber ASCII Art Collections")
    print("=" * 40)
    
    print("\n📱 Random Opening:")
    print(CyberASCIIArt.get_random_opening())
    
    print("\n🔒 Random Closing:")
    print(CyberASCIIArt.get_random_closing())
    
    print("\n📊 Total Pieces:")
    print(f"Opening: {len(CyberASCIIArt.OPENING_ART)}")
    print(f"Closing: {len(CyberASCIIArt.CLOSING_ART)}")
