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
        """,
        
        # 11. Hacker cat (classic hacker mascot)
        """
      /\\_/\\
     ( o.o )
      > ^ <
        """,
        
        # 12. Coffee cup (hacker fuel)
        """
      )))
     ((((
    ┌─────┐
    │ ☕  │
    └─────┘
        """,
        
        # 13. Skull (compact)
        """
      ████
    ██    ██
    ██ ██ ██
    ██    ██
      ████
        """,
        
        # 14. Keyboard keys
        """
    ┌─┐┌─┐┌─┐
    │█││█││█│
    └─┘└─┘└─┘
        """,
        
        # 15. Anonymous mask (compact)
        """
      ████
    ██    ██
    ██ ██ ██
    ██    ██
      ████
        """,
        
        # 16. Terminal cursor
        """
    ┌─[hack@void]─[~]
    └──╼ █
        """,
        
        # 17. Matrix rain
        """
    01010101
    10101010
    01010101
        """,
        
        # 18. Hack symbols
        """
    ██  ██  ██
    ██  ██  ██
        """,
        
        # 19. Code brackets
        """
    {  }  {  }
    {  }  {  }
        """,
        
        # 20. Digital pattern
        """
    ██░░██░░
    ░██░░██░
    ██░░██░░
        """,
        
        # 21. Stack overflow (developer humor)
        """
    ████████
    ██    ██
    ██ ██ ██
    ██    ██
    ████████
        """,
        
        # 22. Infinite loop
        """
      ████
    ██    ██
    ██ ██ ██
    ██    ██
      ████
        """,
        
        # 23. Bug in code
        """
    ██  ██
    ██  ██
    ██  ██
    ██  ██
        """,
        
        # 24. Root access
        """
    ┌─[root@system]─[~]
    └──╼ #
        """,
        
        # 25. Kernel panic
        """
    ████████
    ██    ██
    ██ ██ ██
    ██    ██
    ████████
        """,
        
        # 26. Memory leak
        """
    ██░░██░░
    ░██░░██░
    ██░░██░░
    ░██░░██░
        """,
        
        # 27. Git commit
        """
    ┌─[dev@repo]─[main]
    └──╼ git commit
        """,
        
        # 28. Zero-day exploit
        """
    00000000
    00000000
    00000000
        """,
        
        # 29. Rainbow table
        """
    ██░░██░░
    ░██░░██░
    ██░░██░░
    ░██░░██░
        """,
        
        # 30. Buffer overflow
        """
    ████████
    ██    ██
    ██ ██ ██
    ██    ██
    ████████
        """,
        
        # 31. Social engineering
        """
      /\\_/\\
     ( ^.^ )
      > ^ <
        """,
        
        # 32. Phishing hook
        """
      ████
    ██    ██
    ██ ██ ██
    ██    ██
      ████
        """,
        
        # 33. Man-in-the-middle
        """
    ██  ██  ██
    ██  ██  ██
    ██  ██  ██
        """,
        
        # 34. DDoS attack
        """
    ██░░██░░
    ░██░░██░
    ██░░██░░
    ░██░░██░
        """,
        
        # 35. SQL injection
        """
    {  }  {  }
    {  }  {  }
    {  }  {  }
        """,
        
        # 36. Cross-site scripting
        """
    ██  ██  ██
    ██  ██  ██
    ██  ██  ██
        """,
        
        # 37. Port scanning
        """
    ┌─┐┌─┐┌─┐
    │█││█││█│
    └─┘└─┘└─┘
        """,
        
        # 38. Packet sniffing
        """
    ██░░██░░
    ░██░░██░
    ██░░██░░
    ░██░░██░
        """,
        
        # 39. Reverse engineering
        """
      ████
    ██    ██
    ██ ██ ██
    ██    ██
      ████
        """,
        
        # 40. Steganography
        """
    ██░░██░░
    ░██░░██░
    ██░░██░░
    ░██░░██░
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
        """,
        
        # 11. Hacker cat goodbye
        """
      /\\_/\\
     ( -.- )
      > ^ <
        """,
        
        # 12. Coffee empty
        """
      )))
     ((((
    ┌─────┐
    │     │
    └─────┘
        """,
        
        # 13. Skull goodbye
        """
      ████
    ██    ██
    ██ ██ ██
    ██    ██
      ████
        """,
        
        # 14. Keys locked
        """
    ┌─┐┌─┐┌─┐
    │█││█││█│
    └─┘└─┘└─┘
        """,
        
        # 15. Anonymous exit
        """
      ████
    ██    ██
    ██ ██ ██
    ██    ██
      ████
        """,
        
        # 16. Terminal logout
        """
    ┌─[hack@void]─[~]
    └──╼ logout
        """,
        
        # 17. Matrix fade
        """
    01010101
    10101010
    01010101
        """,
        
        # 18. Hack complete
        """
    ██  ██  ██
    ██  ██  ██
        """,
        
        # 19. Code end
        """
    {  }  {  }
    {  }  {  }
        """,
        
        # 20. Digital goodbye
        """
    ██░░██░░
    ░██░░██░
    ██░░██░░
        """,
        
        # 21. Stack underflow (developer humor)
        """
    ████████
    ██    ██
    ██ ██ ██
    ██    ██
    ████████
        """,
        
        # 22. Loop complete
        """
      ████
    ██    ██
    ██ ██ ██
    ██    ██
      ████
        """,
        
        # 23. Bug fixed
        """
    ██  ██
    ██  ██
    ██  ██
    ██  ██
        """,
        
        # 24. Root logout
        """
    ┌─[root@system]─[~]
    └──╼ exit
        """,
        
        # 25. Kernel stable
        """
    ████████
    ██    ██
    ██ ██ ██
    ██    ██
    ████████
        """,
        
        # 26. Memory freed
        """
    ██░░██░░
    ░██░░██░
    ██░░██░░
    ░██░░██░
        """,
        
        # 27. Git push
        """
    ┌─[dev@repo]─[main]
    └──╼ git push
        """,
        
        # 28. Patch applied
        """
    11111111
    11111111
    11111111
        """,
        
        # 29. Hash cracked
        """
    ██░░██░░
    ░██░░██░
    ██░░██░░
    ░██░░██░
        """,
        
        # 30. Buffer secured
        """
    ████████
    ██    ██
    ██ ██ ██
    ██    ██
    ████████
        """,
        
        # 31. Social engineering failed
        """
      /\\_/\\
     ( -.- )
      > ^ <
        """,
        
        # 32. Phishing blocked
        """
      ████
    ██    ██
    ██ ██ ██
    ██    ██
      ████
        """,
        
        # 33. Man-in-the-middle detected
        """
    ██  ██  ██
    ██  ██  ██
    ██  ██  ██
        """,
        
        # 34. DDoS mitigated
        """
    ██░░██░░
    ░██░░██░
    ██░░██░░
    ░██░░██░
        """,
        
        # 35. SQL injection blocked
        """
    {  }  {  }
    {  }  {  }
    {  }  {  }
        """,
        
        # 36. XSS prevented
        """
    ██  ██  ██
    ██  ██  ██
    ██  ██  ██
        """,
        
        # 37. Ports closed
        """
    ┌─┐┌─┐┌─┐
    │█││█││█│
    └─┘└─┘└─┘
        """,
        
        # 38. Packets secured
        """
    ██░░██░░
    ░██░░██░
    ██░░██░░
    ░██░░██░
        """,
        
        # 39. Reverse engineering complete
        """
      ████
    ██    ██
    ██ ██ ██
    ██    ██
      ████
        """,
        
        # 40. Steganography revealed
        """
    ██░░██░░
    ░██░░██░
    ██░░██░░
    ░██░░██░
        """
    ]
    
    @classmethod
    def get_random_opening(cls) -> str:
        """Get a random opening ASCII art."""
        art = random.choice(cls.OPENING_ART).strip()
        return cls._add_indentation(art)
    
    @classmethod
    def get_random_closing(cls) -> str:
        """Get a random closing ASCII art."""
        art = random.choice(cls.CLOSING_ART).strip()
        return cls._add_indentation(art)
    
    @classmethod
    def _add_indentation(cls, art: str) -> str:
        """Add consistent 4-space indentation to ASCII art."""
        lines = art.split('\n')
        indented_lines = []
        for line in lines:
            if line.strip():  # Only add indentation to non-empty lines
                indented_lines.append('    ' + line.strip())
            else:
                indented_lines.append('')
        return '\n'.join(indented_lines)
    
    @classmethod
    def get_all_openings(cls) -> List[str]:
        """Get all opening ASCII art pieces."""
        return [cls._add_indentation(art.strip()) for art in cls.OPENING_ART]
    
    @classmethod
    def get_all_closings(cls) -> List[str]:
        """Get all closing ASCII art pieces."""
        return [cls._add_indentation(art.strip()) for art in cls.CLOSING_ART]

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
