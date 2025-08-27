#!/usr/bin/env python3
"""
Security Sanitizer - Sanitizes URLs and filenames for safe output
"""

import re
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urlparse
import hashlib

class SecuritySanitizer:
    """Sanitizes URLs and filenames for security."""
    
    def __init__(self):
        # Patterns for potentially malicious URLs
        self.malicious_patterns = [
            r'\.(exe|bat|cmd|com|pif|scr|vbs|js|jar|msi|dll|sys)$',
            r'(virus|malware|trojan|spyware|ransomware)',
            r'(phishing|scam|fake|fraud)',
            r'(\.tk|\.ml|\.ga|\.cf|\.gq)$',  # Free domains often used for malware
            r'(bit\.ly|tinyurl|goo\.gl|t\.co)',  # URL shorteners
        ]
        
        # Compile patterns
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.malicious_patterns]
        
        # Trusted domains (whitelist)
        self.trusted_domains = {
            'github.com', 'stackoverflow.com', 'microsoft.com', 'google.com',
            'mozilla.org', 'python.org', 'nist.gov', 'mitre.org', 'cisa.gov'
        }
    
    def sanitize_url(self, url: str) -> str:
        """Sanitize a URL by adding brackets if potentially malicious."""
        if not url or not isinstance(url, str):
            return url
        
        # Check if it's a valid URL
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return url
        except:
            return url
        
        # Check if it's a trusted domain
        domain = parsed.netloc.lower()
        if domain in self.trusted_domains:
            return url
        
        # Check for malicious patterns
        is_malicious = any(pattern.search(url) for pattern in self.compiled_patterns)
        
        if is_malicious:
            # Add brackets to make it non-clickable
            return f"[{url}]"
        
        return url
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize a filename by adding .sanitize extension if potentially dangerous."""
        if not filename or not isinstance(filename, str):
            return filename
        
        # Check for dangerous file extensions
        dangerous_extensions = {'.exe', '.bat', '.cmd', '.com', '.pif', '.scr', 
                              '.vbs', '.js', '.jar', '.msi', '.dll', '.sys'}
        
        file_path = Path(filename)
        extension = file_path.suffix.lower()
        
        if extension in dangerous_extensions:
            return f"{filename}.sanitize"
        
        # Check for suspicious patterns in filename
        suspicious_patterns = [
            r'(virus|malware|trojan|spyware|ransomware)',
            r'(phishing|scam|fake|fraud)',
            r'(\.pcap|\.pcapng)$',  # Network capture files
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                return f"{filename}.sanitize"
        
        return filename
    
    def sanitize_text(self, text: str) -> str:
        """Sanitize text content, replacing URLs and filenames."""
        if not text or not isinstance(text, str):
            return text
        
        # Find and sanitize URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        text = re.sub(url_pattern, lambda m: self.sanitize_url(m.group()), text)
        
        # Find and sanitize filenames
        filename_pattern = r'\b[\w\-\.]+\.(exe|bat|cmd|com|pif|scr|vbs|js|jar|msi|dll|sys|pcap|pcapng)\b'
        text = re.sub(filename_pattern, lambda m: self.sanitize_filename(m.group()), text)
        
        return text
    
    def sanitize_output(self, output: Any) -> Any:
        """Recursively sanitize output data structures."""
        if isinstance(output, str):
            return self.sanitize_text(output)
        elif isinstance(output, list):
            return [self.sanitize_output(item) for item in output]
        elif isinstance(output, dict):
            return {key: self.sanitize_output(value) for key, value in output.items()}
        else:
            return output
    
    def make_clickable_link(self, url: str, text: str = None) -> str:
        """Create a clickable link for trusted URLs."""
        if not text:
            text = url
        
        # Check if URL is trusted
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain in self.trusted_domains:
                return f"[{text}]({url})"
        except:
            pass
        
        # For untrusted URLs, return sanitized version
        return self.sanitize_url(url)
    
    def get_sanitization_report(self, original: Any, sanitized: Any) -> Dict[str, Any]:
        """Generate a report of what was sanitized."""
        report = {
            'urls_sanitized': 0,
            'filenames_sanitized': 0,
            'trusted_urls': 0,
            'details': []
        }
        
        # This is a simplified report - in practice you'd want more detailed analysis
        return report

# Example usage
if __name__ == "__main__":
    sanitizer = SecuritySanitizer()
    
    # Test URL sanitization
    test_urls = [
        "https://github.com/example/repo",
        "https://malicious-site.com/virus.exe",
        "https://bit.ly/suspicious",
        "https://microsoft.com/security"
    ]
    
    print("URL Sanitization Test:")
    for url in test_urls:
        sanitized = sanitizer.sanitize_url(url)
        print(f"  {url} -> {sanitized}")
    
    # Test filename sanitization
    test_files = [
        "document.pdf",
        "virus.exe",
        "capture.pcap",
        "malware.bat"
    ]
    
    print("\nFilename Sanitization Test:")
    for filename in test_files:
        sanitized = sanitizer.sanitize_filename(filename)
        print(f"  {filename} -> {sanitized}")
    
    # Test text sanitization
    test_text = "Check out this file: virus.exe and visit https://malicious-site.com"
    sanitized_text = sanitizer.sanitize_text(test_text)
    print(f"\nText Sanitization Test:")
    print(f"  Original: {test_text}")
    print(f"  Sanitized: {sanitized_text}")
