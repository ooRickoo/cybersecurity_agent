#!/usr/bin/env python3
"""
Session Viewer MCP Tools for Cybersecurity Agent
Provides tools for managing the session viewer from workflows
"""

import json
import webbrowser
from typing import Dict, Any, List
from pathlib import Path

try:
    from session_viewer_manager import get_session_viewer_manager
    SESSION_VIEWER_AVAILABLE = True
except ImportError:
    SESSION_VIEWER_AVAILABLE = False

class SessionViewerMCPTools:
    """MCP tools for session viewer management."""
    
    def __init__(self):
        self.manager = get_session_viewer_manager() if SESSION_VIEWER_AVAILABLE else None
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get available MCP tools."""
        if not SESSION_VIEWER_AVAILABLE:
            return []
        
        return [
            {
                "type": "function",
                "function": {
                    "name": "launch_session_viewer",
                    "description": "Launch the interactive session viewer for browsing workflow outputs",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "auto_open_browser": {
                                "type": "boolean",
                                "description": "Whether to automatically open the browser",
                                "default": True
                            },
                            "reason": {
                                "type": "string",
                                "description": "Reason for launching the viewer (for logging)"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_session_viewer_status",
                    "description": "Get the current status of the session viewer",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "stop_session_viewer",
                    "description": "Stop the session viewer and reclaim resources",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Reason for stopping the viewer (for logging)"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "extend_session_viewer",
                    "description": "Extend the session viewer session (reset auto-shutdown timer)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Reason for extending the session"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "open_session_viewer_url",
                    "description": "Open the session viewer URL in the default browser",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ]
    
    def launch_session_viewer(self, auto_open_browser: bool = True, reason: str = None) -> Dict[str, Any]:
        """Launch the session viewer."""
        if not SESSION_VIEWER_AVAILABLE:
            return {
                "success": False,
                "error": "Session viewer not available",
                "message": "Node.js or session viewer dependencies not available"
            }
        
        try:
            # Launch the viewer
            result = self.manager.start_viewer(auto_open=auto_open_browser)
            
            if result['success']:
                # Create clickable link for CLI
                url = result['url']
                clickable_link = f"\n🔗 **Session Viewer Launched!**\n"
                clickable_link += f"   **URL:** {url}\n"
                clickable_link += f"   **Status:** {result['status']}\n"
                clickable_link += f"   **Auto-shutdown:** {result.get('auto_shutdown', 'Unknown')}\n"
                
                if reason:
                    clickable_link += f"   **Reason:** {reason}\n"
                
                clickable_link += f"\n💡 **Usage:**\n"
                clickable_link += f"   • Click the URL above to open in browser\n"
                clickable_link += f"   • Browse session outputs and files\n"
                clickable_link += f"   • Viewer will auto-close after 5 minutes of inactivity\n"
                clickable_link += f"   • Use 'extend_session_viewer' to keep it open longer\n"
                
                return {
                    "success": True,
                    "message": "Session viewer launched successfully",
                    "url": url,
                    "status": result['status'],
                    "auto_shutdown": result.get('auto_shutdown'),
                    "cli_output": clickable_link,
                    "reason": reason
                }
            else:
                return {
                    "success": False,
                    "error": result.get('error', 'Unknown error'),
                    "message": result.get('message', 'Failed to launch session viewer'),
                    "cli_output": f"❌ **Failed to launch session viewer:** {result.get('message', 'Unknown error')}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error launching session viewer: {e}",
                "cli_output": f"❌ **Error launching session viewer:** {e}"
            }
    
    def get_session_viewer_status(self) -> Dict[str, Any]:
        """Get the current status of the session viewer."""
        if not SESSION_VIEWER_AVAILABLE:
            return {
                "success": False,
                "error": "Session viewer not available",
                "message": "Node.js or session viewer dependencies not available"
            }
        
        try:
            status = self.manager.get_status()
            
            if status['running']:
                cli_output = f"🟢 **Session Viewer Status:**\n"
                cli_output += f"   **Status:** Running\n"
                cli_output += f"   **URL:** {status['url']}\n"
                cli_output += f"   **Uptime:** {status['uptime']}\n"
                cli_output += f"   **Auto-shutdown:** {status['auto_shutdown']}\n"
                cli_output += f"\n💡 **Actions:**\n"
                cli_output += f"   • Click URL to open in browser\n"
                cli_output += f"   • Use 'extend_session_viewer' to keep open\n"
                cli_output += f"   • Use 'stop_session_viewer' to close now\n"
            else:
                cli_output = f"🔴 **Session Viewer Status:**\n"
                cli_output += f"   **Status:** Not running\n"
                cli_output += f"   **URL:** {status['url']}\n"
                cli_output += f"\n💡 **Actions:**\n"
                cli_output += f"   • Use 'launch_session_viewer' to start\n"
            
            return {
                "success": True,
                "status": status,
                "cli_output": cli_output
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error getting status: {e}",
                "cli_output": f"❌ **Error getting status:** {e}"
            }
    
    def stop_session_viewer(self, reason: str = None) -> Dict[str, Any]:
        """Stop the session viewer."""
        if not SESSION_VIEWER_AVAILABLE:
            return {
                "success": False,
                "error": "Session viewer not available",
                "message": "Node.js or session viewer dependencies not available"
            }
        
        try:
            result = self.manager.stop_viewer()
            
            cli_output = f"🛑 **Session Viewer Stopped**\n"
            cli_output += f"   **Status:** {result['status']}\n"
            cli_output += f"   **Message:** {result['message']}\n"
            
            if reason:
                cli_output += f"   **Reason:** {reason}\n"
            
            cli_output += f"\n💡 **Resources reclaimed successfully**\n"
            
            return {
                "success": result['success'],
                "status": result['status'],
                "message": result['message'],
                "cli_output": cli_output,
                "reason": reason
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error stopping session viewer: {e}",
                "cli_output": f"❌ **Error stopping session viewer:** {e}"
            }
    
    def extend_session_viewer(self, reason: str = None) -> Dict[str, Any]:
        """Extend the session viewer session."""
        if not SESSION_VIEWER_AVAILABLE:
            return {
                "success": False,
                "error": "Session viewer not available",
                "message": "Node.js or session viewer dependencies not available"
            }
        
        try:
            result = self.manager.extend_session()
            
            if result['success']:
                cli_output = f"⏰ **Session Extended**\n"
                cli_output += f"   **Message:** {result['message']}\n"
                
                if reason:
                    cli_output += f"   **Reason:** {reason}\n"
                
                cli_output += f"\n💡 **Auto-shutdown timer reset**\n"
            else:
                cli_output = f"❌ **Failed to extend session**\n"
                cli_output += f"   **Error:** {result.get('error', 'Unknown error')}\n"
                cli_output += f"   **Message:** {result.get('message', 'Unknown error')}\n"
            
            return {
                "success": result['success'],
                "message": result.get('message'),
                "error": result.get('error'),
                "cli_output": cli_output,
                "reason": reason
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error extending session: {e}",
                "cli_output": f"❌ **Error extending session:** {e}"
            }
    
    def open_session_viewer_url(self) -> Dict[str, Any]:
        """Open the session viewer URL in the default browser."""
        if not SESSION_VIEWER_AVAILABLE:
            return {
                "success": False,
                "error": "Session viewer not available",
                "message": "Node.js or session viewer dependencies not available"
            }
        
        try:
            status = self.manager.get_status()
            
            if not status['running']:
                return {
                    "success": False,
                    "error": "Viewer not running",
                    "message": "Session viewer is not running. Use 'launch_session_viewer' first.",
                    "cli_output": f"❌ **Session viewer not running**\n   Use 'launch_session_viewer' first to start it."
                }
            
            # Open browser
            webbrowser.open(status['url'])
            
            cli_output = f"🔗 **Browser Opened**\n"
            cli_output += f"   **URL:** {status['url']}\n"
            cli_output += f"   **Status:** Browser opened successfully\n"
            
            return {
                "success": True,
                "message": "Browser opened successfully",
                "url": status['url'],
                "cli_output": cli_output
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error opening browser: {e}",
                "cli_output": f"❌ **Error opening browser:** {e}"
            }

# Global instance
_session_viewer_tools = None

def get_session_viewer_tools() -> SessionViewerMCPTools:
    """Get or create the global session viewer tools instance."""
    global _session_viewer_tools
    if _session_viewer_tools is None:
        _session_viewer_tools = SessionViewerMCPTools()
    return _session_viewer_tools

if __name__ == "__main__":
    # Test the tools
    tools = get_session_viewer_tools()
    
    print("🧪 Testing Session Viewer MCP Tools...")
    
    # Test status
    status_result = tools.get_session_viewer_status()
    print(f"Status: {status_result}")
    
    if status_result['success'] and not status_result['status']['running']:
        # Test launch
        launch_result = tools.launch_session_viewer(reason="Testing MCP tools")
        print(f"Launch: {launch_result}")
        
        if launch_result['success']:
            # Test status again
            status_result = tools.get_session_viewer_status()
            print(f"Status after launch: {status_result}")
            
            # Test extend
            extend_result = tools.extend_session_viewer(reason="Testing extend")
            print(f"Extend: {extend_result}")
            
            # Test stop
            stop_result = tools.stop_session_viewer(reason="Testing stop")
            print(f"Stop: {stop_result}")
    else:
        print("Session viewer already running or not available")
