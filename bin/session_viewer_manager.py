#!/usr/bin/env python3
"""
Auto-Managing Session Viewer Manager for Cybersecurity Agent
Automatically starts/stops the web viewer based on workflow needs
"""

import os
import sys
import time
import signal
import subprocess
import threading
import webbrowser
import tempfile
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import psutil

class SessionViewerManager:
    """Manages the session viewer lifecycle automatically."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.viewer_dir = self.project_root / "session-viewer"
        self.server_process: Optional[subprocess.Popen] = None
        self.server_port = 3001
        self.is_running = False
        self.last_activity = datetime.now()
        self.auto_shutdown_timer: Optional[threading.Timer] = None
        self.shutdown_delay = 300  # 5 minutes of inactivity
        self.lock = threading.Lock()
        
        # Check if Node.js is available
        self.nodejs_available = self._check_nodejs()
        if not self.nodejs_available:
            print("⚠️  Node.js not available - session viewer will not be available")
    
    def _check_nodejs(self) -> bool:
        """Check if Node.js is available."""
        try:
            result = subprocess.run(['node', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def start_viewer(self, auto_open: bool = True) -> Dict[str, Any]:
        """Start the session viewer if not already running."""
        with self.lock:
            if self.is_running and self.server_process:
                # Check if process is still alive
                if self.server_process.poll() is None:
                    self._update_activity()
                    return {
                        'success': True,
                        'status': 'already_running',
                        'url': f'http://localhost:{self.server_port}',
                        'message': 'Session viewer is already running'
                    }
                else:
                    # Process died, clean up
                    self.is_running = False
                    self.server_process = None
            
            if not self.nodejs_available:
                return {
                    'success': False,
                    'error': 'Node.js not available',
                    'message': 'Please install Node.js to use the session viewer'
                }
            
            try:
                # Ensure dependencies are installed
                if not self._ensure_dependencies():
                    return {
                        'success': False,
                        'error': 'Dependencies not available',
                        'message': 'Failed to install session viewer dependencies'
                    }
                
                # Start the server
                print(f"🚀 Starting session viewer on port {self.server_port}...")
                self.server_process = self._start_server()
                
                if not self.server_process:
                    return {
                        'success': False,
                        'error': 'Server start failed',
                        'message': 'Failed to start session viewer server'
                    }
                
                self.is_running = True
                self._update_activity()
                
                # Wait for server to be ready
                if self._wait_for_server():
                    # Auto-open browser if requested
                    if auto_open:
                        self._open_browser()
                    
                    # Start auto-shutdown timer
                    self._start_auto_shutdown_timer()
                    
                    return {
                        'success': True,
                        'status': 'started',
                        'url': f'http://localhost:{self.server_port}',
                        'message': f'Session viewer started successfully on port {self.server_port}',
                        'auto_shutdown': f'{self.shutdown_delay // 60} minutes'
                    }
                else:
                    self.stop_viewer()
                    return {
                        'success': False,
                        'error': 'Server not ready',
                        'message': 'Server started but failed to respond'
                    }
                    
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'message': f'Error starting session viewer: {e}'
                }
    
    def stop_viewer(self) -> Dict[str, Any]:
        """Stop the session viewer."""
        with self.lock:
            if not self.is_running or not self.server_process:
                return {
                    'success': True,
                    'status': 'not_running',
                    'message': 'Session viewer is not running'
                }
            
            try:
                print("🛑 Stopping session viewer...")
                
                # Cancel auto-shutdown timer
                if self.auto_shutdown_timer:
                    self.auto_shutdown_timer.cancel()
                    self.auto_shutdown_timer = None
                
                # Stop the server process
                if self.server_process:
                    if os.name == 'nt':  # Windows
                        self.server_process.terminate()
                    else:  # Unix/Linux/macOS
                        self.server_process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        self.server_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print("⚠️  Force killing server process...")
                        self.server_process.kill()
                
                self.is_running = False
                self.server_process = None
                
                return {
                    'success': True,
                    'status': 'stopped',
                    'message': 'Session viewer stopped successfully'
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'message': f'Error stopping session viewer: {e}'
                }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current viewer status."""
        with self.lock:
            if not self.is_running or not self.server_process:
                return {
                    'running': False,
                    'port': self.server_port,
                    'url': f'http://localhost:{self.server_port}',
                    'uptime': None,
                    'auto_shutdown': None
                }
            
            # Check if process is still alive
            if self.server_process.poll() is not None:
                self.is_running = False
                self.server_process = None
                return {
                    'running': False,
                    'port': self.server_port,
                    'url': f'http://localhost:{self.server_port}',
                    'uptime': None,
                    'auto_shutdown': None
                }
            
            uptime = datetime.now() - self.last_activity
            auto_shutdown = self.shutdown_delay - uptime.total_seconds()
            
            return {
                'running': True,
                'port': self.server_port,
                'url': f'http://localhost:{self.server_port}',
                'uptime': str(uptime).split('.')[0],
                'auto_shutdown': f'{int(auto_shutdown // 60)}m {int(auto_shutdown % 60)}s' if auto_shutdown > 0 else 'Imminent'
            }
    
    def _ensure_dependencies(self) -> bool:
        """Ensure all dependencies are installed."""
        try:
            # Check if package.json exists
            package_json = self.viewer_dir / "package.json"
            if not package_json.exists():
                print("❌ Session viewer not found")
                return False
            
            # Check if node_modules exists
            node_modules = self.viewer_dir / "node_modules"
            if not node_modules.exists():
                print("📦 Installing session viewer dependencies...")
                os.chdir(self.viewer_dir)
                result = subprocess.run(['npm', 'install'], 
                                      capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    print(f"❌ Failed to install dependencies: {result.stderr}")
                    return False
            
            # Check if client is built
            build_dir = self.viewer_dir / "client" / "build"
            if not build_dir.exists() or not any(build_dir.iterdir()):
                print("🔨 Building React client...")
                os.chdir(self.viewer_dir)
                result = subprocess.run(['npm', 'run', 'build-client'], 
                                      capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    print(f"❌ Failed to build client: {result.stderr}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error ensuring dependencies: {e}")
            return False
    
    def _start_server(self) -> Optional[subprocess.Popen]:
        """Start the Node.js server."""
        try:
            os.chdir(self.viewer_dir)
            
            # Set environment variables
            env = os.environ.copy()
            env['PORT'] = str(self.server_port)
            
            # Start the server
            process = subprocess.Popen(
                ['npm', 'start'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )
            
            return process
            
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            return None
    
    def _wait_for_server(self, timeout: int = 30) -> bool:
        """Wait for server to be ready."""
        import socket
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with socket.create_connection(('localhost', self.server_port), timeout=1):
                    return True
            except (socket.timeout, socket.error):
                time.sleep(1)
        
        return False
    
    def _open_browser(self):
        """Open browser to session viewer."""
        try:
            url = f'http://localhost:{self.server_port}'
            webbrowser.open(url)
            print(f"🔗 Opened browser: {url}")
        except Exception as e:
            print(f"⚠️  Could not open browser: {e}")
    
    def _update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()
        
        # Restart auto-shutdown timer
        if self.auto_shutdown_timer:
            self.auto_shutdown_timer.cancel()
        self._start_auto_shutdown_timer()
    
    def _start_auto_shutdown_timer(self):
        """Start auto-shutdown timer."""
        if self.auto_shutdown_timer:
            self.auto_shutdown_timer.cancel()
        
        self.auto_shutdown_timer = threading.Timer(self.shutdown_delay, self._auto_shutdown)
        self.auto_shutdown_timer.daemon = True
        self.auto_shutdown_timer.start()
    
    def _auto_shutdown(self):
        """Automatically shutdown after inactivity."""
        print(f"⏰ Auto-shutdown after {self.shutdown_delay} seconds of inactivity")
        self.stop_viewer()
    
    def extend_session(self) -> Dict[str, Any]:
        """Extend the current session (reset auto-shutdown timer)."""
        with self.lock:
            if not self.is_running:
                return {
                    'success': False,
                    'error': 'Not running',
                    'message': 'Session viewer is not running'
                }
            
            self._update_activity()
            return {
                'success': True,
                'message': f'Session extended by {self.shutdown_delay // 60} minutes'
            }
    
    def keep_alive(self):
        """Keep the session viewer running and handle user interruption."""
        try:
            print("🔄 Session viewer is running...")
            print("   Press Ctrl+C to stop the viewer")
            
            # Keep the process running until interrupted
            while self.is_running and self.server_process and self.server_process.poll() is None:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping session viewer...")
            self.stop_viewer()
            print("✅ Session viewer stopped.")
        except Exception as e:
            print(f"❌ Error in keep_alive: {e}")
            self.stop_viewer()
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop_viewer()

# Global instance
_session_viewer_manager = None

def get_session_viewer_manager() -> SessionViewerManager:
    """Get or create the global session viewer manager."""
    global _session_viewer_manager
    if _session_viewer_manager is None:
        _session_viewer_manager = SessionViewerManager()
    return _session_viewer_manager

# Signal handlers for graceful shutdown
def _signal_handler(signum, frame):
    """Handle shutdown signals."""
    print(f"\n📡 Received signal {signum}")
    if _session_viewer_manager:
        _session_viewer_manager.cleanup()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

if __name__ == "__main__":
    # Test the manager
    manager = get_session_viewer_manager()
    
    print("🧪 Testing Session Viewer Manager...")
    
    # Start viewer
    result = manager.start_viewer()
    print(f"Start result: {result}")
    
    if result['success']:
        print("✅ Viewer started successfully")
        
        # Show status
        status = manager.get_status()
        print(f"Status: {status}")
        
        # Wait for user input
        input("Press Enter to stop viewer...")
        
        # Stop viewer
        result = manager.stop_viewer()
        print(f"Stop result: {result}")
    else:
        print(f"❌ Failed to start viewer: {result['error']}")
