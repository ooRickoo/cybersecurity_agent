#!/usr/bin/env python3
"""
Session Viewer Launcher for Cybersecurity Agent
Launches the Node.js React session viewer when requested by the agent
"""

import os
import sys
import subprocess
import time
import webbrowser
import signal
import threading
from pathlib import Path

class SessionViewerLauncher:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.viewer_dir = Path(__file__).parent
        self.server_process = None
        self.is_running = False
        
    def start_viewer(self, port=3001):
        """Start the session viewer server and open browser."""
        try:
            print("🚀 Starting Cybersecurity Agent Session Viewer...")
            
            # Check if Node.js is available
            if not self._check_nodejs():
                print("❌ Node.js not found. Please install Node.js 16+ to use the session viewer.")
                return False
            
            # Check if dependencies are installed
            if not self._check_dependencies():
                print("📦 Installing dependencies...")
                if not self._install_dependencies():
                    print("❌ Failed to install dependencies.")
                    return False
            
            # Build the React client if needed
            if not self._check_build():
                print("🔨 Building React client...")
                if not self._build_client():
                    print("❌ Failed to build React client.")
                    return False
            
            # Start the server
            print(f"🌐 Starting server on port {port}...")
            self.server_process = self._start_server(port)
            
            if not self.server_process:
                print("❌ Failed to start server.")
                return False
            
            self.is_running = True
            
            # Wait a moment for server to start
            time.sleep(3)
            
            # Open browser
            url = f"http://localhost:{port}"
            print(f"🔗 Opening browser: {url}")
            webbrowser.open(url)
            
            print("✅ Session viewer started successfully!")
            print("   - Close this terminal to stop the viewer")
            print("   - Or press Ctrl+C to stop")
            print("   - Browser tab can be closed and reopened")
            
            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Keep the process running
            try:
                while self.is_running and self.server_process.poll() is None:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop_viewer()
                
            return True
            
        except Exception as e:
            print(f"❌ Error starting session viewer: {e}")
            return False
    
    def stop_viewer(self):
        """Stop the session viewer server."""
        if self.server_process and self.is_running:
            print("\n🛑 Stopping session viewer...")
            self.is_running = False
            
            try:
                # Send SIGTERM to the Node.js process
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
                
                print("✅ Session viewer stopped.")
                
            except Exception as e:
                print(f"⚠️  Error stopping server: {e}")
                try:
                    self.server_process.kill()
                except:
                    pass
    
    def _check_nodejs(self):
        """Check if Node.js is available."""
        try:
            result = subprocess.run(['node', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"✅ Node.js found: {version}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return False
    
    def _check_dependencies(self):
        """Check if dependencies are installed."""
        package_lock = self.viewer_dir / "package-lock.json"
        node_modules = self.viewer_dir / "node_modules"
        return package_lock.exists() and node_modules.exists()
    
    def _install_dependencies(self):
        """Install Node.js dependencies."""
        try:
            os.chdir(self.viewer_dir)
            result = subprocess.run(['npm', 'install'], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("✅ Dependencies installed successfully")
                return True
            else:
                print(f"❌ Failed to install dependencies: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error installing dependencies: {e}")
            return False
    
    def _check_build(self):
        """Check if React client is built."""
        build_dir = self.viewer_dir / "client" / "build"
        return build_dir.exists() and any(build_dir.iterdir())
    
    def _build_client(self):
        """Build the React client."""
        try:
            os.chdir(self.viewer_dir)
            result = subprocess.run(['npm', 'run', 'build-client'], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("✅ React client built successfully")
                return True
            else:
                print(f"❌ Failed to build client: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error building client: {e}")
            return False
    
    def _start_server(self, port):
        """Start the Node.js server."""
        try:
            os.chdir(self.viewer_dir)
            
            # Set environment variable for port
            env = os.environ.copy()
            env['PORT'] = str(port)
            
            # Start the server
            process = subprocess.Popen(
                ['npm', 'start'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )
            
            # Check if process started successfully
            time.sleep(2)
            if process.poll() is None:
                return process
            else:
                stdout, stderr = process.communicate()
                print(f"❌ Server failed to start: {stderr}")
                return None
                
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            return None
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n📡 Received signal {signum}")
        self.stop_viewer()
        sys.exit(0)

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch Cybersecurity Agent Session Viewer')
    parser.add_argument('--port', type=int, default=3001, help='Port to run the viewer on (default: 3001)')
    parser.add_argument('--no-browser', action='store_true', help='Don\'t open browser automatically')
    
    args = parser.parse_args()
    
    launcher = SessionViewerLauncher()
    
    try:
        success = launcher.start_viewer(args.port)
        if success:
            print("\n🎉 Session viewer is running!")
            print("   Use Ctrl+C to stop the viewer")
            
            # Keep running until interrupted
            while launcher.is_running:
                time.sleep(1)
        else:
            print("❌ Failed to start session viewer")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        launcher.stop_viewer()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        launcher.stop_viewer()
        sys.exit(1)

if __name__ == "__main__":
    main()
