"""
Session Output Manager for Cybersecurity Agent

Manages automatic creation of session folders and output file generation
for chat sessions, workflows, and tool executions.
"""

import os
import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
import uuid

logger = logging.getLogger(__name__)

@dataclass
class OutputFile:
    """Represents an output file to be saved."""
    filename: str
    content: Union[str, bytes, Dict[str, Any]]
    content_type: str  # 'text', 'json', 'csv', 'html', 'markdown', 'binary'
    session_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SessionOutput:
    """Represents a session's output collection."""
    session_id: str
    session_name: str
    created_at: datetime
    output_files: List[OutputFile] = field(default_factory=list)
    session_metadata: Dict[str, Any] = field(default_factory=dict)

class SessionOutputManager:
    """Manages session output creation and file management."""
    
    def __init__(self, base_output_dir: str = "session-output"):
        """Initialize the session output manager."""
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Track active sessions
        self.active_sessions: Dict[str, SessionOutput] = {}
        
        # Output file counters
        self.session_file_counts: Dict[str, int] = {}
        
        logger.info(f"🚀 Session Output Manager initialized at {self.base_output_dir.absolute()}")
    
    def create_session(self, session_name: str = None, session_metadata: Dict[str, Any] = None) -> str:
        """Create a new session for output management."""
        session_id = str(uuid.uuid4())
        session_name = session_name or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session_output = SessionOutput(
            session_id=session_id,
            session_name=session_name,
            created_at=datetime.now(),
            session_metadata=session_metadata or {}
        )
        
        self.active_sessions[session_id] = session_output
        self.session_file_counts[session_id] = 0
        
        logger.info(f"📁 Created session: {session_name} (ID: {session_id})")
        
        return session_id
    
    def add_output_file(self, session_id: str, filename: str, content: Union[str, bytes, Dict[str, Any]], 
                       content_type: str = "text", metadata: Dict[str, Any] = None) -> Optional[str]:
        """Add an output file to a session."""
        if session_id not in self.active_sessions:
            logger.warning(f"⚠️  Session {session_id} not found, cannot add output file")
            return None
        
        # Create output file object
        output_file = OutputFile(
            filename=filename,
            content=content,
            content_type=content_type,
            session_id=session_id,
            metadata=metadata or {}
        )
        
        # Add to session
        self.active_sessions[session_id].output_files.append(output_file)
        self.session_file_counts[session_id] += 1
        
        logger.info(f"📄 Added output file: {filename} to session {session_id}")
        
        return output_file.filename
    
    def save_session_outputs(self, session_id: str, force_save: bool = False) -> Dict[str, Any]:
        """Save all output files for a session to disk."""
        if session_id not in self.active_sessions:
            return {"success": False, "error": f"Session {session_id} not found"}
        
        session = self.active_sessions[session_id]
        
        # Only create session folder if there are output files
        if not session.output_files and not force_save:
            logger.info(f"📁 No output files for session {session_id}, skipping folder creation")
            return {
                "success": True, 
                "message": "No output files to save",
                "session_folder": None
            }
        
        try:
            # Create session folder - use just the session ID for simplicity
            session_folder = self.base_output_dir / session_id
            session_folder.mkdir(exist_ok=True)
            
            saved_files = []
            
            # Save each output file
            for output_file in session.output_files:
                # Extract just the filename from the full path
                filename = Path(output_file.filename).name
                file_path = session_folder / filename
                
                if output_file.content_type == "json":
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(output_file.content, f, indent=2, default=str)
                elif output_file.content_type == "binary":
                    with open(file_path, 'wb') as f:
                        f.write(output_file.content)
                else:  # text, csv, html, markdown
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(str(output_file.content))
                
                saved_files.append(str(file_path))
                logger.info(f"💾 Saved: {file_path}")
            
            # Save session metadata
            metadata_file = session_folder / "session_metadata.json"
            session_data = {
                "session_id": session.session_id,
                "session_name": session.session_name,
                "created_at": session.created_at.isoformat(),
                "total_files": len(saved_files),
                "files": [
                    {
                        "filename": of.filename,
                        "content_type": of.content_type,
                        "timestamp": of.timestamp.isoformat(),
                        "metadata": of.metadata
                    }
                    for of in session.output_files
                ],
                "session_metadata": session.session_metadata
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            logger.info(f"✅ Session {session_id} saved to {session_folder}")
            
            return {
                "success": True,
                "session_folder": str(session_folder),
                "saved_files": saved_files,
                "total_files": len(saved_files),
                "metadata_file": str(metadata_file)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to save session {session_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def save_all_sessions(self) -> Dict[str, Any]:
        """Save all active sessions."""
        results = {}
        total_saved = 0
        
        for session_id in list(self.active_sessions.keys()):
            result = self.save_session_outputs(session_id)
            results[session_id] = result
            if result.get("success"):
                total_saved += 1
        
        logger.info(f"💾 Saved {total_saved} out of {len(self.active_sessions)} sessions")
        
        return {
            "success": True,
            "total_sessions": len(self.active_sessions),
            "saved_sessions": total_saved,
            "results": results
        }
    
    def end_session(self, session_id: str, save_outputs: bool = True) -> Dict[str, Any]:
        """End a session and optionally save outputs."""
        if session_id not in self.active_sessions:
            return {"success": False, "error": f"Session {session_id} not found"}
        
        session = self.active_sessions[session_id]
        
        # Save outputs if requested
        if save_outputs:
            save_result = self.save_session_outputs(session_id)
        else:
            save_result = {"success": True, "message": "Outputs not saved"}
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        if session_id in self.session_file_counts:
            del self.session_file_counts[session_id]
        
        logger.info(f"🔚 Ended session: {session.session_name} (ID: {session_id})")
        
        return {
            "success": True,
            "session_name": session.session_name,
            "session_id": session_id,
            "total_files": len(session.output_files),
            "save_result": save_result
        }
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of a specific session."""
        if session_id not in self.active_sessions:
            return {"success": False, "error": f"Session {session_id} not found"}
        
        session = self.active_sessions[session_id]
        
        return {
            "success": True,
            "session_id": session_id,
            "session_name": session.session_name,
            "created_at": session.created_at.isoformat(),
            "total_files": len(session.output_files),
            "file_types": list(set(of.content_type for of in session.output_files)),
            "session_metadata": session.session_metadata
        }
    
    def get_all_sessions_status(self) -> Dict[str, Any]:
        """Get status of all active sessions."""
        sessions_status = {}
        
        for session_id, session in self.active_sessions.items():
            sessions_status[session_id] = {
                "session_name": session.session_name,
                "created_at": session.session_name,
                "total_files": len(session.output_files),
                "file_types": list(set(of.content_type for of in session.output_files))
            }
        
        return {
            "success": True,
            "total_sessions": len(self.active_sessions),
            "sessions": sessions_status
        }
    
    def cleanup_old_sessions(self, max_age_days: int = 30) -> Dict[str, Any]:
        """Clean up old session folders."""
        try:
            cutoff_date = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
            removed_folders = []
            
            for folder in self.base_output_dir.iterdir():
                if folder.is_dir():
                    # Check if folder is old enough to remove
                    try:
                        folder_age = folder.stat().st_mtime
                        if folder_age < cutoff_date:
                            import shutil
                            shutil.rmtree(folder)
                            removed_folders.append(str(folder))
                            logger.info(f"🗑️  Removed old session folder: {folder}")
                    except Exception as e:
                        logger.warning(f"⚠️  Could not check/remove folder {folder}: {e}")
            
            return {
                "success": True,
                "removed_folders": removed_folders,
                "total_removed": len(removed_folders)
            }
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return {"success": False, "error": str(e)}

# Convenience functions for easy integration
_session_manager = None

def get_session_manager() -> SessionOutputManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionOutputManager()
    return _session_manager

def create_session(session_name: str = None, metadata: Dict[str, Any] = None) -> str:
    """Create a new session."""
    return get_session_manager().create_session(session_name, metadata)

def add_output_file(session_id: str, filename: str, content: Union[str, bytes, Dict[str, Any]], 
                   content_type: str = "text", metadata: Dict[str, Any] = None) -> Optional[str]:
    """Add an output file to a session."""
    return get_session_manager().add_output_file(session_id, filename, content, content_type, metadata)

def save_session(session_id: str) -> Dict[str, Any]:
    """Save a session's outputs."""
    return get_session_manager().save_session_outputs(session_id)

def end_session(session_id: str, save_outputs: bool = True) -> Dict[str, Any]:
    """End a session."""
    return get_session_manager().end_session(session_id, save_outputs)

if __name__ == "__main__":
    # Test the session output manager
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Testing Session Output Manager")
    print("=" * 50)
    
    # Create manager
    manager = SessionOutputManager()
    
    # Create session
    session_id = manager.create_session("test_session", {"purpose": "testing"})
    print(f"✅ Created session: {session_id}")
    
    # Add some output files
    manager.add_output_file(session_id, "test.txt", "Hello, World!", "text")
    manager.add_output_file(session_id, "data.json", {"key": "value", "number": 42}, "json")
    manager.add_output_file(session_id, "report.md", "# Test Report\n\nThis is a test.", "markdown")
    
    # Save session
    result = manager.save_session_outputs(session_id)
    print(f"✅ Save result: {result}")
    
    # End session
    end_result = manager.end_session(session_id)
    print(f"✅ End result: {end_result}")
    
    print("�� Test completed!")
