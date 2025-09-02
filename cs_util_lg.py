#!/usr/bin/env python3
"""
Comprehensive CLI interface for LangGraph Cybersecurity Agent
Combines simple interface with advanced workflow capabilities.
Supports CSV processing, interactive mode, and advanced workflows.
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path
import json
from datetime import datetime
import uuid

# Import cyber ASCII art
try:
    from bin.cyber_ascii_art import CyberASCIIArt
    ASCII_ART_AVAILABLE = True
except ImportError:
    ASCII_ART_AVAILABLE = False

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

# Import the main agent class
try:
    from bin.langgraph_cybersecurity_agent import LangGraphCybersecurityAgent
    LANGGRAPH_AGENT_AVAILABLE = True
except ImportError:
    LANGGRAPH_AGENT_AVAILABLE = False

def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []
    
    try:
        import pandas as pd
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        from bin.langgraph_cybersecurity_agent import LangGraphCybersecurityAgent
        return True, None
    except ImportError:
        missing_deps.append("langgraph_cybersecurity_agent")
    
    return False, missing_deps

def create_session_directories():
    """Create necessary session directories."""
    dirs = ["session-logs", "session-outputs", "knowledge-objects"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    print(f"✅ Created session directories: {', '.join(dirs)}")

def save_session_log(session_id: str, session_name: str, log_data: dict):
    """Save session log to file."""
    timestamp = datetime.now()
    log_filename = f"session_{timestamp.strftime('%Y%m%d_%H%M%S')}_{session_id[:8]}.json"
    log_file_path = Path("session-logs") / log_filename
    
    # Ensure the session-logs directory exists
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    log_data.update({
        "session_metadata": {
            "session_id": session_id,
            "session_name": session_name,
            "start_time": timestamp.isoformat(),
            "version": "2.0.0",
            "framework": "LangGraph Cybersecurity Agent",
            "end_time": None,
            "duration_ms": None
        }
    })
    
    with open(log_file_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    return log_file_path

def save_output_file(session_id: str, filename: str, content: str, content_type: str = "text"):
    """Save output file to session directory."""
    # Use absolute path to ensure files are created in the correct location
    # Get the project root directory (where cs_util_lg.py is located)
    project_root = Path(__file__).parent
    session_dir = project_root / "session-outputs" / session_id
    session_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (session_dir / "data").mkdir(exist_ok=True)
    (session_dir / "visualizations").mkdir(exist_ok=True)
    (session_dir / "text").mkdir(exist_ok=True)
    
    # Determine output directory based on content type
    if content_type == "data":
        output_dir = session_dir / "data"
    elif content_type == "visualization":
        output_dir = session_dir / "visualizations"
    else:
        output_dir = session_dir / "text"
    
    output_path = output_dir / filename
    with open(output_path, 'w') as f:
        f.write(content)
    
    return output_path

async def handle_memory_commands(user_input: str):
    """Handle memory-related commands with enhanced logging."""
    from bin.enhanced_knowledge_memory import enhanced_knowledge_memory
    
    # Handle both memory:command and memory command formats
    if ':' in user_input:
        # Format: memory:command
        command_parts = user_input.lower().split(':', 1)
        if len(command_parts) < 2:
            return "❌ Invalid memory command. Use: memory:<command> [options]"
        command = command_parts[1].split()[0]  # Get the command part
        args = command_parts[1].split()[1:] if len(command_parts[1].split()) > 1 else []
    else:
        # Format: memory command
        command_parts = user_input.lower().split()
        if len(command_parts) < 2:
            return "❌ Invalid memory command. Use: memory <command> [options]"
        command = command_parts[1]
        args = command_parts[2:] if len(command_parts) > 2 else []
    
    try:
        
        if command == "stats":
            stats = await enhanced_knowledge_memory.get_memory_stats()
            return f"📊 **Memory Statistics**\n\n{json.dumps(stats, indent=2, default=str)}"
                
        elif command == "search":
            if len(args) < 1:
                return "❌ Use: memory search <query> or memory:search <query>"
            query = " ".join(args)
            results = await enhanced_knowledge_memory.search_memory(query)
            return f"🔍 **Memory Search Results for '{query}'**\n\n{json.dumps(results, indent=2, default=str)}"
                
        elif command == "context":
            if len(args) < 1:
                return "❌ Use: memory context <query> or memory:context <query>"
            query = " ".join(args)
            context = await enhanced_knowledge_memory.get_llm_context(query)
            return f"🧠 **Memory Context for '{query}'**\n\n{json.dumps(context, indent=2, default=str)}"
                
        elif command == "import":
            if len(args) < 2:
                return "❌ Use: memory import <type> <file_path> or memory:import <type> <file_path>"
            import_type = args[0]
            file_path = args[1]
            
            # Log import start
            print(f"📥 **Starting {import_type.upper()} Import**")
            print(f"📁 File: {file_path}")
            print(f"⏱️  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            result = await enhanced_knowledge_memory.import_data_file(file_path, import_type)
            
            # Log import completion
            print(f"✅ **Import Completed**")
            print(f"📊 Results: {json.dumps(result, indent=2, default=str)}")
            
            return f"📥 **{import_type.upper()} Import Complete**\n\n{json.dumps(result, indent=2, default=str)}"
                
        elif command == "encryption":
            status = await enhanced_knowledge_memory.get_encryption_status()
            return f"🔐 **Encryption Status**\n\n{json.dumps(status, indent=2, default=str)}"
                
        elif command == "re-encrypt":
            if len(args) < 1:
                return "❌ Use: memory re-encrypt <new_password> or memory:re-encrypt <new_password>"
            new_password = args[0]
            
            print(f"🔐 **Starting Re-encryption**")
            print(f"⏱️  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            result = await enhanced_knowledge_memory.re_encrypt_memory_data(new_password)
            
            print(f"✅ **Re-encryption Completed**")
            return f"🔐 **Re-encryption Complete**\n\n{json.dumps(result, indent=2, default=str)}"
                
        else:
            return f"❌ Unknown memory command: {command}\n\nAvailable commands:\n• memory:stats - Show memory statistics\n• memory:search <query> - Search memory\n• memory:context <query> - Get LLM context\n• memory:import <type> <file> - Import data\n• memory:encryption - Show encryption status\n• memory:re-encrypt <password> - Re-encrypt data"
                
    except Exception as e:
        error_msg = f"Error in memory command '{command}': {str(e)}"
        print(f"❌ **Memory Command Error**: {error_msg}")
        return f"❌ **Error**: {error_msg}"

def list_advanced_workflows():
    """List available advanced workflow types."""
    workflows = [
        ("threat_hunting", "Automated threat hunting and analysis"),
        ("incident_response", "Incident response workflow orchestration"),
        ("compliance", "Compliance validation and reporting"),
        ("risk_assessment", "Risk assessment and analysis"),
        ("investigation", "Security investigation workflows"),
        ("analysis", "General security analysis workflows")
    ]
    
    print("\n🔧 Available Advanced Workflows:")
    for workflow_id, description in workflows:
        print(f"  • {workflow_id}: {description}")
    
    return workflows

async def execute_advanced_workflow(workflow_type: str, problem: str, agent=None, input_file_info=None, workflow_config=None):
    """Execute an advanced workflow."""
    if not agent:
        return f"⚠️  Advanced workflows require full agent initialization. Running in simplified mode.\n\nWorkflow Type: {workflow_type}\nProblem: {problem}\n\nTo enable full features:\n1. Install required dependencies: pip install -r requirements.txt\n2. Ensure LangGraph agent can initialize\n3. Restart the CLI"
    
    try:
        # Enhanced workflow execution with file input support
        workflow_context = {
            "workflow_type": workflow_type,
            "problem_description": problem,
            "input_file": input_file_info,
            "configuration": workflow_config or {},
            "timestamp": datetime.now().isoformat()
        }
        
        # For file-based workflows, actually execute the workflow
        if input_file_info and input_file_info['type'] in ['csv', 'json', 'yaml', 'xml', 'pcap', 'log']:
            if workflow_type == 'data_conversion':
                # Execute the actual data conversion workflow
                print(f"🚀 Executing {workflow_type} workflow for: {problem}")
                print(f"📁 Input file: {input_file_info['path']} ({input_file_info['type'].upper()})")
                print(f"⚙️  Configuration: {workflow_config}")
                print("\n🔄 Processing workflow...")
                
                try:
                    # Execute the workflow through the agent's chat method
                    workflow_result = await agent.chat(
                        message=problem,
                        workflow=workflow_type
                    )
                    
                    # Launch session viewer automatically using the session viewer manager
                    try:
                        from bin.session_viewer_manager import get_session_viewer_manager
                        svm = get_session_viewer_manager()
                        
                        # Start session viewer if not running
                        if not svm.get_status()['running']:
                            print("🚀 Starting session viewer...")
                            svm.start_viewer()
                            print("⏳ Waiting for session viewer to start...")
                            import time
                            time.sleep(3)  # Give it time to start
                        
                        # Open browser to the current session
                        session_id = getattr(agent.session_logger, 'session_id', 'unknown') if hasattr(agent, 'session_logger') else 'unknown'
                        session_url = f"http://localhost:3001/session/{session_id}"
                        print(f"🌐 Opening session viewer in browser: {session_url}")
                        
                        import webbrowser
                        webbrowser.open(session_url)
                        
                        print("✅ Session viewer launched successfully!")
                        print(f"📁 Session: {session_id}")
                        print(f"🌐 URL: {session_url}")
                        print("\n💡 Keep this terminal open to keep the session viewer running.")
                        print("   Press Ctrl+C to stop the session viewer.")
                        
                        # Add user prompt for session viewer management (patent analysis specific)
                        if workflow_type == 'patent_analysis':
                            print(f"\n" + "="*60)
                            print(f"🌐 Session Viewer is running in the background")
                            print(f"📱 URL: {session_url}")
                            print(f"")
                            print(f"Would you like to:")
                            print(f"1. Keep the session viewer running (current session)")
                            print(f"2. Shut down the server and continue")
                            print(f"3. Open the session viewer in your browser")
                            print(f"")
                            print(f"Enter your choice (1, 2, or 3) or press Enter to keep running:")
                            
                            try:
                                user_choice = input().strip()
                                if user_choice == "2":
                                    print(f"🛑 Shutting down session viewer server...")
                                    # The server will be stopped when the process exits
                                    print(f"✅ Session viewer server stopped. You can continue with other tasks.")
                                elif user_choice == "3":
                                    print(f"🌐 Opening session viewer in browser...")
                                    import webbrowser
                                    webbrowser.open(session_url)
                                    print(f"✅ Browser opened. Session viewer will continue running in background.")
                                else:
                                    print(f"✅ Session viewer will continue running in background.")
                                    print(f"📱 URL: {session_url}")
                            except KeyboardInterrupt:
                                print(f"\n🛑 Session viewer server stopped by user.")
                            except Exception as e:
                                print(f"⚠️  Error handling user choice: {e}")
                                print(f"✅ Session viewer will continue running in background.")
                        
                    except Exception as e:
                        print(f"⚠️  Warning: Could not launch session viewer: {e}")
                        print("   Session results are still available in the session outputs directory.")
                    
                    return f"✅ Workflow completed successfully!\n\n{workflow_result}\n\n💡 Session viewer has been launched to show detailed results."
                    
                except Exception as e:
                    return f"❌ Error executing workflow: {e}\n\n💡 Check the session logs for more details."
            
            elif workflow_type == 'patent_analysis':
                # Execute patent analysis workflow - PATENT_ANALYSIS_SPECIFIC_SECTION
                print(f"🚀 Executing {workflow_type} workflow for: {problem}")
                print(f"📁 Input file: {input_file_info['path']} ({input_file_info['type'].upper()})")
                
                # Handle output file parameter
                output_file = workflow_config.get('output_file') if workflow_config else None
                if output_file:
                    print(f"📁 Output file: {output_file}")
                else:
                    output_file = "enhanced_patent_analysis.csv"
                    print(f"📁 Output file: {output_file} (default)")
                
                print(f"⚙️  Configuration: {workflow_config}")
                print("\n🔄 Processing workflow...")
                
                try:
                    # Store output file path in agent context for the workflow to use
                    if hasattr(agent, 'session_logger') and agent.session_logger:
                        # Create a workflow context file with the output path
                        workflow_context = {
                            "output_file": output_file,
                            "input_file": input_file_info['path'],
                            "workflow_type": workflow_type,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Save workflow context to session
                        context_file = Path(f"session-outputs/{agent.session_logger.session_id}/workflow_context.json")
                        context_file.parent.mkdir(parents=True, exist_ok=True)
                        with open(context_file, 'w') as f:
                            json.dump(workflow_context, f, indent=2)
                    
                    # Execute the workflow through the agent's chat method
                    workflow_result = await agent.chat(
                        message=problem,
                        workflow=workflow_type
                    )
                    
                    # Launch session viewer automatically using the session viewer manager
                    try:
                        from bin.session_viewer_manager import get_session_viewer_manager
                        svm = get_session_viewer_manager()
                        
                        # Start session viewer if not running
                        if not svm.get_status()['running']:
                            print("🚀 Starting session viewer...")
                            svm.start_viewer()
                            print("⏳ Waiting for session viewer to start...")
                            import time
                            time.sleep(3)  # Give it time to start
                        
                        # Open browser to the current session
                        session_id = getattr(agent.session_logger, 'session_id', 'unknown') if hasattr(agent, 'session_logger') else 'unknown'
                        session_url = f"http://localhost:3001/session/{session_id}"
                        print(f"🌐 Opening session viewer in browser: {session_url}")
                        
                        import webbrowser
                        webbrowser.open(session_url)
                        
                        print("✅ Session viewer launched successfully!")
                        print(f"📁 Session: {session_id}")
                        print(f"🌐 URL: {session_url}")
                        print("\n💡 Keep this terminal open to keep the session viewer running.")
                        print("   Press Ctrl+C to stop the session viewer.")
                        
                        # Add user prompt for session viewer management (patent analysis specific)
                        if workflow_type == 'patent_analysis':
                            print(f"\n" + "="*60)
                            print(f"🌐 Session Viewer is running in the background")
                            print(f"📱 URL: {session_url}")
                            print(f"")
                            print(f"Would you like to:")
                            print(f"1. Keep the session viewer running (current session)")
                            print(f"2. Shut down the server and continue")
                            print(f"3. Open the session viewer in your browser")
                            print(f"")
                            print(f"Enter your choice (1, 2, or 3) or press Enter to keep running:")
                            
                            try:
                                user_choice = input().strip()
                                if user_choice == "2":
                                    print(f"🛑 Shutting down session viewer server...")
                                    # The server will be stopped when the process exits
                                    print(f"✅ Session viewer server stopped. You can continue with other tasks.")
                                elif user_choice == "3":
                                    print(f"🌐 Opening session viewer in browser...")
                                    import webbrowser
                                    webbrowser.open(session_url)
                                    print(f"✅ Browser opened. Session viewer will continue running in background.")
                                else:
                                    print(f"✅ Session viewer will continue running in background.")
                                    print(f"📱 URL: {session_url}")
                            except KeyboardInterrupt:
                                print(f"\n🛑 Session viewer server stopped by user.")
                            except Exception as e:
                                print(f"⚠️  Error handling user choice: {e}")
                                print(f"✅ Session viewer will continue running in background.")
                        
                    except Exception as e:
                        print(f"⚠️  Warning: Could not launch session viewer: {e}")
                        print("   Session results are still available in the session outputs directory.")
                    
                    return f"✅ Workflow completed successfully!\n\n{workflow_result}\n\n💡 Session viewer has been launched to show detailed results."
                    
                except Exception as e:
                    return f"❌ Error executing workflow: {e}\n\n💡 Check the session logs for more details."
            
            elif workflow_type == 'threat_analysis':
                # Execute threat analysis workflow
                print(f"🚀 Executing {workflow_type} workflow for: {problem}")
                print(f"📁 Input file: {input_file_info['path']} ({input_file_info['type'].upper()})")
                print(f"⚙️  Configuration: {workflow_config}")
                print("\n🔄 Processing workflow...")
                
                try:
                    # Execute the workflow through the agent's chat method
                    workflow_result = await agent.chat(
                        message=problem,
                        workflow=workflow_type
                    )
                    
                    # Launch session viewer automatically using the session viewer manager
                    try:
                        from bin.session_viewer_manager import get_session_viewer_manager
                        svm = get_session_viewer_manager()
                        
                        # Start session viewer if not running
                        if not svm.get_status()['running']:
                            print("🚀 Starting session viewer...")
                            svm.start_viewer()
                            print("⏳ Waiting for session viewer to start...")
                            import time
                            time.sleep(3)  # Give it time to start
                        
                        # Open browser to the current session
                        session_id = getattr(agent.session_logger, 'session_id', 'unknown') if hasattr(agent, 'session_logger') else 'unknown'
                        session_url = f"http://localhost:3001/session/{session_id}"
                        print(f"🌐 Opening session viewer in browser: {session_url}")
                        
                        import webbrowser
                        webbrowser.open(session_url)
                        
                        print("✅ Session viewer launched successfully!")
                        print(f"📁 Session: {session_id}")
                        print(f"🌐 URL: {session_url}")
                        print("\n💡 Keep this terminal open to keep the session viewer running.")
                        print("   Press Ctrl+C to stop the session viewer.")
                        
                        # Add user prompt for session viewer management (patent analysis specific)
                        if workflow_type == 'patent_analysis':
                            print(f"\n" + "="*60)
                            print(f"🌐 Session Viewer is running in the background")
                            print(f"📱 URL: {session_url}")
                            print(f"")
                            print(f"Would you like to:")
                            print(f"1. Keep the session viewer running (current session)")
                            print(f"2. Shut down the server and continue")
                            print(f"3. Open the session viewer in your browser")
                            print(f"")
                            print(f"Enter your choice (1, 2, or 3) or press Enter to keep running:")
                            
                            try:
                                user_choice = input().strip()
                                if user_choice == "2":
                                    print(f"🛑 Shutting down session viewer server...")
                                    # The server will be stopped when the process exits
                                    print(f"✅ Session viewer server stopped. You can continue with other tasks.")
                                elif user_choice == "3":
                                    print(f"🌐 Opening session viewer in browser...")
                                    import webbrowser
                                    webbrowser.open(session_url)
                                    print(f"✅ Browser opened. Session viewer will continue running in background.")
                                else:
                                    print(f"✅ Session viewer will continue running in background.")
                                    print(f"📱 URL: {session_url}")
                            except KeyboardInterrupt:
                                print(f"\n🛑 Session viewer server stopped by user.")
                            except Exception as e:
                                print(f"⚠️  Error handling user choice: {e}")
                                print(f"✅ Session viewer will continue running in background.")
                        
                    except Exception as e:
                        print(f"⚠️  Warning: Could not launch session viewer: {e}")
                        print("   Session results are still available in the session outputs directory.")
                    
                    return f"✅ Workflow completed successfully!\n\n{workflow_result}\n\n💡 Session viewer has been launched to show detailed results."
                    
                except Exception as e:
                    return f"❌ Error executing workflow: {e}\n\n💡 Check the session logs for more details."
            
            elif workflow_type == 'bulk_import':
                # Execute bulk import workflow
                print(f"🚀 Executing {workflow_type} workflow for: {problem}")
                print(f"📁 Input file: {input_file_info['path']} ({input_file_info['type'].upper()})")
                print(f"⚙️  Configuration: {workflow_config}")
                print("\n🔄 Processing workflow...")
                
                try:
                    # Execute the workflow through the agent's chat method
                    workflow_result = await agent.chat(
                        message=problem,
                        workflow=workflow_type
                    )
                    
                    # Launch session viewer automatically using the session viewer manager
                    try:
                        from bin.session_viewer_manager import get_session_viewer_manager
                        svm = get_session_viewer_manager()
                        
                        # Start session viewer if not running
                        if not svm.get_status()['running']:
                            print("🚀 Starting session viewer...")
                            svm.start_viewer()
                            print("⏳ Waiting for session viewer to start...")
                            import time
                            time.sleep(3)  # Give it time to start
                        
                        # Open browser to the current session
                        session_id = getattr(agent.session_logger, 'session_id', 'unknown') if hasattr(agent, 'session_logger') else 'unknown'
                        session_url = f"http://localhost:3001/session/{session_id}"
                        print(f"🌐 Opening session viewer in browser: {session_url}")
                        
                        import webbrowser
                        webbrowser.open(session_url)
                        
                        print("✅ Session viewer launched successfully!")
                        print(f"📁 Session: {session_id}")
                        print(f"🌐 URL: {session_url}")
                        print("\n💡 Keep this terminal open to keep the session viewer running.")
                        print("   Press Ctrl+C to stop the session viewer.")
                        
                    except Exception as e:
                        print(f"⚠️  Warning: Could not launch session viewer: {e}")
                        print("   Session results are still available in the session outputs directory.")
                    
                    return f"✅ Workflow completed successfully!\n\n{workflow_result}\n\n💡 Session viewer has been launched to show detailed results."
                    
                except Exception as e:
                    return f"❌ Error executing workflow: {e}\n\n💡 Check the session logs for more details."
            
            elif workflow_type in ['network_analysis', 'analysis', 'threat_hunting', 'incident_response', 'vulnerability_assessment', 'malware_analysis', 'vulnerability_scan']:
                # Execute other workflows
                print(f"🚀 Executing {workflow_type} workflow for: {problem}")
                print(f"📁 Input file: {input_file_info['path']} ({input_file_info['type'].upper()})")
                print(f"⚙️  Configuration: {workflow_config}")
                print("\n🔄 Processing workflow...")
                
                try:
                    # Execute the workflow through the agent's chat method
                    workflow_result = await agent.chat(
                        message=problem,
                        workflow=workflow_type
                    )
                    
                    # Launch session viewer automatically using the session viewer manager
                    try:
                        from bin.session_viewer_manager import get_session_viewer_manager
                        svm = get_session_viewer_manager()
                        
                        # Start session viewer if not running
                        if not svm.get_status()['running']:
                            print("🚀 Starting session viewer...")
                            svm.start_viewer()
                            print("⏳ Waiting for session viewer to start...")
                            import time
                            time.sleep(3)  # Give it time to start
                        
                        # Open browser to the current session
                        session_id = getattr(agent.session_logger, 'session_id', 'unknown') if hasattr(agent, 'session_logger') else 'unknown'
                        session_url = f"http://localhost:3001/session/{session_id}"
                        print(f"🌐 Opening session viewer in browser: {session_url}")
                        
                        import webbrowser
                        webbrowser.open(session_url)
                        
                        print("✅ Session viewer launched successfully!")
                        print(f"📁 Session: {session_id}")
                        print(f"🌐 URL: {session_url}")
                        print("\n💡 Keep this terminal open to keep the session viewer running.")
                        print("   Press Ctrl+C to stop the session viewer.")
                        
                        # Add user prompt for session viewer management (patent analysis specific)
                        if workflow_type == 'patent_analysis':
                            print(f"\n" + "="*60)
                            print(f"🌐 Session Viewer is running in the background")
                            print(f"📱 URL: {session_url}")
                            print(f"")
                            print(f"Would you like to:")
                            print(f"1. Keep the session viewer running (current session)")
                            print(f"2. Shut down the server and continue")
                            print(f"3. Open the session viewer in your browser")
                            print(f"")
                            print(f"Enter your choice (1, 2, or 3) or press Enter to keep running:")
                            
                            try:
                                user_choice = input().strip()
                                if user_choice == "2":
                                    print(f"🛑 Shutting down session viewer server...")
                                    # The server will be stopped when the process exits
                                    print(f"✅ Session viewer server stopped. You can continue with other tasks.")
                                elif user_choice == "3":
                                    print(f"🌐 Opening session viewer in browser...")
                                    import webbrowser
                                    webbrowser.open(session_url)
                                    print(f"✅ Browser opened. Session viewer will continue running in background.")
                                else:
                                    print(f"✅ Session viewer will continue running in background.")
                                    print(f"📱 URL: {session_url}")
                            except KeyboardInterrupt:
                                print(f"\n🛑 Session viewer server stopped by user.")
                            except Exception as e:
                                print(f"⚠️  Error handling user choice: {e}")
                                print(f"✅ Session viewer will continue running in background.")
                        
                    except Exception as e:
                        print(f"⚠️  Warning: Could not launch session viewer: {e}")
                        print("   Session results are still available in the session outputs directory.")
                    
                    return f"✅ Workflow completed successfully!\n\n{workflow_result}\n\n💡 Session viewer has been launched to show detailed results."
                    
                except Exception as e:
                    return f"❌ Error executing workflow: {e}\n\n💡 Check the session logs for more details."
        
        # Default workflow execution
        return f"🚀 Executing {workflow_type} workflow for: {problem}\n\n⚙️  Configuration: {workflow_config}\n\nThis feature requires full agent initialization and workflow templates.\n\n💡 For file-based workflows, ensure the input file is accessible and in the correct format."
        
    except Exception as e:
        return f"❌ Error executing workflow: {e}"

def check_activation():
    """Check if the agent is activated on this host."""
    try:
        from bin.activation_manager import ActivationManager
        activation_manager = ActivationManager()
        
        # Check if activation file exists
        if not activation_manager.activation_file.exists():
            return False, "No activation file found"
        
        # Get activation status
        status = activation_manager.get_activation_status()
        if not status['activated']:
            return False, "Agent not activated on this host"
        
        return True, "Activation verified"
        
    except ImportError:
        # If activation manager is not available, allow access (for development)
        return True, "Activation check bypassed (development mode)"
    except Exception as e:
        return False, f"Activation check failed: {str(e)}"

async def main():
    """Main CLI function."""
    # Check activation first
    print("🔐 Checking activation...")
    is_activated, activation_message = check_activation()
    
    if not is_activated:
        print(f"❌ {activation_message}")
        print("\n🛡️  Cybersecurity Agent requires activation")
        print("   This tool is bound to your specific host for security")
        print("   Please run the activation utility first:")
        print("   python bin/activate_agent.py")
        print("\n   Or use the activation manager directly:")
        print("   python bin/activation_manager.py create")
        return  # Exit gracefully instead of sys.exit
    
    print(f"✅ {activation_message}")
    print()
    
    parser = argparse.ArgumentParser(
        description="Cybersecurity Agent CLI - Simple Interface with Advanced Capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python cs_util_lg.py
  
  # Process CSV file with specific output
  python cs_util_lg.py -csv policies.csv -output analysis_report.csv
  
  # Process CSV with JSON output
  python cs_util_lg.py -csv threats.csv -output threat_analysis.json
  
  # Import file into knowledge graph memory
  python cs_util_lg.py -memory applications.json -memory-config '{"flatten_nested": true, "create_relationships": true}'
  
  # Execute advanced workflow
  python cs_util_lg.py -workflow threat_hunting -problem "Investigate APT29 activity"
  
  # Execute workflow with file input (Google Chronicle to Splunk ES conversion)
  python cs_util_lg.py -workflow data_conversion -problem "Convert Google Chronicle content catalog to Splunk ES format" -input-file chronicle_catalog.json -input-type json
  
  # Execute workflow with CSV input
  python cs_util_lg.py -workflow threat_analysis -problem "Analyze threat indicators" -input-file threat_data.csv -input-type csv
  
  # Execute workflow with custom configuration
  python cs_util_lg.py -workflow incident_response -problem "Investigate security incident" -input-file incident_logs.json -workflow-config '{"priority": "high", "scope": "network"}'
  
  # Execute patent analysis workflow
  python cs_util_lg.py -workflow patent_analysis -problem "Analyze cybersecurity patents" -input-file patent_list.csv -input-type csv
  
  # List available workflows
  python cs_util_lg.py -list-workflows
  
  # Interactive workflow mode
  python cs_util_lg.py -workflow interactive -problem "Security investigation"
  
  # Launch session viewer for specific session
  python cs_util_lg.py -session-viewer 20250829_141043_eee64e92
        """
    )
    
    parser.add_argument(
        '-csv', '--csv-file',
        type=str,
        help='CSV file to process'
    )
    
    parser.add_argument(
        '-output', '--output-file',
        type=str,
        help='Output file name'
    )
    
    parser.add_argument(
        '-prompt', '--prompt',
        type=str,
        help='Specific prompt/question for the agent'
    )
    
    parser.add_argument(
        '-workflow', '--workflow-type',
        type=str,
        choices=['threat_hunting', 'incident_response', 'compliance', 'risk_assessment', 'investigation', 'analysis', 'interactive', 'data_conversion', 'threat_analysis', 'bulk_import', 'network_analysis', 'vulnerability_assessment', 'patent_analysis', 'malware_analysis', 'vulnerability_scan'],
        help='Advanced workflow type to execute'
    )
    
    parser.add_argument(
        '-problem', '--problem-description',
        type=str,
        help='Problem description for workflow execution'
    )
    
    parser.add_argument(
        '-list-workflows', '--list-workflows',
        action='store_true',
        help='List all available advanced workflows'
    )
    
    parser.add_argument(
        '-memory', '--memory-import',
        type=str,
        help='Import file into knowledge graph memory (CSV, JSON, YAML, XML)'
    )
    
    parser.add_argument(
        '-memory-config', '--memory-config',
        type=str,
        help='Memory import configuration (JSON string)'
    )
    
    parser.add_argument(
        '-session-viewer', '--session-viewer',
        type=str,
        help='Launch session viewer for a specific session folder (e.g., 20250829_141043_eee64e92)'
    )
    
    parser.add_argument(
        '-input-file', '--input-file',
        type=str,
        help='Input file for workflow processing (CSV, JSON, YAML, XML, PCAP, etc.)'
    )
    
    parser.add_argument(
        '-input-type', '--input-type',
        type=str,
        choices=['csv', 'json', 'yaml', 'xml', 'pcap', 'log', 'auto'],
        default='auto',
        help='Input file type (auto-detection if not specified)'
    )
    
    parser.add_argument(
        '-workflow-config', '--workflow-config',
        type=str,
        help='Additional workflow configuration (JSON string)'
    )
    
    args = parser.parse_args()
    
    # Check dependencies and create directories
    print("🔍 Checking dependencies...")
    langgraph_available, missing_deps = check_dependencies()
    
    if not langgraph_available:
        print(f"⚠️  Warning: LangGraph agent not available: {', '.join(missing_deps)}")
        print("🔄 Running in simplified mode...")
    
    # Create session directories
    create_session_directories()
    
    # Initialize agent if available
    agent = None
    if langgraph_available and LANGGRAPH_AGENT_AVAILABLE:
        try:
            agent = LangGraphCybersecurityAgent()
            await agent.start()
            print("🚀 LangGraph agent initialized successfully!")
        except Exception as e:
            print(f"⚠️  Warning: Agent initialization failed: {e}")
            print("🔄 Running in simplified mode...")
            agent = None
    elif not LANGGRAPH_AGENT_AVAILABLE:
        print("⚠️  Warning: LangGraph agent module not available")
        print("🔄 Running in simplified mode...")
        agent = None
    
    # Handle session viewer launch
    if args.session_viewer:
        session_folder = args.session_viewer
        print(f"🔍 Launching session viewer for session: {session_folder}")
        
        # Check if session folder exists
        session_path = Path("session-outputs") / session_folder
        if not session_path.exists():
            print(f"❌ Error: Session folder '{session_folder}' not found in session-outputs/")
            print("Available sessions:")
            available_sessions = list(Path("session-outputs").glob("*"))
            for session in available_sessions[:10]:  # Show first 10
                print(f"  • {session.name}")
            if len(available_sessions) > 10:
                print(f"  ... and {len(available_sessions) - 10} more")
            return
        
        # Launch session viewer
        try:
            from bin.session_viewer_manager import get_session_viewer_manager
            svm = get_session_viewer_manager()
            
            # Start session viewer if not running
            if not svm.get_status()['running']:
                print("🚀 Starting session viewer...")
                svm.start_viewer()
                print("⏳ Waiting for session viewer to start...")
                import time
                time.sleep(3)  # Give it time to start
            
            # Open browser to specific session
            session_url = f"http://localhost:3001/session/{session_folder}"
            print(f"🌐 Opening session viewer in browser: {session_url}")
            
            import webbrowser
            webbrowser.open(session_url)
            
            print("✅ Session viewer launched successfully!")
            print(f"📁 Session: {session_folder}")
            print(f"🌐 URL: {session_url}")
            print("\n💡 Keep this terminal open to keep the session viewer running.")
            print("   Press Ctrl+C to stop the session viewer.")
            
            # Keep session viewer running
            try:
                svm.keep_alive()
            except KeyboardInterrupt:
                print("\n🛑 Stopping session viewer...")
                svm.stop_viewer()
                print("✅ Session viewer stopped.")
            
        except Exception as e:
            print(f"❌ Error launching session viewer: {e}")
            print("💡 Make sure the session viewer dependencies are installed.")
        
        return
    
    # Handle list-workflows command
    if args.list_workflows:
        list_advanced_workflows()
        return
    
    # Handle advanced workflow execution
    if args.workflow_type and args.problem_description:
        # Use the agent's session ID if available, otherwise generate new one
        if agent and hasattr(agent, 'session_logger') and agent.session_logger:
            session_id = agent.session_logger.session_id
        else:
            session_id = str(uuid.uuid4())
        session_name = f"{args.workflow_type}_workflow"
        
        # Validate input file if provided
        input_file_info = None
        if args.input_file:
            input_path = Path(args.input_file).resolve()  # Resolve to absolute path
            if not input_path.exists():
                print(f"❌ Error: Input file '{args.input_file}' not found.")
                return
            
            # Auto-detect file type if not specified
            if args.input_type == 'auto':
                file_extension = input_path.suffix.lower()
                if file_extension == '.csv':
                    detected_type = 'csv'
                elif file_extension == '.json':
                    detected_type = 'json'
                elif file_extension == '.yaml' or file_extension == '.yml':
                    detected_type = 'yaml'
                elif file_extension == '.xml':
                    detected_type = 'xml'
                elif file_extension == '.pcap':
                    detected_type = 'pcap'
                elif file_extension in ['.log', '.txt']:
                    detected_type = 'log'
                else:
                    detected_type = 'unknown'
                print(f"🔍 Auto-detected file type: {detected_type}")
            else:
                detected_type = args.input_type
            
            input_file_info = {
                'path': str(input_path),
                'type': detected_type,
                'size': input_path.stat().st_size,
                'exists': True
            }
            
            print(f"📁 Input file: {args.input_file} ({detected_type.upper()})")
        
        # Parse workflow configuration if provided
        workflow_config = {}
        if args.workflow_config:
            try:
                workflow_config = json.loads(args.workflow_config)
                print(f"⚙️  Workflow configuration: {workflow_config}")
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Invalid workflow configuration JSON: {e}")
                print("   Continuing with default configuration...")
        
        # Add output file to workflow configuration if provided
        if args.output_file:
            workflow_config['output_file'] = args.output_file
            print(f"📁 Output file specified: {args.output_file}")
        
        # Show cyber ASCII art at workflow startup
        if ASCII_ART_AVAILABLE:
            print("\n" + CyberASCIIArt.get_random_opening())
            print()
        
        print(f"🚀 Executing {args.workflow_type} workflow...")
        print(f"📋 Problem: {args.problem_description}")
        if input_file_info:
            print(f"📁 Input: {input_file_info['path']} ({input_file_info['type'].upper()})")
        
        # Set the session ID in the agent to match the CLI session ID
        if agent and hasattr(agent, 'set_session_id'):
            agent.set_session_id(session_id)
        
        # Copy input file to session output directory FIRST (before workflow execution)
        if input_file_info and input_file_info['exists']:
            try:
                import shutil
                input_filename = Path(input_file_info['path']).name
                # Use absolute path to ensure files are created in the correct location
                project_root = Path(__file__).parent
                session_input_dir = project_root / "session-outputs" / session_id / "input"
                session_input_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(input_file_info['path'], session_input_dir / input_filename)
                print(f"📁 Input file copied to session: {session_input_dir / input_filename}")
            except Exception as e:
                print(f"⚠️  Warning: Could not copy input file to session: {e}")
        
        # Now execute the workflow (after input file is copied)
        response = await execute_advanced_workflow(args.workflow_type, args.problem_description, agent, input_file_info, workflow_config)
        
        # Save response to output file
        output_filename = f"{args.workflow_type}_workflow_{session_id[:8]}.txt"
        output_path = save_output_file(session_id, output_filename, response, "text")
        
        print(f"\n📊 Workflow Response:")
        print("-" * 60)
        print(response)
        print("-" * 60)
        print(f"\n📁 Session ID: {session_id}")
        print(f"💾 Response saved to: {output_path}")
        
        # Show cyber ASCII art at workflow completion
        if ASCII_ART_AVAILABLE:
            print("\n" + CyberASCIIArt.get_random_closing())
            print()
        
        # Save session log
        log_data = {
            "agent_interactions": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "level": "info",
                    "category": "workflow_execution",
                    "action": args.workflow_type,
                    "details": {
                        "workflow_type": args.workflow_type,
                        "problem": args.problem_description,
                        "input_file": input_file_info,
                        "workflow_config": workflow_config,
                        "response": response
                    },
                    "session_id": session_id,
                    "agent_type": "LangGraphAgent" if agent else "SimplifiedMode",
                    "workflow_step": "workflow_execution"
                }
            ]
        }
        
        save_session_log(session_id, session_name, log_data)
        return
    
    # If CSV file is provided, process it
    if args.csv_file:
        csv_path = Path(args.csv_file)
        if not csv_path.exists():
            print(f"❌ Error: CSV file '{args.csv_file}' not found.")
            return
        
        # Create session
        session_id = str(uuid.uuid4())
        session_name = "csv_processing"
        
        # Read CSV file
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            csv_content = df.to_csv(index=False)
            
            # Default prompt if none provided
            if not args.prompt:
                args.prompt = f"Please analyze this CSV file with {len(df)} rows and {len(df.columns)} columns. Provide insights and recommendations."
            
            # Show cyber ASCII art at CSV processing startup
            if ASCII_ART_AVAILABLE:
                print("\n" + CyberASCIIArt.get_random_opening())
                print()
            
            # Process with agent or fallback
            print(f"📁 Processing CSV file: {args.csv_file}")
            print(f"📊 File contains {len(df)} rows and {len(df.columns)} columns")
            print(f"🤖 Agent prompt: {args.prompt}")
            print("\n🔄 Processing...")
            
            if agent:
                response = await agent.chat(args.prompt, session_id)
            else:
                # Fallback processing
                response = f"""CSV Analysis Results:
                
File: {args.csv_file}
Rows: {len(df)}
Columns: {len(df.columns)}
Column Names: {', '.join(df.columns.tolist())}

Analysis:
- This appears to be a dataset with {len(df)} records
- The data structure includes {len(df.columns)} fields
- Consider reviewing the data quality and completeness
- For detailed analysis, ensure LangGraph agent is available

Recommendations:
- Validate data integrity
- Check for missing values
- Consider data normalization if needed
- Review column data types for consistency"""
            
            # Save response to output file if specified
            if args.output_file:
                output_path = save_output_file(
                    session_id, 
                    args.output_file, 
                    response, 
                    "text"
                )
                print(f"\n💾 Response saved to: {output_path}")
            
            print(f"\n🤖 Agent Response:")
            print("-" * 60)
            print(response)
            print("-" * 60)
            print(f"\n📁 Session ID: {session_id}")
            print(f"📝 Session logs: session-logs/")
            print(f"💾 Session outputs: session-outputs/{session_id}/")
            
            # Show cyber ASCII art at CSV processing completion
            if ASCII_ART_AVAILABLE:
                print("\n" + CyberASCIIArt.get_random_closing())
                print()
            
            # Save session log
            log_data = {
                "agent_interactions": [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "level": "info",
                        "category": "csv_processing",
                        "action": "csv_analysis",
                        "details": {
                            "csv_file": args.csv_file,
                            "rows": len(df),
                            "columns": len(df.columns),
                            "prompt": args.prompt
                        },
                        "session_id": session_id,
                        "agent_type": "LangGraphAgent" if agent else "SimplifiedMode",
                        "workflow_step": "csv_analysis"
                    }
                ]
            }
            
            save_session_log(session_id, session_name, log_data)
            
        except ImportError:
            print("❌ Error: pandas not available. Please install with: pip install pandas")
            return
        except Exception as e:
            print(f"❌ Error processing CSV file: {e}")
            return
    
    # Handle memory import
    if args.memory_import:
        memory_path = Path(args.memory_import)
        if not memory_path.exists():
            print(f"❌ Error: Memory import file '{args.memory_import}' not found.")
            return
        
        # Create session
        session_id = str(uuid.uuid4())
        session_name = "memory_import"
        
        try:
            # Import enhanced knowledge memory
            from bin.enhanced_knowledge_memory import enhanced_knowledge_memory
            
            # Parse memory config if provided
            import_config = {
                "flatten_nested": True,
                "create_relationships": True,
                "extract_entities": True,
                "normalize_fields": True,
                "max_depth": 5
            }
            
            if args.memory_config:
                try:
                    user_config = json.loads(args.memory_config)
                    import_config.update(user_config)
                except json.JSONDecodeError:
                    print("⚠️  Warning: Invalid memory config JSON, using defaults")
            
            # Show cyber ASCII art at memory import startup
            if ASCII_ART_AVAILABLE:
                print("\n" + CyberASCIIArt.get_random_opening())
                print()
            
            print(f"🧠 Importing file into knowledge graph memory: {args.memory_import}")
            print(f"📋 Import configuration: {json.dumps(import_config, indent=2)}")
            print("\n🔄 Processing...")
            
            # Import data
            result = await enhanced_knowledge_memory.import_data_file(str(memory_path), import_config)
            
            print(f"\n✅ Memory Import Results:")
            print("-" * 60)
            print(f"📁 Source file: {result.source_file}")
            print(f"📊 File type: {result.import_type}")
            print(f"🔗 Nodes created: {result.nodes_created}")
            print(f"🔗 Relationships created: {result.relationships_created}")
            print(f"⏱️  Processing time: {result.processing_time:.2f}s")
            
            if result.errors:
                print(f"❌ Errors: {result.errors}")
            if result.warnings:
                print(f"⚠️  Warnings: {result.warnings}")
            
            print("-" * 60)
            print(f"\n📁 Session ID: {session_id}")
            print(f"📝 Session logs: session-logs/")
            print(f"💾 Session outputs: session-outputs/{session_id}/")
            
            # Show cyber ASCII art at memory import completion
            if ASCII_ART_AVAILABLE:
                print("\n" + CyberASCIIArt.get_random_closing())
                print()
            
            # Save session log
            log_data = {
                "agent_interactions": [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "level": "info",
                        "category": "memory_import",
                        "action": "bulk_import",
                        "details": {
                            "source_file": str(memory_path),
                            "import_config": import_config,
                            "result": {
                                "nodes_created": result.nodes_created,
                                "relationships_created": result.relationships_created,
                                "processing_time": result.processing_time
                            }
                        },
                        "session_id": session_id,
                        "agent_type": "EnhancedMemorySystem",
                        "workflow_step": "memory_import"
                    }
                ]
            }
            
            save_session_log(session_id, session_name, log_data)
            return
            
        except ImportError:
            print("❌ Error: Enhanced knowledge memory system not available")
            print("   Please ensure all dependencies are installed")
            return
        except Exception as e:
            print(f"❌ Error during memory import: {e}")
            return
    
    # If no specific command, enter interactive mode
    else:
        # Show cyber ASCII art at startup
        if ASCII_ART_AVAILABLE:
            print("\n" + CyberASCIIArt.get_random_opening())
            print()
        
        print("="*60)
        print("🛡️  Cybersecurity Agent - LangGraph Cybersecurity Agent")
        print("="*60)
        
        if agent:
            print("🔐 Welcome, Security Professional! I'm your AI-powered cybersecurity companion.")
            print("🛡️  I can help you with threat hunting, policy analysis, incident response, and more.")
            print("\nAvailable workflows:")
            
            try:
                for template_name in agent.workflow_manager.list_templates():
                    template = agent.workflow_manager.get_template(template_name)
                    print(f"  • {template.name}: {template.description}")
            except:
                print("  • Basic workflow templates (advanced features require full initialization)")
        else:
            print("🔐 Welcome, Security Professional! Running in simplified mode.")
            print("🛡️  Basic functionality available. For full features, ensure LangGraph agent is properly initialized.")
            print("\nAvailable in simplified mode:")
            print("  • CSV file processing")
            print("  • Basic session management")
            print("  • File output and logging")
            print("  • Advanced workflow templates (when agent available)")
            print("  • Knowledge Graph Memory operations")
        
        print("\n🔍 What would you like to investigate today? (Type 'quit' to exit)")
        print("-" * 60)
        
        # Check if input is piped (non-interactive mode)
        import sys
        is_piped = not sys.stdin.isatty()
        
        if is_piped:
            # Handle piped input
            try:
                user_input = input().strip()
                if user_input.lower().startswith('memory'):
                    response = await handle_memory_commands(user_input)
                    print(response)
                else:
                    print(f"Processed input: {user_input}")
                return
            except EOFError:
                print("No input provided")
                return
        
        # Interactive mode
        while True:
            try:
                print("\n🔒 Analyst: ", end="", flush=True)
                user_input = input().strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    # Show cyber ASCII art at closing
                    if ASCII_ART_AVAILABLE:
                        print("\n" + CyberASCIIArt.get_random_closing())
                        print()
                    print("🛡️  Stay vigilant! Goodbye, Security Professional!")
                    break
                
                # Handle memory commands
                if user_input.lower().startswith('memory'):
                    response = await handle_memory_commands(user_input)
                    print(response)
                    continue
                
                if not user_input:
                    continue
                
                print("\n🔄 Processing...")
                
                if agent:
                    response = await agent.chat(user_input)
                else:
                    response = f"""Simplified Mode Response:
                    
Your query: "{user_input}"

Note: Running in simplified mode. For full AI-powered analysis, ensure the LangGraph agent is properly initialized.

Available actions:
- Process CSV files with -csv parameter
- Save outputs with -output parameter
- Execute advanced workflows with -workflow parameter
- Basic session logging and management

To enable full features:
1. Install required dependencies: pip install -r requirements.txt
2. Ensure LangGraph agent can initialize
3. Restart the CLI"""
                
                print(f"\n🛡️  Cybersecurity Agent: {response}")
                
            except KeyboardInterrupt:
                # Show cyber ASCII art at closing
                if ASCII_ART_AVAILABLE:
                    print("\n" + CyberASCIIArt.get_random_closing())
                    print()
                print("\n🛡️  Stay vigilant! Goodbye, Security Professional!")
                break
            except EOFError:
                print("\n❌ Error: EOF when reading a line")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue

if __name__ == "__main__":
    asyncio.run(main())
