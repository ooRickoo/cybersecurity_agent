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

def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []
    
    try:
        import pandas as pd
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        from langgraph_cybersecurity_agent import LangGraphCybersecurityAgent
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
    session_dir = Path("session-outputs") / session_id
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

def execute_advanced_workflow(workflow_type: str, problem: str, agent=None):
    """Execute an advanced workflow."""
    if not agent:
        return f"⚠️  Advanced workflows require full agent initialization. Running in simplified mode.\n\nWorkflow Type: {workflow_type}\nProblem: {problem}\n\nTo enable full features:\n1. Install required dependencies: pip install -r requirements.txt\n2. Ensure LangGraph agent can initialize\n3. Restart the CLI"
    
    try:
        # This would integrate with the agent's workflow system
        return f"🚀 Executing {workflow_type} workflow for: {problem}\n\nThis feature requires full agent initialization and workflow templates."
    except Exception as e:
        return f"❌ Error executing workflow: {e}"

async def main():
    """Main CLI function."""
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
  
  # Execute advanced workflow
  python cs_util_lg.py -workflow threat_hunting -problem "Investigate APT29 activity"
  
  # List available workflows
  python cs_util_lg.py -list-workflows
  
  # Interactive workflow mode
  python cs_util_lg.py -workflow interactive -problem "Security investigation"
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
        choices=['threat_hunting', 'incident_response', 'compliance', 'risk_assessment', 'investigation', 'analysis', 'interactive'],
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
    if langgraph_available:
        try:
            agent = LangGraphCybersecurityAgent()
            await agent.start()
            print("🚀 LangGraph agent initialized successfully!")
        except Exception as e:
            print(f"⚠️  Warning: Agent initialization failed: {e}")
            print("🔄 Running in simplified mode...")
            agent = None
    
    # Handle list-workflows command
    if args.list_workflows:
        list_advanced_workflows()
        return
    
    # Handle advanced workflow execution
    if args.workflow_type and args.problem_description:
        session_id = str(uuid.uuid4())
        session_name = f"{args.workflow_type}_workflow"
        
        # Show cyber ASCII art at workflow startup
        if ASCII_ART_AVAILABLE:
            print("\n" + CyberASCIIArt.get_random_opening())
            print()
        
        print(f"🚀 Executing {args.workflow_type} workflow...")
        print(f"📋 Problem: {args.problem_description}")
        
        response = execute_advanced_workflow(args.workflow_type, args.problem_description, agent)
        
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
            sys.exit(1)
        
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
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error processing CSV file: {e}")
            sys.exit(1)
    
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
        
        print("\n🔍 What would you like to investigate today? (Type 'quit' to exit)")
        print("-" * 60)
        
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
