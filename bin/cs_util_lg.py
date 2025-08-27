#!/usr/bin/env python3
"""
Simple CLI interface for LangGraph Cybersecurity Agent
Supports -csv and -output parameters for non-interactive use.
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from langgraph_cybersecurity_agent import LangGraphCybersecurityAgent

async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="LangGraph Cybersecurity Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python cybersecurity_cli.py
  
  # Process CSV file with specific output
  python cybersecurity_cli.py -csv policies.csv -output analysis_report.csv
  
  # Process CSV with JSON output
  python cybersecurity_cli.py -csv threats.csv -output threat_analysis.json
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
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = LangGraphCybersecurityAgent()
    await agent.start()
    
    # If CSV file is provided, process it
    if args.csv_file:
        csv_path = Path(args.csv_file)
        if not csv_path.exists():
            print(f"❌ Error: CSV file '{args.csv_file}' not found.")
            sys.exit(1)
        
        # Read CSV file
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            csv_content = df.to_csv(index=False)
            
            # Create session
            session_id = agent.session_manager.create_session("csv_processing")
            
            # Default prompt if none provided
            if not args.prompt:
                args.prompt = f"Please analyze this CSV file with {len(df)} rows and {len(df.columns)} columns. Provide insights and recommendations."
            
            # Process with agent
            print(f"📁 Processing CSV file: {args.csv_file}")
            print(f"📊 File contains {len(df)} rows and {len(df.columns)} columns")
            print(f"🤖 Agent prompt: {args.prompt}")
            print("\n🔄 Processing...")
            
            response = await agent.chat(args.prompt, session_id)
            
            # Save response to output file if specified
            if args.output_file:
                output_path = agent.session_manager.save_output_file(
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
            
        except Exception as e:
            print(f"❌ Error processing CSV file: {e}")
            sys.exit(1)
    
    # If no CSV file, enter interactive mode
    else:
        print("\n" + "="*60)
        print("🛡️  Cybersecurity Agent - LangGraph Cybersecurity Agent")
        print("="*60)
        print("🔐 Welcome, Security Professional! I'm your AI-powered cybersecurity companion.")
        print("🛡️  I can help you with threat hunting, policy analysis, incident response, and more.")
        print("\nAvailable workflows:")
        
        for template_name in agent.workflow_manager.list_templates():
            template = agent.workflow_manager.get_template(template_name)
            print(f"  • {template.name}: {template.description}")
        
        print("\n🔍 What would you like to investigate today? (Type 'quit' to exit)")
        print("-" * 60)
        
        while True:
            try:
                print("\n🔒 Analyst: ", end="", flush=True)
                user_input = input().strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("🛡️  Stay vigilant! Goodbye, Security Professional!")
                    break
                
                if not user_input:
                    continue
                
                print("\n🔄 Processing...")
                response = await agent.chat(user_input)
                print(f"\n🛡️  Cybersecurity Agent: {response}")
                
            except KeyboardInterrupt:
                print("\n\n🛡️  Stay vigilant! Goodbye, Security Professional!")
                break
            except EOFError:
                print("\n❌ Error: EOF when reading a line")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue

if __name__ == "__main__":
    asyncio.run(main())
