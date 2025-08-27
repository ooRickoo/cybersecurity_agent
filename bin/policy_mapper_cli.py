#!/usr/bin/env python3
"""
Policy Mapper CLI - Standalone tool for policy mapping with session files

This tool provides the same functionality as the ADK agent but can be run directly
to create session files and logs.
"""

import sys
import json
import argparse
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from comprehensive_session_manager import (
    create_comprehensive_session, 
    add_workflow_execution, 
    add_detailed_workflow_step,
    add_workflow_progress,
    finalize_comprehensive_session
)
from adk_python_bridge import execute_policy_mapping_bridge

def main():
    """Main CLI interface for policy mapping."""
    parser = argparse.ArgumentParser(description='Policy Mapper CLI - MITRE ATT&CK Policy Mapping')
    parser.add_argument('--csv-file', help='Path to CSV file with policies (optional - uses sample data if not provided)')
    parser.add_argument('--output-format', choices=['json', 'text', 'table'], default='table', 
                       help='Output format')
    parser.add_argument('--create-session', action='store_true', default=True,
                       help='Create session files and logs (default: true)')
    
    args = parser.parse_args()
    
    try:
        print("🚀 Policy Mapper CLI - MITRE ATT&CK Policy Mapping")
        print("=" * 60)
        
        # Create comprehensive session if requested
        session_id = None
        if args.create_session:
            print("📁 Creating comprehensive session...")
            session_id = create_comprehensive_session("policy_mapper_cli_session")
            print(f"✅ Session created: {session_id}")
            
            # Add workflow execution start
            add_workflow_execution(session_id, "policy_mapping_cli", {"status": "started"})
        
        # Execute policy mapping with detailed workflow logging
        print("🔍 Executing policy mapping analysis...")
        
        # Step 1: Data Preparation
        if session_id:
            add_workflow_progress(session_id, 1, 6, "Data Preparation", {
                "step_description": "Preparing policy data for analysis",
                "data_source": "sample_data" if not args.csv_file else "csv_file",
                "file_path": args.csv_file if args.csv_file else "None (using sample data)"
            })
        
        if args.csv_file:
            # Read CSV file
            print("📂 Reading CSV file...")
            with open(args.csv_file, 'r') as f:
                csv_data = f.read()
            
            if session_id:
                add_detailed_workflow_step(session_id, "csv_file_reading", {
                    "file_path": args.csv_file,
                    "file_size_bytes": len(csv_data),
                    "file_size_lines": len(csv_data.split('\n')),
                    "operation": "file_read",
                    "status": "success"
                }, "data_operation", "policy_mapping_cli")
        else:
            # Use sample data
            print("📋 Using sample policy data...")
            if session_id:
                add_detailed_workflow_step(session_id, "sample_data_preparation", {
                    "data_source": "built_in_sample",
                    "sample_policies_count": 5,
                    "operation": "sample_data_loaded",
                    "status": "success"
                }, "data_operation", "policy_mapping_cli")
        
        # Step 2: MITRE ATT&CK Mapping Execution
        if session_id:
            add_workflow_progress(session_id, 2, 6, "MITRE ATT&CK Mapping Execution", {
                "step_description": "Executing policy mapping to MITRE ATT&CK framework",
                "mapping_engine": "adk_python_bridge",
                "execution_method": "direct_function_call"
            })
        
        print("🗺️ Executing MITRE ATT&CK mapping...")
        if args.csv_file:
            result = execute_policy_mapping_bridge(csv_data, session_id)
        else:
            result = execute_policy_mapping_bridge(None, session_id)
        
        # Step 3: Result Validation
        if session_id:
            add_workflow_progress(session_id, 3, 6, "Result Validation", {
                "step_description": "Validating policy mapping results",
                "success": result.get("success", False),
                "error_message": result.get("error", None)
            })
        
        if not result.get("success"):
            print(f"❌ Policy mapping failed: {result.get('error', 'Unknown error')}")
            if session_id:
                add_workflow_execution(session_id, "policy_mapping_cli", {"error": result.get("error")})
                finalize_comprehensive_session(session_id, success=False, error_message=result.get("error"))
            sys.exit(1)
        
        # Step 4: Data Analysis
        if session_id:
            add_workflow_progress(session_id, 4, 6, "Data Analysis", {
                "step_description": "Analyzing mapping results and calculating metrics",
                "total_policies": result.get("total_policies", 0),
                "mapped_policies": result.get("mapped_policies", 0),
                "average_confidence": result.get("average_confidence", 0.0),
                "mitre_techniques_found": len(set([p.get("mitre_technique_id") for p in result.get("enriched_data", []) if p.get("mitre_technique_id")]))
            })
        
        # Step 5: Output Generation
        if session_id:
            add_workflow_progress(session_id, 5, 6, "Output Generation", {
                "step_description": "Generating formatted output and display",
                "output_format": args.output_format,
                "display_method": "console_output"
            })
        
        # Display results based on format
        if args.output_format == 'json':
            print(json.dumps(result, indent=2))
        elif args.output_format == 'text':
            display_text_results(result)
        else:  # table format
            display_table_results(result)
        
        # Step 6: Session Finalization
        if session_id:
            add_workflow_progress(session_id, 6, 6, "Session Finalization", {
                "step_description": "Finalizing session and saving all outputs",
                "session_id": session_id,
                "output_files_created": True,
                "session_log_completed": True
            })
            
            # Add workflow execution completion with detailed metrics
            add_workflow_execution(session_id, "policy_mapping_cli", {
                "policies_processed": result.get("total_policies", 0),
                "mitre_mappings": result.get("mapped_policies", 0),
                "average_confidence": result.get("average_confidence", 0.0),
                "workflow_steps_completed": 6,
                "execution_environment": "external_cli",
                "code_controlled": True
            })
            
            # Finalize session
            finalize_comprehensive_session(session_id, success=True)
            print(f"\n📁 Session finalized: {session_id}")
            print(f"📝 Session log created in: session-logs/")
            print(f"💾 Session output created in: session-output/{session_id}/")
        
        print("\n✅ Policy mapping completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if session_id:
            try:
                # Log the error with detailed information
                add_detailed_workflow_step(session_id, "error_handling", {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "error_location": "main_execution_loop",
                    "stack_trace": str(e),
                    "recovery_attempted": False
                }, "error_handling", "policy_mapping_cli")
                
                add_workflow_execution(session_id, "policy_mapping_cli", {"error": str(e)})
                finalize_comprehensive_session(session_id, success=False, error_message=str(e))
            except Exception as log_error:
                print(f"⚠️  Failed to log error: {log_error}")
        sys.exit(1)

def display_text_results(result):
    """Display results in text format."""
    print(f"\n📊 Policy Mapping Results")
    print(f"Total Policies: {result.get('total_policies', 0)}")
    print(f"Mapped Policies: {result.get('mapped_policies', 0)}")
    print(f"Average Confidence: {result.get('average_confidence', 0.0):.2f}")
    
    if 'enriched_data' in result:
        print(f"\n📋 Policy Details:")
        for policy in result['enriched_data']:
            print(f"\nPolicy: {policy.get('policy_name', 'Unknown')}")
            print(f"  ID: {policy.get('policy_id', 'Unknown')}")
            print(f"  Category: {policy.get('category', 'Unknown')}")
            print(f"  MITRE Technique: {policy.get('mitre_technique_id', 'Unknown')} - {policy.get('mitre_technique_name', 'Unknown')}")
            print(f"  MITRE Tactic: {policy.get('mitre_tactic_id', 'Unknown')} - {policy.get('mitre_tactic_name', 'Unknown')}")
            print(f"  Confidence: {policy.get('mitre_confidence_score', 0.0):.2f}")
            print(f"  Reasoning: {policy.get('mapping_reasoning', 'No reasoning provided')}")

def display_table_results(result):
    """Display results in table format."""
    print(f"\n📊 Policy Mapping Results")
    print(f"Total Policies: {result.get('total_policies', 0)}")
    print(f"Mapped Policies: {result.get('mapped_policies', 0)}")
    print(f"Average Confidence: {result.get('average_confidence', 0.0):.2f}")
    
    if 'enriched_data' in result:
        print(f"\n📋 Policy Details:")
        print("-" * 120)
        print(f"{'Policy ID':<10} {'Policy Name':<30} {'MITRE Tech':<15} {'MITRE Tactic':<20} {'Confidence':<10}")
        print("-" * 120)
        
        for policy in result['enriched_data']:
            policy_id = policy.get('policy_id', 'Unknown')[:9]
            policy_name = policy.get('policy_name', 'Unknown')[:29]
            mitre_tech = f"{policy.get('mitre_technique_id', 'Unknown')} - {policy.get('mitre_technique_name', 'Unknown')[:10]}"
            mitre_tactic = f"{policy.get('mitre_tactic_id', 'Unknown')} - {policy.get('mitre_tactic_name', 'Unknown')[:17]}"
            confidence = f"{policy.get('mitre_confidence_score', 0.0):.2f}"
            
            print(f"{policy_id:<10} {policy_name:<30} {mitre_tech:<15} {mitre_tactic:<20} {confidence:<10}")
        
        print("-" * 120)

if __name__ == "__main__":
    main()
