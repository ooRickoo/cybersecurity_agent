#!/usr/bin/env python3
"""
Workflow Verification Integration Example
Shows how to integrate the verification system with the cybersecurity agent
"""

import asyncio
import time
from typing import Dict, Any, List

# Add bin directory to path for imports
import sys
from pathlib import Path
bin_path = Path(__file__).parent
if str(bin_path) not in sys.path:
    sys.path.insert(0, str(bin_path))

try:
    from langgraph_cybersecurity_agent import LangGraphCybersecurityAgent
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False

def example_threat_analysis_workflow():
    """Example of a threat analysis workflow with verification."""
    print("🔍 **Example: Threat Analysis Workflow with Verification**\n")
    
    if not AGENT_AVAILABLE:
        print("❌ Agent not available - please check your installation")
        return
    
    # Initialize the agent
    agent = LangGraphCybersecurityAgent()
    
    # Simulate a threat analysis workflow
    print("📋 **Step 1: Threat Intelligence Gathering**")
    threat_intel_step = agent.track_workflow_step(
        step_type="data_collection",
        description="Gather threat intelligence from multiple sources",
        tools_used=["threat_intel_tool", "osint_tools", "dark_web_monitoring"],
        inputs={"sources": ["threat_feeds", "osint", "dark_web"]},
        outputs={"threat_indicators": ["malware_hashes", "ip_addresses", "domains"]},
        execution_time=5.2
    )
    
    print("📊 **Step 2: Threat Pattern Analysis**")
    pattern_analysis_step = agent.track_workflow_step(
        step_type="analysis",
        description="Analyze threat patterns and behaviors",
        tools_used=["malware_analysis", "behavioral_analysis", "pattern_recognition"],
        inputs={"threat_data": "malware_indicators"},
        outputs={"patterns": ["ransomware_family", "attack_vectors", "target_industries"]},
        execution_time=8.7
    )
    
    print("🎯 **Step 3: Risk Assessment**")
    risk_assessment_step = agent.track_workflow_step(
        step_type="assessment",
        description="Assess threat level and potential impact",
        tools_used=["risk_assessment_tool", "threat_modeling"],
        inputs={"threat_patterns": "analyzed_patterns"},
        outputs={"risk_score": 8.5, "impact_assessment": "high", "mitigation_priority": "critical"},
        execution_time=6.3
    )
    
    print("📝 **Step 4: Mitigation Planning**")
    mitigation_step = agent.track_workflow_step(
        step_type="planning",
        description="Develop mitigation strategies and response plan",
        tools_used=["planning_tools", "documentation", "stakeholder_management"],
        inputs={"risk_assessment": "risk_results"},
        outputs={"mitigation_plan": "comprehensive_response_strategy", "timeline": "immediate_action_required"},
        execution_time=4.8
    )
    
    # Create agent state with workflow steps
    from langgraph_cybersecurity_agent import AgentState
    
    state = AgentState(
        messages=[{"role": "user", "content": "What is the current threat landscape and how should we respond?"}],
        session_id="example_session_001",
        workflow_steps=[threat_intel_step, pattern_analysis_step, risk_assessment_step, mitigation_step],
        original_question="What is the current threat landscape and how should we respond?",
        final_answer="Based on comprehensive threat intelligence analysis, we've identified a high-risk ransomware campaign targeting financial institutions. The campaign uses sophisticated techniques and has been observed in multiple regions. Immediate action is required including network segmentation, enhanced monitoring, and employee training. Risk score: 8.5/10 with critical priority."
    )
    
    print("\n🔄 **Executing Workflow Verification...**")
    
    # Execute verification
    try:
        # Run the verification node
        updated_state = agent._workflow_verification_node(state)
        
        print("\n✅ **Verification Complete!**")
        print(f"Execution ID: {updated_state.execution_id}")
        print(f"Accuracy Score: {updated_state.verification_result.get('accuracy_score', 'N/A')}")
        print(f"Confidence Level: {updated_state.verification_result.get('confidence_level', 'N/A')}")
        print(f"Needs Backtrack: {updated_state.needs_backtrack}")
        
        if updated_state.needs_backtrack:
            print("\n⚠️ **Backtracking Required**")
            print(f"Alternative Template: {updated_state.backtrack_result.get('template_name', 'N/A')}")
            print(f"Confidence: {updated_state.backtrack_result.get('confidence', 'N/A')}")
        
        # Show verification summary
        if updated_state.verification_result:
            print("\n📊 **Verification Details:**")
            if updated_state.verification_result.get('issues_found'):
                print("Issues Found:")
                for issue in updated_state.verification_result['issues_found']:
                    print(f"  • {issue}")
            
            if updated_state.verification_result.get('recommendations'):
                print("Recommendations:")
                for rec in updated_state.verification_result['recommendations']:
                    print(f"  • {rec}")
        
        # Show final messages
        print("\n💬 **Final Messages:**")
        for msg in updated_state.messages:
            if msg.get('role') == 'assistant' and 'verification' in msg.get('content', '').lower():
                print(msg['content'])
        
    except Exception as e:
        print(f"❌ **Verification Error**: {e}")
    
    return updated_state

def example_incident_response_workflow():
    """Example of an incident response workflow with verification."""
    print("\n🚨 **Example: Incident Response Workflow with Verification**\n")
    
    if not AGENT_AVAILABLE:
        print("❌ Agent not available - please check your installation")
        return
    
    # Initialize the agent
    agent = LangGraphCybersecurityAgent()
    
    # Simulate an incident response workflow
    print("🚨 **Step 1: Incident Detection**")
    detection_step = agent.track_workflow_step(
        step_type="detection",
        description="Detect and classify security incident",
        tools_used=["siem", "alerting_system", "monitoring_tools"],
        inputs={"alert_source": "siem", "alert_type": "suspicious_activity"},
        outputs={"incident_classified": True, "severity": "high", "type": "data_breach"},
        execution_time=2.1
    )
    
    print("🔍 **Step 2: Initial Assessment**")
    assessment_step = agent.track_workflow_step(
        step_type="assessment",
        description="Perform initial incident assessment",
        tools_used=["forensic_tools", "log_analysis", "network_analysis"],
        inputs={"incident_data": "detection_results"},
        outputs={"scope_defined": True, "impact_assessed": True, "affected_systems": ["web_server", "database"]},
        execution_time=6.5
    )
    
    print("🛡️ **Step 3: Containment**")
    containment_step = agent.track_workflow_step(
        step_type="response",
        description="Contain the incident and prevent further damage",
        tools_used=["network_isolation", "access_control", "system_quarantine"],
        inputs={"affected_systems": "identified_systems"},
        outputs={"incident_contained": True, "further_damage_prevented": True},
        execution_time=4.2
    )
    
    print("🧹 **Step 4: Eradication**")
    eradication_step = agent.track_workflow_step(
        step_type="response",
        description="Remove threat and restore systems",
        tools_used=["malware_removal", "system_restoration", "security_patching"],
        inputs={"contained_systems": "isolated_systems"},
        outputs={"threat_removed": True, "systems_restored": True, "patches_applied": True},
        execution_time=8.9
    )
    
    # Create agent state
    from langgraph_cybersecurity_agent import AgentState
    
    state = AgentState(
        messages=[{"role": "user", "content": "How should we respond to this data breach?"}],
        session_id="example_session_002",
        workflow_steps=[detection_step, assessment_step, containment_step, eradication_step],
        original_question="How should we respond to this data breach?",
        final_answer="We've successfully contained and eradicated the data breach incident. The threat has been removed from all affected systems, security patches have been applied, and systems have been restored. The incident was contained within 4 hours, preventing further data loss. All affected systems (web server and database) have been secured and are operational."
    )
    
    print("\n🔄 **Executing Workflow Verification...**")
    
    # Execute verification
    try:
        updated_state = agent._workflow_verification_node(state)
        
        print("\n✅ **Verification Complete!**")
        print(f"Execution ID: {updated_state.execution_id}")
        print(f"Accuracy Score: {updated_state.verification_result.get('accuracy_score', 'N/A')}")
        print(f"Confidence Level: {updated_state.verification_result.get('confidence_level', 'N/A')}")
        
        # Show verification summary
        if updated_state.verification_result:
            print("\n📊 **Verification Details:**")
            if updated_state.verification_result.get('issues_found'):
                print("Issues Found:")
                for issue in updated_state.verification_result['issues_found']:
                    print(f"  • {issue}")
            
            if updated_state.verification_result.get('recommendations'):
                print("Recommendations:")
                for rec in updated_state.verification_result['recommendations']:
                    print(f"  • {rec}")
        
    except Exception as e:
        print(f"❌ **Verification Error**: {e}")
    
    return updated_state

def example_template_selection():
    """Example of workflow template selection."""
    print("\n📋 **Example: Workflow Template Selection**\n")
    
    if not AGENT_AVAILABLE:
        print("❌ Agent not available - please check your installation")
        return
    
    # Initialize the agent
    agent = LangGraphCybersecurityAgent()
    
    # Test different question types
    questions = [
        "What is the current threat landscape?",
        "How should we respond to this security incident?",
        "What vulnerabilities exist in our web applications?",
        "Are we compliant with industry security standards?",
        "What's the weather like today?"
    ]
    
    for question in questions:
        print(f"🔍 **Question**: {question}")
        
        try:
            template_result = agent.select_workflow_template(question)
            
            if template_result.get("success"):
                template = template_result["template"]
                print(f"✅ **Selected Template**: {template['name']}")
                print(f"   Type: {template['template_type']}")
                print(f"   Success Rate: {template['success_rate']:.1%}")
                print(f"   Steps: {len(template['steps'])}")
            else:
                print(f"❌ **No Template Found**: {template_result.get('error', 'Unknown error')}")
        
        except Exception as e:
            print(f"❌ **Error**: {e}")
        
        print()

def example_verification_summary():
    """Example of getting verification summary."""
    print("\n📊 **Example: Verification Summary**\n")
    
    if not AGENT_AVAILABLE:
        print("❌ Agent not available - please check your installation")
        return
    
    # Initialize the agent
    agent = LangGraphCybersecurityAgent()
    
    # Get execution history
    try:
        history_result = agent.get_execution_history(limit=5)
        
        if history_result.get("success"):
            history = history_result["history"]
            print(f"📚 **Recent Executions**: {len(history)} found")
            
            for i, execution in enumerate(history, 1):
                print(f"\n{i}. **Execution**: {execution['execution_id'][:8]}...")
                print(f"   Question: {execution['question']}")
                print(f"   Status: {execution['status']}")
                print(f"   Accuracy: {execution['accuracy']:.2f}")
                print(f"   Steps: {execution['steps']}")
                print(f"   Completed: {execution['completed_at'][:10]}")
        else:
            print(f"❌ **Error**: {history_result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ **Error**: {e}")

async def main():
    """Main function to run all examples."""
    print("🚀 **Workflow Verification Integration Examples**\n")
    print("This demonstrates how the verification system integrates with the cybersecurity agent.\n")
    
    # Run examples
    example_threat_analysis_workflow()
    example_incident_response_workflow()
    example_template_selection()
    example_verification_summary()
    
    print("\n🎉 **Examples Complete!**")
    print("\nThe verification system is now integrated with your agent and will:")
    print("✅ Automatically verify every workflow execution")
    print("✅ Detect accuracy issues and provide recommendations")
    print("✅ Prevent loops by tracking execution paths")
    print("✅ Suggest alternatives when workflows fail verification")
    print("✅ Learn and improve template performance over time")

if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
