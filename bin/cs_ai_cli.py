#!/usr/bin/env python3
"""
CS AI CLI - Dynamic Agentic Workflow Tool with Advanced Workflow System
Integrates with MCP server for self-describing tools and dynamic agent workflows.
Now includes advanced workflow capabilities based on Google ADK framework.
Enhanced with agentic workflow system for automated and manual execution paths.
"""

import sys
import os
import json
import asyncio
import argparse
from pathlib import Path
from typing import Optional, List

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class CSAgentCLI:
    """Enhanced CLI with MCP server integration and advanced workflow capabilities."""
    
    def __init__(self):
        self.tool_manager = None
        self.mcp_server = None
        self.advanced_workflow_cli = None
        self.agentic_workflow_cli = None
        self.dynamic_workflow_generator = None
        self.session_logger = None
        
        try:
            # Try to import the main tools first
            sys.path.append("bin")
            from bin.cs_ai_tools import tool_manager
            self.tool_manager = tool_manager
            self.mcp_server = tool_manager.mcp_server
            
            # Initialize session logger
            if hasattr(self.tool_manager, 'session_logger'):
                self.session_logger = self.tool_manager.session_logger
                print("📝 Session logging initialized")
            
            # Trigger tool discovery to ensure all tools are registered
            if self.mcp_server:
                self.mcp_server.discover_available_tools()
            
            # Explicitly access PCAP tools to trigger their registration
            if hasattr(self.tool_manager, 'pcap_analysis_tools'):
                _ = self.tool_manager.pcap_analysis_tools
            
            print("🚀 CS AI CLI initialized with MCP server integration")
        except ImportError as e:
            print(f"⚠️  Main tools not available: {e}")
        
        try:
            # Try to import advanced workflow system
            sys.path.append("bin")
            from workflow_templates import WorkflowTemplateLibrary
            self.advanced_workflow_cli = WorkflowTemplateLibrary()
            print("🚀 Advanced workflow system initialized")
        except ImportError as e:
            print(f"⚠️  Advanced workflow system not available: {e}")
        
        try:
            # Try to import agentic workflow system
            sys.path.append("bin")
            from agentic_workflow_system import AgenticWorkflowSystem
            self.agentic_workflow_cli = AgenticWorkflowSystem(".")
            print("🚀 Agentic workflow system initialized")
        except ImportError as e:
            print(f"⚠️  Agentic workflow system not available: {e}")
        
        try:
            # Try to import dynamic workflow generator
            sys.path.append("bin")
            from dynamic_workflow_generator import DynamicWorkflowGenerator
            self.dynamic_workflow_generator = DynamicWorkflowGenerator()
            print("🚀 Dynamic workflow generator initialized")
        except ImportError as e:
            print(f"⚠️  Dynamic workflow generator not available: {e}")
        
        if not self.tool_manager and not self.advanced_workflow_cli and not self.agentic_workflow_cli:
            print("❌ No tools available. Please check your installation.")
            sys.exit(1)
    
    def list_tools(self, category=None, tags=None, detailed=False):
        """List available MCP tools with filtering options."""
        if not self.mcp_server:
            print("❌ MCP server not available")
            return
            
        try:
            tools = self.mcp_server._list_tools_handler(
                category=category, 
                tags=tags, 
                detailed=detailed
            )
            
            if tools['success']:
                print(f"\n🔧 Available Tools ({tools['total_count']} total):")
                print("=" * 80)
                
                # Group tools by category
                tools_by_category = {}
                for tool_name, tool_info in tools['tools'].items():
                    cat = tool_info['category']
                    if cat not in tools_by_category:
                        tools_by_category[cat] = []
                    tools_by_category[cat].append((tool_name, tool_info))
                
                for category, category_tools in tools_by_category.items():
                    print(f"\n📂 {category.upper()} ({len(category_tools)} tools):")
                    print("-" * 40)
                    
                    for tool_name, tool_info in category_tools:
                        print(f"  📌 {tool_name}")
                        print(f"     Description: {tool_info['description']}")
                        print(f"     Tags: {', '.join(tool_info['tags'])}")
                        
                        if detailed and 'inputSchema' in tool_info:
                            print(f"     Schema: {json.dumps(tool_info['inputSchema'], indent=6)}")
                        print()
            else:
                print(f"❌ Error listing tools: {tools['error']}")
                
        except Exception as e:
            print(f"❌ Error in list_tools: {e}")
    
    def list_advanced_workflows(self):
        """List available advanced workflow templates."""
        if not self.advanced_workflow_cli:
            print("❌ Advanced workflow system not available")
            return
            
        try:
            templates = self.advanced_workflow_cli.list_workflow_templates()
            print(f"\n📋 Available Workflow Templates ({len(templates)} total):")
            print("=" * 80)
            
            for template_id, template_info in templates.items():
                print(f"🔹 {template_id}")
                print(f"   Strategy: {template_info['strategy']}")
                print(f"   Nodes: {template_info['nodes_count']}")
                print(f"   Description: {template_info['description']}")
                print()
                
        except Exception as e:
            print(f"❌ Error listing workflow templates: {e}")
    
    def list_advanced_mcp_tools(self, category=None):
        """List MCP tools from the advanced workflow system."""
        if not self.advanced_workflow_cli:
            print("❌ Advanced workflow system not available")
            return
            
        try:
            from mcp_integration_layer import MCPToolCategory
            
            if category:
                cat_enum = MCPToolCategory(category)
                tools = self.advanced_workflow_cli.list_mcp_tools(cat_enum)
            else:
                tools = self.advanced_workflow_cli.list_mcp_tools()
            
            print(f"\n🔧 Advanced MCP Tools ({len(tools)} total):")
            print("=" * 80)
            
            for tool_id, tool_info in tools.items():
                print(f"🔹 {tool_id}")
                print(f"   Name: {tool_info['name']}")
                print(f"   Description: {tool_info['description']}")
                print(f"   Category: {tool_info['category']}")
                print(f"   Capabilities: {', '.join(tool_info['capabilities'])}")
                print(f"   Usage: {tool_info['usage_count']} times")
                print(f"   Success Rate: {tool_info['success_rate']:.2%}")
                print()
                
        except Exception as e:
            print(f"❌ Error listing advanced MCP tools: {e}")
    
    def get_tool_schema(self, tool_name):
        """Get detailed schema for a specific tool."""
        if not self.mcp_server:
            print("❌ MCP server not available")
            return
            
        try:
            schema = self.mcp_server._get_tool_schema_handler(tool_name=tool_name)
            
            if schema['success']:
                tool = schema['tool']
                print(f"\n📋 Tool Schema: {tool['name']}")
                print("=" * 80)
                print(f"Description: {tool['description']}")
                print(f"Category: {tool['category']}")
                print(f"Tags: {', '.join(tool['tags'])}")
                print(f"\nInput Schema:")
                print(json.dumps(tool['inputSchema'], indent=2))
            else:
                print(f"❌ Error getting schema: {schema['error']}")
                
        except Exception as e:
            print(f"❌ Error in get_tool_schema: {e}")
    
    def get_server_info(self, detailed=False):
        """Get MCP server information and capabilities."""
        if not self.mcp_server:
            print("❌ MCP server not available")
            return
            
        try:
            info = self.mcp_server._get_server_info_handler(detailed=detailed)
            
            if info['success']:
                server_info = info['server_info']
                print(f"\nℹ️  MCP Server Information")
                print("=" * 80)
                print(f"Server: {server_info['server_name']}")
                print(f"Version: {server_info['version']}")
                print(f"Tool Count: {server_info['tool_count']}")
                print(f"Categories: {', '.join(server_info['categories'])}")
                print(f"Capabilities: {', '.join(server_info['capabilities'])}")
                
                if detailed:
                    print(f"\n📊 Detailed Tool Information:")
                    for tool_name, tool_details in server_info['tools'].items():
                        print(f"  {tool_name}: {tool_details['category']} - {', '.join(tool_details['tags'])}")
            else:
                print(f"❌ Error getting server info: {info['error']}")
                
        except Exception as e:
            print(f"❌ Error in get_server_info: {e}")
    
    async def execute_tool(self, tool_name, arguments_json):
        """Execute an MCP tool with the given arguments."""
        if not self.mcp_server:
            print("❌ MCP server not available")
            return
            
        try:
            # Parse arguments JSON
            if arguments_json:
                arguments = json.loads(arguments_json)
            else:
                arguments = {}
            
            print(f"\n🚀 Executing tool: {tool_name}")
            print(f"Arguments: {json.dumps(arguments, indent=2)}")
            
            result = await self.mcp_server._call_tool_handler(
                name=tool_name, 
                arguments=arguments
            )
            
            if result['success']:
                print(f"✅ Tool execution successful")
                print(f"Tool: {result['tool']}")
                print(f"Execution time: {result.get('execution_time', 'N/A')}")
                print(f"\nResult:")
                print(json.dumps(result['result'], indent=2))
            else:
                print(f"❌ Tool execution failed: {result['error']}")
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON arguments: {e}")
        except Exception as e:
            print(f"❌ Error executing tool: {e}")
    
    async def execute_advanced_workflow(self, workflow_type, problem_description, priority=1, complexity=1):
        """Execute an advanced workflow using templates."""
        if not self.advanced_workflow_cli:
            print("❌ Advanced workflow system not available")
            return
            
        try:
            from workflow_templates import ProblemType
            
            # Map workflow type to ProblemType enum
            problem_type_map = {
                'threat_hunting': ProblemType.THREAT_HUNTING,
                'incident_response': ProblemType.INCIDENT_RESPONSE,
                'compliance': ProblemType.COMPLIANCE,
                'risk_assessment': ProblemType.RISK_ASSESSMENT,
                'investigation': ProblemType.INVESTIGATION,
                'analysis': ProblemType.ANALYSIS
            }
            
            if workflow_type not in problem_type_map:
                print(f"❌ Unknown workflow type: {workflow_type}")
                print(f"Available types: {', '.join(problem_type_map.keys())}")
                return
            
            problem_type = problem_type_map[workflow_type]
            context = {'priority': priority, 'complexity': complexity}
            
            print(f"\n🚀 Executing {workflow_type} workflow")
            print(f"Problem: {problem_description}")
            print(f"Priority: {priority}, Complexity: {complexity}")
            
            result = await self.advanced_workflow_cli.execute_template_workflow(
                problem_type, problem_description, context
            )
            
            print(f"\n📊 Workflow Result:")
            print(json.dumps(result, indent=2, default=str))
            
            return result
            
        except Exception as e:
            print(f"❌ Error executing advanced workflow: {e}")
    
    async def execute_advanced_mcp_workflow(self, problem_description, tools, priority=1, complexity=1):
        """Execute a workflow using specific MCP tools from the advanced system."""
        if not self.advanced_workflow_cli:
            print("❌ Advanced workflow system not available")
            return
            
        try:
            required_tools = [tool.strip() for tool in tools.split(",")]
            context = {'priority': priority, 'complexity': complexity}
            
            print(f"\n🔧 Executing MCP workflow")
            print(f"Problem: {problem_description}")
            print(f"Tools: {', '.join(required_tools)}")
            
            result = await self.advanced_workflow_cli.execute_with_mcp_tools(
                problem_description, required_tools, context
            )
            
            print(f"\n📊 MCP Workflow Result:")
            print(json.dumps(result, indent=2, default=str))
            
            return result
            
        except Exception as e:
            print(f"❌ Error executing MCP workflow: {e}")
    
    # New Agentic Workflow Methods
    
    async def execute_automated_workflow(self, csv_file_path, problem_description, output_file, priority=1, complexity=1):
        """Execute automated workflow for CSV processing."""
        if not self.agentic_workflow_cli:
            print("❌ Agentic workflow system not available")
            return
            
        try:
            result = await self.agentic_workflow_cli.execute_automated_workflow(
                csv_file_path, problem_description, output_file, priority, complexity
            )
            
            if result.get('success'):
                print(f"\n📊 Automated Workflow Result:")
                print(json.dumps(result, indent=2, default=str))
            else:
                print(f"\n❌ Workflow failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error executing automated workflow: {e}")
    
    async def execute_manual_workflow(self, problem_description, priority=1, complexity=1, interactive=False):
        """Execute manual workflow for interactive problem solving."""
        if not self.agentic_workflow_cli:
            print("❌ Agentic workflow system not available")
            return
            
        try:
            result = await self.agentic_workflow_cli.execute_manual_workflow(
                problem_description, priority, complexity, interactive
            )
            
            if result.get('success'):
                print(f"\n📊 Manual Workflow Result:")
                print(json.dumps(result, indent=2, default=str))
            else:
                print(f"\n❌ Workflow failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error executing manual workflow: {e}")
    
    async def execute_hybrid_workflow(self, csv_file_path, problem_description, output_file, priority=1, complexity=1):
        """Execute hybrid workflow combining automated and manual approaches."""
        if not self.agentic_workflow_cli:
            print("❌ Agentic workflow system not available")
            return
            
        try:
            result = await self.agentic_workflow_cli.execute_hybrid_workflow(
                csv_file_path, problem_description, output_file, priority, complexity
            )
            
            if result.get('success'):
                print(f"\n📊 Hybrid Workflow Result:")
                print(json.dumps(result, indent=2, default=str))
            else:
                print(f"\n❌ Workflow failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error executing hybrid workflow: {e}")
    
    async def execute_csv_enrichment_workflow(self, input_csv, output_csv, enrichment_prompt, 
                                           batch_size=100, max_retries=3, quality_threshold=0.8):
        """Execute CSV enrichment workflow with LLM processing."""
        try:
            print(f"🚀 Starting CSV Enrichment Workflow")
            print(f"   Input: {input_csv}")
            print(f"   Output: {output_csv}")
            print(f"   Prompt: {enrichment_prompt}")
            print(f"   Batch size: {batch_size}")
            print(f"   Max retries: {max_retries}")
            print(f"   Quality threshold: {quality_threshold}")
            
            # Import CSV enrichment executor
            sys.path.append("bin")
            from bin.csv_enrichment_executor import CSVEnrichmentExecutor
            
            # Create executor
            executor = CSVEnrichmentExecutor()
            
            # Prepare context
            context = {
                'input_csv_path': input_csv,
                'output_csv_path': output_csv,
                'enrichment_prompt': enrichment_prompt,
                'batch_size': batch_size,
                'max_retries': max_retries,
                'quality_threshold': quality_threshold
            }
            
            # Execute workflow nodes
            print(f"\n📥 Step 1: Importing CSV...")
            import_result = await executor.execute_csv_import(context)
            if not import_result['success']:
                print(f"❌ CSV import failed: {import_result['error']}")
                return import_result
            
            print(f"✅ Imported {import_result['rows_imported']} rows with {len(import_result['columns_imported'])} columns")
            
            print(f"\n🔍 Step 2: Analyzing columns...")
            analysis_result = await executor.execute_column_analysis(context)
            if not analysis_result['success']:
                print(f"❌ Column analysis failed: {analysis_result['error']}")
                return analysis_result
            
            print(f"✅ Identified {len(analysis_result['new_columns'])} new columns: {analysis_result['new_columns']}")
            
            print(f"\n🏗️ Step 3: Creating new columns...")
            creation_result = await executor.execute_column_creation(context)
            if not creation_result['success']:
                print(f"❌ Column creation failed: {creation_result['error']}")
                return creation_result
            
            print(f"✅ Created {len(creation_result['columns_created'])} new columns")
            
            print(f"\n🤖 Step 4: Processing with LLM...")
            processing_result = await executor.execute_llm_processing(context)
            if not processing_result['success']:
                print(f"❌ LLM processing failed: {processing_result['error']}")
                return processing_result
            
            print(f"✅ Processed {processing_result['rows_processed']} rows, enriched {processing_result['rows_enriched']} rows")
            print(f"   Quality score: {processing_result['quality_score']:.2%}")
            
            print(f"\n🔍 Step 5: Validating data...")
            validation_result = await executor.execute_data_validation(context)
            if not validation_result['success']:
                print(f"❌ Data validation failed: {validation_result['error']}")
                return validation_result
            
            print(f"✅ Validation completed with quality score: {validation_result['quality_score']:.2%}")
            
            print(f"\n💾 Step 6: Exporting results...")
            export_result = await executor.execute_export_results(context)
            if not export_result['success']:
                print(f"❌ Export failed: {export_result['error']}")
                return export_result
            
            print(f"✅ Exported {export_result['rows_exported']} rows to {export_result['output_path']}")
            
            # Summary
            print(f"\n🎉 CSV Enrichment Workflow Completed Successfully!")
            print(f"   Total rows processed: {processing_result['rows_processed']}")
            print(f"   Rows enriched: {processing_result['rows_enriched']}")
            print(f"   Final quality score: {validation_result['quality_score']:.2%}")
            print(f"   Output file: {export_result['output_path']}")
            
            return {
                'success': True,
                'import_result': import_result,
                'analysis_result': analysis_result,
                'creation_result': creation_result,
                'processing_result': processing_result,
                'validation_result': validation_result,
                'export_result': export_result
            }
            
        except Exception as e:
            print(f"❌ Error executing CSV enrichment workflow: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    async def run_interactive_mode(self):
        """Run interactive cybersecurity AI helper mode."""
        try:
            # Log session start
            if self.session_logger:
                self.session_logger.log_info("session_start", "Interactive cybersecurity AI helper session started")
            
            # Display welcome message
            self._display_welcome()
            
            # Main interactive loop
            while True:
                try:
                    # Display menu
                    self._display_menu()
                    
                    # Get user input
                    user_input = input("\n🔍 What would you like to do? (or type 'help' for options): ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Log user input
                    if self.session_logger:
                        self.session_logger.log_agent_question(
                            question=user_input,
                            context="interactive_mode",
                            metadata={"input_type": "menu_selection"}
                        )
                    
                    # Process user input
                    if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                        if self.session_logger:
                            self.session_logger.log_info("session_end", "User requested session termination")
                        self._display_goodbye()
                        break
                    elif user_input.lower() in ['help', 'h', '?']:
                        self._display_help()
                        if self.session_logger:
                            self.session_logger.log_info("help_requested", "User requested help")
                    elif user_input.lower() in ['menu', 'm']:
                        self._display_menu()
                    elif user_input.lower() in ['tools', 't']:
                        self._show_available_tools()
                        if self.session_logger:
                            self.session_logger.log_info("tools_listed", "User requested tools list")
                    elif user_input.lower() in ['workflows', 'w']:
                        self._show_available_workflows()
                        if self.session_logger:
                            self.session_logger.log_info("workflows_listed", "User requested workflows list")
                    elif user_input.lower() in ['status', 's']:
                        self._show_system_status()
                        if self.session_logger:
                            self.session_logger.log_info("status_requested", "User requested system status")
                    elif user_input.lower().startswith('csv-enrich'):
                        await self._handle_csv_enrichment_interactive(user_input)
                    elif user_input.lower().startswith('analyze'):
                        await self._handle_analysis_request(user_input)
                    elif user_input.lower().startswith('categorize'):
                        await self._handle_categorization_request(user_input)
                    elif user_input.lower().startswith('summarize'):
                        await self._handle_summarization_request(user_input)
                    elif user_input.lower() in ['clear', 'cls']:
                        import os
                        os.system('clear' if os.name == 'posix' else 'cls')
                        if self.session_logger:
                            self.session_logger.log_info("screen_cleared", "User cleared screen")
                    else:
                        # Try to handle as a general query
                        await self._handle_general_query(user_input)
                        
                except KeyboardInterrupt:
                    print("\n\n⚠️  Interrupted by user")
                    if self.session_logger:
                        self.session_logger.log_info("user_interruption", "User interrupted with Ctrl+C")
                    continue
                except EOFError:
                    print("\n\n👋 Goodbye!")
                    if self.session_logger:
                        self.session_logger.log_info("session_end", "Session ended due to EOF")
                    break
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    if self.session_logger:
                        self.session_logger.log_error("interactive_mode_error", str(e), metadata={"error_type": "user_input_processing"})
                    continue
                    
        except Exception as e:
            print(f"\n❌ Fatal error in interactive mode: {e}")
            if self.session_logger:
                self.session_logger.log_error("fatal_error", str(e), metadata={"error_type": "interactive_mode", "severity": "critical"})
            import traceback
            traceback.print_exc()
        finally:
            # Log session end
            if self.session_logger:
                self.session_logger.log_info("session_end", "Interactive session ended")
                # Get session summary
                try:
                    summary = self.session_logger.get_session_summary()
                    self.session_logger.end_session(summary)
                except Exception as summary_error:
                    print(f"⚠️  Could not save session summary: {summary_error}")
    
    def _display_welcome(self):
        """Display a random welcome message."""
        import random
        
        welcome_messages = [
            "🛡️  Welcome to the Cybersecurity AI Helper! Ready to defend the digital realm?",
            "🚀 Greetings, cyber warrior! Your AI assistant is online and ready for action.",
            "🔒 Hello there! Time to secure, analyze, and protect. What's on your mind?",
            "⚡ Welcome to the command center! Let's make cybersecurity simple and effective.",
            "🕵️  Greetings, digital detective! Ready to investigate and secure?",
            "🛡️  Hello, security specialist! Your AI partner is here to help.",
            "🚀 Welcome aboard! Let's navigate the cybersecurity landscape together.",
            "🔒 Greetings! Time to turn complex security challenges into simple solutions.",
            "⚡ Hello there! Ready to accelerate your security operations?",
            "🕵️  Welcome, cyber investigator! Let's solve security mysteries together."
        ]
        
        selected_message = random.choice(welcome_messages)
        print(f"\n{selected_message}")
        print("💡 Type 'help' for available options or 'menu' to see the main menu.")
    
    def _display_goodbye(self):
        """Display a random goodbye message."""
        import random
        
        goodbye_messages = [
            "🛡️  Stay secure out there! Until next time, cyber warrior.",
            "🚀 Mission accomplished! Keep defending the digital frontier.",
            "🔒 Stay vigilant and secure! Your AI assistant will be here when you return.",
            "⚡ Powering down... but your security knowledge remains active!",
            "🕵️  Case closed for now! Stay sharp and stay safe.",
            "🛡️  Logging out... but the security never stops!",
            "🚀 Shutting down systems... until our next security mission!",
            "🔒 Disconnecting... but your security awareness stays connected!",
            "⚡ Powering off... but your cyber defenses remain online!",
            "🕵️  Signing off... but the investigation never truly ends!"
        ]
        
        selected_message = random.choice(goodbye_messages)
        print(f"\n{selected_message}")
        print("👋 Goodbye!")
    
    def _display_menu(self):
        """Display the main interactive menu."""
        print("\n" + "="*60)
        print("🛡️  CYBERSECURITY AI HELPER - MAIN MENU")
        print("="*60)
        print("📋 Available Options:")
        print("  🔧 tools          - Show available MCP tools")
        print("  🔄 workflows      - Show available workflows")
        print("  📊 status         - Show system status")
        print("  📈 csv-enrich     - CSV enrichment workflow")
        print("  🔍 analyze        - Analyze data or text")
        print("  🏷️  categorize     - Categorize data or text")
        print("  📝 summarize      - Summarize data or text")
        print("  ❓ help           - Show detailed help")
        print("  🧹 clear          - Clear screen")
        print("  🚪 quit/exit      - Exit interactive mode")
        print("="*60)
        print("💡 You can also ask questions directly!")
        print("   Example: 'Analyze this log file for threats'")
    
    def _display_help(self):
        """Display detailed help information."""
        print("\n" + "="*60)
        print("🛡️  CYBERSECURITY AI HELPER - HELP")
        print("="*60)
        print("📖 How to use the interactive mode:")
        print("\n🔧 Tool Commands:")
        print("  tools              - List all available MCP tools")
        print("  workflows          - Show available workflow templates")
        print("  status             - Display system health and status")
        
        print("\n📊 Workflow Commands:")
        print("  csv-enrich         - Start CSV enrichment workflow")
        print("  analyze [query]    - Analyze data using local tools")
        print("  categorize [query] - Categorize data using local tools")
        print("  summarize [query]  - Summarize data using local tools")
        
        print("\n💬 General Usage:")
        print("  • Ask questions directly: 'What tools can analyze network traffic?'")
        print("  • Use natural language: 'Help me understand this log file'")
        print("  • Request workflows: 'I need to enrich some threat data'")
        
        print("\n⚡ Smart Features:")
        print("  • Local tools used when possible for speed")
        print("  • LLM tasks only when complex analysis needed")
        print("  • Dynamic workflow generation for complex problems")
        print("  • Intelligent tool selection based on your needs")
        
        print("\n🔍 Examples:")
        print("  'Show me tools for network analysis'")
        print("  'Analyze this CSV for security threats'")
        print("  'Categorize these log entries by severity'")
        print("  'Summarize the security findings'")
        print("="*60)
    
    def _show_available_tools(self):
        """Show available MCP tools in an interactive format."""
        print("\n🔧 Available MCP Tools:")
        print("-" * 40)
        
        try:
            tools = self.mcp_server.get_dynamic_tools() if self.mcp_server else {}
            tool_list = tools.get('tools', {})
            
            if not tool_list:
                print("❌ No tools available. Try running 'discover-tools' first.")
                return
            
            # Group tools by category
            categories = {}
            for tool_name, tool_info in tool_list.items():
                category = tool_info.get('category', 'uncategorized')
                if category not in categories:
                    categories[category] = []
                categories[category].append((tool_name, tool_info.get('description', 'No description')))
            
            # Display tools by category
            for category, tools_in_category in categories.items():
                print(f"\n📁 {category.upper()}:")
                for tool_name, description in tools_in_category:
                    print(f"  • {tool_name}: {description}")
                    
        except Exception as e:
            print(f"❌ Error showing tools: {e}")
    
    def _show_available_workflows(self):
        """Show available workflow templates."""
        print("\n🔄 Available Workflows:")
        print("-" * 40)
        
        try:
            if hasattr(self, 'agentic_workflow_cli') and self.agentic_workflow_cli:
                print("✅ Agentic Workflow System: Available")
                print("  • Automated CSV processing")
                print("  • Manual problem solving")
                print("  • Hybrid workflow execution")
            else:
                print("❌ Agentic Workflow System: Not available")
            
            print("\n📋 Built-in Workflows:")
            print("  • CSV Enrichment (LLM-based)")
            print("  • Threat Analysis")
            print("  • Security Assessment")
            print("  • Data Classification")
            
        except Exception as e:
            print(f"❌ Error showing workflows: {e}")
    
    def _show_system_status(self):
        """Show system status and health."""
        print("\n📊 System Status:")
        print("-" * 40)
        
        try:
            # MCP Server status
            if self.mcp_server:
                print("✅ MCP Server: Running")
                tools = self.mcp_server.get_dynamic_tools()
                tool_count = len(tools.get('tools', {}))
                print(f"   Tools available: {tool_count}")
            else:
                print("❌ MCP Server: Not available")
            
            # Agentic workflow status
            if hasattr(self, 'agentic_workflow_cli') and self.agentic_workflow_cli:
                print("✅ Agentic Workflow: Available")
            else:
                print("❌ Agentic Workflow: Not available")
            
            # Tool manager status
            if hasattr(self, 'tool_manager') and self.tool_manager:
                print("✅ Tool Manager: Available")
                status = self.tool_manager.get_tool_status()
                for tool, available in status.items():
                    status_icon = "✅" if available else "❌"
                    print(f"   {status_icon} {tool}")
            else:
                print("❌ Tool Manager: Not available")
                
        except Exception as e:
            print(f"❌ Error showing status: {e}")
    
    async def _handle_csv_enrichment_interactive(self, user_input):
        """Handle CSV enrichment requests interactively."""
        try:
            print("\n📊 CSV Enrichment Workflow")
            print("-" * 40)
            
            # Get input file
            input_file = input("📥 Input CSV file path: ").strip()
            if not input_file:
                print("❌ Input file path required")
                return
            
            # Get output file
            output_file = input("📤 Output CSV file path: ").strip()
            if not output_file:
                print("❌ Output file path required")
                return
            
            # Get enrichment prompt
            prompt = input("🤖 Enrichment prompt: ").strip()
            if not prompt:
                print("❌ Enrichment prompt required")
                return
            
            # Get optional parameters
            batch_size_input = input("📦 Batch size (default 100): ").strip()
            batch_size = int(batch_size_input) if batch_size_input.isdigit() else 100
            
            max_retries_input = input("🔄 Max retries (default 3): ").strip()
            max_retries = int(max_retries_input) if max_retries_input.isdigit() else 3
            
            quality_threshold_input = input("🎯 Quality threshold 0.0-1.0 (default 0.8): ").strip()
            quality_threshold = float(quality_threshold_input) if quality_threshold_input else 0.8
            
            print(f"\n🚀 Starting CSV enrichment workflow...")
            print(f"   Input: {input_file}")
            print(f"   Output: {output_file}")
            print(f"   Prompt: {prompt}")
            print(f"   Batch size: {batch_size}")
            print(f"   Max retries: {max_retries}")
            print(f"   Quality threshold: {quality_threshold}")
            
            # Execute workflow
            result = await self.execute_csv_enrichment_workflow(
                input_file, output_file, prompt, batch_size, max_retries, quality_threshold
            )
            
            if result and result.get('success'):
                print("\n✅ CSV enrichment completed successfully!")
            else:
                print(f"\n❌ CSV enrichment failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error in CSV enrichment: {e}")
    
    async def _handle_analysis_request(self, user_input):
        """Handle analysis requests using local tools when possible."""
        try:
            query = user_input.replace('analyze', '').strip()
            if not query:
                query = input("🔍 What would you like to analyze? ").strip()
            
            if not query:
                print("❌ Analysis query required")
                return
            
            print(f"\n🔍 Analyzing: {query}")
            
            # Try to use local tools first
            if self._can_handle_locally(query):
                print("⚡ Using local tools for fast analysis...")
                result = await self._local_analysis(query)
                print(f"✅ Local analysis result: {result}")
            else:
                print("🤖 Using LLM for complex analysis...")
                result = await self._llm_analysis(query)
                print(f"✅ LLM analysis result: {result}")
                
        except Exception as e:
            print(f"❌ Error in analysis: {e}")
    
    async def _handle_categorization_request(self, user_input):
        """Handle categorization requests using local tools when possible."""
        try:
            query = user_input.replace('categorize', '').strip()
            if not query:
                query = input("🔍 What would you like to categorize? ").strip()
            if not query:
                print("❌ Categorization query required")
                return
            
            print(f"\n🏷️  Categorizing: {query}")
            
            # Try to use local tools first
            if self._can_handle_locally(query):
                print("⚡ Using local tools for fast categorization...")
                result = await self._local_categorization(query)
                print(f"✅ Local categorization result: {result}")
            else:
                print("🤖 Using LLM for complex categorization...")
                result = await self._llm_categorization(query)
                print(f"✅ LLM categorization result: {result}")
                
        except Exception as e:
            print(f"❌ Error in categorization: {e}")
    
    async def _handle_summarization_request(self, user_input):
        """Handle summarization requests using local tools when possible."""
        try:
            query = user_input.replace('summarize', '').strip()
            if not query:
                query = input("📝 What would you like to summarize? ").strip()
            
            if not query:
                print("❌ Summarization query required")
                return
            
            print(f"\n📝 Summarizing: {query}")
            
            # Try to use local tools first
            if self._can_handle_locally(query):
                print("⚡ Using local tools for fast summarization...")
                result = await self._local_summarization(query)
                print(f"✅ Local summarization result: {result}")
            else:
                print("🤖 Using LLM for complex summarization...")
                result = await self._llm_summarization(query)
                print(f"✅ LLM summarization result: {result}")
                
        except Exception as e:
            print(f"❌ Error in summarization: {e}")
    
    async def _handle_general_query(self, user_input):
        """Handle general queries intelligently using problem-driven orchestration."""
        try:
            print(f"\n🤔 Processing query: {user_input}")
            
            # Log query processing start
            if self.session_logger:
                self.session_logger.log_info("query_processing_start", f"Processing general query: {user_input[:100]}...")
            
            # Check if this is a file-based analysis request
            if self._is_file_analysis_request(user_input):
                if self.session_logger:
                    self.session_logger.log_info("file_analysis_detected", "File-based analysis request detected")
                await self._handle_file_analysis_request(user_input)
                return
            
            # Try dynamic workflow generation first
            if self.dynamic_workflow_generator:
                print("🔍 Attempting dynamic workflow generation...")
                try:
                    # Extract potential input files and outputs from the query
                    input_files = self._extract_potential_files(user_input)
                    desired_outputs = self._extract_desired_outputs(user_input)
                    
                    # Generate dynamic workflow
                    dynamic_workflow = await self.dynamic_workflow_generator.generate_workflow(
                        problem_description=user_input,
                        input_files=input_files,
                        desired_outputs=desired_outputs
                    )
                    
                    if dynamic_workflow.confidence_score > 0.7:
                        print(f"🚀 Dynamic workflow generated with {dynamic_workflow.confidence_score:.2f} confidence!")
                        print(f"   Components: {len(dynamic_workflow.components)}")
                        print(f"   Estimated time: {dynamic_workflow.estimated_duration:.1f}s")
                        print(f"   Execution plan: {len(dynamic_workflow.execution_plan)} steps")
                        
                        # Log dynamic workflow generation
                        if self.session_logger:
                            self.session_logger.log_workflow_execution(
                                workflow_name=f"dynamic_{dynamic_workflow.workflow_id}",
                                step_name="workflow_generated",
                                status="success",
                                metadata={
                                    "confidence_score": dynamic_workflow.confidence_score,
                                    "component_count": len(dynamic_workflow.components),
                                    "estimated_duration": dynamic_workflow.estimated_duration,
                                    "problem_type": dynamic_workflow.problem_description[:100]
                                }
                            )
                        
                        # Offer to execute the dynamic workflow
                        execute_dynamic = input("\n🔧 Execute this dynamic workflow? (y/n): ").strip().lower()
                        if execute_dynamic in ['y', 'yes']:
                            await self._execute_dynamic_workflow(dynamic_workflow, user_input)
                            return
                        else:
                            print("⏭️  Skipping dynamic workflow execution")
                            
                except Exception as dynamic_error:
                    print(f"⚠️  Dynamic workflow generation failed: {dynamic_error}")
                    if self.session_logger:
                        self.session_logger.log_error("dynamic_workflow_generation_failed", str(dynamic_error))
            
            # Fall back to problem-driven orchestration
            print("🔍 Using problem-driven workflow orchestration...")
            result = await self._orchestrate_workflow(user_input)
            
            if result:
                print(f"✅ Workflow orchestrated: {result}")
                if self.session_logger:
                    self.session_logger.log_info("workflow_orchestrated", f"Problem-driven workflow orchestrated: {result}")
            else:
                print("❌ No suitable workflow found for this query")
                if self.session_logger:
                    self.session_logger.log_info("no_workflow_found", "No suitable workflow found for query")
                
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            if self.session_logger:
                self.session_logger.log_error("general_query_error", str(e), metadata={"query": user_input[:100]})
    
    def _is_file_analysis_request(self, user_input: str) -> bool:
        """Check if the query is a file-based analysis request."""
        file_indicators = [
            "analyze file", "check file", "process file", "map file", "enrich file",
            "mitre", "attack framework", "policy catalog", "content catalog",
            "map to", "categorize file", "classify file"
        ]
        
        user_input_lower = user_input.lower()
        return any(indicator in user_input_lower for indicator in file_indicators)
    
    async def _handle_file_analysis_request(self, user_input: str):
        """Handle file-based analysis requests with automatic workflow selection."""
        try:
            print("\n📁 File Analysis Request Detected")
            print("-" * 40)
            
            # Extract file path from user input
            file_path = self._extract_file_path(user_input)
            
            if not file_path:
                file_path = input("📂 Enter the file path to analyze: ").strip()
            
            if not file_path:
                print("❌ File path required")
                return
            
            # Check if file exists
            if not Path(file_path).exists():
                print(f"❌ File not found: {file_path}")
                return
            
            # Get output file path
            output_path = input("📤 Enter output file path (or press Enter for auto-generated): ").strip()
            if not output_path:
                # Auto-generate output path
                input_path = Path(file_path)
                output_path = input_path.parent / f"{input_path.stem}_analyzed{input_path.suffix}"
            
            # Analyze the problem and recommend workflow
            print(f"\n🔍 Analyzing problem and selecting workflow...")
            
            # Import problem orchestrator
            sys.path.append("bin")
            from bin.problem_driven_orchestrator import ProblemDrivenOrchestrator
            
            orchestrator = ProblemDrivenOrchestrator(self.tool_manager)
            
            # Analyze problem
            context = await orchestrator.analyze_problem(user_input, [file_path])
            
            print(f"✅ Problem analyzed:")
            print(f"   Type: {[dt.value for dt in context.data_types]}")
            print(f"   Complexity: {context.complexity_level}")
            print(f"   Desired outputs: {context.desired_outputs}")
            
            # Get workflow recommendation
            workflow = await orchestrator.recommend_workflow(context)
            
            print(f"\n🚀 Workflow Recommendation:")
            print(f"   Workflow: {workflow.workflow_name}")
            print(f"   Confidence: {workflow.confidence_score:.2f}")
            print(f"   Reasoning: {workflow.reasoning}")
            print(f"   Estimated time: {workflow.estimated_time}")
            
            # Execute the recommended workflow
            print(f"\n🚀 Executing recommended workflow...")
            
            if workflow.workflow_name == "mitre_attack_mapping":
                await self._execute_mitre_mapping_workflow(file_path, output_path, user_input)
            elif workflow.workflow_name == "csv_enrichment":
                await self._execute_csv_enrichment_workflow(file_path, output_path, user_input)
            else:
                print(f"🤖 Using general workflow: {workflow.workflow_name}")
                result = await orchestrator.execute_workflow(context, workflow)
                print(f"✅ Workflow result: {result}")
                
        except Exception as e:
            print(f"❌ Error in file analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_file_path(self, user_input: str) -> Optional[str]:
        """Extract file path from user input."""
        # Look for file paths in the input
        import re
        
        # Common file path patterns
        patterns = [
            r'["\']([^"\']*\.(?:csv|txt|json|md|docx?|pdf))["\']',  # Quoted paths
            r'([a-zA-Z0-9/._-]+\.(?:csv|txt|json|md|docx?|pdf))',  # Unquoted paths
            r'file\s+([a-zA-Z0-9/._-]+\.(?:csv|txt|json|md|docx?|pdf))',  # "file path.ext"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_potential_files(self, user_input: str) -> List[str]:
        """Extract potential input files from user input."""
        files = []
        words = user_input.split()
        for word in words:
            if word.endswith(('.csv', '.json', '.xlsx', '.txt', '.log', '.pcap')):
                files.append(word)
        return files
    
    def _extract_desired_outputs(self, user_input: str) -> List[str]:
        """Extract desired outputs from user input."""
        outputs = []
        user_input_lower = user_input.lower()
        
        # Look for output indicators
        if any(word in user_input_lower for word in ["csv", "excel", "spreadsheet"]):
            outputs.append("csv")
        if any(word in user_input_lower for word in ["json", "api"]):
            outputs.append("json")
        if any(word in user_input_lower for word in ["report", "summary", "analysis"]):
            outputs.append("report")
        if any(word in user_input_lower for word in ["enriched", "enhanced", "processed"]):
            outputs.append("enriched_data")
        if any(word in user_input_lower for word in ["mitre", "attack framework", "mapped"]):
            outputs.append("mitre_mapping")
        
        return outputs
    
    async def _execute_dynamic_workflow(self, workflow, user_input: str):
        """Execute a dynamically generated workflow."""
        try:
            print(f"\n🚀 Executing dynamic workflow: {workflow.workflow_id}")
            print(f"Problem: {workflow.problem_description}")
            print(f"Components: {len(workflow.components)}")
            
            # Log workflow execution start
            if self.session_logger:
                self.session_logger.log_workflow_execution(
                    workflow_name=f"dynamic_{workflow.workflow_id}",
                    step_name="execution_started",
                    status="in_progress",
                    metadata={
                        "component_count": len(workflow.components),
                        "estimated_duration": workflow.estimated_duration,
                        "confidence_score": workflow.confidence_score
                    }
                )
            
            # Execute each component in the workflow
            for i, step in enumerate(workflow.execution_plan):
                print(f"\n📋 Step {step['step']}: {step['name']}")
                print(f"   Components: {', '.join(step['components'])}")
                print(f"   Execution: {step['execution_type']}")
                print(f"   Estimated time: {step['estimated_time']:.1f}s")
                
                # Log step execution
                if self.session_logger:
                    self.session_logger.log_workflow_execution(
                        workflow_name=f"dynamic_{workflow.workflow_id}",
                        step_name=f"step_{step['step']}_{step['name'].lower().replace(' ', '_')}",
                        status="in_progress",
                        metadata={
                            "step_number": step['step'],
                            "step_name": step['name'],
                            "components": step['components'],
                            "execution_type": step['execution_type'],
                            "estimated_time": step['estimated_time']
                        }
                    )
                
                # Simulate component execution (in real implementation, this would call actual components)
                print(f"   🔄 Executing components...")
                await asyncio.sleep(1)  # Simulate execution time
                
                print(f"   ✅ Step completed")
                
                # Log step completion
                if self.session_logger:
                    self.session_logger.log_workflow_execution(
                        workflow_name=f"dynamic_{workflow.workflow_id}",
                        step_name=f"step_{step['step']}_{step['name'].lower().replace(' ', '_')}",
                        status="completed",
                        metadata={
                            "step_number": step['step'],
                            "step_name": step['name'],
                            "execution_time": 1.0  # Simulated time
                        }
                    )
            
            print(f"\n🎉 Dynamic workflow completed successfully!")
            print(f"   Total steps: {len(workflow.execution_plan)}")
            print(f"   Total time: {workflow.estimated_duration:.1f}s")
            
            # Log workflow completion
            if self.session_logger:
                self.session_logger.log_workflow_execution(
                    workflow_name=f"dynamic_{workflow.workflow_id}",
                    step_name="execution_completed",
                    status="completed",
                    metadata={
                        "total_steps": len(workflow.execution_plan),
                        "total_time": workflow.estimated_duration,
                        "success": True
                    }
                )
                
        except Exception as e:
            print(f"❌ Error executing dynamic workflow: {e}")
            if self.session_logger:
                self.session_logger.log_error("dynamic_workflow_execution_failed", str(e), metadata={
                    "workflow_id": workflow.workflow_id,
                    "problem_description": workflow.problem_description[:100]
                })
    
    async def execute_dynamic_workflow(self, problem_description: str, input_files: List[str] = None, 
                                     outputs: List[str] = None, constraints: str = None, 
                                     execute: bool = False, export: bool = False):
        """Execute dynamic workflow generation and optionally execution."""
        try:
            if not self.dynamic_workflow_generator:
                print("❌ Dynamic workflow generator not available")
                return
            
            print(f"\n🚀 Dynamic Workflow Generation")
            print(f"Problem: {problem_description}")
            if input_files:
                print(f"Input files: {', '.join(input_files)}")
            if outputs:
                print(f"Desired outputs: {', '.join(outputs)}")
            
            # Parse constraints if provided
            constraints_dict = None
            if constraints:
                try:
                    constraints_dict = json.loads(constraints)
                    print(f"Constraints: {json.dumps(constraints_dict, indent=2)}")
                except json.JSONDecodeError:
                    print(f"⚠️  Invalid constraints JSON: {constraints}")
            
            # Generate workflow
            print("\n🔍 Generating dynamic workflow...")
            workflow = await self.dynamic_workflow_generator.generate_workflow(
                problem_description=problem_description,
                input_files=input_files,
                desired_outputs=outputs,
                constraints=constraints_dict
            )
            
            # Display workflow details
            print(f"\n✅ Dynamic Workflow Generated!")
            print(f"   ID: {workflow.workflow_id}")
            print(f"   Confidence: {workflow.confidence_score:.2f}")
            print(f"   Components: {len(workflow.components)}")
            print(f"   Estimated time: {workflow.estimated_duration:.1f}s")
            print(f"   Steps: {len(workflow.execution_plan)}")
            
            print(f"\n📋 Execution Plan:")
            for step in workflow.execution_plan:
                print(f"   Step {step['step']}: {step['name']}")
                print(f"   Components: {', '.join(step['components'])}")
                print(f"   Type: {step['execution_type']}")
                print(f"   Time: {step['estimated_time']:.1f}s")
            
            if workflow.adaptation_points:
                print(f"\n🔄 Adaptation Points:")
                for point in workflow.adaptation_points:
                    print(f"   • {point['description']}")
            
            # Export workflow if requested
            if export:
                workflow_json = self.dynamic_workflow_generator.export_workflow(workflow, "json")
                export_file = f"dynamic_workflow_{workflow.workflow_id}.json"
                with open(export_file, 'w') as f:
                    f.write(workflow_json)
                print(f"\n💾 Workflow exported to: {export_file}")
            
            # Execute workflow if requested
            if execute:
                print(f"\n🔧 Executing dynamic workflow...")
                await self._execute_dynamic_workflow(workflow, problem_description)
            else:
                print(f"\n💡 Use --execute to run this workflow immediately")
            
            # Show performance metrics
            metrics = self.dynamic_workflow_generator.get_performance_metrics()
            if metrics:
                print(f"\n📊 Performance Metrics:")
                print(f"   Total workflows generated: {metrics['total_workflows_generated']}")
                print(f"   Average generation time: {metrics['average_generation_time']:.3f}s")
                print(f"   Cache hit rate: {metrics['cache_hit_rate']:.1%}")
                
        except Exception as e:
            print(f"❌ Error in dynamic workflow execution: {e}")
            if self.session_logger:
                self.session_logger.log_error("dynamic_workflow_command_failed", str(e), metadata={
                    "problem_description": problem_description[:100],
                    "input_files": input_files,
                    "outputs": outputs
                })
    
    async def _execute_mitre_mapping_workflow(self, input_file: str, output_file: str, user_query: str):
        """Execute MITRE ATT&CK mapping workflow."""
        try:
            print(f"\n🛡️  Executing MITRE ATT&CK Mapping Workflow")
            print(f"   Input: {input_file}")
            print(f"   Output: {output_file}")
            
            # Import MITRE mapping workflow
            from bin.mitre_attack_mapping_workflow import MitreAttackMappingWorkflow
            
            # Create workflow instance
            workflow = MitreAttackMappingWorkflow()
            
            # Auto-detect content column
            content_column = input("🔍 Content column name (or press Enter for auto-detection): ").strip()
            if not content_column:
                content_column = None
            
            # Get batch size
            batch_size_input = input("📦 Batch size (default 100): ").strip()
            batch_size = int(batch_size_input) if batch_size_input.isdigit() else 100
            
            print(f"\n🚀 Starting MITRE ATT&CK mapping...")
            print(f"   Content column: {content_column or 'Auto-detected'}")
            print(f"   Batch size: {batch_size}")
            
            # Execute workflow
            result = await workflow.execute_mapping_workflow(
                input_file=input_file,
                output_file=output_file,
                content_column=content_column,
                batch_size=batch_size
            )
            
            if result.success:
                print(f"\n✅ MITRE ATT&CK mapping completed successfully!")
                print(f"   Total items: {result.total_items}")
                print(f"   Mapped items: {result.mapped_items}")
                print(f"   Mapping time: {result.mapping_time:.2f} seconds")
                print(f"   Output file: {result.output_file}")
                
                # Show sample of mappings
                if result.mappings:
                    print(f"\n📊 Sample Mappings:")
                    print("-" * 40)
                    for i, mapping in enumerate(result.mappings[:3]):  # Show first 3
                        print(f"{i+1}. {mapping.technique_name} ({mapping.technique_id})")
                        print(f"   Tactic: {mapping.tactic_name}")
                        print(f"   Confidence: {mapping.confidence_score:.2f}")
                        print(f"   Reasoning: {mapping.reasoning[:100]}...")
                        print()
            else:
                print(f"\n❌ MITRE mapping failed: {result.error_message}")
                
        except Exception as e:
            print(f"❌ Error executing MITRE mapping workflow: {e}")
            import traceback
            traceback.print_exc()
    
    async def _orchestrate_workflow(self, user_input: str):
        """Use problem-driven orchestration to select and execute workflows."""
        try:
            # Import problem orchestrator
            sys.path.append("bin")
            from bin.problem_driven_orchestrator import ProblemDrivenOrchestrator
            
            orchestrator = ProblemDrivenOrchestrator(self.tool_manager)
            
            # Analyze problem
            context = await orchestrator.analyze_problem(user_input)
            
            # Get workflow recommendation
            workflow = await orchestrator.recommend_workflow(context)
            
            # Get tool recommendations
            tools = await orchestrator.recommend_tools(context, workflow)
            
            print(f"\n🔍 Problem Analysis:")
            print(f"   Type: {context.problem_description[:100]}...")
            print(f"   Complexity: {context.complexity_level}")
            print(f"   Urgency: {context.urgency}")
            
            print(f"\n🚀 Workflow Recommendation:")
            print(f"   Workflow: {workflow.workflow_name}")
            print(f"   Confidence: {workflow.confidence_score:.2f}")
            print(f"   Reasoning: {workflow.reasoning}")
            print(f"   Estimated time: {workflow.estimated_time}")
            
            print(f"\n🔧 Tool Recommendations:")
            for tool in tools[:5]:  # Show top 5 tools
                print(f"   • {tool.tool_name}: {tool.reasoning}")
            
            # Ask user if they want to execute
            execute = input(f"\n🚀 Execute recommended workflow '{workflow.workflow_name}'? (y/n): ").strip().lower()
            
            if execute in ['y', 'yes']:
                print(f"\n🚀 Executing workflow...")
                result = await orchestrator.execute_workflow(context, workflow)
                return result
            else:
                print("⏸️  Workflow execution cancelled by user")
                return None
                
        except Exception as e:
            print(f"❌ Error in workflow orchestration: {e}")
            return None
    
    def _can_handle_locally(self, query):
        """Determine if a query can be handled with local tools."""
        simple_keywords = ['list', 'show', 'count', 'basic', 'simple', 'status', 'tools']
        complex_keywords = ['analyze', 'understand', 'explain', 'why', 'how', 'complex', 'relationship']
        
        query_lower = query.lower()
        
        # Check for simple operations
        if any(keyword in query_lower for keyword in simple_keywords):
            return True
        
        # Check for complex operations
        if any(keyword in query_lower for keyword in complex_keywords):
            return False
        
        # Default to local for short queries
        return len(query.split()) < 10
    
    def _assess_query_complexity(self, query):
        """Assess the complexity of a user query."""
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Simple queries
        if word_count < 5 and any(word in query_lower for word in ['show', 'list', 'status', 'help']):
            return 'simple'
        
        # Complex queries
        if word_count > 15 or any(word in query_lower for word in ['analyze', 'explain', 'understand', 'relationship', 'why', 'how']):
            return 'complex'
        
        # Moderate queries
        return 'moderate'
    
    async def _local_analysis(self, query):
        """Perform local analysis using available tools."""
        # Mock implementation - replace with actual local tool calls
        return f"Local analysis result for: {query}"
    
    async def _llm_analysis(self, query):
        """Perform LLM-based analysis."""
        # Mock implementation - replace with actual LLM calls
        return f"LLM analysis result for: {query}"
    
    async def _local_categorization(self, query):
        """Perform local categorization using available tools."""
        # Mock implementation - replace with actual local tool calls
        return f"Local categorization result for: {query}"
    
    async def _llm_categorization(self, query):
        """Perform LLM-based categorization."""
        # Mock implementation - replace with actual LLM calls
        return f"LLM categorization result for: {query}"
    
    async def _local_summarization(self, query):
        """Perform local summarization using available tools."""
        # Mock implementation - replace with actual local tool calls
        return f"Local summarization result for: {query}"
    
    async def _llm_summarization(self, query):
        """Perform LLM-based summarization."""
        # Mock implementation - replace with actual LLM calls
        return f"LLM summarization result for: {query}"
    
    async def _local_query_handling(self, query):
        """Handle simple queries with local tools."""
        # Mock implementation - replace with actual local tool calls
        return f"Local handling result for: {query}"
    
    async def _hybrid_query_handling(self, query):
        """Handle moderate queries with hybrid approach."""
        # Mock implementation - replace with actual hybrid tool calls
        return f"Hybrid handling result for: {query}"
    
    async def _llm_query_handling(self, query):
        """Handle complex queries with LLM."""
        # Mock implementation - replace with actual LLM calls
        return f"LLM handling result for: {query}"
    
    def get_agentic_system_status(self):
        """Get agentic workflow system status."""
        if not self.agentic_workflow_cli:
            print("❌ Agentic workflow system not available")
            return
            
        try:
            status = self.agentic_workflow_cli.get_system_status()
            print(f"\n📊 Agentic System Status:")
            print("=" * 80)
            print(json.dumps(status, indent=2, default=str))
            
        except Exception as e:
            print(f"❌ Error getting agentic system status: {e}")
    
    def get_advanced_workflow_metrics(self):
        """Get performance metrics from the advanced workflow system."""
        if not self.advanced_workflow_cli:
            print("❌ Advanced workflow system not available")
            return
            
        try:
            metrics = self.advanced_workflow_cli.get_performance_metrics()
            print(f"\n📊 Advanced Workflow Performance Metrics:")
            print("=" * 80)
            print(json.dumps(metrics, indent=2, default=str))
            
        except Exception as e:
            print(f"❌ Error getting workflow metrics: {e}")
    
    async def discover_advanced_tools(self):
        """Discover new MCP tools in the advanced workflow system."""
        if not self.advanced_workflow_cli:
            print("❌ Advanced workflow system not available")
            return
            
        try:
            discovered_tools = await self.advanced_workflow_cli.discover_mcp_tools()
            print(f"\n🔍 Tool Discovery Results:")
            print("=" * 80)
            for tool in discovered_tools:
                print(f"🔹 {tool.get('tool_id', 'Unknown')}")
                print(f"   Category: {tool.get('category', 'Unknown')}")
                print(f"   Capabilities: {', '.join(tool.get('capabilities', []))}")
                print()
                
        except Exception as e:
            print(f"❌ Error discovering tools: {e}")
    
    def show_tool_categories(self):
        """Show available tool categories with counts."""
        if not self.mcp_server:
            print("❌ MCP server not available")
            return
            
        try:
            tools = self.mcp_server._list_tools_handler()
            
            if tools['success']:
                categories = {}
                for tool_name, tool_info in tools['tools'].items():
                    cat = tool_info['category']
                    categories[cat] = categories.get(cat, 0) + 1
                
                print(f"\n📂 Tool Categories ({len(categories)} total):")
                print("=" * 50)
                
                for category, count in sorted(categories.items()):
                    print(f"  {category}: {count} tools")
                    
        except Exception as e:
            print(f"❌ Error showing categories: {e}")
    
    def show_tool_tags(self):
        """Show available tool tags with usage counts."""
        if not self.mcp_server:
            print("❌ MCP server not available")
            return
            
        try:
            tools = self.mcp_server._list_tools_handler()
            
            if tools['success']:
                tags = {}
                for tool_name, tool_info in tools['tools'].items():
                    for tag in tool_info['tags']:
                        tags[tag] = tags.get(tag, 0) + 1
                
                print(f"\n🏷️  Tool Tags ({len(tags)} total):")
                print("=" * 50)
                
                for tag, count in sorted(tags.items()):
                    print(f"  {tag}: {count} tools")
                    
        except Exception as e:
            print(f"❌ Error showing tags: {e}")
    
    def search_tools(self, query):
        """Search tools by name, description, or tags."""
        if not self.mcp_server:
            print("❌ MCP server not available")
            return
            
        try:
            tools = self.mcp_server._list_tools_handler(detailed=True)
            
            if tools['success']:
                matching_tools = {}
                query_lower = query.lower()
                
                for tool_name, tool_info in tools['tools'].items():
                    # Search in name, description, and tags
                    if (query_lower in tool_name.lower() or
                        query_lower in tool_info['description'].lower() or
                        any(query_lower in tag.lower() for tag in tool_info['tags'])):
                        matching_tools[tool_name] = tool_info
                
                if matching_tools:
                    print(f"\n🔍 Search Results for '{query}' ({len(matching_tools)} matches):")
                    print("=" * 80)
                    
                    for tool_name, tool_info in matching_tools.items():
                        print(f"📌 {tool_name}")
                        print(f"   Description: {tool_info['description']}")
                        print(f"   Category: {tool_info['category']}")
                        print(f"   Tags: {', '.join(tool_info['tags'])}")
                        print()
                else:
                    print(f"\n🔍 No tools found matching '{query}'")
                    
        except Exception as e:
            print(f"❌ Error searching tools: {e}")
    
    def show_workflow_examples(self):
        """Show example workflows using the available tools."""
        print(f"\n📚 Example Workflows")
        print("=" * 80)
        
        print("1. Data Analysis Workflow:")
        print("   - Create DataFrame from CSV → query_dataframe → manipulate_dataframe")
        print("   - Convert results to HTML report → write_html_report")
        
        print("\n2. Security Investigation Workflow:")
        print("   - Extract archive → extract_archive")
        print("   - Analyze logs → create_dataframe → query_dataframe")
        print("   - Build threat graph → create_node → query_graph")
        
        print("\n3. Report Generation Workflow:")
        print("   - Query multiple data sources")
        print("   - Correlate results")
        print("   - Generate comprehensive report → write_html_report")
        
        print("\n4. Data Pipeline Workflow:")
        print("   - Extract → Transform → Load")
        print("   - Use convert_file for format transformations")
        print("   - Use compression tools for storage optimization")
        
        if self.advanced_workflow_cli:
            print("\n5. Advanced Workflow Examples:")
            print("   - Threat Hunting: python cs_ai_cli.py workflow --type threat_hunting --problem 'Investigate APT29'")
            print("   - Incident Response: python cs_ai_cli.py workflow --type incident_response --problem 'Respond to breach'")
            print("   - Compliance: python cs_ai_cli.py workflow --type compliance --problem 'Assess policy compliance'")
        
        if self.agentic_workflow_cli:
            print("\n6. Agentic Workflow Examples:")
            print("   - Automated CSV Processing:")
            print("     python cs_ai_cli.py automated --csv threat_data.csv --problem 'Analyze threats' --output enriched.csv")
            print("   - Manual Problem Solving:")
            print("     python cs_ai_cli.py manual --problem 'Investigate security incident'")
            print("   - Hybrid Workflow:")
            print("     python cs_ai_cli.py hybrid --csv data.csv --problem 'Adaptive analysis' --output result.csv")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CS AI CLI - Dynamic Agentic Workflow Tool with Advanced Workflow System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available tools
  python cs_ai_cli.py list-tools
  
  # List advanced workflow templates
  python cs_ai_cli.py list-workflows
  
  # List advanced MCP tools
  python cs_ai_cli.py list-advanced-tools
  
  # Execute advanced workflow
  python cs_ai_cli.py workflow --type threat_hunting --problem "Investigate APT29 activity"
  
  # Execute MCP workflow
  python cs_ai_cli.py mcp-workflow --problem "Get threat context" --tools "get_workflow_context,search_memories"
  
  # Execute automated CSV workflow
  python cs_ai_cli.py automated --csv input.csv --problem "Analyze data" --output enriched.csv
  
  # Execute manual workflow
  python cs_ai_cli.py manual --problem "Investigate security incident"
  
  # Execute hybrid workflow
  python cs_ai_cli.py hybrid --csv data.csv --problem "Adaptive analysis" --output result.csv
  
  # Get workflow metrics
  python cs_ai_cli.py workflow-metrics
  
  # Get agentic system status
  python cs_ai_cli.py agentic-status
  
  # Discover new tools
  python cs_ai_cli.py discover-tools
  
  # Traditional MCP tool operations
  python cs_ai_cli.py list-tools --category dataframe
  python cs_ai_cli.py execute --tool create_dataframe --args '{"name": "test", "data": "sample.csv"}'
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Advanced workflow commands
    workflow_parser = subparsers.add_parser('workflow', help='Execute advanced workflow')
    workflow_parser.add_argument('--type', required=True, 
                               choices=['threat_hunting', 'incident_response', 'compliance', 'risk_assessment', 'investigation', 'analysis'],
                               help='Workflow type')
    workflow_parser.add_argument('--problem', required=True, help='Problem description')
    workflow_parser.add_argument('--priority', type=int, default=1, help='Problem priority (1-5)')
    workflow_parser.add_argument('--complexity', type=int, default=1, help='Problem complexity (1-10)')
    
    mcp_workflow_parser = subparsers.add_parser('mcp-workflow', help='Execute MCP workflow')
    mcp_workflow_parser.add_argument('--problem', required=True, help='Problem description')
    mcp_workflow_parser.add_argument('--tools', required=True, help='Comma-separated list of required tools')
    mcp_workflow_parser.add_argument('--priority', type=int, default=1, help='Problem priority (1-5)')
    mcp_workflow_parser.add_argument('--complexity', type=int, default=1, help='Problem complexity (1-10)')
    
    # Agentic workflow commands
    automated_parser = subparsers.add_parser('automated', help='Execute automated CSV processing workflow')
    automated_parser.add_argument('--csv', required=True, help='Input CSV file path')
    automated_parser.add_argument('--problem', required=True, help='Problem description and instructions')
    automated_parser.add_argument('--output', required=True, help='Output CSV file path')
    automated_parser.add_argument('--priority', type=int, default=1, help='Problem priority (1-5)')
    automated_parser.add_argument('--complexity', type=int, default=1, help='Problem complexity (1-10)')
    
    # CSV Enrichment workflow command
    csv_enrichment_parser = subparsers.add_parser('csv-enrichment', help='Execute CSV enrichment workflow with LLM processing')
    csv_enrichment_parser.add_argument('--input', required=True, help='Input CSV file path')
    csv_enrichment_parser.add_argument('--output', required=True, help='Output CSV file path')
    csv_enrichment_parser.add_argument('--prompt', required=True, help='Enrichment prompt describing what to add')
    csv_enrichment_parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing (default: 100)')
    csv_enrichment_parser.add_argument('--max-retries', type=int, default=3, help='Maximum retries for failed rows (default: 3)')
    csv_enrichment_parser.add_argument('--quality-threshold', type=float, default=0.8, help='Quality threshold for validation (default: 0.8)')
    
    # Dynamic workflow generation command
    dynamic_parser = subparsers.add_parser('dynamic-workflow', help='Generate and execute dynamic workflows')
    dynamic_parser.add_argument('problem_description', help='Description of the problem to solve')
    dynamic_parser.add_argument('--input-files', '-i', nargs='+', help='Input files for analysis')
    dynamic_parser.add_argument('--outputs', '-o', nargs='+', help='Desired output formats/types')
    dynamic_parser.add_argument('--constraints', '-c', help='JSON string of constraints')
    dynamic_parser.add_argument('--execute', '-e', action='store_true', help='Execute the generated workflow immediately')
    dynamic_parser.add_argument('--export', action='store_true', help='Export the workflow to JSON file')
    
    manual_parser = subparsers.add_parser('manual', help='Execute manual interactive workflow')
    manual_parser.add_argument('--problem', required=True, help='Problem description')
    manual_parser.add_argument('--priority', type=int, default=1, help='Problem priority (1-5)')
    manual_parser.add_argument('--complexity', type=int, default=1, help='Problem complexity (1-10)')
    manual_parser.add_argument('--interactive', action='store_true', help='Enable interactive mode')
    
    hybrid_parser = subparsers.add_parser('hybrid', help='Execute hybrid workflow (automated + manual)')
    hybrid_parser.add_argument('--csv', help='Input CSV file path (optional)')
    hybrid_parser.add_argument('--problem', required=True, help='Problem description')
    hybrid_parser.add_argument('--output', help='Output CSV file path (if CSV input provided)')
    hybrid_parser.add_argument('--priority', type=int, default=1, help='Problem priority (1-5)')
    hybrid_parser.add_argument('--complexity', type=int, default=1, help='Problem complexity (1-10)')
    
    # System management commands
    subparsers.add_parser('list-workflows', help='List available workflow templates')
    subparsers.add_parser('list-advanced-tools', help='List advanced MCP tools')
    subparsers.add_parser('workflow-metrics', help='Get workflow performance metrics')
    subparsers.add_parser('agentic-status', help='Get agentic workflow system status')
    subparsers.add_parser('discover-tools', help='Discover new MCP tools')
    
    # Traditional MCP commands
    list_parser = subparsers.add_parser('list-tools', help='List available MCP tools')
    list_parser.add_argument('--category', help='Filter by tool category')
    list_parser.add_argument('--tags', nargs='+', help='Filter by tags')
    list_parser.add_argument('--detailed', action='store_true', help='Show detailed tool information')
    
    schema_parser = subparsers.add_parser('get-schema', help='Get tool schema')
    schema_parser.add_argument('--tool', required=True, help='Tool name')
    
    execute_parser = subparsers.add_parser('execute', help='Execute an MCP tool')
    execute_parser.add_argument('--tool', required=True, help='Tool name to execute')
    execute_parser.add_argument('--args', help='JSON string of tool arguments')
    
    info_parser = subparsers.add_parser('server-info', help='Get MCP server information')
    info_parser.add_argument('--detailed', action='store_true', help='Show detailed information')
    
    subparsers.add_parser('categories', help='Show available tool categories')
    subparsers.add_parser('tags', help='Show available tool tags')
    
    search_parser = subparsers.add_parser('search', help='Search tools')
    search_parser.add_argument('--query', required=True, help='Search query')
    
    subparsers.add_parser('examples', help='Show example workflows')
    
    args = parser.parse_args()
    
    if not args.command:
        # No command provided - run interactive mode
        print("🚀 Starting Interactive Cybersecurity AI Helper...")
        cli = CSAgentCLI()
        asyncio.run(cli.run_interactive_mode())
        return
    
    # Initialize CLI
    cli = CSAgentCLI()
    
    # Execute commands
    if args.command == 'workflow':
        asyncio.run(cli.execute_advanced_workflow(
            args.type, args.problem, args.priority, args.complexity
        ))
    
    elif args.command == 'mcp-workflow':
        asyncio.run(cli.execute_advanced_mcp_workflow(
            args.problem, args.tools, args.priority, args.complexity
        ))
    
    elif args.command == 'automated':
        asyncio.run(cli.execute_automated_workflow(
            args.csv, args.problem, args.output, args.priority, args.complexity
        ))
    
    elif args.command == 'manual':
        asyncio.run(cli.execute_manual_workflow(
            args.problem, args.priority, args.complexity, args.interactive
        ))
    
    elif args.command == 'hybrid':
        asyncio.run(cli.execute_hybrid_workflow(
            args.csv, args.problem, args.output, args.priority, args.complexity
        ))
    
    elif args.command == 'csv-enrichment':
        asyncio.run(cli.execute_csv_enrichment_workflow(
            args.input, args.output, args.prompt, args.batch_size, args.max_retries, args.quality_threshold
        ))
    
    elif args.command == 'dynamic-workflow':
        asyncio.run(cli.execute_dynamic_workflow(
            args.problem_description, args.input_files, args.outputs, args.constraints, args.execute, args.export
        ))
    
    elif args.command == 'list-workflows':
        cli.list_advanced_workflows()
    
    elif args.command == 'list-advanced-tools':
        cli.list_advanced_mcp_tools()
    
    elif args.command == 'workflow-metrics':
        cli.get_advanced_workflow_metrics()
    
    elif args.command == 'agentic-status':
        cli.get_agentic_system_status()
    
    elif args.command == 'discover-tools':
        asyncio.run(cli.discover_advanced_tools())
    
    elif args.command == 'list-tools':
        cli.list_tools(
            category=args.category,
            tags=args.tags,
            detailed=args.detailed
        )
    
    elif args.command == 'get-schema':
        cli.get_tool_schema(args.tool)
    
    elif args.command == 'execute':
        asyncio.run(cli.execute_tool(args.tool, args.args))
    
    elif args.command == 'server-info':
        cli.get_server_info(detailed=args.detailed)
    
    elif args.command == 'categories':
        cli.show_tool_categories()
    
    elif args.command == 'tags':
        cli.show_tool_tags()
    
    elif args.command == 'search':
        cli.search_tools(args.query)
    
    elif args.command == 'examples':
        cli.show_workflow_examples()

if __name__ == "__main__":
    main()
