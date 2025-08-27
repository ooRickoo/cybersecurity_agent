#!/usr/bin/env python3
"""
Comprehensive Splunk Integration System
Full-featured tools for interacting with on-prem and cloud Splunk instances.
"""

import os
import sys
import json
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import base64

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import splunklib.client as client
    import splunklib.results as results
    SPLUNK_SDK_AVAILABLE = True
except ImportError:
    SPLUNK_SDK_AVAILABLE = False
    print("⚠️  Splunk SDK not available. Install with: pip install splunk-sdk")

class SplunkInstance:
    """Represents a Splunk instance with connection and authentication."""
    
    def __init__(self, name: str, host: str, port: int = 8089, 
                 username: str = None, password: str = None, 
                 token: str = None, is_cloud: bool = False):
        self.name = name
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.token = token
        self.is_cloud = is_cloud
        self.service = None
        self.session_key = None
        
        # Connection status
        self.connected = False
        self.last_connection = None
        self.connection_errors = []
        
        # Instance capabilities
        self.capabilities = {}
        self.version = None
        self.license_info = {}
    
    def connect(self) -> bool:
        """Connect to the Splunk instance."""
        try:
            if self.token:
                # Use token-based authentication
                self.service = client.connect(
                    host=self.host,
                    port=self.port,
                    token=self.token,
                    scheme='https' if self.is_cloud else 'http'
                )
            elif self.username and self.password:
                # Use username/password authentication
                self.service = client.connect(
                    host=self.host,
                    port=self.port,
                    username=self.username,
                    password=self.password,
                    scheme='https' if self.is_cloud else 'http'
                )
            else:
                raise ValueError("Either token or username/password must be provided")
            
            # Test connection
            self.service.info()
            self.connected = True
            self.last_connection = datetime.now()
            
            # Get instance information
            self._get_instance_info()
            
            print(f"✅ Connected to Splunk instance: {self.name} ({self.host}:{self.port})")
            return True
            
        except Exception as e:
            error_msg = f"Connection failed: {e}"
            self.connection_errors.append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            })
            print(f"❌ {error_msg}")
            return False
    
    def _get_instance_info(self):
        """Get instance information and capabilities."""
        try:
            # Get version
            info = self.service.info()
            self.version = info.get('version', 'Unknown')
            
            # Get license info
            try:
                license_info = self.service.get_license_info()
                self.license_info = {
                    'type': license_info.get('type', 'Unknown'),
                    'quota': license_info.get('quota', 'Unknown'),
                    'used': license_info.get('used', 'Unknown')
                }
            except:
                self.license_info = {'type': 'Unknown'}
            
            # Determine capabilities
            self.capabilities = {
                'search': True,
                'scheduled_jobs': True,
                'knowledge_objects': True,
                'data_models': True,
                'indexes': True,
                'sourcetypes': True
            }
            
        except Exception as e:
            print(f"⚠️  Warning: Could not get instance info: {e}")
    
    def disconnect(self):
        """Disconnect from the Splunk instance."""
        try:
            if self.service:
                self.service.logout()
                self.service = None
            self.connected = False
            print(f"✅ Disconnected from Splunk instance: {self.name}")
        except Exception as e:
            print(f"⚠️  Error during disconnect: {e}")
    
    def is_connected(self) -> bool:
        """Check if instance is connected."""
        if not self.connected or not self.service:
            return False
        
        try:
            # Test connection with a simple API call
            self.service.info()
            return True
        except:
            self.connected = False
            return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information."""
        return {
            'name': self.name,
            'host': self.host,
            'port': self.port,
            'is_cloud': self.is_cloud,
            'connected': self.connected,
            'last_connection': self.last_connection.isoformat() if self.last_connection else None,
            'version': self.version,
            'license_info': self.license_info,
            'capabilities': self.capabilities,
            'connection_errors': self.connection_errors[-5:]  # Last 5 errors
        }

class SplunkIntegration:
    """Comprehensive Splunk integration system."""
    
    def __init__(self, session_manager=None):
        self.session_manager = session_manager
        self.instances = {}
        self.active_instance = None
        
        # Search history and caching
        self.search_history = []
        self.search_cache = {}
        
        # Data discovery cache
        self.discovery_cache = {}
        
        # Performance metrics
        self.performance_metrics = {
            'searches_executed': 0,
            'total_search_time': 0,
            'average_search_time': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def add_instance(self, name: str, host: str, port: int = 8089,
                    username: str = None, password: str = None,
                    token: str = None, is_cloud: bool = False) -> bool:
        """Add a Splunk instance."""
        try:
            instance = SplunkInstance(
                name=name,
                host=host,
                port=port,
                username=username,
                password=password,
                token=token,
                is_cloud=is_cloud
            )
            
            # Test connection
            if instance.connect():
                self.instances[name] = instance
                if not self.active_instance:
                    self.active_instance = name
                print(f"✅ Added Splunk instance: {name}")
                return True
            else:
                print(f"❌ Failed to add instance: {name}")
                return False
                
        except Exception as e:
            print(f"❌ Error adding instance {name}: {e}")
            return False
    
    def connect_instance(self, name: str) -> bool:
        """Connect to a specific Splunk instance."""
        if name not in self.instances:
            print(f"❌ Instance not found: {name}")
            return False
        
        instance = self.instances[name]
        if instance.connect():
            self.active_instance = name
            return True
        return False
    
    def get_active_instance(self) -> Optional[SplunkInstance]:
        """Get the currently active Splunk instance."""
        if not self.active_instance or self.active_instance not in self.instances:
            return None
        return self.instances[self.active_instance]
    
    def list_instances(self) -> List[Dict[str, Any]]:
        """List all configured Splunk instances."""
        instance_list = []
        for name, instance in self.instances.items():
            info = instance.get_connection_info()
            info['is_active'] = (name == self.active_instance)
            instance_list.append(info)
        return instance_list
    
    def execute_search(self, spl_query: str, instance_name: str = None,
                      earliest_time: str = "-24h", latest_time: str = "now",
                      max_results: int = 1000, output_mode: str = "json",
                      cache_results: bool = True) -> Dict[str, Any]:
        """Execute a Splunk search query."""
        try:
            start_time = time.time()
            
            # Determine instance to use
            if instance_name and instance_name in self.instances:
                instance = self.instances[instance_name]
            else:
                instance = self.get_active_instance()
            
            if not instance or not instance.is_connected():
                return {'error': 'No connected Splunk instance available', 'success': False}
            
            # Check cache first
            cache_key = f"{instance.name}:{spl_query}:{earliest_time}:{latest_time}:{max_results}"
            if cache_results and cache_key in self.search_cache:
                self.performance_metrics['cache_hits'] += 1
                cached_result = self.search_cache[cache_key]
                cached_result['from_cache'] = True
                return cached_result
            
            self.performance_metrics['cache_misses'] += 1
            
            # Log search start
            if self.session_manager:
                self.session_manager.log_workflow_step(
                    "splunk_search",
                    {"query": spl_query, "instance": instance.name},
                    inputs={"query": spl_query, "instance": instance.name, "time_range": f"{earliest_time} to {latest_time}"},
                    status="started"
                )
            
            # Execute search
            print(f"🔍 Executing Splunk search on {instance.name}...")
            print(f"   Query: {spl_query}")
            print(f"   Time range: {earliest_time} to {latest_time}")
            
            # Create search job
            search_query = f"search {spl_query} | head {max_results}"
            job = instance.service.jobs.create(
                search_query,
                earliest_time=earliest_time,
                latest_time=latest_time,
                output_mode=output_mode
            )
            
            # Wait for completion
            while not job.is_done():
                time.sleep(1)
            
            # Get results
            if output_mode == "json":
                result_stream = job.results()
                search_results = list(result_stream)
            else:
                # For other output modes, get raw results
                result_stream = job.results()
                search_results = list(result_stream)
            
            # Process results
            processed_results = self._process_search_results(search_results, output_mode)
            
            # Create result object
            search_result = {
                'success': True,
                'query': spl_query,
                'instance': instance.name,
                'time_range': f"{earliest_time} to {latest_time}",
                'results_count': len(processed_results),
                'results': processed_results,
                'search_metadata': {
                    'job_id': job.sid,
                    'execution_time': time.time() - start_time,
                    'timestamp': datetime.now().isoformat(),
                    'output_mode': output_mode
                }
            }
            
            # Cache results if requested
            if cache_results:
                self.search_cache[cache_key] = search_result
            
            # Update performance metrics
            self.performance_metrics['searches_executed'] += 1
            self.performance_metrics['total_search_time'] += search_result['search_metadata']['execution_time']
            self.performance_metrics['average_search_time'] = (
                self.performance_metrics['total_search_time'] / self.performance_metrics['searches_executed']
            )
            
            # Add to search history
            self.search_history.append({
                'timestamp': datetime.now().isoformat(),
                'query': spl_query,
                'instance': instance.name,
                'execution_time': search_result['search_metadata']['execution_time'],
                'results_count': len(processed_results)
            })
            
            # Log search completion
            if self.session_manager:
                self.session_manager.log_workflow_step(
                    "splunk_search",
                    {"query": spl_query, "instance": instance.name},
                    outputs=search_result,
                    duration=search_result['search_metadata']['execution_time'],
                    status="completed"
                )
            
            print(f"✅ Search completed in {search_result['search_metadata']['execution_time']:.2f}s")
            print(f"   Results: {len(processed_results)}")
            
            return search_result
            
        except Exception as e:
            error_msg = f"Search execution failed: {e}"
            print(f"❌ {error_msg}")
            
            if self.session_manager:
                self.session_manager.log_error('splunk_search', error_msg, str(e), {'query': spl_query})
            
            return {'error': error_msg, 'success': False}
    
    def _process_search_results(self, raw_results: List, output_mode: str) -> List[Dict[str, Any]]:
        """Process raw search results into structured format."""
        processed_results = []
        
        try:
            if output_mode == "json":
                for result in raw_results:
                    if hasattr(result, 'content'):
                        # Parse JSON content
                        try:
                            content = json.loads(result.content)
                            processed_results.append(content)
                        except json.JSONDecodeError:
                            # Fallback to raw content
                            processed_results.append({'raw_content': str(result.content)})
                    else:
                        processed_results.append({'raw_result': str(result)})
            else:
                # For other output modes, store as-is
                for result in raw_results:
                    processed_results.append({'raw_result': str(result)})
            
        except Exception as e:
            print(f"⚠️  Warning: Error processing results: {e}")
            processed_results = [{'error': f'Processing failed: {e}'}]
        
        return processed_results
    
    def get_scheduled_jobs(self, instance_name: str = None) -> Dict[str, Any]:
        """Get list of scheduled jobs from Splunk instance."""
        try:
            instance = self._get_instance(instance_name)
            if not instance:
                return {'error': 'No connected Splunk instance available', 'success': False}
            
            # Get scheduled jobs
            jobs = instance.service.saved_searches.list()
            
            scheduled_jobs = []
            for job in jobs:
                if hasattr(job, 'content') and job.content.get('is_scheduled'):
                    scheduled_jobs.append({
                        'name': job.name,
                        'search': job.content.get('search', ''),
                        'cron_schedule': job.content.get('cron_schedule', ''),
                        'dispatch_time': job.content.get('dispatch_time', ''),
                        'next_run': job.content.get('next_run', ''),
                        'enabled': job.content.get('disabled', '0') == '0'
                    })
            
            return {
                'success': True,
                'instance': instance.name,
                'scheduled_jobs_count': len(scheduled_jobs),
                'scheduled_jobs': scheduled_jobs
            }
            
        except Exception as e:
            error_msg = f"Failed to get scheduled jobs: {e}"
            print(f"❌ {error_msg}")
            return {'error': error_msg, 'success': False}
    
    def create_scheduled_job(self, job_name: str, search_query: str, cron_schedule: str,
                           instance_name: str = None, enabled: bool = True) -> Dict[str, Any]:
        """Create a new scheduled job on Splunk instance."""
        try:
            instance = self._get_instance(instance_name)
            if not instance:
                return {'error': 'No connected Splunk instance available', 'success': False}
            
            # Create saved search with scheduling
            saved_search = instance.service.saved_searches.create(
                name=job_name,
                search=search_query,
                cron_schedule=cron_schedule,
                disabled=not enabled
            )
            
            return {
                'success': True,
                'instance': instance.name,
                'job_name': job_name,
                'message': f'Scheduled job "{job_name}" created successfully'
            }
            
        except Exception as e:
            error_msg = f"Failed to create scheduled job: {e}"
            print(f"❌ {error_msg}")
            return {'error': error_msg, 'success': False}
    
    def copy_scheduled_jobs(self, source_instance: str, target_instance: str,
                           job_names: List[str] = None) -> Dict[str, Any]:
        """Copy scheduled jobs from one instance to another."""
        try:
            if source_instance not in self.instances or target_instance not in self.instances:
                return {'error': 'Source or target instance not found', 'success': False}
            
            source = self.instances[source_instance]
            target = self.instances[target_instance]
            
            if not source.is_connected() or not target.is_connected():
                return {'error': 'Source or target instance not connected', 'success': False}
            
            # Get source jobs
            source_jobs = self.get_scheduled_jobs(source_instance)
            if not source_jobs.get('success'):
                return source_jobs
            
            # Filter jobs if specific names provided
            if job_names:
                source_jobs['scheduled_jobs'] = [
                    job for job in source_jobs['scheduled_jobs'] 
                    if job['name'] in job_names
                ]
            
            # Copy jobs to target
            copied_jobs = []
            failed_jobs = []
            
            for job in source_jobs['scheduled_jobs']:
                try:
                    result = self.create_scheduled_job(
                        job_name=job['name'],
                        search_query=job['search'],
                        cron_schedule=job['cron_schedule'],
                        instance_name=target_instance,
                        enabled=job['enabled']
                    )
                    
                    if result.get('success'):
                        copied_jobs.append(job['name'])
                    else:
                        failed_jobs.append({'name': job['name'], 'error': result.get('error')})
                        
                except Exception as e:
                    failed_jobs.append({'name': job['name'], 'error': str(e)})
            
            return {
                'success': True,
                'source_instance': source_instance,
                'target_instance': target_instance,
                'total_jobs': len(source_jobs['scheduled_jobs']),
                'copied_jobs': copied_jobs,
                'failed_jobs': failed_jobs,
                'message': f'Copied {len(copied_jobs)} jobs from {source_instance} to {target_instance}'
            }
            
        except Exception as e:
            error_msg = f"Failed to copy scheduled jobs: {e}"
            print(f"❌ {error_msg}")
            return {'error': error_msg, 'success': False}
    
    def discover_data_sources(self, instance_name: str = None, 
                            cache_results: bool = True) -> Dict[str, Any]:
        """Discover data sources, indexes, and sourcetypes."""
        try:
            instance = self._get_instance(instance_name)
            if not instance:
                return {'error': 'No connected Splunk instance available', 'success': False}
            
            # Check cache
            cache_key = f"data_discovery:{instance.name}"
            if cache_results and cache_key in self.discovery_cache:
                return self.discovery_cache[cache_key]
            
            print(f"🔍 Discovering data sources on {instance.name}...")
            
            # Get indexes
            indexes = []
            try:
                index_list = instance.service.indexes.list()
                for index in index_list:
                    if hasattr(index, 'content'):
                        indexes.append({
                            'name': index.name,
                            'total_size': index.content.get('total_size', 'Unknown'),
                            'max_size': index.content.get('max_size', 'Unknown'),
                            'status': index.content.get('status', 'Unknown')
                        })
            except Exception as e:
                print(f"⚠️  Warning: Could not get indexes: {e}")
            
            # Get sourcetypes
            sourcetypes = []
            try:
                sourcetype_list = instance.service.sourcetypes.list()
                for st in sourcetype_list:
                    if hasattr(st, 'content'):
                        sourcetypes.append({
                            'name': st.name,
                            'description': st.content.get('description', ''),
                            'category': st.content.get('category', '')
                        })
            except Exception as e:
                print(f"⚠️  Warning: Could not get sourcetypes: {e}")
            
            # Get apps
            apps = []
            try:
                app_list = instance.service.apps.list()
                for app in app_list:
                    if hasattr(app, 'content'):
                        apps.append({
                            'name': app.name,
                            'version': app.content.get('version', ''),
                            'description': app.content.get('description', ''),
                            'author': app.content.get('author', '')
                        })
            except Exception as e:
                print(f"⚠️  Warning: Could not get apps: {e}")
            
            # Compile discovery results
            discovery_results = {
                'success': True,
                'instance': instance.name,
                'timestamp': datetime.now().isoformat(),
                'indexes': {
                    'count': len(indexes),
                    'list': indexes
                },
                'sourcetypes': {
                    'count': len(sourcetypes),
                    'list': sourcetypes
                },
                'apps': {
                    'count': len(apps),
                    'list': apps
                }
            }
            
            # Cache results
            if cache_results:
                self.discovery_cache[cache_key] = discovery_results
            
            print(f"✅ Data discovery completed:")
            print(f"   Indexes: {len(indexes)}")
            print(f"   Sourcetypes: {len(sourcetypes)}")
            print(f"   Apps: {len(apps)}")
            
            return discovery_results
            
        except Exception as e:
            error_msg = f"Data discovery failed: {e}"
            print(f"❌ {error_msg}")
            return {'error': error_msg, 'success': False}
    
    def analyze_data_flow(self, index_name: str, sourcetype: str = None,
                          time_range: str = "-7d", instance_name: str = None) -> Dict[str, Any]:
        """Analyze data flow and volume for specific index/sourcetype."""
        try:
            instance = self._get_instance(instance_name)
            if not instance:
                return {'error': 'No connected Splunk instance available', 'success': False}
            
            print(f"📊 Analyzing data flow for index: {index_name}")
            if sourcetype:
                print(f"   Sourcetype: {sourcetype}")
            print(f"   Time range: {time_range}")
            
            # Build search query
            if sourcetype:
                search_query = f"index={index_name} sourcetype={sourcetype}"
            else:
                search_query = f"index={index_name}"
            
            # Add time-based analysis
            search_query += f" | stats count by _time | timechart span=1h count"
            
            # Execute search
            result = self.execute_search(
                spl_query=search_query,
                instance_name=instance.name,
                earliest_time=time_range,
                latest_time="now",
                max_results=10000
            )
            
            if not result.get('success'):
                return result
            
            # Process results into DataFrame
            if result['results']:
                df = pd.DataFrame(result['results'])
                
                # Save to session if available
                if self.session_manager:
                    filename = f"data_flow_analysis_{index_name}_{sourcetype or 'all'}.csv"
                    file_path = self.session_manager.save_dataframe(
                        df, filename,
                        f"Data flow analysis for index {index_name}"
                    )
                    result['dataframe_file'] = file_path
                
                return {
                    'success': True,
                    'index': index_name,
                    'sourcetype': sourcetype,
                    'time_range': time_range,
                    'data_points': len(df),
                    'dataframe': df,
                    'summary': {
                        'total_events': df['count'].sum() if 'count' in df.columns else 0,
                        'average_events_per_hour': df['count'].mean() if 'count' in df.columns else 0,
                        'peak_hour': df.loc[df['count'].idxmax(), '_time'] if 'count' in df.columns and '_time' in df.columns else 'Unknown'
                    }
                }
            else:
                return {
                    'success': True,
                    'index': index_name,
                    'sourcetype': sourcetype,
                    'time_range': time_range,
                    'data_points': 0,
                    'message': 'No data found for the specified criteria'
                }
            
        except Exception as e:
            error_msg = f"Data flow analysis failed: {e}"
            print(f"❌ {error_msg}")
            return {'error': error_msg, 'success': False}
    
    def compare_instances(self, instance1: str, instance2: str) -> Dict[str, Any]:
        """Compare two Splunk instances for differences."""
        try:
            if instance1 not in self.instances or instance2 not in self.instances:
                return {'error': 'One or both instances not found', 'success': False}
            
            print(f"🔍 Comparing Splunk instances: {instance1} vs {instance2}")
            
            # Get data discovery for both instances
            discovery1 = self.discover_data_sources(instance1, cache_results=False)
            discovery2 = self.discover_data_sources(instance2, cache_results=False)
            
            if not discovery1.get('success') or not discovery2.get('success'):
                return {'error': 'Failed to get data discovery for one or both instances', 'success': False}
            
            # Compare indexes
            indexes1 = set(idx['name'] for idx in discovery1['indexes']['list'])
            indexes2 = set(idx['name'] for idx in discovery2['indexes']['list'])
            
            index_diff = {
                'only_in_instance1': list(indexes1 - indexes2),
                'only_in_instance2': list(indexes2 - indexes1),
                'common': list(indexes1 & indexes2)
            }
            
            # Compare sourcetypes
            sourcetypes1 = set(st['name'] for st in discovery1['sourcetypes']['list'])
            sourcetypes2 = set(st['name'] for st in discovery2['sourcetypes']['list'])
            
            sourcetype_diff = {
                'only_in_instance1': list(sourcetypes1 - sourcetypes2),
                'only_in_instance2': list(sourcetypes2 - sourcetypes1),
                'common': list(sourcetypes1 & sourcetypes2)
            }
            
            # Compare apps
            apps1 = set(app['name'] for app in discovery1['apps']['list'])
            apps2 = set(app['name'] for app in discovery2['apps']['list'])
            
            app_diff = {
                'only_in_instance1': list(apps1 - apps2),
                'only_in_instance2': list(apps2 - apps1),
                'common': list(apps1 & apps2)
            }
            
            comparison_results = {
                'success': True,
                'instance1': instance1,
                'instance2': instance2,
                'timestamp': datetime.now().isoformat(),
                'indexes': {
                    'instance1_count': len(indexes1),
                    'instance2_count': len(indexes2),
                    'differences': index_diff
                },
                'sourcetypes': {
                    'instance1_count': len(sourcetypes1),
                    'instance2_count': len(sourcetypes2),
                    'differences': sourcetype_diff
                },
                'apps': {
                    'instance1_count': len(apps1),
                    'instance2_count': len(apps2),
                    'differences': app_diff
                }
            }
            
            # Save comparison to session if available
            if self.session_manager:
                # Create comparison summary DataFrame
                comparison_data = []
                
                # Index differences
                for idx in index_diff['only_in_instance1']:
                    comparison_data.append({
                        'category': 'index',
                        'item': idx,
                        'status': f'only_in_{instance1}',
                        'instance1': 'Yes',
                        'instance2': 'No'
                    })
                
                for idx in index_diff['only_in_instance2']:
                    comparison_data.append({
                        'category': 'index',
                        'item': idx,
                        'status': f'only_in_{instance2}',
                        'instance1': 'No',
                        'instance2': 'Yes'
                    })
                
                # Sourcetype differences
                for st in sourcetype_diff['only_in_instance1']:
                    comparison_data.append({
                        'category': 'sourcetype',
                        'item': st,
                        'status': f'only_in_{instance1}',
                        'instance1': 'Yes',
                        'instance2': 'No'
                    })
                
                for st in sourcetype_diff['only_in_instance2']:
                    comparison_data.append({
                        'category': 'sourcetype',
                        'item': st,
                        'status': f'only_in_{instance2}',
                        'instance1': 'No',
                        'instance2': 'Yes'
                    })
                
                if comparison_data:
                    df_comparison = pd.DataFrame(comparison_data)
                    comparison_file = self.session_manager.save_dataframe(
                        df_comparison,
                        f"instance_comparison_{instance1}_vs_{instance2}.csv",
                        f"Comparison between {instance1} and {instance2}"
                    )
                    comparison_results['comparison_file'] = comparison_file
            
            print(f"✅ Instance comparison completed:")
            print(f"   Indexes: {len(index_diff['common'])} common, {len(index_diff['only_in_instance1'])} only in {instance1}, {len(index_diff['only_in_instance2'])} only in {instance2}")
            print(f"   Sourcetypes: {len(sourcetype_diff['common'])} common, {len(sourcetype_diff['only_in_instance1'])} only in {instance1}, {len(sourcetype_diff['only_in_instance2'])} only in {instance2}")
            
            return comparison_results
            
        except Exception as e:
            error_msg = f"Instance comparison failed: {e}"
            print(f"❌ {error_msg}")
            return {'error': error_msg, 'success': False}
    
    def _get_instance(self, instance_name: str = None) -> Optional[SplunkInstance]:
        """Get Splunk instance by name or active instance."""
        if instance_name and instance_name in self.instances:
            return self.instances[instance_name]
        return self.get_active_instance()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            'searches_executed': self.performance_metrics['searches_executed'],
            'total_search_time': self.performance_metrics['total_search_time'],
            'average_search_time': self.performance_metrics['average_search_time'],
            'cache_hits': self.performance_metrics['cache_hits'],
            'cache_misses': self.performance_metrics['cache_misses'],
            'cache_hit_rate': (
                self.performance_metrics['cache_hits'] / 
                (self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'])
                if (self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses']) > 0 else 0
            )
        }
    
    def cleanup(self):
        """Clean up resources."""
        try:
            for instance in self.instances.values():
                instance.disconnect()
            print("✅ Splunk integration cleaned up")
        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")

# Example usage
if __name__ == "__main__":
    # Test Splunk integration
    splunk = SplunkIntegration()
    
    print("🔍 Splunk Integration System")
    print("=" * 40)
    
    # Check capabilities
    print(f"Splunk SDK available: {SPLUNK_SDK_AVAILABLE}")
    
    if SPLUNK_SDK_AVAILABLE:
        print("✅ Splunk integration ready")
        
        # Example: Add instances (would need real credentials)
        # splunk.add_instance("onprem", "splunk-onprem.company.com", username="admin", password="password")
        # splunk.add_instance("cloud", "splunk-cloud.company.com", token="your-token", is_cloud=True)
        
        print("💡 To use Splunk integration:")
        print("   1. Add instances with add_instance()")
        print("   2. Connect to instances with connect_instance()")
        print("   3. Execute searches with execute_search()")
        print("   4. Discover data sources with discover_data_sources()")
        print("   5. Compare instances with compare_instances()")
    else:
        print("❌ Splunk SDK not available")
        print("   Install with: pip install splunk-sdk")
    
    print("\n✅ Splunk integration test completed!")
