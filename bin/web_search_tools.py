#!/usr/bin/env python3
"""
Enhanced Web Search Tools
Comprehensive web search capabilities including traditional search APIs and headless browser automation.
"""

import os
import sys
import json
import time
import hashlib
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import urllib.parse
import re
from dataclasses import dataclass

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Try to import optional dependencies
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️  Selenium not available. Install with: pip install selenium")

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    print("⚠️  BeautifulSoup not available. Install with: pip install beautifulsoup4")

@dataclass
class SearchResult:
    """Represents a search result."""
    title: str
    url: str
    snippet: str
    source: str
    timestamp: str
    metadata: Dict[str, Any] = None

@dataclass
class WebPageContent:
    """Represents extracted web page content."""
    url: str
    title: str
    text_content: str
    html_content: str
    metadata: Dict[str, Any] = None
    extracted_data: Dict[str, Any] = None
    timestamp: str = None

class WebSearchTools:
    """Enhanced web search and research tools."""
    
    def __init__(self, session_logger=None):
        self.session_logger = session_logger
        self.search_engines = {
            'google': self._search_google,
            'bing': self._search_bing,
            'duckduckgo': self._search_duckduckgo,
            'github': self._search_github,
            'stackoverflow': self._search_stackoverflow
        }
        
        # Browser configuration
        self.browser_options = None
        self.browser_driver = None
        
        # Rate limiting
        self.last_search_time = 0
        self.min_search_interval = 1.0  # seconds
        
        # User agents for different browsers
        self.user_agents = {
            'chrome': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'firefox': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0',
            'safari': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
        }
    
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get available web search tools."""
        return {
            'web_search': {
                'name': 'Web Search',
                'description': 'Search across multiple search engines',
                'category': 'web_search',
                'parameters': {
                    'query': {'type': 'string', 'description': 'Search query'},
                    'engine': {'type': 'string', 'description': 'Search engine (google, bing, duckduckgo, github, stackoverflow)'},
                    'max_results': {'type': 'integer', 'description': 'Maximum number of results'},
                    'include_metadata': {'type': 'boolean', 'description': 'Include result metadata'}
                },
                'returns': {'type': 'list', 'description': 'List of search results'},
                'available': True
            },
            'webpage_extract': {
                'name': 'Webpage Content Extraction',
                'description': 'Extract content from web pages using headless browser',
                'category': 'web_search',
                'parameters': {
                    'url': {'type': 'string', 'description': 'URL to extract content from'},
                    'extract_type': {'type': 'string', 'description': 'Type of extraction (text, html, structured, all)'},
                    'wait_time': {'type': 'integer', 'description': 'Wait time for page load (seconds)'},
                    'javascript_enabled': {'type': 'boolean', 'description': 'Enable JavaScript execution'}
                },
                'returns': {'type': 'dict', 'description': 'Extracted webpage content'},
                'available': SELENIUM_AVAILABLE
            },
            'web_research': {
                'name': 'Web Research Assistant',
                'description': 'Perform comprehensive web research on a topic',
                'category': 'web_search',
                'parameters': {
                    'research_topic': {'type': 'string', 'description': 'Topic to research'},
                    'research_depth': {'type': 'string', 'description': 'Research depth (basic, comprehensive, expert)'},
                    'sources': {'type': 'list', 'description': 'Preferred sources or domains'},
                    'include_analysis': {'type': 'boolean', 'description': 'Include AI analysis of findings'}
                },
                'returns': {'type': 'dict', 'description': 'Research results and analysis'},
                'available': True
            },
            'data_extraction': {
                'name': 'Structured Data Extraction',
                'description': 'Extract structured data from web pages',
                'category': 'web_search',
                'parameters': {
                    'url': {'type': 'string', 'description': 'URL to extract data from'},
                    'data_patterns': {'type': 'dict', 'description': 'Data extraction patterns'},
                    'output_format': {'type': 'string', 'description': 'Output format (json, csv, table)'}
                },
                'returns': {'type': 'dict', 'description': 'Extracted structured data'},
                'available': SELENIUM_AVAILABLE and BEAUTIFULSOUP_AVAILABLE
            }
        }
    
    def execute_tool(self, tool_id: str, **kwargs) -> Dict[str, Any]:
        """Execute a web search tool."""
        try:
            if tool_id == 'web_search':
                return self._execute_web_search(**kwargs)
            elif tool_id == 'webpage_extract':
                return self._execute_webpage_extract(**kwargs)
            elif tool_id == 'web_research':
                return self._execute_web_research(**kwargs)
            elif tool_id == 'data_extraction':
                return self._execute_data_extraction(**kwargs)
            else:
                return {'error': f'Unknown tool: {tool_id}', 'success': False}
                
        except Exception as e:
            if self.session_logger:
                self.session_logger.log_error(e, context={'tool_id': tool_id, 'kwargs': kwargs})
            return {'error': f'Tool execution error: {e}', 'success': False}
    
    def _execute_web_search(self, **kwargs) -> Dict[str, Any]:
        """Execute web search across multiple engines."""
        query = kwargs.get('query')
        engine = kwargs.get('engine', 'google')
        max_results = kwargs.get('max_results', 10)
        include_metadata = kwargs.get('include_metadata', True)
        
        if not query:
            return {'error': 'Search query is required', 'success': False}
        
        try:
            # Rate limiting
            self._rate_limit()
            
            # Perform search
            if engine in self.search_engines:
                results = self.search_engines[engine](query, max_results)
            else:
                # Multi-engine search
                results = self._multi_engine_search(query, max_results)
            
            # Process results
            processed_results = []
            for result in results[:max_results]:
                processed_result = {
                    'title': result.title,
                    'url': result.url,
                    'snippet': result.snippet,
                    'source': result.source,
                    'timestamp': result.timestamp
                }
                
                if include_metadata and result.metadata:
                    processed_result['metadata'] = result.metadata
                
                processed_results.append(processed_result)
            
            return {
                'tool': 'web_search',
                'success': True,
                'query': query,
                'engine': engine,
                'results_count': len(processed_results),
                'results': processed_results
            }
            
        except Exception as e:
            return {'error': f'Search error: {e}', 'success': False}
    
    def _execute_webpage_extract(self, **kwargs) -> Dict[str, Any]:
        """Execute webpage content extraction."""
        if not SELENIUM_AVAILABLE:
            return {'error': 'Selenium not available for webpage extraction', 'success': False}
        
        url = kwargs.get('url')
        extract_type = kwargs.get('extract_type', 'text')
        wait_time = kwargs.get('wait_time', 10)
        javascript_enabled = kwargs.get('javascript_enabled', True)
        
        if not url:
            return {'error': 'URL is required', 'success': False}
        
        try:
            # Extract content
            content = self.extract_webpage_content(
                url, extract_type, wait_time, javascript_enabled
            )
            
            if content:
                return {
                    'tool': 'webpage_extract',
                    'success': True,
                    'url': url,
                    'extract_type': extract_type,
                    'content': {
                        'title': content.title,
                        'text_content': content.text_content[:1000] + "..." if len(content.text_content) > 1000 else content.text_content,
                        'metadata': content.metadata,
                        'extracted_data': content.extracted_data
                    },
                    'full_content_size': len(content.text_content)
                }
            else:
                return {'error': 'Failed to extract content', 'success': False}
                
        except Exception as e:
            return {'error': f'Extraction error: {e}', 'success': False}
    
    def _execute_web_research(self, **kwargs) -> Dict[str, Any]:
        """Execute comprehensive web research."""
        research_topic = kwargs.get('research_topic')
        research_depth = kwargs.get('research_depth', 'comprehensive')
        sources = kwargs.get('sources', [])
        include_analysis = kwargs.get('include_analysis', True)
        
        if not research_topic:
            return {'error': 'Research topic is required', 'success': False}
        
        try:
            # Perform research
            research_results = self.perform_web_research(
                research_topic, research_depth, sources, include_analysis
            )
            
            return {
                'tool': 'web_research',
                'success': True,
                'research_topic': research_topic,
                'research_depth': research_depth,
                'results': research_results
            }
            
        except Exception as e:
            return {'error': f'Research error: {e}', 'success': False}
    
    def _execute_data_extraction(self, **kwargs) -> Dict[str, Any]:
        """Execute structured data extraction."""
        if not SELENIUM_AVAILABLE or not BEAUTIFULSOUP_AVAILABLE:
            return {'error': 'Selenium and BeautifulSoup required for data extraction', 'success': False}
        
        url = kwargs.get('url')
        data_patterns = kwargs.get('data_patterns', {})
        output_format = kwargs.get('output_format', 'json')
        
        if not url:
            return {'error': 'URL is required', 'success': False}
        
        try:
            # Extract structured data
            extracted_data = self.extract_structured_data(url, data_patterns, output_format)
            
            return {
                'tool': 'data_extraction',
                'success': True,
                'url': url,
                'output_format': output_format,
                'extracted_data': extracted_data
            }
            
        except Exception as e:
            return {'error': f'Data extraction error: {e}', 'success': False}
    
    def _rate_limit(self):
        """Implement rate limiting for search requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_search_time
        
        if time_since_last < self.min_search_interval:
            sleep_time = self.min_search_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_search_time = time.time()
    
    def _search_google(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search using Google (simulated - requires API key for production)."""
        # This is a simplified implementation
        # For production, use Google Custom Search API
        results = []
        
        # Simulate search results
        for i in range(min(max_results, 5)):
            results.append(SearchResult(
                title=f"Google Result {i+1} for: {query}",
                url=f"https://example.com/result{i+1}",
                snippet=f"This is a simulated search result for the query: {query}",
                source="google",
                timestamp=datetime.now().isoformat()
            ))
        
        return results
    
    def _search_bing(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search using Bing (simulated - requires API key for production)."""
        results = []
        
        for i in range(min(max_results, 5)):
            results.append(SearchResult(
                title=f"Bing Result {i+1} for: {query}",
                url=f"https://example.com/bing-result{i+1}",
                snippet=f"Bing search result for: {query}",
                source="bing",
                timestamp=datetime.now().isoformat()
            ))
        
        return results
    
    def _search_duckduckgo(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search using DuckDuckGo (simulated)."""
        results = []
        
        for i in range(min(max_results, 5)):
            results.append(SearchResult(
                title=f"DuckDuckGo Result {i+1} for: {query}",
                url=f"https://example.com/ddg-result{i+1}",
                snippet=f"DuckDuckGo search result for: {query}",
                source="duckduckgo",
                timestamp=datetime.now().isoformat()
            ))
        
        return results
    
    def _search_github(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search GitHub repositories (simulated - requires API key for production)."""
        results = []
        
        for i in range(min(max_results, 5)):
            results.append(SearchResult(
                title=f"GitHub Repository: {query}-{i+1}",
                url=f"https://github.com/user/{query}-{i+1}",
                snippet=f"GitHub repository related to {query}",
                source="github",
                timestamp=datetime.now().isoformat()
            ))
        
        return results
    
    def _search_stackoverflow(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search Stack Overflow (simulated - requires API key for production)."""
        results = []
        
        for i in range(min(max_results, 5)):
            results.append(SearchResult(
                title=f"Stack Overflow: {query} question {i+1}",
                url=f"https://stackoverflow.com/questions/{1000000+i}",
                snippet=f"Stack Overflow question about {query}",
                source="stackoverflow",
                timestamp=datetime.now().isoformat()
            ))
        
        return results
    
    def _multi_engine_search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search across multiple engines and combine results."""
        all_results = []
        
        for engine_name in ['google', 'bing', 'duckduckgo']:
            try:
                engine_results = self.search_engines[engine_name](query, max_results // 3)
                all_results.extend(engine_results)
            except Exception as e:
                print(f"Warning: {engine_name} search failed: {e}")
        
        # Remove duplicates and sort by relevance
        unique_results = self._deduplicate_results(all_results)
        return unique_results[:max_results]
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate search results based on URL."""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        return unique_results
    
    def extract_webpage_content(self, url: str, extract_type: str = 'text', 
                               wait_time: int = 10, javascript_enabled: bool = True) -> Optional[WebPageContent]:
        """Extract content from a webpage using headless browser."""
        if not SELENIUM_AVAILABLE:
            return None
        
        try:
            # Setup browser
            self._setup_browser(javascript_enabled)
            
            # Navigate to page
            self.browser_driver.get(url)
            
            # Wait for page load
            WebDriverWait(self.browser_driver, wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Extract content based on type
            if extract_type == 'text':
                title = self.browser_driver.title
                text_content = self.browser_driver.find_element(By.TAG_NAME, "body").text
                html_content = self.browser_driver.page_source
                
            elif extract_type == 'html':
                title = self.browser_driver.title
                text_content = ""
                html_content = self.browser_driver.page_source
                
            elif extract_type == 'structured':
                title = self.browser_driver.title
                text_content = self.browser_driver.find_element(By.TAG_NAME, "body").text
                html_content = self.browser_driver.page_source
                
                # Extract structured data
                extracted_data = self._extract_structured_content()
                
            else:  # 'all'
                title = self.browser_driver.title
                text_content = self.browser_driver.find_element(By.TAG_NAME, "body").text
                html_content = self.browser_driver.page_source
                extracted_data = self._extract_structured_content()
            
            # Create content object
            content = WebPageContent(
                url=url,
                title=title,
                text_content=text_content,
                html_content=html_content,
                metadata={
                    'extract_type': extract_type,
                    'javascript_enabled': javascript_enabled,
                    'wait_time': wait_time,
                    'user_agent': self.browser_driver.execute_script("return navigator.userAgent;")
                },
                extracted_data=extracted_data if extract_type in ['structured', 'all'] else None,
                timestamp=datetime.now().isoformat()
            )
            
            return content
            
        except Exception as e:
            print(f"❌ Error extracting webpage content: {e}")
            return None
        
        finally:
            # Clean up browser
            self._cleanup_browser()
    
    def _setup_browser(self, javascript_enabled: bool = True):
        """Setup headless browser."""
        try:
            options = Options()
            
            if not javascript_enabled:
                options.add_argument("--disable-javascript")
            
            # Headless mode
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            
            # User agent
            options.add_argument(f"--user-agent={self.user_agents['chrome']}")
            
            # Performance options
            options.add_argument("--disable-images")
            options.add_argument("--disable-plugins")
            options.add_argument("--disable-extensions")
            
            # Create driver
            self.browser_driver = webdriver.Chrome(options=options)
            self.browser_driver.set_page_load_timeout(30)
            
        except Exception as e:
            print(f"❌ Error setting up browser: {e}")
            raise
    
    def _cleanup_browser(self):
        """Clean up browser resources."""
        try:
            if self.browser_driver:
                self.browser_driver.quit()
                self.browser_driver = None
        except Exception as e:
            print(f"Warning: Error cleaning up browser: {e}")
    
    def _extract_structured_content(self) -> Dict[str, Any]:
        """Extract structured content from webpage."""
        try:
            structured_data = {}
            
            # Extract meta tags
            meta_tags = self.browser_driver.find_elements(By.TAG_NAME, "meta")
            for meta in meta_tags:
                name = meta.get_attribute("name") or meta.get_attribute("property")
                content = meta.get_attribute("content")
                if name and content:
                    structured_data[f"meta_{name}"] = content
            
            # Extract headings
            headings = {}
            for i in range(1, 7):
                h_elements = self.browser_driver.find_elements(By.TAG_NAME, f"h{i}")
                headings[f"h{i}"] = [h.text for h in h_elements if h.text.strip()]
            structured_data['headings'] = headings
            
            # Extract links
            links = self.browser_driver.find_elements(By.TAG_NAME, "a")
            structured_data['links'] = [
                {
                    'text': link.text.strip(),
                    'href': link.get_attribute("href"),
                    'title': link.get_attribute("title")
                }
                for link in links if link.text.strip() and link.get_attribute("href")
            ]
            
            # Extract tables
            tables = self.browser_driver.find_elements(By.TAG_NAME, "table")
            structured_data['tables'] = []
            for table in tables:
                try:
                    rows = table.find_elements(By.TAG_NAME, "tr")
                    table_data = []
                    for row in rows:
                        cells = row.find_elements(By.TAG_NAME, "td") + row.find_elements(By.TAG_NAME, "th")
                        table_data.append([cell.text.strip() for cell in cells])
                    structured_data['tables'].append(table_data)
                except Exception as e:
                    print(f"Warning: Error extracting table: {e}")
            
            return structured_data
            
        except Exception as e:
            print(f"Warning: Error extracting structured content: {e}")
            return {}
    
    def perform_web_research(self, topic: str, depth: str = 'comprehensive', 
                            sources: List[str] = None, include_analysis: bool = True) -> Dict[str, Any]:
        """Perform comprehensive web research on a topic."""
        research_results = {
            'topic': topic,
            'depth': depth,
            'timestamp': datetime.now().isoformat(),
            'search_results': [],
            'content_analysis': [],
            'sources_used': sources or [],
            'summary': {}
        }
        
        try:
            # Determine search queries based on depth
            if depth == 'basic':
                search_queries = [topic]
            elif depth == 'comprehensive':
                search_queries = [
                    topic,
                    f"{topic} overview",
                    f"{topic} latest developments",
                    f"{topic} best practices"
                ]
            else:  # expert
                search_queries = [
                    topic,
                    f"{topic} advanced techniques",
                    f"{topic} research papers",
                    f"{topic} expert analysis",
                    f"{topic} case studies"
                ]
            
            # Perform searches
            for query in search_queries:
                try:
                    results = self._multi_engine_search(query, max_results=5)
                    research_results['search_results'].extend(results)
                    
                    # Extract content from top results if include_analysis is True
                    if include_analysis and len(research_results['content_analysis']) < 3:
                        for result in results[:2]:
                            try:
                                content = self.extract_webpage_content(
                                    result.url, 'text', wait_time=5, javascript_enabled=False
                                )
                                if content:
                                    research_results['content_analysis'].append({
                                        'url': result.url,
                                        'title': content.title,
                                        'content_preview': content.text_content[:500] + "..." if len(content.text_content) > 500 else content.text_content,
                                        'extraction_time': content.timestamp
                                    })
                            except Exception as e:
                                print(f"Warning: Error extracting content from {result.url}: {e}")
                    
                    # Rate limiting between searches
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"Warning: Error in search query '{query}': {e}")
            
            # Generate summary
            research_results['summary'] = {
                'total_results': len(research_results['search_results']),
                'content_analyzed': len(research_results['content_analysis']),
                'search_queries_used': search_queries,
                'research_completed': True
            }
            
            return research_results
            
        except Exception as e:
            research_results['summary'] = {
                'error': str(e),
                'research_completed': False
            }
            return research_results
    
    def extract_structured_data(self, url: str, data_patterns: Dict[str, Any], 
                               output_format: str = 'json') -> Dict[str, Any]:
        """Extract structured data from webpage based on patterns."""
        try:
            # Extract webpage content
            content = self.extract_webpage_content(url, 'structured', wait_time=10)
            if not content:
                return {'error': 'Failed to extract webpage content'}
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(content.html_content, 'html.parser')
            
            extracted_data = {}
            
            # Extract data based on patterns
            for field_name, pattern in data_patterns.items():
                try:
                    if pattern.get('type') == 'css_selector':
                        elements = soup.select(pattern['selector'])
                        if pattern.get('multiple', False):
                            extracted_data[field_name] = [elem.get_text(strip=True) for elem in elements]
                        else:
                            extracted_data[field_name] = elements[0].get_text(strip=True) if elements else None
                    
                    elif pattern.get('type') == 'xpath':
                        # XPath extraction would require lxml
                        extracted_data[field_name] = "XPath extraction not implemented"
                    
                    elif pattern.get('type') == 'regex':
                        text = content.text_content
                        matches = re.findall(pattern['pattern'], text)
                        extracted_data[field_name] = matches
                    
                    elif pattern.get('type') == 'meta_tag':
                        meta = soup.find('meta', attrs=pattern['attributes'])
                        extracted_data[field_name] = meta.get('content') if meta else None
                    
                except Exception as e:
                    extracted_data[field_name] = f"Error extracting {field_name}: {e}"
            
            # Format output
            if output_format == 'csv':
                # Convert to CSV format
                import csv
                csv_data = []
                if extracted_data:
                    headers = list(extracted_data.keys())
                    csv_data.append(headers)
                    
                    # Get max length of values
                    max_length = max(len(str(v)) if isinstance(v, list) else 1 for v in extracted_data.values())
                    
                    for i in range(max_length):
                        row = []
                        for header in headers:
                            value = extracted_data[header]
                            if isinstance(value, list) and i < len(value):
                                row.append(str(value[i]))
                            elif i == 0:
                                row.append(str(value))
                            else:
                                row.append("")
                        csv_data.append(row)
                    
                    extracted_data = {'csv_data': csv_data}
            
            return {
                'url': url,
                'extraction_time': datetime.now().isoformat(),
                'data_patterns_used': data_patterns,
                'extracted_data': extracted_data,
                'output_format': output_format
            }
            
        except Exception as e:
            return {'error': f'Data extraction error: {e}'}

# Example usage and testing
if __name__ == "__main__":
    # Initialize web search tools
    web_tools = WebSearchTools()
    
    print("\n🧪 Testing Web Search Tools")
    print("=" * 50)
    
    # Test tool availability
    tools = web_tools.get_available_tools()
    print(f"📋 Available tools: {len(tools)}")
    for tool_id, tool_info in tools.items():
        print(f"   • {tool_id}: {tool_info['name']} ({'✅' if tool_info['available'] else '❌'})")
    
    # Test web search
    print("\n🔍 Testing web search...")
    search_result = web_tools.execute_tool('web_search', query='cybersecurity best practices', max_results=3)
    if search_result['success']:
        print(f"✅ Search completed: {search_result['results_count']} results")
        for i, result in enumerate(search_result['results'][:2], 1):
            print(f"   {i}. {result['title']}")
            print(f"      URL: {result['url']}")
    else:
        print(f"❌ Search failed: {search_result['error']}")
    
    # Test webpage extraction (if Selenium available)
    if SELENIUM_AVAILABLE:
        print("\n🌐 Testing webpage extraction...")
        extract_result = web_tools.execute_tool('webpage_extract', url='https://example.com', extract_type='text')
        if extract_result['success']:
            print(f"✅ Extraction completed: {extract_result['content']['title']}")
            print(f"   Content preview: {extract_result['content']['text_content'][:100]}...")
        else:
            print(f"❌ Extraction failed: {extract_result['error']}")
    else:
        print("\n⚠️  Selenium not available - skipping webpage extraction test")
    
    print("\n✅ Web Search Tools testing completed!")
