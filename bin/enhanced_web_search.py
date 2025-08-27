#!/usr/bin/env python3
"""
Enhanced Web Search System
Combines traditional search APIs with Selenium for comprehensive web research.
"""

import os
import sys
import json
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from urllib.parse import urlparse, quote_plus
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
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

class EnhancedWebSearch:
    """Enhanced web search combining APIs and Selenium for comprehensive research."""
    
    def __init__(self, session_manager=None):
        self.session_manager = session_manager
        
        # Search API configurations
        self.search_apis = {
            'google': {
                'base_url': 'https://www.google.com/search',
                'params': {'q': '', 'num': 10}
            },
            'bing': {
                'base_url': 'https://www.bing.com/search',
                'params': {'q': '', 'count': 10}
            },
            'duckduckgo': {
                'base_url': 'https://duckduckgo.com/',
                'params': {'q': ''}
            }
        }
        
        # Selenium configuration
        self.selenium_driver = None
        self.selenium_options = None
        self._setup_selenium()
        
        # Search history and caching
        self.search_history = []
        self.content_cache = {}
        
        # User agents for different browsers
        self.user_agents = {
            'chrome': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'firefox': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7; rv:89.0) Gecko/20100101 Firefox/89.0',
            'safari': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        }
    
    def _setup_selenium(self):
        """Setup Selenium WebDriver with Chrome options."""
        if not SELENIUM_AVAILABLE:
            return
        
        try:
            # Chrome options for headless browsing
            self.selenium_options = Options()
            self.selenium_options.add_argument('--headless')
            self.selenium_options.add_argument('--no-sandbox')
            self.selenium_options.add_argument('--disable-dev-shm-usage')
            self.selenium_options.add_argument('--disable-gpu')
            self.selenium_options.add_argument('--window-size=1920,1080')
            self.selenium_options.add_argument('--user-agent=' + self.user_agents['chrome'])
            
            # Additional options for better performance
            self.selenium_options.add_argument('--disable-extensions')
            self.selenium_options.add_argument('--disable-plugins')
            self.selenium_options.add_argument('--disable-images')  # Faster loading
            self.selenium_options.add_argument('--disable-javascript')  # For basic content extraction
            
            print("✅ Selenium configured successfully")
            
        except Exception as e:
            print(f"⚠️  Selenium setup failed: {e}")
            self.selenium_options = None
    
    def _get_selenium_driver(self):
        """Get or create Selenium WebDriver."""
        if not SELENIUM_AVAILABLE or not self.selenium_options:
            return None
        
        try:
            if self.selenium_driver is None:
                self.selenium_driver = webdriver.Chrome(options=self.selenium_options)
                print("✅ Selenium WebDriver initialized")
            return self.selenium_driver
        except Exception as e:
            print(f"⚠️  Failed to create WebDriver: {e}")
            return None
    
    def search_web(self, query: str, search_engine: str = 'google', 
                   max_results: int = 10, use_selenium: bool = False) -> Dict[str, Any]:
        """Perform web search using specified engine."""
        try:
            start_time = time.time()
            
            # Log search start
            if self.session_manager:
                self.session_manager.log_workflow_step(
                    "web_search",
                    {"query": query, "search_engine": search_engine, "use_selenium": use_selenium},
                    inputs={"query": query, "search_engine": search_engine},
                    status="started"
                )
            
            # Check cache first
            cache_key = f"{search_engine}:{query}:{max_results}"
            if cache_key in self.content_cache:
                print(f"📋 Using cached results for: {query}")
                return self.content_cache[cache_key]
            
            if use_selenium and self._get_selenium_driver():
                results = self._search_with_selenium(query, search_engine, max_results)
            else:
                results = self._search_with_api(query, search_engine, max_results)
            
            # Add metadata
            results['search_metadata'] = {
                'query': query,
                'search_engine': search_engine,
                'use_selenium': use_selenium,
                'max_results': max_results,
                'search_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache results
            self.content_cache[cache_key] = results
            
            # Log search completion
            if self.session_manager:
                self.session_manager.log_workflow_step(
                    "web_search",
                    {"query": query, "search_engine": search_engine, "use_selenium": use_selenium},
                    outputs=results,
                    duration=time.time() - start_time,
                    status="completed"
                )
            
            return results
            
        except Exception as e:
            error_msg = f"Search failed: {e}"
            print(f"❌ {error_msg}")
            
            if self.session_manager:
                self.session_manager.log_error('web_search', error_msg, str(e), {'query': query, 'search_engine': search_engine})
            
            return {'error': error_msg, 'results': [], 'success': False}
    
    def _search_with_api(self, query: str, search_engine: str, max_results: int) -> Dict[str, Any]:
        """Perform search using traditional HTTP requests."""
        try:
            if search_engine not in self.search_apis:
                return {'error': f'Unsupported search engine: {search_engine}', 'results': []}
            
            api_config = self.search_apis[search_engine]
            params = api_config['params'].copy()
            params['q'] = query
            
            if 'num' in params:
                params['num'] = max_results
            elif 'count' in params:
                params['count'] = max_results
            
            # Perform search request
            headers = {'User-Agent': self.user_agents['chrome']}
            response = requests.get(api_config['base_url'], params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse results based on search engine
            if search_engine == 'google':
                results = self._parse_google_results(response.text, max_results)
            elif search_engine == 'bing':
                results = self._parse_bing_results(response.text, max_results)
            elif search_engine == 'duckduckgo':
                results = self._parse_duckduckgo_results(response.text, max_results)
            else:
                results = []
            
            return {
                'success': True,
                'results': results,
                'total_results': len(results),
                'search_method': 'api'
            }
            
        except Exception as e:
            return {'error': f'API search failed: {e}', 'results': [], 'success': False}
    
    def _search_with_selenium(self, query: str, search_engine: str, max_results: int) -> Dict[str, Any]:
        """Perform search using Selenium WebDriver."""
        try:
            driver = self._get_selenium_driver()
            if not driver:
                return {'error': 'Selenium WebDriver not available', 'results': []}
            
            # Navigate to search engine
            if search_engine == 'google':
                search_url = f"https://www.google.com/search?q={quote_plus(query)}&num={max_results}"
            elif search_engine == 'bing':
                search_url = f"https://www.bing.com/search?q={quote_plus(query)}&count={max_results}"
            elif search_engine == 'duckduckgo':
                search_url = f"https://duckduckgo.com/?q={quote_plus(query)}"
            else:
                return {'error': f'Unsupported search engine for Selenium: {search_engine}', 'results': []}
            
            print(f"🔍 Navigating to {search_engine} search...")
            driver.get(search_url)
            
            # Wait for results to load
            wait = WebDriverWait(driver, 10)
            if search_engine == 'google':
                wait.until(EC.presence_of_element_located((By.ID, "search")))
            elif search_engine == 'bing':
                wait.until(EC.presence_of_element_located((By.ID, "b_results")))
            elif search_engine == 'duckduckgo':
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "results")))
            
            # Extract search results
            results = self._extract_selenium_results(driver, search_engine, max_results)
            
            return {
                'success': True,
                'results': results,
                'total_results': len(results),
                'search_method': 'selenium'
            }
            
        except Exception as e:
            return {'error': f'Selenium search failed: {e}', 'results': [], 'success': False}
    
    def _parse_google_results(self, html_content: str, max_results: int) -> List[Dict[str, Any]]:
        """Parse Google search results from HTML."""
        results = []
        
        if not BEAUTIFULSOUP_AVAILABLE:
            # Fallback to basic parsing
            return self._basic_html_parse(html_content, 'google')
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find search result containers
            search_results = soup.find_all('div', class_='g')
            
            for result in search_results[:max_results]:
                try:
                    title_elem = result.find('h3')
                    link_elem = result.find('a')
                    snippet_elem = result.find('div', class_='VwiC3b')
                    
                    if title_elem and link_elem:
                        title = title_elem.get_text(strip=True)
                        link = link_elem.get('href', '')
                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                        
                        results.append({
                            'title': title,
                            'url': link,
                            'snippet': snippet,
                            'source': 'google'
                        })
                except Exception as e:
                    continue
            
        except Exception as e:
            print(f"⚠️  Google parsing failed: {e}")
        
        return results
    
    def _parse_bing_results(self, html_content: str, max_results: int) -> List[Dict[str, Any]]:
        """Parse Bing search results from HTML."""
        results = []
        
        if not BEAUTIFULSOUP_AVAILABLE:
            return self._basic_html_parse(html_content, 'bing')
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find search result containers
            search_results = soup.find_all('li', class_='b_algo')
            
            for result in search_results[:max_results]:
                try:
                    title_elem = result.find('h2')
                    link_elem = result.find('a')
                    snippet_elem = result.find('p')
                    
                    if title_elem and link_elem:
                        title = title_elem.get_text(strip=True)
                        link = link_elem.get('href', '')
                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                        
                        results.append({
                            'title': title,
                            'url': link,
                            'snippet': snippet,
                            'source': 'bing'
                        })
                except Exception as e:
                    continue
            
        except Exception as e:
            print(f"⚠️  Bing parsing failed: {e}")
        
        return results
    
    def _parse_duckduckgo_results(self, html_content: str, max_results: int) -> List[Dict[str, Any]]:
        """Parse DuckDuckGo search results from HTML."""
        results = []
        
        if not BEAUTIFULSOUP_AVAILABLE:
            return self._basic_html_parse(html_content, 'duckduckgo')
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find search result containers
            search_results = soup.find_all('div', class_='result')
            
            for result in search_results[:max_results]:
                try:
                    title_elem = result.find('a', class_='result__a')
                    snippet_elem = result.find('div', class_='result__snippet')
                    
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        link = title_elem.get('href', '')
                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                        
                        results.append({
                            'title': title,
                            'url': link,
                            'snippet': snippet,
                            'source': 'duckduckgo'
                        })
                except Exception as e:
                    continue
            
        except Exception as e:
            print(f"⚠️  DuckDuckGo parsing failed: {e}")
        
        return results
    
    def _basic_html_parse(self, html_content: str, search_engine: str) -> List[Dict[str, Any]]:
        """Basic HTML parsing fallback."""
        results = []
        
        # Simple regex-based extraction
        import re
        
        # Extract URLs
        url_pattern = r'https?://[^\s<>"]+'
        urls = re.findall(url_pattern, html_content)
        
        # Extract titles (basic approach)
        title_pattern = r'<h[1-6][^>]*>([^<]+)</h[1-6]>'
        titles = re.findall(title_pattern, html_content)
        
        # Combine results
        for i, (url, title) in enumerate(zip(urls[:max_results], titles[:max_results])):
            results.append({
                'title': title.strip(),
                'url': url,
                'snippet': f'Result from {search_engine}',
                'source': search_engine
            })
        
        return results
    
    def _extract_selenium_results(self, driver, search_engine: str, max_results: int) -> List[Dict[str, Any]]:
        """Extract search results using Selenium."""
        results = []
        
        try:
            if search_engine == 'google':
                # Google results
                result_elements = driver.find_elements(By.CSS_SELECTOR, 'div.g')
                
                for result in result_elements[:max_results]:
                    try:
                        title_elem = result.find_element(By.CSS_SELECTOR, 'h3')
                        link_elem = result.find_element(By.CSS_SELECTOR, 'a')
                        
                        title = title_elem.text if title_elem else ''
                        link = link_elem.get_attribute('href') if link_elem else ''
                        
                        # Try to get snippet
                        try:
                            snippet_elem = result.find_element(By.CSS_SELECTOR, 'div.VwiC3b')
                            snippet = snippet_elem.text if snippet_elem else ''
                        except:
                            snippet = ''
                        
                        if title and link:
                            results.append({
                                'title': title,
                                'url': link,
                                'snippet': snippet,
                                'source': 'google_selenium'
                            })
                    except Exception as e:
                        continue
            
            elif search_engine == 'bing':
                # Bing results
                result_elements = driver.find_elements(By.CSS_SELECTOR, 'li.b_algo')
                
                for result in result_elements[:max_results]:
                    try:
                        title_elem = result.find_element(By.CSS_SELECTOR, 'h2 a')
                        snippet_elem = result.find_element(By.CSS_SELECTOR, 'p')
                        
                        title = title_elem.text if title_elem else ''
                        link = title_elem.get_attribute('href') if title_elem else ''
                        snippet = snippet_elem.text if snippet_elem else ''
                        
                        if title and link:
                            results.append({
                                'title': title,
                                'url': link,
                                'snippet': snippet,
                                'source': 'bing_selenium'
                            })
                    except Exception as e:
                        continue
            
        except Exception as e:
            print(f"⚠️  Selenium extraction failed: {e}")
        
        return results
    
    def extract_page_content(self, url: str, wait_time: int = 5, 
                           extract_method: str = 'selenium') -> Dict[str, Any]:
        """Extract content from a specific URL."""
        try:
            start_time = time.time()
            
            if self.session_manager:
                self.session_manager.log_workflow_step(
                    "page_content_extraction",
                    {"url": url, "extract_method": extract_method},
                    inputs={"url": url},
                    status="started"
                )
            
            if extract_method == 'selenium' and self._get_selenium_driver():
                content = self._extract_with_selenium(url, wait_time)
            else:
                content = self._extract_with_requests(url)
            
            # Add metadata
            content['extraction_metadata'] = {
                'url': url,
                'extract_method': extract_method,
                'extraction_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            # Log extraction completion
            if self.session_manager:
                self.session_manager.log_workflow_step(
                    "page_content_extraction",
                    {"url": url, "extract_method": extract_method},
                    outputs=content,
                    duration=time.time() - start_time,
                    status="completed"
                )
            
            return content
            
        except Exception as e:
            error_msg = f"Content extraction failed: {e}"
            print(f"❌ {error_msg}")
            
            if self.session_manager:
                self.session_manager.log_error('page_content_extraction', error_msg, str(e), {'url': url})
            
            return {'error': error_msg, 'success': False}
    
    def _extract_with_selenium(self, url: str, wait_time: int) -> Dict[str, Any]:
        """Extract page content using Selenium."""
        try:
            driver = self._get_selenium_driver()
            if not driver:
                return {'error': 'Selenium WebDriver not available', 'success': False}
            
            print(f"🌐 Loading page: {url}")
            driver.get(url)
            
            # Wait for page to load
            time.sleep(wait_time)
            
            # Get page information
            title = driver.title
            current_url = driver.current_url
            
            # Extract text content
            try:
                body = driver.find_element(By.TAG_NAME, 'body')
                text_content = body.text
            except:
                text_content = ""
            
            # Extract HTML content
            html_content = driver.page_source
            
            # Extract links
            try:
                links = driver.find_elements(By.TAG_NAME, 'a')
                link_urls = [link.get_attribute('href') for link in links if link.get_attribute('href')]
            except:
                link_urls = []
            
            return {
                'success': True,
                'title': title,
                'url': current_url,
                'text_content': text_content,
                'html_content': html_content,
                'links': link_urls,
                'extraction_method': 'selenium'
            }
            
        except Exception as e:
            return {'error': f'Selenium extraction failed: {e}', 'success': False}
    
    def _extract_with_requests(self, url: str) -> Dict[str, Any]:
        """Extract page content using requests."""
        try:
            headers = {'User-Agent': self.user_agents['chrome']}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            html_content = response.text
            
            if BEAUTIFULSOUP_AVAILABLE:
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Extract title
                title = soup.title.string if soup.title else ''
                
                # Extract text content
                text_content = soup.get_text(separator=' ', strip=True)
                
                # Extract links
                links = soup.find_all('a', href=True)
                link_urls = [link['href'] for link in links]
                
            else:
                # Basic extraction
                title = ''
                text_content = html_content
                link_urls = []
            
            return {
                'success': True,
                'title': title,
                'url': url,
                'text_content': text_content,
                'html_content': html_content,
                'links': link_urls,
                'extraction_method': 'requests'
            }
            
        except Exception as e:
            return {'error': f'Requests extraction failed: {e}', 'success': False}
    
    def research_topic(self, topic: str, search_engines: List[str] = None, 
                      max_results_per_engine: int = 5, extract_content: bool = True,
                      save_to_session: bool = True) -> Dict[str, Any]:
        """Comprehensive topic research across multiple search engines."""
        try:
            if search_engines is None:
                search_engines = ['google', 'bing']
            
            start_time = time.time()
            
            if self.session_manager:
                self.session_manager.log_workflow_step(
                    "topic_research",
                    {"topic": topic, "search_engines": search_engines},
                    inputs={"topic": topic},
                    status="started"
                )
            
            all_results = []
            content_extractions = []
            
            # Search across engines
            for engine in search_engines:
                print(f"🔍 Searching {engine} for: {topic}")
                search_results = self.search_web(
                    topic, 
                    search_engine=engine, 
                    max_results=max_results_per_engine,
                    use_selenium=True  # Use Selenium for better results
                )
                
                if search_results.get('success'):
                    all_results.extend(search_results['results'])
                    
                    # Extract content from top results if requested
                    if extract_content:
                        top_results = search_results['results'][:3]  # Top 3 results
                        for result in top_results:
                            if result.get('url'):
                                print(f"📄 Extracting content from: {result['title']}")
                                content = self.extract_page_content(result['url'])
                                if content.get('success'):
                                    content['search_result'] = result
                                    content_extractions.append(content)
                                
                                # Small delay between extractions
                                time.sleep(1)
            
            # Compile research results
            research_results = {
                'topic': topic,
                'search_engines_used': search_engines,
                'total_search_results': len(all_results),
                'search_results': all_results,
                'content_extractions': content_extractions,
                'research_metadata': {
                    'total_time': time.time() - start_time,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            # Save to session if requested
            if save_to_session and self.session_manager:
                # Save search results as DataFrame
                if all_results:
                    df_results = pd.DataFrame(all_results)
                    results_file = self.session_manager.save_dataframe(
                        df_results, 
                        f"research_results_{topic.replace(' ', '_')}.csv",
                        f"Research results for topic: {topic}"
                    )
                    research_results['results_file'] = results_file
                
                # Save content extractions
                if content_extractions:
                    content_data = []
                    for content in content_extractions:
                        if content.get('success'):
                            content_data.append({
                                'url': content.get('url', ''),
                                'title': content.get('title', ''),
                                'text_length': len(content.get('text_content', '')),
                                'links_count': len(content.get('links', [])),
                                'extraction_method': content.get('extraction_method', '')
                            })
                    
                    if content_data:
                        df_content = pd.DataFrame(content_data)
                        content_file = self.session_manager.save_dataframe(
                            df_content,
                            f"content_extractions_{topic.replace(' ', '_')}.csv",
                            f"Content extraction summary for topic: {topic}"
                        )
                        research_results['content_file'] = content_file
            
            # Log research completion
            if self.session_manager:
                self.session_manager.log_workflow_step(
                    "topic_research",
                    {"topic": topic, "search_engines": search_engines},
                    outputs=research_results,
                    duration=time.time() - start_time,
                    status="completed"
                )
            
            return research_results
            
        except Exception as e:
            error_msg = f"Topic research failed: {e}"
            print(f"❌ {error_msg}")
            
            if self.session_manager:
                self.session_manager.log_error('topic_research', error_msg, str(e), {'topic': topic})
            
            return {'error': error_msg, 'success': False}
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.selenium_driver:
                self.selenium_driver.quit()
                self.selenium_driver = None
                print("✅ Selenium WebDriver cleaned up")
        except Exception as e:
            print(f"⚠️  Error cleaning up Selenium: {e}")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get system capabilities."""
        return {
            'selenium_available': SELENIUM_AVAILABLE,
            'beautifulsoup_available': BEAUTIFULSOUP_AVAILABLE,
            'search_engines': list(self.search_apis.keys()),
            'extraction_methods': ['api', 'selenium', 'requests'],
            'features': [
                'multi-engine_search',
                'dynamic_content_extraction',
                'javascript_rendering',
                'content_parsing',
                'session_integration'
            ]
        }

# Example usage
if __name__ == "__main__":
    # Test enhanced web search
    web_search = EnhancedWebSearch()
    
    print("🔍 Enhanced Web Search System")
    print("=" * 40)
    
    # Check capabilities
    capabilities = web_search.get_capabilities()
    print(f"Capabilities: {capabilities}")
    
    # Test search
    print("\n🔍 Testing web search...")
    results = web_search.search_web("cybersecurity threats 2024", use_selenium=True)
    
    if results.get('success'):
        print(f"✅ Found {len(results['results'])} results")
        for i, result in enumerate(results['results'][:3], 1):
            print(f"  {i}. {result['title']}")
            print(f"     {result['url']}")
            print(f"     {result['snippet'][:100]}...")
    else:
        print(f"❌ Search failed: {results.get('error')}")
    
    # Test content extraction
    if results.get('success') and results['results']:
        print("\n📄 Testing content extraction...")
        first_result = results['results'][0]
        if first_result.get('url'):
            content = web_search.extract_page_content(first_result['url'])
            if content.get('success'):
                print(f"✅ Extracted content from: {content['title']}")
                print(f"   Text length: {len(content['text_content'])} characters")
                print(f"   Links found: {len(content['links'])}")
            else:
                print(f"❌ Content extraction failed: {content.get('error')}")
    
    # Cleanup
    web_search.cleanup()
    print("\n✅ Web search test completed!")
