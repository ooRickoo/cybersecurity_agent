#!/usr/bin/env python3
"""
US Patent Lookup Tools for LangGraph Cybersecurity Agent
Provides tools to fetch detailed patent information from external APIs.
"""

import os
import sys
import json
import csv
import time
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PatentData:
    """Data structure for patent information."""
    patent_number: str
    publication_number: str
    title: str
    abstract: str
    inventors: List[str]
    assignee: str
    filing_date: str
    publication_date: str
    patent_status: str
    claims_count: int
    classification: str
    legal_status: str
    api_source: str
    fetch_timestamp: str

class USPatentLookupTool:
    """Tool for fetching US patent information from external APIs."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Cybersecurity-Agent/1.0 (Patent Research Tool)'
        })
        
        # API endpoints (using public USPTO and Google Patents APIs)
        self.uspto_api_base = "https://developer.uspto.gov/ds-api"
        self.google_patents_base = "https://patents.google.com"
        
        # Rate limiting
        self.request_delay = 1.0  # seconds between requests
        self.max_retries = 3
        
    def fetch_patent_by_number(self, patent_number: str, publication_number: str = None) -> Optional[PatentData]:
        """
        Fetch patent details by patent number and optionally publication number.
        
        Args:
            patent_number: US patent number (e.g., "US10123456")
            publication_number: Publication number if available
            
        Returns:
            PatentData object or None if not found
        """
        try:
            logger.info(f"Fetching patent: {patent_number}")
            if publication_number:
                logger.info(f"Associated publication number: {publication_number}")
            
            # Try USPTO API first with patent number
            patent_data = self._fetch_from_uspto(patent_number)
            if patent_data:
                # If we have a publication number, try to enhance the data
                if publication_number and not patent_data.publication_number:
                    patent_data.publication_number = publication_number
                return patent_data
            
            # If USPTO failed and we have a publication number, try Google Patents with publication number
            if publication_number:
                logger.info(f"Trying Google Patents with publication number: {publication_number}")
                patent_data = self._fetch_from_google_patents_by_publication(publication_number)
                if patent_data:
                    # Ensure the patent number matches
                    if patent_data.patent_number != patent_number:
                        patent_data.patent_number = patent_number
                    return patent_data
            
            # Fallback to Google Patents with patent number
            patent_data = self._fetch_from_google_patents(patent_number, publication_number)
            if patent_data:
                return patent_data
            
            logger.warning(f"Patent {patent_number} not found in any source")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching patent {patent_number}: {e}")
            return None
    
    def _fetch_from_uspto(self, patent_number: str) -> Optional[PatentData]:
        """Fetch patent data from USPTO API."""
        try:
            # Clean patent number
            clean_number = patent_number.replace("US", "").replace("Patent", "").strip()
            
            # USPTO Patent Application Full-Text and Image Database
            url = f"{self.uspto_api_base}/patents/v1/patents/{clean_number}"
            
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_uspto_response(data, patent_number)
            else:
                logger.debug(f"USPTO API returned {response.status_code} for {patent_number}")
                return None
                
        except Exception as e:
            logger.debug(f"USPTO API error for {patent_number}: {e}")
            return None
    
    def _fetch_publication_details(self, publication_number: str) -> Optional[Dict]:
        """Fetch additional publication details including PDF links."""
        try:
            # Clean publication number
            clean_pub = publication_number.replace("US", "").replace("A1", "").replace("B1", "").replace("B2", "").strip()
            
            # Try USPTO Publication API
            url = f"{self.uspto_api_base}/publications/v1/publications/{publication_number}"
            
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_publication_response(data, publication_number)
            else:
                logger.debug(f"USPTO Publication API returned {response.status_code} for {publication_number}")
                
                # Fallback: Try to construct Google Patents PDF URL
                try:
                    # Google Patents PDF URL format: https://patents.google.com/patent/US{patent_number}/en.pdf
                    patent_num = publication_number.replace("US", "").replace("A1", "").replace("B1", "").replace("B2", "").strip()
                    google_pdf_url = f"https://patents.google.com/patent/US{patent_num}/en.pdf"
                    
                    # Test if PDF is accessible
                    pdf_response = self.session.head(google_pdf_url, timeout=10)
                    if pdf_response.status_code == 200:
                        return {
                            'publication_number': publication_number,
                            'pdf_url': google_pdf_url,
                            'api_source': 'Google_Patents_PDF',
                            'note': 'PDF available via Google Patents'
                        }
                except Exception as pdf_e:
                    logger.debug(f"Google Patents PDF fallback failed: {pdf_e}")
                
                return None
                
        except Exception as e:
            logger.debug(f"USPTO Publication API error for {publication_number}: {e}")
            return None
    
    def _fetch_from_google_patents(self, patent_number: str, publication_number: str = None) -> Optional[PatentData]:
        """Fetch patent data from Google Patents (web scraping fallback)."""
        try:
            # Clean patent number
            clean_number = patent_number.replace("US", "").replace("Patent", "").strip()
            
            # Google Patents URL
            url = f"{self.google_patents_base}/patent/US{clean_number}"
            
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                # Parse HTML response for key information
                return self._parse_google_patents_response(response.text, patent_number, publication_number)
            else:
                logger.debug(f"Google Patents returned {response.status_code} for {patent_number}")
                return None
                
        except Exception as e:
            logger.debug(f"Google Patents error for {patent_number}: {e}")
            return None
    
    def _fetch_from_google_patents_by_publication(self, publication_number: str) -> Optional[PatentData]:
        """Fetch patent data from Google Patents using publication number."""
        try:
            # Clean publication number
            clean_number = publication_number.replace("US", "").replace("A1", "").replace("B1", "").replace("B2", "").strip()
            
            # Google Patents URL with publication number
            url = f"{self.google_patents_base}/patent/{publication_number}"
            
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                # Parse HTML response for key information
                return self._parse_google_patents_response(response.text, f"US{clean_number}", publication_number)
            else:
                logger.debug(f"Google Patents returned {response.status_code} for publication {publication_number}")
                return None
                
        except Exception as e:
            logger.debug(f"Google Patents error for publication {publication_number}: {e}")
            return None
    
    def _parse_uspto_response(self, data: Dict, patent_number: str) -> PatentData:
        """Parse USPTO API response into PatentData."""
        try:
            # Extract data from USPTO response
            patent_info = data.get('patent', {})
            
            return PatentData(
                patent_number=patent_number,
                publication_number=patent_info.get('publicationNumber', ''),
                title=patent_info.get('inventionTitle', ''),
                abstract=patent_info.get('abstract', ''),
                inventors=[inv.get('name', '') for inv in patent_info.get('inventors', [])],
                assignee=patent_info.get('assignee', {}).get('name', ''),
                filing_date=patent_info.get('filingDate', ''),
                publication_date=patent_info.get('publicationDate', ''),
                patent_status=patent_info.get('patentStatus', ''),
                claims_count=len(patent_info.get('claims', [])),
                classification=patent_info.get('classification', ''),
                legal_status=patent_info.get('legalStatus', ''),
                api_source='USPTO',
                fetch_timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Error parsing USPTO response: {e}")
            return None
    
    def _parse_publication_response(self, data: Dict, publication_number: str) -> Dict:
        """Parse USPTO publication response for additional details."""
        try:
            pub_info = data.get('publication', {})
            
            return {
                'publication_number': publication_number,
                'title': pub_info.get('inventionTitle', ''),
                'abstract': pub_info.get('abstract', ''),
                'inventors': [inv.get('name', '') for inv in pub_info.get('inventors', [])],
                'assignee': pub_info.get('assignee', {}).get('name', ''),
                'filing_date': pub_info.get('filingDate', ''),
                'publication_date': pub_info.get('publicationDate', ''),
                'patent_status': pub_info.get('patentStatus', ''),
                'legal_status': pub_info.get('legalStatus', ''),
                'pdf_url': pub_info.get('pdfUrl', ''),
                'full_text_url': pub_info.get('fullTextUrl', ''),
                'claims': pub_info.get('claims', []),
                'classification': pub_info.get('classification', ''),
                'api_source': 'USPTO_Publication'
            }
        except Exception as e:
            logger.debug(f"Error parsing publication response: {e}")
            return {}
    
    def _parse_google_patents_response(self, html_content: str, patent_number: str, publication_number: str = None) -> PatentData:
        """Parse Google Patents HTML response into PatentData."""
        try:
            # This is a simplified parser - in production you'd want more robust HTML parsing
            # For now, we'll extract basic information using string methods
            
            # Extract title (look for title in meta tags or specific HTML elements)
            title = self._extract_from_html(html_content, 'title') or 'Title not available'
            
            # Extract abstract (look for abstract in meta description or specific divs)
            abstract = self._extract_from_html(html_content, 'abstract') or 'Abstract not available'
            
            # Extract inventors (look for inventor information)
            inventors = self._extract_inventors_from_html(html_content)
            
            # Extract filing date
            filing_date = self._extract_from_html(html_content, 'filing') or 'Date not available'
            
            # Extract publication date
            publication_date = self._extract_from_html(html_content, 'publication') or 'Date not available'
            
            # Use provided publication number if available
            final_publication_number = publication_number if publication_number else ''
            
            # Try to extract more fields
            assignee = self._extract_from_html(html_content, 'assignee') or 'Assignee not available'
            patent_status = self._extract_from_html(html_content, 'status') or 'Status not available'
            legal_status = self._extract_from_html(html_content, 'legal') or 'Legal status not available'
            classification = self._extract_from_html(html_content, 'classification') or 'Classification not available'
            
            return PatentData(
                patent_number=patent_number,
                publication_number=final_publication_number,
                title=title,
                abstract=abstract,
                inventors=inventors,
                assignee=assignee,
                filing_date=filing_date,
                publication_date=publication_date,
                patent_status=patent_status,
                claims_count=0,
                classification=classification,
                legal_status=legal_status,
                api_source='Google Patents',
                fetch_timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Error parsing Google Patents response: {e}")
            return None
    
    def _extract_from_html(self, html_content: str, field: str) -> str:
        """Extract specific field from HTML content."""
        try:
            # Look for common patterns in HTML
            if field == 'title':
                # Look for title in meta tags or title element
                if '<title>' in html_content:
                    start = html_content.find('<title>') + 7
                    end = html_content.find('</title>', start)
                    if start > 6 and end > start:
                        title = html_content[start:end].strip()
                        # Clean up title - remove extra whitespace and newlines
                        title = ' '.join(title.split())
                        return title
                
                # Look for meta description
                if 'name="description"' in html_content:
                    start = html_content.find('content="', html_content.find('name="description"')) + 9
                    end = html_content.find('"', start)
                    if start > 8 and end > start:
                        return html_content[start:end].strip()
                
                # Look for h1 or other heading elements
                if '<h1' in html_content:
                    start = html_content.find('<h1') + 4
                    start = html_content.find('>', start) + 1
                    end = html_content.find('</h1>', start)
                    if start > 0 and end > start:
                        title = html_content[start:end].strip()
                        title = ' '.join(title.split())
                        return title
            
            elif field == 'abstract':
                # Look for abstract in meta description
                if 'name="description"' in html_content:
                    start = html_content.find('content="', html_content.find('name="description"')) + 9
                    end = html_content.find('"', start)
                    if start > 8 and end > start:
                        return html_content[start:end].strip()
                
                # Look for abstract in specific divs or sections
                abstract_patterns = [
                    'abstract',
                    'summary',
                    'description'
                ]
                
                for pattern in abstract_patterns:
                    if pattern in html_content.lower():
                        start = html_content.lower().find(pattern)
                        if start > 0:
                            # Look for text after the pattern
                            text_after = html_content[start:start+1000]
                            # Find the next HTML tag or end of content
                            tag_end = text_after.find('<', 100)
                            if tag_end > 0:
                                abstract_text = text_after[100:tag_end].strip()
                                if len(abstract_text) > 20:  # Only return if we have substantial text
                                    return abstract_text[:500]  # Limit length
            
            elif field in ['filing', 'publication']:
                # Look for date patterns
                date_patterns = [
                    r'(\d{1,2}/\d{1,2}/\d{4})',  # MM/DD/YYYY
                    r'(\d{4}-\d{2}-\d{2})',      # YYYY-MM-DD
                    r'(\w+ \d{1,2}, \d{4})'      # Month DD, YYYY
                ]
                
                for pattern in date_patterns:
                    import re
                    match = re.search(pattern, html_content)
                    if match:
                        return match.group(1)
            
            elif field == 'assignee':
                # Look for assignee information
                assignee_patterns = [
                    'assignee',
                    'assignee:',
                    'assigned to',
                    'owner',
                    'owner:'
                ]
                
                for pattern in assignee_patterns:
                    if pattern in html_content.lower():
                        start = html_content.lower().find(pattern)
                        if start > 0:
                            text_after = html_content[start:start+300]
                            # Extract text after the pattern
                            tag_end = text_after.find('<', 50)
                            if tag_end > 0:
                                assignee_text = text_after[50:tag_end].strip()
                                if len(assignee_text) > 3:
                                    # Clean the extracted text
                                    cleaned_text = self._clean_html_text(assignee_text)
                                    if cleaned_text and len(cleaned_text) > 3:
                                        return cleaned_text[:100]
            
            elif field == 'status':
                # Look for patent status
                status_patterns = [
                    'patent status',
                    'status:',
                    'current status',
                    'patent state'
                ]
                
                for pattern in status_patterns:
                    if pattern in html_content.lower():
                        start = html_content.lower().find(pattern)
                        if start > 0:
                            text_after = html_content[start:start+200]
                            tag_end = text_after.find('<', 50)
                            if tag_end > 0:
                                status_text = text_after[50:tag_end].strip()
                                if len(status_text) > 3:
                                    # Clean the extracted text
                                    cleaned_text = self._clean_html_text(status_text)
                                    if cleaned_text and len(cleaned_text) > 3:
                                        return cleaned_text[:100]
            
            elif field == 'legal':
                # Look for legal status
                legal_patterns = [
                    'legal status',
                    'legal:',
                    'legal state',
                    'patent legal'
                ]
                
                for pattern in legal_patterns:
                    if pattern in html_content.lower():
                        start = html_content.lower().find(pattern)
                        if start > 0:
                            text_after = html_content[start:start+200]
                            tag_end = text_after.find('<', 50)
                            if tag_end > 0:
                                legal_text = text_after[50:tag_end].strip()
                                if len(legal_text) > 3:
                                    # Clean the extracted text
                                    cleaned_text = self._clean_html_text(legal_text)
                                    if cleaned_text and len(cleaned_text) > 3:
                                        return cleaned_text[:100]
            
            elif field == 'classification':
                # Look for patent classification
                class_patterns = [
                    'classification',
                    'class:',
                    'patent class',
                    'uspc',
                    'cpc'
                ]
                
                for pattern in class_patterns:
                    if pattern in html_content.lower():
                        start = html_content.lower().find(pattern)
                        if start > 0:
                            text_after = html_content[start:start+200]
                            tag_end = text_after.find('<', 50)
                            if tag_end > 0:
                                class_text = text_after[50:tag_end].strip()
                                if len(class_text) > 3:
                                    # Clean the extracted text
                                    cleaned_text = self._clean_html_text(class_text)
                                    if cleaned_text and len(cleaned_text) > 3:
                                        return cleaned_text[:100]
            
            return ''
        except Exception as e:
            logger.debug(f"Error extracting {field} from HTML: {e}")
            return ''
    
    def _clean_html_text(self, text: str) -> str:
        """Clean extracted HTML text by removing markup and cleaning up content."""
        try:
            if not text:
                return ""
            
            # Remove HTML tags
            import re
            
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            
            # Remove HTML entities
            text = re.sub(r'&[a-zA-Z]+;', '', text)
            text = re.sub(r'&#\d+;', '', text)
            
            # Remove CSS selectors and properties
            text = re.sub(r'[a-zA-Z-]+\s*:\s*[^;]+;?', '', text)
            text = re.sub(r'[a-zA-Z-]+\s*=\s*["\'][^"\']*["\']', '', text)
            
            # Remove common HTML attributes
            text = re.sub(r'\b(?:class|id|style|href|src|alt|title)\s*=\s*["\'][^"\']*["\']', '', text)
            
            # Remove JavaScript code
            text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
            text = re.sub(r'function\s*\([^)]*\)\s*\{[^}]*\}', '', text)
            
            # Remove CSS URLs and imports
            text = re.sub(r'url\s*\(\s*[^)]+\s*\)', '', text)
            text = re.sub(r'@import\s+[^;]+;?', '', text)
            
            # Remove extra whitespace and normalize
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            # Remove text that's clearly not meaningful
            if len(text) < 3:
                return ""
            
            # Remove text that's mostly punctuation or special characters
            if re.match(r'^[^\w\s]*$', text):
                return ""
            
            # Remove text that looks like CSS or HTML structure
            if re.match(r'^[<>{}[\]]+$', text):
                return ""
            
            return text
            
        except Exception as e:
            logger.debug(f"Error cleaning HTML text: {e}")
            return text if text else ""
    
    def _extract_inventors_from_html(self, html_content: str) -> List[str]:
        """Extract inventor names from HTML content."""
        try:
            inventors = []
            
            # Look for inventor patterns in HTML with more specific targeting
            inventor_patterns = [
                'inventor',
                'invented by',
                'inventors:',
                'inventor:',
                'inventor name',
                'inventor names'
            ]
            
            for pattern in inventor_patterns:
                if pattern in html_content.lower():
                    # Extract text around inventor information
                    start = html_content.lower().find(pattern)
                    if start > 0:
                        # Look for names after the pattern
                        text_after = html_content[start:start+1000]
                        
                        # Try to find actual names using various patterns
                        # Look for text that might be names (capitalized words, 2-3 words)
                        import re
                        
                        # Pattern for potential names (2-3 capitalized words)
                        name_pattern = r'\b([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
                        potential_names = re.findall(name_pattern, text_after)
                        
                        for name in potential_names:
                            # Filter out common non-name words
                            if not any(word.lower() in ['the', 'and', 'or', 'for', 'with', 'from', 'by', 'in', 'on', 'at', 'to'] for word in name.split()):
                                if len(name) > 5 and len(name) < 50:  # Reasonable name length
                                    inventors.append(name.strip())
                        
                        # If we found names, break
                        if inventors:
                            break
            
            # If no inventors found, try alternative approach
            if not inventors:
                # Look for structured data or JSON in the HTML
                if 'application/ld+json' in html_content:
                    try:
                        # Extract JSON-LD data which often contains inventor information
                        json_start = html_content.find('application/ld+json') + 20
                        json_end = html_content.find('</script>', json_start)
                        if json_end > json_start:
                            json_content = html_content[json_start:json_end].strip()
                            if json_content.startswith('{'):
                                import json
                                data = json.loads(json_content)
                                if 'inventor' in data:
                                    inventors.append(str(data['inventor']))
                    except:
                        pass
            
            if not inventors:
                inventors.append('Inventors not available')
            
            return inventors[:5]  # Limit to 5 inventors max
        except Exception as e:
            logger.debug(f"Error extracting inventors from HTML: {e}")
            return ['Inventors not available']
    
    def batch_fetch_patents(self, patent_list: List[Dict[str, str]]) -> List[PatentData]:
        """
        Fetch multiple patents in batch.
        
        Args:
            patent_list: List of dicts with 'patent_number' and optionally 'publication_number'
            
        Returns:
            List of PatentData objects
        """
        results = []
        
        for i, patent_info in enumerate(patent_list):
            patent_number = patent_info.get('patent_number')
            publication_number = patent_info.get('publication_number')
            
            if not patent_number:
                logger.warning(f"Skipping entry {i}: missing patent_number")
                continue
            
            logger.info(f"Processing patent {i+1}/{len(patent_list)}: {patent_number}")
            
            patent_data = self.fetch_patent_by_number(patent_number, publication_number)
            if patent_data:
                results.append(patent_data)
            else:
                # Create a placeholder entry for failed fetches
                placeholder = PatentData(
                    patent_number=patent_number,
                    publication_number=publication_number or '',
                    title='Fetch failed',
                    abstract='Patent information could not be retrieved',
                    inventors=['Unknown'],
                    assignee='Unknown',
                    filing_date='Unknown',
                    publication_date='Unknown',
                    patent_status='Unknown',
                    claims_count=0,
                    classification='Unknown',
                    legal_status='Unknown',
                    api_source='Failed',
                    fetch_timestamp=datetime.now().isoformat()
                )
                results.append(placeholder)
            
            # Rate limiting
            if i < len(patent_list) - 1:
                time.sleep(self.request_delay)
        
        return results
    
    def export_to_csv(self, patent_data_list: List[PatentData], output_path: str) -> str:
        """
        Export patent data to CSV file.
        
        Args:
            patent_data_list: List of PatentData objects
            output_path: Path to save CSV file
            
        Returns:
            Path to saved CSV file
        """
        try:
            # Convert to list of dictionaries for CSV export
            csv_data = []
            for patent in patent_data_list:
                csv_data.append({
                    'Patent Number': patent.patent_number,
                    'Publication Number': patent.publication_number,
                    'Title': patent.title,
                    'Abstract': patent.abstract,
                    'Inventors': '; '.join(patent.inventors),
                    'Assignee': patent.assignee,
                    'Filing Date': patent.filing_date,
                    'Publication Date': patent.publication_date,
                    'Patent Status': patent.patent_status,
                    'Claims Count': patent.claims_count,
                    'Classification': patent.classification,
                    'Legal Status': patent.legal_status,
                    'API Source': patent.api_source,
                    'Fetch Timestamp': patent.fetch_timestamp
                })
            
            # Create output directory if it doesn't exist
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write CSV
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = csv_data[0].keys() if csv_data else []
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            
            logger.info(f"Exported {len(patent_data_list)} patents to {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            raise

    async def export_enhanced_csv(self, patent_data_list: List[PatentData], output_path: str, llm_client=None) -> str:
        """
        Export patent data to enhanced CSV with LLM-generated insights.
        
        Args:
            patent_data_list: List of PatentData objects
            output_path: Path to save CSV file
            llm_client: LLM client for enhanced analysis (optional)
            
        Returns:
            Path to saved CSV file
        """
        try:
            # Convert to list of dictionaries for CSV export with enhanced columns
            csv_data = []
            for patent in patent_data_list:
                # Generate LLM insights if available
                key_value_prop = "Not analyzed"
                patent_category = "Not categorized"
                
                if llm_client and patent.title and patent.abstract:
                    try:
                        # Use standardized AI tools if available, otherwise fall back to direct OpenAI client
                        if hasattr(llm_client, 'call_tool') and hasattr(llm_client, 'get_tool'):
                            # Use standardized AI tools from cs_ai_tools.py
                            try:
                                # Call the patent analysis tool
                                analysis_result = llm_client.call_tool("ai_patent_analysis", {
                                    "patent_title": patent.title,
                                    "patent_abstract": patent.abstract,
                                    "analysis_type": "both",
                                    "model": "gpt-4",
                                    "temperature": 0.3
                                })
                                
                                if analysis_result.get("success"):
                                    key_value_prop = analysis_result.get("value_proposition", "Not analyzed")
                                    patent_category = analysis_result.get("category", "Not categorized")
                                else:
                                    key_value_prop = f"Analysis failed: {analysis_result.get('error', 'Unknown error')}"
                                    patent_category = "Categorization failed"
                                    
                            except Exception as tool_error:
                                logger.warning(f"Standardized AI tools failed for patent {patent.patent_number}: {tool_error}")
                                # Fall back to direct OpenAI client
                                key_value_prop, patent_category = await self._generate_llm_insights_direct(patent, llm_client)
                        else:
                            # Use direct OpenAI client (legacy method)
                            key_value_prop, patent_category = await self._generate_llm_insights_direct(patent, llm_client)
                        
                        logger.info(f"Generated LLM insights for patent {patent.patent_number}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to generate LLM insights for patent {patent.patent_number}: {e}")
                        key_value_prop = f"Analysis failed: {str(e)}"
                        patent_category = "Categorization failed"
                
                csv_data.append({
                    # Key columns at the beginning
                    'Patent Number': patent.patent_number,
                    'Publication Number': patent.publication_number,
                    'Title': patent.title,
                    'Key Value Proposition': key_value_prop,
                    'Patent Category': patent_category,
                    'Assignee': patent.assignee,
                    
                    # Additional details
                    'Inventors': '; '.join(patent.inventors),
                    'Abstract': patent.abstract,
                    'Filing Date': patent.filing_date,
                    'Publication Date': patent.publication_date,
                    'Patent Status': patent.patent_status,
                    'Claims Count': patent.claims_count,
                    'Classification': patent.classification,
                    'Legal Status': patent.legal_status,
                    'API Source': patent.api_source,
                    'Fetch Timestamp': patent.fetch_timestamp
                })
            
            # Create output directory if it doesn't exist
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write CSV
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = csv_data[0].keys() if csv_data else []
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            
            logger.info(f"Exported {len(patent_data_list)} patents to enhanced CSV: {output_file}")
            
            # Create additional summary artifact with PDF links and resources
            summary_artifact_path = await self._create_summary_artifact(patent_data_list, output_path, llm_client)
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error exporting to enhanced CSV: {e}")
            raise
    
    async def _create_summary_artifact(self, patent_data_list: List[PatentData], csv_output_path: str, llm_client=None) -> str:
        """
        Create a comprehensive summary artifact with PDF links and additional resources.
        
        Args:
            patent_data_list: List of PatentData objects
            csv_output_path: Path to the CSV output file
            llm_client: LLM client for additional insights
            
        Returns:
            Path to the summary artifact file
        """
        try:
            # Create output directory for resources
            output_dir = Path(csv_output_path).parent
            resources_dir = output_dir / "patent_resources"
            resources_dir.mkdir(exist_ok=True)
            
            # Create summary artifact
            summary_path = output_dir / "patent_analysis_summary.md"
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("# Patent Analysis Summary Report\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Total Patents Analyzed:** {len(patent_data_list)}\n\n")
                
                # Summary statistics
                successful_fetches = len([p for p in patent_data_list if p.api_source != 'Failed'])
                f.write(f"**Success Rate:** {successful_fetches}/{len(patent_data_list)} ({successful_fetches/len(patent_data_list)*100:.1f}%)\n\n")
                
                # Patent details with resources
                f.write("## Patent Details and Resources\n\n")
                
                for i, patent in enumerate(patent_data_list, 1):
                    f.write(f"### {i}. {patent.patent_number} - {patent.title}\n\n")
                    f.write(f"**Publication Number:** {patent.publication_number}\n")
                    f.write(f"**Assignee:** {patent.assignee}\n")
                    f.write(f"**Inventors:** {'; '.join(patent.inventors)}\n")
                    f.write(f"**Filing Date:** {patent.filing_date}\n")
                    f.write(f"**Publication Date:** {patent.publication_date}\n")
                    f.write(f"**Patent Status:** {patent.patent_status}\n")
                    f.write(f"**Legal Status:** {patent.legal_status}\n")
                    f.write(f"**Classification:** {patent.classification}\n")
                    f.write(f"**Claims Count:** {patent.claims_count}\n")
                    f.write(f"**API Source:** {patent.api_source}\n\n")
                    
                    # Generate LLM insights for summary if available
                    if llm_client and patent.title and patent.abstract:
                        try:
                            # Generate key value proposition using OpenAI client
                            kvp_prompt = f"""Analyze this patent and provide a concise 1-3 line summary of its key value proposition:

Title: {patent.title}
Abstract: {patent.abstract}

Focus on:
- What problem does it solve?
- What makes it innovative?
- What are the key benefits?

Provide a clear, business-focused summary in 1-3 lines:"""

                            kvp_response = await llm_client.ainvoke(kvp_prompt, llm_client.get_patent_kvp_system_prompt())
                            key_value_prop = kvp_response if isinstance(kvp_response, str) else str(kvp_response)
                            
                            # Generate patent category using OpenAI client
                            category_prompt = f"""Based on this patent information, assign it to ONE of these cybersecurity categories:

Title: {patent.title}
Abstract: {patent.abstract}

Categories:
- Network Security
- Endpoint Protection
- Identity & Access Management
- Data Protection & Encryption
- Threat Detection & Response
- Vulnerability Management
- Security Operations
- Compliance & Governance
- Application Security
- Infrastructure Security
- Cloud Security
- IoT Security
- Mobile Security
- Incident Response
- Forensic Analysis
- Other

Choose the SINGLE most appropriate category:"""

                            category_response = await llm_client.ainvoke(category_prompt, llm_client.get_patent_category_system_prompt())
                            patent_category = category_response if isinstance(category_response, str) else str(category_response)
                            
                            # Add LLM insights to summary
                            f.write(f"**🤖 AI-Generated Insights:**\n")
                            f.write(f"**Value Proposition:** {key_value_prop}\n\n")
                            f.write(f"**Category:** {patent_category}\n\n")
                            
                        except Exception as e:
                            logger.warning(f"Failed to generate LLM insights for {patent.patent_number}: {e}")
                            f.write(f"**🤖 AI-Generated Insights:** *Analysis failed*\n\n")
                    
                    if patent.abstract:
                        f.write(f"**Abstract:** {patent.abstract[:300]}{'...' if len(patent.abstract) > 300 else ''}\n\n")
                    
                    # Add resource links
                    f.write("**Available Resources:**\n")
                    
                    # Try to fetch additional publication details
                    if patent.publication_number:
                        # Always try Google Patents PDF download first (more reliable)
                        try:
                            # Try to construct Google Patents PDF URL
                            google_pdf_url = f"https://patents.google.com/patent/{patent.publication_number}/en.pdf"
                            pdf_filename = f"{patent.patent_number}_{patent.publication_number}_google.pdf"
                            pdf_path = resources_dir / pdf_filename
                            
                            f.write(f"- [Google Patents PDF]({google_pdf_url})\n")
                            
                            # Try to download the PDF
                            try:
                                pdf_response = self.session.get(google_pdf_url, timeout=60)
                                if pdf_response.status_code == 200:
                                    content_type = pdf_response.headers.get('content-type', '').lower()
                                    content = pdf_response.content
                                    
                                    # Check if it's actually a PDF by looking at the content
                                    is_pdf = (content.startswith(b'%PDF-') or 
                                             'application/pdf' in content_type)
                                    
                                    if is_pdf:
                                        # It's a real PDF
                                        with open(pdf_path, 'wb') as pdf_file:
                                            pdf_file.write(content)
                                        f.write(f"  - [Local PDF Copy](file://{pdf_path.absolute()})\n")
                                        logger.info(f"Downloaded Google Patents PDF for {patent.patent_number}: {pdf_path}")
                                    else:
                                        # It's HTML or other content, save as HTML
                                        html_filename = f"{patent.patent_number}_{patent.publication_number}_google.html"
                                        html_path = resources_dir / html_filename
                                        with open(html_path, 'wb') as html_file:
                                            html_file.write(content)
                                        f.write(f"  - [Local HTML Copy](file://{html_path.absolute()})\n")
                                        logger.info(f"Downloaded Google Patents HTML for {patent.patent_number}: {html_path}")
                                else:
                                    f.write(f"  - Download failed (Status: {pdf_response.status_code})\n")
                            except Exception as e:
                                logger.warning(f"Failed to download Google Patents PDF for {patent.patent_number}: {e}")
                                f.write(f"  - Download failed: {str(e)[:100]}\n")
                        except Exception as e:
                            logger.warning(f"Failed to process Google Patents PDF for {patent.patent_number}: {e}")
                        
                        # Try USPTO publication details as well
                        try:
                            pub_details = self._fetch_publication_details(patent.publication_number)
                            if pub_details:
                                if pub_details.get('pdf_url'):
                                    f.write(f"- [USPTO PDF Document]({pub_details['pdf_url']})\n")
                                if pub_details.get('full_text_url'):
                                    f.write(f"- [USPTO Full Text]({pub_details['full_text_url']})\n")
                                
                                # Download USPTO PDF if available
                                if pub_details.get('pdf_url'):
                                    pdf_filename = f"{patent.patent_number}_{patent.publication_number}_uspto.pdf"
                                    pdf_path = resources_dir / pdf_filename
                                    try:
                                        pdf_response = self.session.get(pub_details['pdf_url'], timeout=60)
                                        if pdf_response.status_code == 200:
                                            content_type = pdf_response.headers.get('content-type', '').lower()
                                            content = pdf_response.content
                                            
                                            # Check if it's actually a PDF by looking at the content
                                            is_pdf = (content.startswith(b'%PDF-') or 
                                                     'application/pdf' in content_type)
                                            
                                            if is_pdf:
                                                # It's a real PDF
                                                with open(pdf_path, 'wb') as pdf_file:
                                                    pdf_file.write(content)
                                                f.write(f"  - [Local USPTO PDF Copy](file://{pdf_path.absolute()})\n")
                                                logger.info(f"Downloaded USPTO PDF for {patent.patent_number}: {pdf_path}")
                                            else:
                                                # It's HTML or other content, save as HTML
                                                html_filename = f"{patent.patent_number}_{patent.publication_number}_uspto.html"
                                                html_path = resources_dir / html_filename
                                                with open(html_path, 'wb') as html_file:
                                                    html_file.write(content)
                                                f.write(f"  - [Local USPTO HTML Copy](file://{html_path.absolute()})\n")
                                                logger.info(f"Downloaded USPTO HTML for {patent.patent_number}: {html_path}")
                                    except Exception as e:
                                        logger.warning(f"Failed to download USPTO PDF for {patent.patent_number}: {e}")
                        except Exception as e:
                            logger.debug(f"Could not fetch USPTO publication details for {patent.publication_number}: {e}")
                    
                    f.write("\n---\n\n")
                
                # Overall insights
                f.write("## Overall Analysis Insights\n\n")
                
                # LLM Analysis Summary
                if llm_client:
                    f.write("### 🤖 AI-Generated Analysis Summary\n\n")
                    f.write("The following insights were generated using AI analysis of patent abstracts and titles:\n\n")
                    
                    # Collect categories and value propositions for summary
                    categories = []
                    value_props = []
                    
                    for patent in patent_data_list:
                        if patent.title and patent.abstract:
                            try:
                                # Generate category for summary
                                category_prompt = f"""Based on this patent information, assign it to ONE of these cybersecurity categories:

Title: {patent.title}
Abstract: {patent.abstract}

Categories:
- Network Security
- Endpoint Protection
- Identity & Access Management
- Data Protection & Encryption
- Threat Detection & Response
- Vulnerability Management
- Security Operations
- Compliance & Governance
- Application Security
- Infrastructure Security
- Cloud Security
- IoT Security
- Mobile Security
- Incident Response
- Forensic Analysis
- Other

Choose the SINGLE most appropriate category:"""

                                category_response = await llm_client.ainvoke(category_prompt, llm_client.get_patent_category_system_prompt())
                                patent_category = category_response if isinstance(category_response, str) else str(category_response)
                                categories.append(patent_category)
                                
                                # Generate value proposition for summary
                                kvp_prompt = f"""Analyze this patent and provide a concise 1-3 line summary of its key value proposition:

Title: {patent.title}
Abstract: {patent.abstract}

Focus on:
- What problem does it solve?
- What makes it innovative?
- What are the key benefits?

Provide a clear, business-focused summary in 1-3 lines:"""

                                kvp_response = await llm_client.ainvoke(kvp_prompt, llm_client.get_patent_kvp_system_prompt())
                                key_value_prop = kvp_response if isinstance(kvp_response, str) else str(kvp_response)
                                value_props.append(key_value_prop)
                                
                            except Exception as e:
                                logger.warning(f"Failed to generate LLM insights for summary: {e}")
                    
                    # Category distribution
                    if categories:
                        from collections import Counter
                        category_counts = Counter(categories)
                        f.write("**Category Distribution:**\n")
                        for category, count in category_counts.most_common():
                            f.write(f"- {category}: {count} patents\n")
                        f.write("\n")
                    
                    # Top value propositions (sample)
                    if value_props:
                        f.write("**Sample Value Propositions:**\n")
                        for i, vp in enumerate(value_props[:3], 1):  # Show first 3
                            f.write(f"{i}. {vp}\n")
                        f.write("\n")
                    
                    f.write("---\n\n")
                
                # Collect unique assignees and categories
                assignees = [p.assignee for p in patent_data_list if p.assignee and p.assignee != 'Unknown']
                unique_assignees = list(set(assignees))
                
                f.write(f"**Unique Assignees:** {len(unique_assignees)}\n")
                for assignee in unique_assignees[:5]:  # Top 5
                    f.write(f"- {assignee}\n")
                f.write("\n")
                
                # API source breakdown
                api_sources = {}
                for patent in patent_data_list:
                    source = patent.api_source
                    api_sources[source] = api_sources.get(source, 0) + 1
                
                f.write("**Data Sources:**\n")
                for source, count in api_sources.items():
                    f.write(f"- {source}: {count} patents\n")
                f.write("\n")
                
                f.write(f"**Output Files:**\n")
                f.write(f"- [Enhanced CSV Data](file://{Path(csv_output_path).absolute()})\n")
                f.write(f"- [Patent Resources](file://{resources_dir.absolute()})\n")
                f.write(f"- [This Summary Report](file://{summary_path.absolute()})\n")
            
            logger.info(f"Created summary artifact: {summary_path}")
            return str(summary_path)
            
        except Exception as e:
            logger.error(f"Error creating summary artifact: {e}")
            return ""
    
    def process_csv_input(self, csv_file_path: str) -> List[Dict[str, str]]:
        """
        Process CSV input file with patent numbers and publication numbers.
        
        Args:
            csv_file_path: Path to CSV file with patent data
            
        Returns:
            List of dictionaries with patent information
        """
        try:
            import pandas as pd
            
            # Read CSV file
            df = pd.read_csv(csv_file_path)
            
            # Validate required columns
            required_columns = ['patent_number', 'publication_number']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert to list of dictionaries
            patent_list = []
            for _, row in df.iterrows():
                patent_info = {
                    'patent_number': str(row['patent_number']).strip(),
                    'publication_number': str(row['publication_number']).strip()
                }
                
                # Add optional description if available
                if 'description' in df.columns:
                    patent_info['description'] = str(row['description']).strip()
                
                patent_list.append(patent_info)
            
            logger.info(f"Processed {len(patent_list)} patents from CSV file: {csv_file_path}")
            return patent_list
            
        except Exception as e:
            logger.error(f"Error processing CSV input file: {e}")
            raise

    def generate_summary_report(self, patent_data_list: List[PatentData]) -> Dict[str, Any]:
        """
        Generate a summary report of the patent data.
        
        Args:
            patent_data_list: List of PatentData objects
            
        Returns:
            Summary statistics and insights
        """
        try:
            if not patent_data_list:
                return {"error": "No patent data to summarize"}
            
            total_patents = len(patent_data_list)
            successful_fetches = len([p for p in patent_data_list if p.api_source != 'Failed'])
            failed_fetches = total_patents - successful_fetches
            
            # Collect unique assignees
            assignees = [p.assignee for p in patent_data_list if p.assignee and p.assignee != 'Unknown']
            unique_assignees = list(set(assignees))
            
            # Collect unique classifications
            classifications = [p.classification for p in patent_data_list if p.classification and p.classification != 'Unknown']
            unique_classifications = list(set(classifications))
            
            # Calculate average claims count
            claims_counts = [p.claims_count for p in patent_data_list if p.claims_count > 0]
            avg_claims = sum(claims_counts) / len(claims_counts) if claims_counts else 0
            
            # API source breakdown
            api_sources = {}
            for patent in patent_data_list:
                source = patent.api_source
                api_sources[source] = api_sources.get(source, 0) + 1
            
            summary = {
                "total_patents": total_patents,
                "successful_fetches": successful_fetches,
                "failed_fetches": failed_fetches,
                "success_rate": (successful_fetches / total_patents * 100) if total_patents > 0 else 0,
                "unique_assignees": len(unique_assignees),
                "average_claims_count": round(avg_claims, 2),
                "api_source_breakdown": api_sources,
                "top_assignees": sorted(unique_assignees, key=lambda x: assignees.count(x), reverse=True)[:5],
                "top_classifications": sorted(unique_classifications, key=lambda x: classifications.count(x), reverse=True)[:5],
                "generated_at": datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            return {"error": f"Failed to generate summary: {str(e)}"}

    def analyze_patent_with_llm(self, patent_data: PatentData, llm_client=None) -> Dict[str, str]:
        """
        Analyze patent using LLM to generate key value proposition and category.
        
        Args:
            patent_data: PatentData object to analyze
            llm_client: LLM client for analysis (optional)
            
        Returns:
            Dictionary with 'key_value_proposition' and 'category'
        """
        try:
            # If no LLM client provided, return basic analysis
            if not llm_client:
                # Basic categorization based on keywords
                category = self._basic_categorization(patent_data)
                key_value = self._basic_value_proposition(patent_data)
                return {
                    'key_value_proposition': key_value,
                    'category': category
                }
            
            # Enhanced LLM-based analysis
            prompt = f"""
            Analyze this US patent and provide:
            1. Key Value Proposition: Summarize the main value proposition in 1-3 lines
            2. Category: Assign a relevant cybersecurity/technology category
            
            Patent Information:
            Title: {patent_data.title}
            Abstract: {patent_data.abstract}
            Inventors: {', '.join(patent_data.inventors)}
            Assignee: {patent_data.assignee}
            Classification: {patent_data.classification}
            
            Please provide your analysis in this exact format:
            KEY_VALUE_PROPOSITION: [1-3 line summary]
            CATEGORY: [specific category]
            """
            
            # This would call the LLM client - for now, return basic analysis
            # In production, you'd call: response = await llm_client.ainvoke(prompt)
            category = self._basic_categorization(patent_data)
            key_value = self._basic_value_proposition(patent_data)
            
            return {
                'key_value_proposition': key_value,
                'category': category
            }
            
        except Exception as e:
            logger.error(f"Error analyzing patent with LLM: {e}")
            return {
                'key_value_proposition': 'Analysis failed',
                'category': 'Unknown'
            }
    
    def _basic_categorization(self, patent_data: PatentData) -> str:
        """Basic categorization based on keywords in title and abstract."""
        try:
            text = f"{patent_data.title} {patent_data.abstract}".lower()
            
            # Cybersecurity categories
            if any(word in text for word in ['threat', 'malware', 'virus', 'attack']):
                return 'Threat Detection & Prevention'
            elif any(word in text for word in ['encryption', 'cryptography', 'key', 'cipher']):
                return 'Cryptography & Encryption'
            elif any(word in text for word in ['network', 'firewall', 'intrusion', 'traffic']):
                return 'Network Security'
            elif any(word in text for word in ['identity', 'authentication', 'authorization', 'access']):
                return 'Identity & Access Management'
            elif any(word in text for word in ['incident', 'response', 'forensics', 'investigation']):
                return 'Incident Response & Forensics'
            elif any(word in text for word in ['compliance', 'audit', 'policy', 'governance']):
                return 'Compliance & Governance'
            elif any(word in text for word in ['data', 'privacy', 'protection', 'breach']):
                return 'Data Protection & Privacy'
            elif any(word in text for word in ['cloud', 'saas', 'virtualization', 'container']):
                return 'Cloud & Virtualization Security'
            else:
                return 'General Cybersecurity'
                
        except Exception as e:
            logger.error(f"Error in basic categorization: {e}")
            return 'Unknown'
    
    def _basic_value_proposition(self, patent_data: PatentData) -> str:
        """Generate basic value proposition from patent data."""
        try:
            title = patent_data.title
            abstract = patent_data.abstract
            
            if title and title != 'Title not available':
                # Extract key benefits from title and abstract
                if 'system' in title.lower():
                    return f"Provides a comprehensive {title.lower()} for enhanced security operations"
                elif 'method' in title.lower():
                    return f"Implements an innovative {title.lower()} to improve security effectiveness"
                elif 'apparatus' in title.lower():
                    return f"Delivers a robust {title.lower()} for security infrastructure"
                else:
                    return f"Offers {title.lower()} to address critical security challenges"
            else:
                return "Patent provides innovative security technology and methodology"
                
        except Exception as e:
            logger.error(f"Error generating value proposition: {e}")
            return "Value proposition analysis unavailable"
    
    async def execute_complete_patent_workflow(self, csv_input_path: str, output_path: str, llm_client=None) -> Dict[str, Any]:
        """
        Execute complete patent analysis workflow from CSV input to enhanced CSV output.
        
        Args:
            csv_input_path: Path to input CSV with patent numbers
            output_path: Path for output enhanced CSV
            llm_client: LLM client for generating insights
            
        Returns:
            Dictionary with workflow results and file paths
        """
        try:
            logger.info("🚀 Starting complete patent analysis workflow...")
            
            # Step 1: Process CSV input
            logger.info("📁 Step 1: Processing CSV input file...")
            patent_list = self.process_csv_input(csv_input_path)
            logger.info(f"✅ Processed {len(patent_list)} patents from input file")
            
            # Step 2: Fetch patent data
            logger.info("🔍 Step 2: Fetching patent data from APIs...")
            patent_data_list = self.batch_fetch_patents(patent_list)
            logger.info(f"✅ Fetched data for {len(patent_data_list)} patents")
            
            # Step 3: Generate summary report
            logger.info("📊 Step 3: Generating summary report...")
            summary_report = self.generate_summary_report(patent_data_list)
            logger.info("✅ Summary report generated")
            
            # Step 4: Export enhanced CSV with LLM insights
            logger.info("🤖 Step 4: Generating LLM insights and exporting enhanced CSV...")
            enhanced_csv_path = await self.export_enhanced_csv(patent_data_list, output_path, llm_client)
            logger.info(f"✅ Enhanced CSV exported to: {enhanced_csv_path}")
            
            # Step 5: Generate comprehensive report
            logger.info("📝 Step 5: Generating comprehensive workflow report...")
            
            workflow_report = {
                "workflow_summary": {
                    "total_patents_processed": len(patent_list),
                    "successful_fetches": len([p for p in patent_data_list if p.api_source != 'Failed']),
                    "failed_fetches": len([p for p in patent_data_list if p.api_source == 'Failed']),
                    "success_rate": f"{(len([p for p in patent_data_list if p.api_source != 'Failed']) / len(patent_data_list) * 100):.1f}%",
                    "llm_insights_generated": llm_client is not None
                },
                "input_file": csv_input_path,
                "output_files": {
                    "enhanced_csv": enhanced_csv_path,
                    "summary_report": summary_report
                },
                "workflow_steps": [
                    "CSV input processing",
                    "Patent data fetching",
                    "Summary report generation", 
                    "LLM insight generation",
                    "Enhanced CSV export"
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("🎉 Patent analysis workflow completed successfully!")
            return workflow_report
            
        except Exception as e:
            error_msg = f"Error in complete patent workflow: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    async def _generate_llm_insights_direct(self, patent: PatentData, llm_client) -> tuple[str, str]:
        """Generate LLM insights using direct OpenAI client (legacy method)."""
        try:
            # Generate key value proposition using OpenAI client
            kvp_prompt = f"""Analyze this patent and provide a concise 1-3 line summary of its key value proposition:

Title: {patent.title}
Abstract: {patent.abstract}

Focus on:
- What problem does it solve?
- What makes it innovative?
- What are the key benefits?

Provide a clear, business-focused summary in 1-3 lines:"""

            kvp_response = await llm_client.ainvoke(kvp_prompt, llm_client.get_patent_kvp_system_prompt())
            key_value_prop = kvp_response if isinstance(kvp_response, str) else str(kvp_response)
            
            # Generate patent category using OpenAI client
            category_prompt = f"""Based on this patent information, assign it to ONE of these cybersecurity categories:

Title: {patent.title}
Abstract: {patent.abstract}

Categories:
- Threat Detection & Prevention
- Network Security
- Identity & Access Management
- Data Protection & Encryption
- Incident Response & Forensics
- Security Analytics & Monitoring
- Cloud Security
- IoT Security
- Mobile Security
- Compliance & Governance

Choose the SINGLE most appropriate category:"""

            category_response = await llm_client.ainvoke(category_prompt, llm_client.get_patent_category_system_prompt())
            patent_category = category_response if isinstance(category_response, str) else str(category_response)
            
            return key_value_prop, patent_category
            
        except Exception as e:
            logger.warning(f"Direct LLM client failed for patent {patent.patent_number}: {e}")
            return f"Analysis failed: {str(e)}", "Categorization failed"

def get_patent_lookup_tools():
    """Get patent lookup tools for MCP integration."""
    return {
        "patent_lookup": {
            "description": "Fetch US patent details by patent number and publication number",
            "class": USPatentLookupTool,
            "methods": [
                "fetch_patent_by_number",
                "batch_fetch_patents", 
                "export_to_csv",
                "export_enhanced_csv",
                "process_csv_input",
                "execute_complete_patent_workflow",
                "generate_summary_report"
            ]
        }
    }

# Example usage and testing
if __name__ == "__main__":
    # Test the patent lookup tool
    tool = USPatentLookupTool()
    
    # Test with a sample patent
    test_patent = "US10123456"
    print(f"Testing patent lookup for: {test_patent}")
    
    patent_data = tool.fetch_patent_by_number(test_patent)
    if patent_data:
        print(f"Found patent: {patent_data.title}")
        print(f"Inventors: {', '.join(patent_data.inventors)}")
        print(f"Assignee: {patent_data.assignee}")
    else:
        print("Patent not found or fetch failed")
