"""
Local Scratch Tools for Cybersecurity Agent

Provides powerful local data processing capabilities for handling cloud resource data,
enabling efficient analysis and manipulation without external dependencies.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import re
import hashlib

from .enhanced_session_manager import EnhancedSessionManager

logger = logging.getLogger(__name__)

class LocalScratchTools:
    """Enhanced local scratch tools for data processing and analysis."""
    
    def __init__(self, session_manager: EnhancedSessionManager):
        self.session_manager = session_manager
        self.data_cache = {}
        self.analysis_cache = {}
        
    def load_data(self, file_path: str, data_type: str = "auto") -> pd.DataFrame:
        """
        Load data from various file formats into a DataFrame.
        
        Args:
            file_path: Path to the data file
            data_type: Type of file (auto, csv, json, parquet, excel)
            
        Returns:
            Loaded DataFrame
        """
        try:
            if data_type == "auto":
                # Auto-detect file type
                if file_path.endswith('.csv'):
                    data_type = "csv"
                elif file_path.endswith('.json'):
                    data_type = "json"
                elif file_path.endswith('.parquet'):
                    data_type = "parquet"
                elif file_path.endswith(('.xlsx', '.xls')):
                    data_type = "excel"
                else:
                    data_type = "csv"  # Default
            
            if data_type == "csv":
                df = pd.read_csv(file_path)
            elif data_type == "json":
                df = pd.read_json(file_path)
            elif data_type == "parquet":
                df = pd.read_parquet(file_path)
            elif data_type == "excel":
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            
            # Cache the loaded data
            self.data_cache[file_path] = df
            
            logger.info(f"Successfully loaded data from {file_path}: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            raise
    
    def save_data(self, df: pd.DataFrame, file_path: str, data_type: str = "csv") -> str:
        """
        Save DataFrame to various file formats.
        
        Args:
            df: DataFrame to save
            file_path: Output file path
            data_type: Output format (csv, json, parquet, excel)
            
        Returns:
            Path to saved file
        """
        try:
            if data_type == "csv":
                df.to_csv(file_path, index=False)
            elif data_type == "json":
                df.to_json(file_path, orient='records', indent=2)
            elif data_type == "parquet":
                df.to_parquet(file_path, index=False)
            elif data_type == "excel":
                df.to_excel(file_path, index=False)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            
            logger.info(f"Successfully saved data to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save data to {file_path}: {e}")
            raise
    
    def analyze_data_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the structure and characteristics of a DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Analysis results dictionary
        """
        try:
            analysis = {
                "shape": df.shape,
                "columns": list(df.columns),
                "data_types": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
                "unique_values": {col: df[col].nunique() for col in df.columns},
                "memory_usage": df.memory_usage(deep=True).sum(),
                "sample_data": df.head(5).to_dict('records'),
                "statistical_summary": df.describe().to_dict() if df.select_dtypes(include=[np.number]).shape[1] > 0 else {}
            }
            
            # Detect potential data quality issues
            quality_issues = []
            for col in df.columns:
                if df[col].isnull().sum() > len(df) * 0.5:
                    quality_issues.append(f"Column '{col}' has >50% missing values")
                if df[col].dtype == 'object' and df[col].nunique() == 1:
                    quality_issues.append(f"Column '{col}' has only one unique value")
                if df[col].dtype == 'object' and df[col].nunique() == len(df):
                    quality_issues.append(f"Column '{col}' appears to be an ID (all values unique)")
            
            analysis["quality_issues"] = quality_issues
            analysis["analysis_timestamp"] = datetime.now().isoformat()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze data structure: {e}")
            raise
    
    def clean_data(self, df: pd.DataFrame, 
                   remove_duplicates: bool = True,
                   handle_missing: str = "drop",
                   fill_value: Any = None) -> pd.DataFrame:
        """
        Clean and preprocess DataFrame.
        
        Args:
            df: DataFrame to clean
            remove_duplicates: Whether to remove duplicate rows
            handle_missing: How to handle missing values (drop, fill, interpolate)
            fill_value: Value to fill missing values with
            
        Returns:
            Cleaned DataFrame
        """
        try:
            df_clean = df.copy()
            
            # Remove duplicates
            if remove_duplicates:
                initial_rows = len(df_clean)
                df_clean = df_clean.drop_duplicates()
                removed_duplicates = initial_rows - len(df_clean)
                logger.info(f"Removed {removed_duplicates} duplicate rows")
            
            # Handle missing values
            if handle_missing == "drop":
                initial_rows = len(df_clean)
                df_clean = df_clean.dropna()
                removed_rows = initial_rows - len(df_clean)
                logger.info(f"Removed {removed_rows} rows with missing values")
            elif handle_missing == "fill":
                df_clean = df_clean.fillna(fill_value)
                logger.info(f"Filled missing values with {fill_value}")
            elif handle_missing == "interpolate":
                df_clean = df_clean.interpolate()
                logger.info("Interpolated missing values")
            
            # Clean column names
            df_clean.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in df_clean.columns]
            
            # Remove empty columns
            empty_cols = [col for col in df_clean.columns if df_clean[col].isnull().all()]
            if empty_cols:
                df_clean = df_clean.drop(columns=empty_cols)
                logger.info(f"Removed empty columns: {empty_cols}")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Failed to clean data: {e}")
            raise
    
    def filter_data(self, df: pd.DataFrame, 
                    filters: Dict[str, Any],
                    operator: str = "AND") -> pd.DataFrame:
        """
        Filter DataFrame based on multiple conditions.
        
        Args:
            df: DataFrame to filter
            filters: Dictionary of column: value pairs for filtering
            operator: Logical operator (AND, OR)
            
        Returns:
            Filtered DataFrame
        """
        try:
            if not filters:
                return df
            
            mask = pd.Series([True] * len(df))
            
            for column, value in filters.items():
                if column in df.columns:
                    if isinstance(value, (list, tuple)):
                        # Multiple values
                        col_mask = df[column].isin(value)
                    elif isinstance(value, dict):
                        # Range or comparison
                        if 'min' in value and 'max' in value:
                            col_mask = (df[column] >= value['min']) & (df[column] <= value['max'])
                        elif 'min' in value:
                            col_mask = df[column] >= value['min']
                        elif 'max' in value:
                            col_mask = df[column] <= value['max']
                        elif 'contains' in value:
                            col_mask = df[column].str.contains(value['contains'], case=False, na=False)
                        elif 'regex' in value:
                            col_mask = df[column].str.match(value['regex'], case=False, na=False)
                        else:
                            col_mask = df[column] == value
                    else:
                        # Single value
                        col_mask = df[column] == value
                    
                    if operator == "AND":
                        mask = mask & col_mask
                    else:  # OR
                        mask = mask | col_mask
            
            filtered_df = df[mask].copy()
            logger.info(f"Filtered data: {len(df)} -> {len(filtered_df)} rows")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Failed to filter data: {e}")
            raise
    
    def aggregate_data(self, df: pd.DataFrame, 
                       group_by: List[str],
                       aggregations: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Aggregate data by grouping columns.
        
        Args:
            df: DataFrame to aggregate
            group_by: Columns to group by
            aggregations: Dictionary of column: [functions] pairs
            
        Returns:
            Aggregated DataFrame
        """
        try:
            if not group_by or not aggregations:
                return df
            
            # Create aggregation dictionary
            agg_dict = {}
            for col, funcs in aggregations.items():
                if col in df.columns:
                    if isinstance(funcs, str):
                        agg_dict[col] = funcs
                    elif isinstance(funcs, list):
                        agg_dict[col] = funcs
                    else:
                        agg_dict[col] = funcs
            
            # Perform aggregation
            aggregated = df.groupby(group_by).agg(agg_dict).reset_index()
            
            # Flatten column names if multiple aggregation functions
            if any(isinstance(funcs, list) and len(funcs) > 1 for funcs in aggregations.values()):
                aggregated.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col 
                                   for col in aggregated.columns]
            
            logger.info(f"Aggregated data by {group_by}: {len(df)} -> {len(aggregated)} rows")
            return aggregated
            
        except Exception as e:
            logger.error(f"Failed to aggregate data: {e}")
            raise
    
    def merge_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame,
                      on: Union[str, List[str]],
                      how: str = "inner",
                      suffixes: Tuple[str, str] = ("_x", "_y")) -> pd.DataFrame:
        """
        Merge two DataFrames.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            on: Column(s) to merge on
            how: Merge type (inner, left, right, outer)
            suffixes: Suffixes for duplicate columns
            
        Returns:
            Merged DataFrame
        """
        try:
            merged = pd.merge(df1, df2, on=on, how=how, suffixes=suffixes)
            
            logger.info(f"Merged datasets: {len(df1)} + {len(df2)} -> {len(merged)} rows")
            return merged
            
        except Exception as e:
            logger.error(f"Failed to merge datasets: {e}")
            raise
    
    def detect_anomalies(self, df: pd.DataFrame, 
                         columns: Optional[List[str]] = None,
                         method: str = "iqr",
                         threshold: float = 1.5) -> Dict[str, Any]:
        """
        Detect anomalies in numerical columns.
        
        Args:
            df: DataFrame to analyze
            columns: Columns to check for anomalies (default: all numerical)
            method: Detection method (iqr, zscore, isolation_forest)
            threshold: Threshold for anomaly detection
            
        Returns:
            Anomaly detection results
        """
        try:
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            anomalies = {}
            
            for col in columns:
                if col in df.columns and df[col].dtype in ['int64', 'float64']:
                    series = df[col].dropna()
                    
                    if method == "iqr":
                        Q1 = series.quantile(0.25)
                        Q3 = series.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        
                        anomaly_mask = (series < lower_bound) | (series > upper_bound)
                        anomaly_indices = series[anomaly_mask].index.tolist()
                        
                    elif method == "zscore":
                        z_scores = np.abs((series - series.mean()) / series.std())
                        anomaly_mask = z_scores > threshold
                        anomaly_indices = series[anomaly_mask].index.tolist()
                    
                    else:
                        anomaly_indices = []
                    
                    anomalies[col] = {
                        "anomaly_count": len(anomaly_indices),
                        "anomaly_percentage": len(anomaly_indices) / len(series) * 100,
                        "anomaly_indices": anomaly_indices,
                        "anomaly_values": series[anomaly_indices].tolist() if anomaly_indices else []
                    }
            
            return {
                "total_anomalies": sum(anomaly["anomaly_count"] for anomaly in anomalies.values()),
                "columns_analyzed": list(anomalies.keys()),
                "anomalies_by_column": anomalies,
                "detection_method": method,
                "threshold": threshold
            }
            
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")
            raise
    
    def extract_patterns(self, df: pd.DataFrame, 
                        columns: Optional[List[str]] = None,
                        pattern_types: List[str] = None) -> Dict[str, Any]:
        """
        Extract patterns from text columns.
        
        Args:
            df: DataFrame to analyze
            columns: Columns to analyze (default: all object columns)
            pattern_types: Types of patterns to extract
            
        Returns:
            Pattern extraction results
        """
        try:
            if columns is None:
                columns = df.select_dtypes(include=['object']).columns.tolist()
            
            if pattern_types is None:
                pattern_types = ["email", "ip", "url", "date", "phone"]
            
            patterns = {}
            
            for col in columns:
                if col in df.columns:
                    series = df[col].dropna().astype(str)
                    col_patterns = {}
                    
                    for pattern_type in pattern_types:
                        if pattern_type == "email":
                            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                            matches = series.str.extractall(email_pattern)
                            col_patterns["email"] = {
                                "count": len(matches),
                                "unique": matches[0].nunique() if not matches.empty else 0
                            }
                        
                        elif pattern_type == "ip":
                            ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
                            matches = series.str.extractall(ip_pattern)
                            col_patterns["ip"] = {
                                "count": len(matches),
                                "unique": matches[0].nunique() if not matches.empty else 0
                            }
                        
                        elif pattern_type == "url":
                            url_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
                            matches = series.str.extractall(url_pattern)
                            col_patterns["url"] = {
                                "count": len(matches),
                                "unique": matches[0].nunique() if not matches.empty else 0
                            }
                        
                        elif pattern_type == "date":
                            date_pattern = r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b'
                            matches = series.str.extractall(date_pattern)
                            col_patterns["date"] = {
                                "count": len(matches),
                                "unique": matches[0].nunique() if not matches.empty else 0
                            }
                        
                        elif pattern_type == "phone":
                            phone_pattern = r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b'
                            matches = series.str.extractall(phone_pattern)
                            col_patterns["phone"] = {
                                "count": len(matches),
                                "unique": len(matches) if not matches.empty else 0
                            }
                    
                    patterns[col] = col_patterns
            
            return {
                "total_columns_analyzed": len(patterns),
                "patterns_by_column": patterns,
                "pattern_types": pattern_types
            }
            
        except Exception as e:
            logger.error(f"Failed to extract patterns: {e}")
            raise
    
    def generate_summary_report(self, df: pd.DataFrame, 
                              include_analysis: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report for a DataFrame.
        
        Args:
            df: DataFrame to summarize
            include_analysis: Whether to include detailed analysis
            
        Returns:
            Summary report dictionary
        """
        try:
            report = {
                "basic_info": {
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
                    "data_types": df.dtypes.value_counts().to_dict()
                },
                "missing_data": {
                    "total_missing": df.isnull().sum().sum(),
                    "missing_by_column": df.isnull().sum().to_dict(),
                    "missing_percentage": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
                },
                "data_quality": {
                    "duplicate_rows": df.duplicated().sum(),
                    "duplicate_percentage": (df.duplicated().sum() / len(df) * 100)
                }
            }
            
            if include_analysis:
                # Add data structure analysis
                structure_analysis = self.analyze_data_structure(df)
                report["structure_analysis"] = structure_analysis
                
                # Add anomaly detection for numerical columns
                numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numerical_cols:
                    anomalies = self.detect_anomalies(df, columns=numerical_cols[:5])  # Limit to first 5 columns
                    report["anomaly_detection"] = anomalies
                
                # Add pattern extraction for text columns
                text_cols = df.select_dtypes(include=['object']).columns.tolist()
                if text_cols:
                    patterns = self.extract_patterns(df, columns=text_cols[:5])  # Limit to first 5 columns
                    report["pattern_extraction"] = patterns
            
            report["generated_at"] = datetime.now().isoformat()
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
            raise
    
    def export_analysis(self, analysis_results: Dict[str, Any], 
                       export_format: str = "json") -> str:
        """
        Export analysis results to various formats.
        
        Args:
            analysis_results: Analysis results to export
            export_format: Export format (json, csv, html)
            
        Returns:
            Path to exported file
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if export_format == "json":
                filename = f"analysis_report_{timestamp}.json"
                self.session_manager.save_text_output(
                    json.dumps(analysis_results, indent=2, default=str),
                    filename.replace('.json', ''),
                    "Data analysis report in JSON format"
                )
            
            elif export_format == "csv":
                # Flatten nested structures for CSV export
                flattened_data = self._flatten_dict(analysis_results)
                df = pd.DataFrame([flattened_data])
                filename = f"analysis_report_{timestamp}.csv"
                self.session_manager.save_dataframe(
                    df,
                    filename.replace('.csv', ''),
                    "Data analysis report in CSV format"
                )
            
            elif export_format == "html":
                html_content = self._generate_html_report(analysis_results)
                filename = f"analysis_report_{timestamp}.html"
                self.session_manager.save_text_output(
                    html_content,
                    filename.replace('.html', ''),
                    "Data analysis report in HTML format"
                )
            
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to export analysis: {e}")
            raise
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _generate_html_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate HTML report from analysis results."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .section h2 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 5px; }
                .metric { margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-radius: 3px; }
                .metric strong { color: #007bff; }
                table { width: 100%; border-collapse: collapse; margin: 10px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .timestamp { color: #666; font-style: italic; }
            </style>
        </head>
        <body>
            <h1>Data Analysis Report</h1>
            <div class="timestamp">Generated at: {timestamp}</div>
        """.format(timestamp=analysis_results.get('generated_at', 'Unknown'))
        
        for section_name, section_data in analysis_results.items():
            if section_name == 'generated_at':
                continue
                
            html += f'<div class="section"><h2>{section_name.replace("_", " ").title()}</h2>'
            
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    if isinstance(value, (dict, list)):
                        html += f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong><br>'
                        html += f'<pre>{json.dumps(value, indent=2, default=str)}</pre></div>'
                    else:
                        html += f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong> {value}</div>'
            else:
                html += f'<div class="metric">{section_data}</div>'
            
            html += '</div>'
        
        html += """
        </body>
        </html>
        """
        
        return html

# MCP Tools for Local Scratch Tools
class LocalScratchMCPTools:
    """MCP-compatible tools for local scratch data processing."""
    
    def __init__(self, scratch_tools: LocalScratchTools):
        self.scratch = scratch_tools
    
    def get_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get MCP tool definitions for local scratch tools."""
        return {
            "scratch_load_data": {
                "name": "scratch_load_data",
                "description": "Load data from various file formats into a DataFrame",
                "parameters": {
                    "file_path": {"type": "string", "description": "Path to the data file"},
                    "data_type": {"type": "string", "description": "Type of file (auto, csv, json, parquet, excel)"}
                },
                "returns": {"type": "object", "description": "Loaded DataFrame information"}
            },
            "scratch_analyze_data": {
                "name": "scratch_analyze_data",
                "description": "Analyze the structure and characteristics of a DataFrame",
                "parameters": {
                    "df": {"type": "object", "description": "DataFrame to analyze"}
                },
                "returns": {"type": "object", "description": "Analysis results"}
            },
            "scratch_clean_data": {
                "name": "scratch_clean_data",
                "description": "Clean and preprocess DataFrame",
                "parameters": {
                    "df": {"type": "object", "description": "DataFrame to clean"},
                    "remove_duplicates": {"type": "boolean", "description": "Whether to remove duplicate rows"},
                    "handle_missing": {"type": "string", "description": "How to handle missing values (drop, fill, interpolate)"},
                    "fill_value": {"type": "any", "description": "Value to fill missing values with"}
                },
                "returns": {"type": "object", "description": "Cleaned DataFrame"}
            },
            "scratch_filter_data": {
                "name": "scratch_filter_data",
                "description": "Filter DataFrame based on multiple conditions",
                "parameters": {
                    "df": {"type": "object", "description": "DataFrame to filter"},
                    "filters": {"type": "object", "description": "Dictionary of column: value pairs for filtering"},
                    "operator": {"type": "string", "description": "Logical operator (AND, OR)"}
                },
                "returns": {"type": "object", "description": "Filtered DataFrame"}
            },
            "scratch_generate_report": {
                "name": "scratch_generate_report",
                "description": "Generate a comprehensive summary report for a DataFrame",
                "parameters": {
                    "df": {"type": "object", "description": "DataFrame to summarize"},
                    "include_analysis": {"type": "boolean", "description": "Whether to include detailed analysis"}
                },
                "returns": {"type": "object", "description": "Summary report"}
            }
        }
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute local scratch MCP tool."""
        if tool_name == "scratch_load_data":
            return self.scratch.load_data(**kwargs)
        elif tool_name == "scratch_analyze_data":
            return self.scratch.analyze_data_structure(**kwargs)
        elif tool_name == "scratch_clean_data":
            return self.scratch.clean_data(**kwargs)
        elif tool_name == "scratch_filter_data":
            return self.scratch.filter_data(**kwargs)
        elif tool_name == "scratch_generate_report":
            return self.scratch.generate_summary_report(**kwargs)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
