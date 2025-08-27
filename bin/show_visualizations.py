#!/usr/bin/env python3
"""
Show Visualizations Script
Displays the generated visualizations for review.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def show_visualization_info():
    """Show information about generated visualizations."""
    
    viz_dir = Path("session-outputs/visualizations")
    if not viz_dir.exists():
        print("❌ No visualizations directory found")
        return
    
    print("🎨 Generated Visualizations")
    print("=" * 60)
    
    # Group files by type
    viz_types = {
        'workflow_diagram': [],
        'neo4j_graph': [],
        'vega_lite': [],
        'dataframe': []
    }
    
    for file_path in viz_dir.glob("*"):
        if file_path.is_file():
            if 'workflow_diagram' in file_path.name:
                viz_types['workflow_diagram'].append(file_path)
            elif 'neo4j_graph' in file_path.name:
                viz_types['neo4j_graph'].append(file_path)
            elif 'vega_lite' in file_path.name:
                viz_types['vega_lite'].append(file_path)
            elif 'dataframe' in file_path.name:
                viz_types['dataframe'].append(file_path)
    
    # Show each type
    for viz_type, files in viz_types.items():
        if files:
            print(f"\n📊 {viz_type.replace('_', ' ').title()}:")
            for file_path in files:
                file_size = file_path.stat().st_size
                if file_path.suffix == '.png':
                    size_mb = file_size / (1024 * 1024)
                    print(f"   🖼️  {file_path.name} ({size_mb:.1f} MB)")
                elif file_path.suffix == '.svg':
                    size_kb = file_size / 1024
                    print(f"   🎨 {file_path.name} ({size_kb:.1f} KB)")
                elif file_path.suffix == '.html':
                    size_kb = file_size / 1024
                    print(f"   🌐 {file_path.name} ({size_kb:.1f} KB)")
    
    print(f"\n💡 To view visualizations:")
    print("   • PNG/SVG files: Open in image viewer or browser")
    print("   • HTML files: Open in web browser")
    print("   • Use 'open' command: open session-outputs/visualizations/FILENAME")

def open_latest_visualizations():
    """Open the latest visualizations of each type."""
    
    viz_dir = Path("session-outputs/visualizations")
    if not viz_dir.exists():
        print("❌ No visualizations directory found")
        return
    
    print("🚀 Opening latest visualizations...")
    
    # Find latest files of each type
    latest_files = {}
    
    for file_path in viz_dir.glob("*"):
        if file_path.is_file():
            if 'workflow_diagram' in file_path.name:
                if 'workflow_diagram' not in latest_files or file_path.stat().st_mtime > latest_files['workflow_diagram'].stat().st_mtime:
                    latest_files['workflow_diagram'] = file_path
            elif 'neo4j_graph' in file_path.name:
                if 'neo4j_graph' not in latest_files or file_path.stat().st_mtime > latest_files['neo4j_graph'].stat().st_mtime:
                    latest_files['neo4j_graph'] = file_path
            elif 'vega_lite' in file_path.name:
                if 'vega_lite' not in latest_files or file_path.stat().st_mtime > latest_files['vega_lite'].stat().st_mtime:
                    latest_files['vega_lite'] = file_path
            elif 'dataframe' in file_path.name:
                if 'dataframe' not in latest_files or file_path.stat().st_mtime > latest_files['dataframe'].stat().st_mtime:
                    latest_files['dataframe'] = file_path
    
    # Open each type
    for viz_type, file_path in latest_files.items():
        print(f"   Opening {viz_type}: {file_path.name}")
        try:
            os.system(f"open '{file_path}'")
        except Exception as e:
            print(f"   ❌ Could not open {file_path}: {e}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Show generated visualizations")
    parser.add_argument('--open', action='store_true', help='Open latest visualizations')
    
    args = parser.parse_args()
    
    if args.open:
        open_latest_visualizations()
    else:
        show_visualization_info()
        print(f"\n💡 Use --open to automatically open the latest visualizations")

if __name__ == "__main__":
    main()
