#!/usr/bin/env python3
"""
Generate high-quality PDF pages from Jupyter notebooks.
Converts notebooks to PDF, then each PDF page to a PNG image.

Usage:
    python generate_notebook_images.py

Requirements:
    pip install nbconvert[webpdf] pdf2image
    # Also needs: apt-get install poppler-utils
"""

import os
import json
import subprocess
from pathlib import Path

# Configuration
NOTEBOOKS_DIR = Path("notebooks")
OUTPUT_DIR = Path("notebook-images")
DPI = 150  # Resolution for PDF to image conversion


def install_dependencies():
    """Install required packages."""
    print("Checking dependencies...")

    # Try to import required packages
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("Installing pdf2image...")
        os.system("pip install pdf2image --break-system-packages 2>/dev/null || pip install pdf2image")

    # Check if poppler is installed
    result = subprocess.run(['which', 'pdftoppm'], capture_output=True)
    if result.returncode != 0:
        print("Installing poppler-utils...")
        os.system("apt-get update && apt-get install -y poppler-utils")


def convert_notebook_to_pdf(notebook_path, output_path):
    """Convert a Jupyter notebook to PDF using nbconvert."""
    print(f"  Converting to PDF...")

    # Use nbconvert with webpdf (requires playwright)
    cmd = [
        'jupyter', 'nbconvert',
        '--to', 'webpdf',
        '--allow-chromium-download',
        '--output', str(output_path.stem),
        '--output-dir', str(output_path.parent),
        str(notebook_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # Fallback: try html first, then use playwright to convert to PDF
        print("  WebPDF failed, trying alternative method...")
        return convert_via_html(notebook_path, output_path)

    return output_path.exists()


def convert_via_html(notebook_path, output_path):
    """Convert notebook to HTML, then to PDF using playwright."""
    import asyncio

    # First convert to HTML
    html_path = output_path.with_suffix('.html')

    # Read notebook and generate HTML
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    html_content = generate_notebook_html(notebook, notebook_path.stem)

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"  Generated HTML: {html_path}")

    # Use playwright to generate PDF
    async def html_to_pdf():
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()

            # Load HTML file
            await page.goto(f'file://{html_path.absolute()}')
            await page.wait_for_load_state('networkidle')

            # Generate PDF
            await page.pdf(
                path=str(output_path),
                format='A4',
                margin={'top': '20mm', 'right': '15mm', 'bottom': '20mm', 'left': '15mm'},
                print_background=True
            )

            await browser.close()

    asyncio.run(html_to_pdf())
    return output_path.exists()


def generate_notebook_html(notebook, name):
    """Generate clean HTML from notebook JSON."""

    css = '''
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@400;500&display=swap');

        * { box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.5;
            color: #333;
            background: #fff;
            margin: 0;
            padding: 0;
        }

        .notebook-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px 30px;
            margin-bottom: 20px;
        }

        .notebook-header h1 {
            margin: 0 0 8px 0;
            font-size: 20pt;
            font-weight: 600;
        }

        .notebook-header p {
            margin: 0;
            opacity: 0.9;
            font-size: 10pt;
        }

        .notebook-content {
            padding: 0 30px 30px 30px;
        }

        .cell {
            margin-bottom: 15px;
            page-break-inside: avoid;
        }

        .markdown-cell {
            padding: 8px 0;
        }

        .markdown-cell h1 { font-size: 16pt; border-bottom: 2px solid #667eea; padding-bottom: 5px; margin: 15px 0 10px 0; color: #333; }
        .markdown-cell h2 { font-size: 14pt; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin: 12px 0 8px 0; color: #444; }
        .markdown-cell h3 { font-size: 12pt; margin: 10px 0 6px 0; color: #555; }
        .markdown-cell p { margin: 6px 0; }
        .markdown-cell ul, .markdown-cell ol { margin: 6px 0; padding-left: 25px; }
        .markdown-cell li { margin: 3px 0; }

        .code-cell {
            background: #fafafa;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
        }

        .input-area {
            display: flex;
            border-bottom: 1px solid #e8e8e8;
        }

        .prompt {
            font-family: 'Source Code Pro', 'Consolas', monospace;
            font-size: 9pt;
            padding: 8px 10px;
            min-width: 70px;
            text-align: right;
            font-weight: 600;
            background: #f5f5f5;
            border-right: 1px solid #e0e0e0;
        }

        .input-prompt { color: #303F9F; }
        .output-prompt { color: #D84315; }

        .code-content {
            flex: 1;
            padding: 8px 12px;
            overflow-x: auto;
        }

        .code-content pre {
            margin: 0;
            font-family: 'Source Code Pro', 'Consolas', monospace;
            font-size: 9pt;
            line-height: 1.4;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        /* Syntax highlighting */
        .keyword { color: #008000; font-weight: bold; }
        .string { color: #BA2121; }
        .number { color: #666666; }
        .comment { color: #408080; font-style: italic; }
        .function { color: #0000FF; }

        .output-area {
            background: #fff;
            border-top: 1px solid #e8e8e8;
        }

        .output-row {
            display: flex;
        }

        .output-content {
            flex: 1;
            padding: 8px 12px;
            overflow-x: auto;
        }

        .output-stream pre,
        .output-text pre {
            margin: 0;
            font-family: 'Source Code Pro', monospace;
            font-size: 9pt;
            line-height: 1.4;
            white-space: pre-wrap;
        }

        .output-image {
            text-align: center;
            padding: 10px;
        }

        .output-image img {
            max-width: 100%;
            height: auto;
        }

        .output-html {
            overflow-x: auto;
        }

        .dataframe {
            border-collapse: collapse;
            font-size: 9pt;
            margin: 0;
        }

        .dataframe th {
            background: #f0f0f0;
            padding: 6px 10px;
            border: 1px solid #ddd;
            text-align: left;
            font-weight: 600;
        }

        .dataframe td {
            padding: 5px 10px;
            border: 1px solid #ddd;
        }

        .dataframe tr:nth-child(even) {
            background: #f9f9f9;
        }

        .output-error {
            background: #fff0f0;
            color: #cc0000;
            padding: 8px 12px;
        }

        .output-error pre {
            margin: 0;
            font-family: 'Source Code Pro', monospace;
            font-size: 9pt;
        }

        @media print {
            .cell { page-break-inside: avoid; }
            .notebook-header { page-break-after: avoid; }
        }
    </style>
    '''

    cells_html = []
    exec_count = 0

    for cell in notebook.get('cells', []):
        cell_type = cell.get('cell_type', 'code')
        source = ''.join(cell.get('source', []))

        if not source.strip():
            continue

        if cell_type == 'markdown':
            html_content = render_markdown(source)
            cells_html.append(f'<div class="cell markdown-cell">{html_content}</div>')

        elif cell_type == 'code':
            exec_count += 1
            execution_count = cell.get('execution_count') or exec_count

            code_escaped = source.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            code_highlighted = highlight_python(code_escaped)

            cell_html = f'''
            <div class="cell code-cell">
                <div class="input-area">
                    <div class="prompt input-prompt">In [{execution_count}]:</div>
                    <div class="code-content"><pre>{code_highlighted}</pre></div>
                </div>
            '''

            outputs = cell.get('outputs', [])
            if outputs:
                cell_html += '<div class="output-area">'
                for output in outputs:
                    cell_html += render_output(output, execution_count)
                cell_html += '</div>'

            cell_html += '</div>'
            cells_html.append(cell_html)

    display_name = name.replace('_', ' ').replace('skML-complete ', '')

    return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{name}</title>
    {css}
</head>
<body>
    <div class="notebook-header">
        <h1>{display_name}</h1>
        <p>Machine Learning Implementation for Network Intrusion Detection</p>
    </div>
    <div class="notebook-content">
        {''.join(cells_html)}
    </div>
</body>
</html>'''


def render_markdown(text):
    """Convert markdown to HTML."""
    import re

    lines = text.split('\n')
    html_lines = []
    in_list = False
    list_type = None

    for line in lines:
        # Headers
        if line.startswith('### '):
            if in_list: html_lines.append(f'</{list_type}>'); in_list = False
            html_lines.append(f'<h3>{line[4:]}</h3>')
        elif line.startswith('## '):
            if in_list: html_lines.append(f'</{list_type}>'); in_list = False
            html_lines.append(f'<h2>{line[3:]}</h2>')
        elif line.startswith('# '):
            if in_list: html_lines.append(f'</{list_type}>'); in_list = False
            html_lines.append(f'<h1>{line[2:]}</h1>')
        # Bullet lists
        elif line.startswith('- ') or line.startswith('* '):
            if not in_list or list_type != 'ul':
                if in_list: html_lines.append(f'</{list_type}>')
                html_lines.append('<ul>')
                in_list = True
                list_type = 'ul'
            content = line[2:]
            content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
            content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', content)
            content = re.sub(r'`(.+?)`', r'<code>\1</code>', content)
            html_lines.append(f'<li>{content}</li>')
        # Numbered lists
        elif re.match(r'^\d+\.\s', line):
            if not in_list or list_type != 'ol':
                if in_list: html_lines.append(f'</{list_type}>')
                html_lines.append('<ol>')
                in_list = True
                list_type = 'ol'
            content = re.sub(r'^\d+\.\s', '', line)
            content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
            html_lines.append(f'<li>{content}</li>')
        # Regular text
        elif line.strip():
            if in_list:
                html_lines.append(f'</{list_type}>')
                in_list = False
            # Bold
            line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
            # Italic
            line = re.sub(r'\*(.+?)\*', r'<em>\1</em>', line)
            # Code
            line = re.sub(r'`(.+?)`', r'<code>\1</code>', line)
            html_lines.append(f'<p>{line}</p>')

    if in_list:
        html_lines.append(f'</{list_type}>')

    return '\n'.join(html_lines)


def highlight_python(code):
    """Apply syntax highlighting to Python code."""
    import re

    keywords = ['import', 'from', 'as', 'def', 'class', 'return', 'if', 'else', 'elif',
                'for', 'while', 'in', 'and', 'or', 'not', 'True', 'False', 'None',
                'try', 'except', 'finally', 'with', 'lambda', 'yield', 'raise', 'pass',
                'break', 'continue', 'global', 'nonlocal', 'assert', 'del', 'is', 'print']

    lines = code.split('\n')
    highlighted = []

    for line in lines:
        # Handle comments first
        if '#' in line:
            idx = line.index('#')
            code_part = line[:idx]
            comment_part = line[idx:]
            line = code_part + f'<span class="comment">{comment_part}</span>'

        # Strings
        line = re.sub(r'(\"\"\".*?\"\"\"|\'\'\'.*?\'\'\')', r'<span class="string">\1</span>', line, flags=re.DOTALL)
        line = re.sub(r'(\"[^\"]*\"|\'[^\']*\')', r'<span class="string">\1</span>', line)

        # Numbers
        line = re.sub(r'\b(\d+\.?\d*)\b', r'<span class="number">\1</span>', line)

        # Keywords
        for kw in keywords:
            line = re.sub(rf'\b({kw})\b', rf'<span class="keyword">\1</span>', line)

        # Function calls
        line = re.sub(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', r'<span class="function">\1</span>(', line)

        highlighted.append(line)

    return '\n'.join(highlighted)


def render_output(output, exec_count):
    """Render notebook cell output."""
    output_type = output.get('output_type', '')
    html = ''

    if output_type == 'stream':
        text = ''.join(output.get('text', []))
        text_escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        html = f'<div class="output-row"><div class="prompt"></div><div class="output-content output-stream"><pre>{text_escaped}</pre></div></div>'

    elif output_type in ('execute_result', 'display_data'):
        data = output.get('data', {})

        if 'image/png' in data:
            img_data = data['image/png']
            html = f'''<div class="output-row">
                <div class="prompt output-prompt">Out[{exec_count}]:</div>
                <div class="output-content output-image"><img src="data:image/png;base64,{img_data}"/></div>
            </div>'''
        elif 'text/html' in data:
            html_content = ''.join(data['text/html'])
            html = f'''<div class="output-row">
                <div class="prompt output-prompt">Out[{exec_count}]:</div>
                <div class="output-content output-html">{html_content}</div>
            </div>'''
        elif 'text/plain' in data:
            text = ''.join(data['text/plain'])
            text_escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            html = f'''<div class="output-row">
                <div class="prompt output-prompt">Out[{exec_count}]:</div>
                <div class="output-content output-text"><pre>{text_escaped}</pre></div>
            </div>'''

    elif output_type == 'error':
        ename = output.get('ename', 'Error')
        evalue = output.get('evalue', '')
        html = f'<div class="output-row"><div class="prompt"></div><div class="output-content output-error"><pre>{ename}: {evalue}</pre></div></div>'

    return html


def pdf_to_images(pdf_path, output_prefix):
    """Convert PDF pages to PNG images."""
    from pdf2image import convert_from_path

    print(f"  Converting PDF to images (DPI={DPI})...")

    pages = convert_from_path(pdf_path, dpi=DPI)

    image_files = []
    for i, page in enumerate(pages, 1):
        output_path = f"{output_prefix}_page_{i:02d}.png"
        page.save(output_path, 'PNG')
        image_files.append(Path(output_path).name)
        print(f"    Page {i}/{len(pages)}: {output_path}")

    return image_files


def process_notebook(notebook_info):
    """Process a single notebook."""
    notebook_path = NOTEBOOKS_DIR / notebook_info['file']

    if not notebook_path.exists():
        print(f"  Notebook not found: {notebook_path}")
        return None

    print(f"\nProcessing: {notebook_info['file']}")

    # Output paths
    pdf_path = OUTPUT_DIR / f"{notebook_info['prefix']}.pdf"
    output_prefix = OUTPUT_DIR / notebook_info['prefix']

    # Convert to PDF
    if convert_via_html(notebook_path, pdf_path):
        print(f"  Generated PDF: {pdf_path}")

        # Convert PDF to images
        pages = pdf_to_images(pdf_path, str(output_prefix))

        return {
            'title': notebook_info['title'],
            'file': notebook_info['file'],
            'pages': pages
        }
    else:
        print(f"  Failed to generate PDF")
        return None


def main():
    print("=" * 60)
    print("Jupyter Notebook to PDF/Image Converter")
    print("=" * 60)

    # Check directory
    if not NOTEBOOKS_DIR.exists():
        print(f"Error: {NOTEBOOKS_DIR} directory not found!")
        return

    # Install dependencies
    install_dependencies()

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    notebooks = [
        {
            'prefix': 'Ensemble',
            'file': 'skML-complete_Ensemble.ipynb',
            'title': 'Ensemble Methods (BaggingClassifier)'
        },
        {
            'prefix': 'NonLinear',
            'file': 'skML-complete_NonLinear.ipynb',
            'title': 'Non-Linear Methods (DecisionTreeClassifier)'
        },
        {
            'prefix': 'SupportVector',
            'file': 'skML-complete_SupportVector.ipynb',
            'title': 'Linear Methods (RidgeClassifier)'
        }
    ]

    manifest = {}

    for nb in notebooks:
        result = process_notebook(nb)
        if result:
            manifest[nb['prefix']] = result

    # Save manifest
    manifest_path = OUTPUT_DIR / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nGenerated in: {OUTPUT_DIR}/")

    for prefix, data in manifest.items():
        print(f"\n{data['title']}: {len(data['pages'])} pages")


if __name__ == '__main__':
    main()
