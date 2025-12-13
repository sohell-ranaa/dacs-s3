#!/usr/bin/env python3
"""
Convert Markdown Report to Professional PDF-styled HTML
Author: Auto-generated for DACS Assignment
Uses regex-based markdown parsing (no external dependencies)
"""

import re
import os
from pathlib import Path
import shutil

# Configuration
INPUT_FILE = "REPORT.md"
OUTPUT_FILE = "REPORT.html"
FIGURES_DIR = "../../figures"

def md_to_html(md_text):
    """Convert markdown to HTML using regex"""
    html = md_text

    # Fix image paths first
    html = html.replace('../../figures/', 'figures/')

    # Escape HTML special chars in code blocks first (preserve them)
    code_blocks = {}
    def save_code_block(match):
        key = f"___CODE_BLOCK_{len(code_blocks)}___"
        lang = match.group(1) or ''
        code = match.group(2)
        code = code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        code_blocks[key] = f'<pre><code class="language-{lang}">{code}</code></pre>'
        return key

    html = re.sub(r'```(\w*)\n(.*?)```', save_code_block, html, flags=re.DOTALL)

    # Inline code
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)

    # Images with captions
    html = re.sub(
        r'!\[([^\]]*)\]\(([^)]+)\)\n\n\*([^*]+)\*',
        r'<figure><img src="\2" alt="\1"><figcaption>\3</figcaption></figure>',
        html
    )

    # Images without captions
    html = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<img src="\2" alt="\1">', html)

    # Links
    html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)

    # Headers
    html = re.sub(r'^###### (.+)$', r'<h6>\1</h6>', html, flags=re.MULTILINE)
    html = re.sub(r'^##### (.+)$', r'<h5>\1</h5>', html, flags=re.MULTILINE)
    html = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

    # Bold and italic
    html = re.sub(r'\*\*\*([^*]+)\*\*\*', r'<strong><em>\1</em></strong>', html)
    html = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', html)

    # Tables
    def convert_table(match):
        table_text = match.group(0)
        lines = table_text.strip().split('\n')
        if len(lines) < 2:
            return table_text

        html_table = '<table>\n<thead>\n<tr>\n'
        # Header row
        headers = [h.strip() for h in lines[0].split('|') if h.strip()]
        for h in headers:
            html_table += f'<th>{h}</th>\n'
        html_table += '</tr>\n</thead>\n<tbody>\n'

        # Data rows (skip separator line)
        for line in lines[2:]:
            if line.strip():
                cells = [c.strip() for c in line.split('|') if c.strip() or c == '']
                cells = [c for c in cells if c != '']
                if cells:
                    html_table += '<tr>\n'
                    for cell in cells:
                        html_table += f'<td>{cell}</td>\n'
                    html_table += '</tr>\n'

        html_table += '</tbody>\n</table>'
        return html_table

    # Match tables
    html = re.sub(r'(\|[^\n]+\|\n)+', convert_table, html)

    # Horizontal rules
    html = re.sub(r'^---+$', r'<hr>', html, flags=re.MULTILINE)

    # Page breaks
    html = re.sub(
        r'<div style="page-break-after: always;"></div>',
        r'<div class="page-break"></div>',
        html
    )

    # Lists - unordered
    def convert_ul(match):
        items = match.group(0).strip().split('\n')
        result = '<ul>\n'
        for item in items:
            item_text = re.sub(r'^[\s]*[-*]\s+', '', item)
            if item_text:
                result += f'<li>{item_text}</li>\n'
        result += '</ul>'
        return result

    html = re.sub(r'(^[\s]*[-*]\s+.+\n?)+', convert_ul, html, flags=re.MULTILINE)

    # Lists - ordered
    def convert_ol(match):
        items = match.group(0).strip().split('\n')
        result = '<ol>\n'
        for item in items:
            item_text = re.sub(r'^[\s]*\d+\.\s+', '', item)
            if item_text:
                result += f'<li>{item_text}</li>\n'
        result += '</ol>'
        return result

    html = re.sub(r'(^[\s]*\d+\.\s+.+\n?)+', convert_ol, html, flags=re.MULTILINE)

    # Paragraphs - wrap text blocks
    lines = html.split('\n')
    result_lines = []
    in_paragraph = False

    for line in lines:
        stripped = line.strip()

        # Check if it's already an HTML tag or special element
        is_html = (stripped.startswith('<') and not stripped.startswith('<em>') and
                   not stripped.startswith('<strong>') and not stripped.startswith('<code>') and
                   not stripped.startswith('<a '))
        is_empty = not stripped
        is_code_placeholder = stripped.startswith('___CODE_BLOCK_')

        if is_html or is_empty or is_code_placeholder:
            if in_paragraph:
                result_lines.append('</p>')
                in_paragraph = False
            result_lines.append(line)
        else:
            if not in_paragraph:
                result_lines.append('<p>')
                in_paragraph = True
            result_lines.append(line)

    if in_paragraph:
        result_lines.append('</p>')

    html = '\n'.join(result_lines)

    # Restore code blocks
    for key, value in code_blocks.items():
        html = html.replace(key, value)

    # Clean up empty paragraphs
    html = re.sub(r'<p>\s*</p>', '', html)
    html = re.sub(r'<p>\s*<(h[1-6]|div|table|figure|ul|ol|pre|hr)', r'<\1', html)
    html = re.sub(r'</(h[1-6]|div|table|figure|ul|ol|pre)>\s*</p>', r'</\1>', html)

    return html

# Read markdown content
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Convert to HTML
html_body = md_to_html(md_content)

# Professional CSS styling for PDF-like appearance
css_styles = """
<style>
    /* Reset and Base Styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    /* Page Setup for PDF */
    @page {
        size: A4;
        margin: 2.5cm 2cm;
        @top-center {
            content: "Machine Learning-Based Network Intrusion Detection System";
            font-size: 10pt;
            color: #666;
        }
        @bottom-center {
            content: "Page " counter(page) " of " counter(pages);
            font-size: 10pt;
            color: #666;
        }
    }

    @media print {
        body {
            print-color-adjust: exact;
            -webkit-print-color-adjust: exact;
        }
        .page-break {
            page-break-after: always;
        }
        h1, h2, h3 {
            page-break-after: avoid;
        }
        table, figure, img {
            page-break-inside: avoid;
        }
    }

    /* Body Styles */
    body {
        font-family: 'Times New Roman', Georgia, serif;
        font-size: 12pt;
        line-height: 1.6;
        color: #333;
        background: #fff;
        max-width: 210mm;
        margin: 0 auto;
        padding: 20mm;
    }

    /* Headings */
    h1 {
        font-size: 24pt;
        font-weight: bold;
        color: #1a1a1a;
        margin-top: 30pt;
        margin-bottom: 18pt;
        border-bottom: 3px solid #2c3e50;
        padding-bottom: 10pt;
        page-break-after: avoid;
    }

    h2 {
        font-size: 18pt;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 24pt;
        margin-bottom: 12pt;
        border-bottom: 2px solid #3498db;
        padding-bottom: 6pt;
        page-break-after: avoid;
    }

    h3 {
        font-size: 14pt;
        font-weight: bold;
        color: #34495e;
        margin-top: 18pt;
        margin-bottom: 10pt;
        page-break-after: avoid;
    }

    h4 {
        font-size: 12pt;
        font-weight: bold;
        color: #555;
        margin-top: 14pt;
        margin-bottom: 8pt;
    }

    /* Paragraphs */
    p {
        margin-bottom: 12pt;
        text-align: justify;
        text-justify: inter-word;
    }

    /* Lists */
    ul, ol {
        margin-left: 25pt;
        margin-bottom: 12pt;
    }

    li {
        margin-bottom: 6pt;
    }

    /* Tables */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 15pt 0;
        font-size: 10pt;
        page-break-inside: avoid;
    }

    th {
        background-color: #2c3e50;
        color: white;
        font-weight: bold;
        padding: 10pt 8pt;
        text-align: left;
        border: 1px solid #2c3e50;
    }

    td {
        padding: 8pt;
        border: 1px solid #ddd;
        vertical-align: top;
    }

    tr:nth-child(even) {
        background-color: #f8f9fa;
    }

    tr:hover {
        background-color: #e8f4f8;
    }

    /* Caption styling for tables */
    table + p em, table + p strong {
        display: block;
        text-align: center;
        font-size: 10pt;
        color: #555;
        margin-top: 8pt;
        margin-bottom: 20pt;
    }

    /* Images and Figures */
    img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 20pt auto;
        border: 1px solid #ddd;
        border-radius: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* Figure captions */
    img + br + em, p > em:only-child {
        display: block;
        text-align: center;
        font-size: 10pt;
        color: #555;
        font-style: italic;
        margin-top: -10pt;
        margin-bottom: 20pt;
    }

    /* Code Blocks */
    pre {
        background-color: #f4f4f4;
        border: 1px solid #ddd;
        border-left: 4px solid #3498db;
        border-radius: 4px;
        padding: 12pt;
        margin: 15pt 0;
        overflow-x: auto;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        font-size: 9pt;
        line-height: 1.4;
        page-break-inside: avoid;
    }

    code {
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        font-size: 9pt;
        background-color: #f4f4f4;
        padding: 2pt 4pt;
        border-radius: 3px;
    }

    pre code {
        background: none;
        padding: 0;
    }

    /* Horizontal Rules */
    hr {
        border: none;
        border-top: 1px solid #ddd;
        margin: 20pt 0;
    }

    /* Strong and Emphasis */
    strong {
        font-weight: bold;
        color: #2c3e50;
    }

    em {
        font-style: italic;
    }

    /* Cover Page Styling */
    .cover-page {
        text-align: center;
        padding: 50pt 0;
        page-break-after: always;
    }

    .cover-page h1 {
        font-size: 28pt;
        border: none;
        margin-bottom: 30pt;
    }

    .cover-page h2 {
        font-size: 22pt;
        border: none;
        color: #3498db;
    }

    .cover-page h3 {
        font-size: 18pt;
        color: #555;
        margin-top: 40pt;
    }

    /* TOC Styling */
    .toc {
        page-break-after: always;
    }

    .toc table {
        font-size: 11pt;
    }

    /* Appendix Styling */
    .appendix h1 {
        background-color: #ecf0f1;
        padding: 15pt;
        border-radius: 4px;
    }

    /* Print-specific adjustments */
    @media print {
        body {
            padding: 0;
            font-size: 11pt;
        }

        h1 { font-size: 20pt; }
        h2 { font-size: 16pt; }
        h3 { font-size: 13pt; }

        pre {
            font-size: 8pt;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        img {
            max-width: 90%;
        }

        table {
            font-size: 9pt;
        }
    }

    /* Blockquote styling */
    blockquote {
        border-left: 4px solid #3498db;
        margin: 15pt 0;
        padding: 10pt 20pt;
        background-color: #f8f9fa;
        font-style: italic;
    }

    /* Links */
    a {
        color: #3498db;
        text-decoration: none;
    }

    a:hover {
        text-decoration: underline;
    }

    /* Page break utility */
    .page-break {
        page-break-after: always;
        height: 0;
        margin: 0;
        border: 0;
    }

    /* Special styling for output blocks */
    .output {
        background-color: #fff;
        border: 1px solid #28a745;
        border-left: 4px solid #28a745;
        padding: 10pt;
        margin: 10pt 0;
        font-family: 'Consolas', monospace;
        font-size: 9pt;
    }

    /* Header info box */
    .info-box {
        background-color: #e7f3ff;
        border: 1px solid #b8daff;
        border-radius: 4px;
        padding: 15pt;
        margin: 15pt 0;
    }

    /* Warning/Note boxes */
    .note {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-left: 4px solid #ffc107;
        padding: 10pt 15pt;
        margin: 15pt 0;
        border-radius: 4px;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 10pt;
        color: #666;
        margin-top: 40pt;
        padding-top: 15pt;
        border-top: 1px solid #ddd;
    }
</style>
"""

# HTML template
html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Machine Learning-Based Network Intrusion Detection System - DACS Group Assignment</title>
    <meta name="author" content="Muhammad Usama Fazal, Imran Shahadat Noble, Md Sohel Rana">
    <meta name="description" content="CT115-3-M Data Analytics in Cyber Security Group Assignment Report">
    {css_styles}
</head>
<body>
    {html_body}

    <div class="footer">
        <p><strong>CT115-3-M Data Analytics in Cyber Security</strong></p>
        <p>Asia Pacific University of Technology and Innovation</p>
        <p>December 2024</p>
    </div>
</body>
</html>
"""

# Post-process HTML to improve formatting
# Convert page break divs
html_template = re.sub(
    r'<div style="page-break-after: always;"></div>',
    '<div class="page-break"></div>',
    html_template
)

# Add proper figure styling
html_template = re.sub(
    r'<p>(<img[^>]+>)</p>\s*<p><em>([^<]+)</em></p>',
    r'<figure>\1<figcaption>\2</figcaption></figure>',
    html_template
)

# Write output
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write(html_template)

print(f"✓ HTML report generated: {OUTPUT_FILE}")
print(f"  - Input: {INPUT_FILE}")
print(f"  - Size: {len(html_template):,} characters")

# Copy figures to the output directory
figures_src = Path("../../figures")
figures_dst = Path("figures")

if figures_src.exists():
    import shutil
    if figures_dst.exists():
        shutil.rmtree(figures_dst)
    shutil.copytree(figures_src, figures_dst)
    print(f"✓ Figures copied to: {figures_dst}")
    print(f"  - {len(list(figures_dst.glob('*.png')))} PNG images")
else:
    print(f"⚠ Figures directory not found at {figures_src}")
    print("  Images will need to be in 'figures/' subdirectory")
