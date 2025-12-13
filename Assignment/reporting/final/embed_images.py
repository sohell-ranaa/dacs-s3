#!/usr/bin/env python3
"""
Embed images as base64 in HTML file
"""

import re
import base64
from pathlib import Path

HTML_FILE = "REPORT.html"
OUTPUT_FILE = "REPORT_embedded.html"

# Read HTML
with open(HTML_FILE, 'r', encoding='utf-8') as f:
    html = f.read()

# Find all image sources and replace with base64
def embed_image(match):
    src = match.group(1)
    img_path = Path(src)

    if img_path.exists():
        with open(img_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')

        # Determine mime type
        ext = img_path.suffix.lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml'
        }
        mime = mime_types.get(ext, 'image/png')

        return f'src="data:{mime};base64,{img_data}"'
    else:
        print(f"Warning: Image not found: {src}")
        return match.group(0)

# Replace all src="figures/..." with base64
html = re.sub(r'src="(figures/[^"]+)"', embed_image, html)

# Write output
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"✓ Created {OUTPUT_FILE} with embedded images")
print(f"  Size: {len(html):,} characters ({len(html)/1024/1024:.1f} MB)")

# Count embedded images
count = html.count('data:image/')
print(f"  Embedded images: {count}")
