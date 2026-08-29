import os
import re
import subprocess
from pathlib import Path
import html
import markdown

def build_printable_html(md_path: Path, output_html: Path):
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert markdown to clean HTML using python-markdown with tables and fenced_code
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc', 'sane_lists', 'nl2br'])
    body_html = md.convert(md_content)

    # Wrap tables in responsive / printable table containers for clean borders and page-breaks
    body_html = re.sub(r'<table>', r'<div class="table-container"><table>', body_html)
    body_html = re.sub(r'</table>', r'</table></div>', body_html)

    # Convert mermaid code blocks to interactive SVG-rendering mermaid divs
    def replace_mermaid(match):
        code_content = match.group(1)
        # Unescape HTML entities that markdown converter introduced so mermaid syntax is pure
        unescaped_code = html.unescape(code_content).strip()
        return f'<div class="mermaid-container"><div class="mermaid">\n{unescaped_code}\n</div></div>'

    body_html = re.sub(r'<pre><code class="language-mermaid">([\s\S]*?)</code></pre>', replace_mermaid, body_html)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Legal Policy & Document Analyzer — Project Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'neutral',
            securityLevel: 'loose',
            fontFamily: 'Inter, sans-serif'
        }});
    </script>
    <style>
        @page {{
            size: A4;
            margin: 16mm 14mm 16mm 14mm;
            @bottom-right {{
                content: counter(page);
                font-family: 'Inter', sans-serif;
                font-size: 8pt;
                color: #6b7280;
            }}
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #1f2937;
            background: #ffffff;
            font-size: 9.5pt;
            line-height: 1.6;
            padding: 2.5rem 3rem;
            max-width: 980px;
            margin: 0 auto;
        }}

        @media screen {{
            body {{
                background: #f9fafb;
                box-shadow: 0 4px 25px rgba(0,0,0,0.08);
                margin: 2rem auto;
                border-radius: 8px;
            }}
        }}

        h1 {{
            font-size: 1.85rem;
            font-weight: 800;
            color: #111827;
            border-bottom: 3px solid #4f46e5;
            padding-bottom: 0.75rem;
            margin-bottom: 1.5rem;
            letter-spacing: -0.02em;
        }}

        h2 {{
            font-size: 1.35rem;
            font-weight: 700;
            color: #1e1b4b;
            border-bottom: 1.5px solid #e0e7ff;
            padding-bottom: 0.4rem;
            margin-top: 2.2rem;
            margin-bottom: 0.9rem;
            page-break-after: avoid;
        }}

        h3 {{
            font-size: 1.08rem;
            font-weight: 600;
            color: #312e81;
            margin-top: 1.4rem;
            margin-bottom: 0.6rem;
            page-break-after: avoid;
        }}

        h4 {{
            font-size: 0.95rem;
            font-weight: 600;
            color: #4338ca;
            margin-top: 1.1rem;
            margin-bottom: 0.4rem;
            page-break-after: avoid;
        }}

        p {{
            margin-bottom: 0.85rem;
            color: #374151;
        }}

        strong {{
            color: #111827;
            font-weight: 600;
        }}

        ul, ol {{
            margin-bottom: 0.95rem;
            padding-left: 1.5rem;
            color: #374151;
        }}

        li {{
            margin-bottom: 0.35rem;
        }}

        /* Table Styling */
        .table-container {{
            margin: 1.25rem 0 1.5rem 0;
            overflow-x: auto;
            page-break-inside: avoid;
            border-radius: 6px;
            border: 1px solid #e5e7eb;
            background: #ffffff;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 8.8pt;
            text-align: left;
        }}

        thead {{
            background: #f1f5f9;
        }}

        th {{
            background: #f1f5f9;
            color: #0f172a;
            font-weight: 700;
            padding: 8px 12px;
            border-bottom: 2px solid #cbd5e1;
            border-right: 1px solid #e2e8f0;
            white-space: nowrap;
        }}

        th:last-child {{
            border-right: none;
        }}

        td {{
            padding: 7px 12px;
            border-bottom: 1px solid #e2e8f0;
            border-right: 1px solid #f1f5f9;
            color: #334155;
            vertical-align: top;
            line-height: 1.5;
        }}

        td:last-child {{
            border-right: none;
        }}

        tr:nth-child(even) td {{
            background: #f8fafc;
        }}

        tr:hover td {{
            background: #f1f5f9;
        }}

        /* Mermaid Diagram Styling */
        .mermaid-container {{
            margin: 1.5rem 0;
            padding: 1.25rem;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow-x: auto;
            page-break-inside: avoid;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}

        .mermaid {{
            width: 100%;
            display: flex;
            justify-content: center;
        }}

        .mermaid svg {{
            max-width: 100%;
            height: auto;
        }}

        /* Code Styling */
        code {{
            font-family: 'JetBrains Mono', Consolas, Monaco, monospace;
            font-size: 8.2pt;
            background: #f1f5f9;
            color: #4338ca;
            padding: 1.5px 5px;
            border-radius: 4px;
            border: 1px solid #e2e8f0;
        }}

        pre {{
            background: #0f172a;
            color: #f8fafc;
            padding: 1rem 1.25rem;
            border-radius: 6px;
            margin: 1rem 0 1.25rem 0;
            overflow-x: auto;
            page-break-inside: avoid;
            border: 1px solid #1e293b;
        }}

        pre code {{
            background: transparent;
            color: #e2e8f0;
            padding: 0;
            border: none;
            font-size: 8pt;
            line-height: 1.5;
        }}

        blockquote {{
            border-left: 3.5px solid #6366f1;
            background: #f5f3ff;
            color: #4338ca;
            padding: 0.6rem 1.1rem;
            margin: 0.9rem 0;
            border-radius: 0 6px 6px 0;
            font-size: 9pt;
        }}

        hr {{
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 1.75rem 0;
        }}

        @media print {{
            body {{
                padding: 0;
                background: #ffffff;
                font-size: 9pt;
            }}
            .table-container {{
                border: 1px solid #cbd5e1;
            }}
            .mermaid-container {{
                border: 1px solid #d1d5db;
                box-shadow: none;
                padding: 0.75rem;
            }}
            th {{
                background: #f1f5f9 !important;
                color: #0f172a !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
            tr:nth-child(even) td {{
                background: #f8fafc !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
            pre {{
                background: #1e293b !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
            blockquote {{
                background: #f5f3ff !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
        }}
    </style>
</head>
<body>
{body_html}
</body>
</html>
"""

    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(full_html)

if __name__ == '__main__':
    project_root = Path(r"c:\Users\apex\Desktop\Project\AI-Legal-Policy-and-Document-Analyzer")
    md_path = project_root / "REPORT.md"
    if not md_path.exists():
        md_path = project_root / "project_report.md"
    html_out = project_root / "project_report.html"
    pdf_out = project_root / "AI_Legal_Analyzer_Project_Report.pdf"

    print(f"Building HTML report with Mermaid diagrams at: {html_out}")
    build_printable_html(md_path, html_out)
    print("HTML report built successfully.")

    # Convert to PDF via headless Edge / Chrome
    browser_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    ]
    
    browser_bin = None
    for p in browser_paths:
        if os.path.exists(p):
            browser_bin = p
            break

    if browser_bin:
        print(f"Generating PDF using headless browser: {browser_bin}")
        cmd = [
            browser_bin,
            "--headless",
            "--disable-gpu",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=3000",
            f"--print-to-pdf={str(pdf_out)}",
            "--no-pdf-header-footer",
            str(html_out)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if pdf_out.exists():
            print(f"SUCCESS: Generated PDF at {pdf_out} (Size: {pdf_out.stat().st_size} bytes)")
        else:
            print(f"PDF generation failed: {res.stderr}")
    else:
        print("No headless browser found to compile PDF.")
