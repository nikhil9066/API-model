#!/usr/bin/env python3
"""
test_md_to_pdf.py
Simple test script to convert markdown to PDF
Usage: python3 test_md_to_pdf.py
"""

import subprocess
import sys
from pathlib import Path

def test_pandoc_installation():
    """Test if pandoc is installed and working"""
    try:
        result = subprocess.run(['pandoc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✅ Pandoc found: {version}")
            return True
        else:
            print("❌ Pandoc installed but not working properly")
            return False
    except FileNotFoundError:
        print("❌ Pandoc not found")
        return False

def create_test_markdown():
    """Create a simple test markdown file"""
    test_md_content = """# Test Report
## AutoML Analysis Test

This is a **test report** to verify markdown to PDF conversion.

### Key Features
- Simple formatting
- Tables support
- Image embedding test

### Sample Table

| Model | Score | Time |
|-------|--------|------|
| Ridge | 0.6662 | 0.015s |
| Lasso | 0.6503 | 0.047s |

### Sample Code Block

```python
def test_function():
    return "Hello World"
```

### Performance Summary

This is a test paragraph with some **bold text** and *italic text*.

> This is a blockquote for testing purposes.

---

*Test report generated successfully*
"""
    
    test_file = Path("test_report.md")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_md_content)
    
    print(f"✅ Test markdown created: {test_file}")
    return test_file

def convert_with_basic_pandoc(md_file, pdf_file):
    """Try basic pandoc conversion"""
    cmd = [
        'pandoc',
        str(md_file),
        '-o', str(pdf_file),
        '--pdf-engine=xelatex'
    ]
    
    print("🔄 Trying basic conversion...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Basic conversion successful: {pdf_file}")
        return True
    else:
        print(f"❌ Basic conversion failed: {result.stderr}")
        return False

def convert_with_minimal_options(md_file, pdf_file):
    """Try minimal pandoc options"""
    cmd = [
        'pandoc',
        str(md_file),
        '-o', str(pdf_file),
        '--pdf-engine=xelatex',
        '-V', 'geometry:margin=1in',
        '--toc'
    ]
    
    print("🔄 Trying minimal options...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Minimal conversion successful: {pdf_file}")
        return True
    else:
        print(f"❌ Minimal conversion failed: {result.stderr}")
        return False

def convert_with_pdflatex(md_file, pdf_file):
    """Try with pdflatex engine"""
    cmd = [
        'pandoc',
        str(md_file),
        '-o', str(pdf_file),
        '--pdf-engine=pdflatex'
    ]
    
    print("🔄 Trying pdflatex engine...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ PDFLaTeX conversion successful: {pdf_file}")
        return True
    else:
        print(f"❌ PDFLaTeX conversion failed: {result.stderr}")
        return False

def try_alternative_methods(md_file):
    """Try alternative conversion methods"""
    print("\n🔧 Trying alternative methods...")
    
    # Method 1: markdown library + weasyprint
    try:
        import markdown
        import weasyprint
        
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        html = markdown.markdown(md_content, extensions=['tables', 'codehilite'])
        html_with_style = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 4px; }}
                blockquote {{ border-left: 4px solid #ccc; margin: 0; padding-left: 20px; }}
            </style>
        </head>
        <body>
        {html}
        </body>
        </html>
        """
        
        pdf_file = md_file.with_suffix('.weasyprint.pdf')
        weasyprint.HTML(string=html_with_style).write_pdf(str(pdf_file))
        print(f"✅ WeasyPrint conversion successful: {pdf_file}")
        return True
        
    except ImportError:
        print("❌ WeasyPrint not available (pip install weasyprint markdown)")
    except Exception as e:
        print(f"❌ WeasyPrint conversion failed: {e}")
    
    # Method 2: markdown2 + pdfkit
    try:
        import markdown2
        import pdfkit
        
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        html = markdown2.markdown(md_content, extras=['tables', 'code-friendly'])
        html_with_style = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
        {html}
        </body>
        </html>
        """
        
        pdf_file = md_file.with_suffix('.pdfkit.pdf')
        pdfkit.from_string(html_with_style, str(pdf_file))
        print(f"✅ PDFKit conversion successful: {pdf_file}")
        return True
        
    except ImportError:
        print("❌ PDFKit not available (pip install pdfkit markdown2)")
    except Exception as e:
        print(f"❌ PDFKit conversion failed: {e}")
    
    return False

def main():
    """Main test function"""
    print("🧪 Markdown to PDF Conversion Test")
    print("=" * 40)
    
    # Test pandoc installation
    if not test_pandoc_installation():
        print("\n📥 Install pandoc:")
        print("   Ubuntu/Debian: sudo apt-get install pandoc texlive-xetex")
        print("   macOS: brew install pandoc basictex")
        print("   Windows: https://pandoc.org/installing.html")
        
        # Try alternative methods
        md_file = create_test_markdown()
        if try_alternative_methods(md_file):
            print("\n✅ Alternative conversion method worked!")
        else:
            print("\n❌ All conversion methods failed")
        return
    
    # Create test file
    md_file = create_test_markdown()
    
    # Try different pandoc approaches
    pdf_file = md_file.with_suffix('.basic.pdf')
    if convert_with_basic_pandoc(md_file, pdf_file):
        print(f"\n🎉 SUCCESS! Check your PDF: {pdf_file}")
        return
    
    pdf_file = md_file.with_suffix('.minimal.pdf')
    if convert_with_minimal_options(md_file, pdf_file):
        print(f"\n🎉 SUCCESS! Check your PDF: {pdf_file}")
        return
    
    pdf_file = md_file.with_suffix('.pdflatex.pdf')
    if convert_with_pdflatex(md_file, pdf_file):
        print(f"\n🎉 SUCCESS! Check your PDF: {pdf_file}")
        return
    
    # Try alternative methods
    if try_alternative_methods(md_file):
        print("\n✅ Alternative method worked!")
    else:
        print("\n❌ All conversion methods failed")
        print("\n🔧 Troubleshooting suggestions:")
        print("   1. Install missing LaTeX packages: sudo apt-get install texlive-full")
        print("   2. Try: pip install weasyprint markdown")
        print("   3. Check pandoc version: pandoc --version")

if __name__ == "__main__":
    main()