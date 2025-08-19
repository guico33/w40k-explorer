#!/usr/bin/env python3
"""Script to parse all sample HTML pages and generate corresponding JSON files."""

import json
import sys
from pathlib import Path

# Add src to path for new architecture
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from w40k.infrastructure.rag.parser import parse_article_html


def parse_sample_pages():
    """Parse all HTML files in data/sample_pages and save as JSON."""

    # Define paths
    root_dir = Path(__file__).parent.parent
    sample_pages_dir = root_dir / "data" / "sample_pages"
    output_dir = root_dir / "data" / "sample_parsed"

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Find all HTML files
    html_files = list(sample_pages_dir.glob("*.html"))

    if not html_files:
        print("❌ No HTML files found in data/sample_pages/")
        return 1

    print(f"🔍 Found {len(html_files)} HTML files to parse")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 50)

    # Parse each file
    success_count = 0
    failed_count = 0

    for html_file in sorted(html_files):
        try:
            print(f"📄 Processing: {html_file.name}")

            # Read HTML content
            with html_file.open("r", encoding="utf-8") as f:
                html_content = f.read()

            # Parse with our parser
            parsed_doc = parse_article_html(html_content)

            # Generate output filename
            json_filename = html_file.stem + ".json"
            json_file = output_dir / json_filename

            # Save parsed JSON
            with json_file.open("w", encoding="utf-8") as f:
                json.dump(parsed_doc, f, ensure_ascii=False, indent=2)

            # Show summary info
            title = parsed_doc.get("title", "No title")
            lead_length = len(parsed_doc.get("lead", ""))
            sections_count = len(parsed_doc.get("sections", []))
            infobox_type = parsed_doc.get("infobox", {}).get("type")

            print(f"  ✅ '{title}'")
            print(f"     Lead: {lead_length} chars, Sections: {sections_count}")
            if infobox_type:
                print(f"     Infobox: {infobox_type}")
            print(f"     💾 Saved: {json_file.name}")

            success_count += 1

        except Exception as e:
            print(f"  ❌ Failed to parse {html_file.name}: {e}")
            failed_count += 1

        print()

    # Summary
    print("=" * 50)
    print("📊 Parsing Summary")
    print("=" * 50)
    print(f"✅ Successfully parsed: {success_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📁 JSON files saved to: {output_dir}")

    if failed_count == 0:
        print("🎉 All sample pages parsed successfully!")
        return 0
    else:
        print(f"⚠️  {failed_count} files failed to parse")
        return 1


if __name__ == "__main__":
    sys.exit(parse_sample_pages())
