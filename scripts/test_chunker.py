#!/usr/bin/env python3
"""Comprehensive test script for the chunker to validate performance across sample files."""

import json
import sys
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.rag.chunker import chunk_article, ChunkParams
from src.database.models import Chunk


def test_chunker():
    """Test chunker against all sample parsed files and validate results."""
    
    # Define paths
    root_dir = Path(__file__).parent.parent
    sample_parsed_dir = root_dir / "data" / "sample_parsed"
    
    if not sample_parsed_dir.exists():
        print("❌ No sample_parsed directory found. Run parse_sample_pages.py first.")
        return 1
    
    # Find all JSON files
    json_files = list(sample_parsed_dir.glob("*.json"))
    
    if not json_files:
        print("❌ No JSON files found in data/sample_parsed/")
        return 1
    
    print(f"🧪 Testing chunker against {len(json_files)} parsed files")
    print("=" * 60)
    
    # Test parameters
    params = ChunkParams()
    
    # Statistics tracking
    stats = {
        "total_files": len(json_files),
        "successful_files": 0,
        "failed_files": 0,
        "total_chunks": 0,
        "chunks_by_type": {},
        "token_stats": {"min": float("inf"), "max": 0, "total": 0},
        "articles_by_id": {},
        "duplicate_uids": set(),
        "all_uids": set()
    }
    
    failed_files = []
    
    for json_file in sorted(json_files):
        try:
            print(f"🔄 Processing: {json_file.name}")
            
            # Load parsed data
            with json_file.open("r", encoding="utf-8") as f:
                parsed_data = json.load(f)
            
            # Generate chunks
            chunks = chunk_article(parsed_data, params)
            
            if not chunks:
                print(f"  ⚠️  No chunks generated")
                continue
            
            # Validate and collect statistics
            article_title = parsed_data.get("title", "Unknown")
            article_id = None
            
            # Extract article_id from first chunk
            if chunks:
                article_id = chunks[0].article_id
                stats["articles_by_id"][article_id] = article_title
            
            print(f"  ✅ Generated {len(chunks)} chunks for '{article_title}'")
            print(f"     Article ID: {article_id}")
            
            # Validate chunk properties
            valid_chunks = 0
            for i, chunk in enumerate(chunks):
                # Check required fields
                if not chunk.chunk_uid:
                    print(f"     ❌ Chunk {i}: Missing chunk_uid")
                    continue
                
                if not chunk.text.strip():
                    print(f"     ❌ Chunk {i}: Empty text")
                    continue
                
                # Check for duplicate UIDs
                if chunk.chunk_uid in stats["all_uids"]:
                    stats["duplicate_uids"].add(chunk.chunk_uid)
                    print(f"     ⚠️  Duplicate chunk_uid: {chunk.chunk_uid[:16]}...")
                else:
                    stats["all_uids"].add(chunk.chunk_uid)
                
                # Track block types
                block_type = str(chunk.block_type)
                stats["chunks_by_type"][block_type] = stats["chunks_by_type"].get(block_type, 0) + 1
                
                # Track token statistics
                if chunk.token_count:
                    stats["token_stats"]["min"] = min(stats["token_stats"]["min"], chunk.token_count)
                    stats["token_stats"]["max"] = max(stats["token_stats"]["max"], chunk.token_count)
                    stats["token_stats"]["total"] += chunk.token_count
                
                valid_chunks += 1
            
            print(f"     Valid chunks: {valid_chunks}/{len(chunks)}")
            
            # Show sample chunk
            if chunks:
                sample = chunks[0]
                section_path = json.loads(sample.section_path) if sample.section_path else []
                print(f"     Sample: {' > '.join(section_path) if section_path else 'Root'}")
                print(f"     Text preview: {sample.text[:80]}...")
            
            stats["total_chunks"] += len(chunks)
            stats["successful_files"] += 1
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            failed_files.append((json_file.name, str(e)))
            stats["failed_files"] += 1
        
        print()
    
    # Final statistics
    print("=" * 60)
    print("📊 Chunker Test Results")
    print("=" * 60)
    
    print(f"📁 Files processed: {stats['total_files']}")
    print(f"✅ Successful: {stats['successful_files']}")
    print(f"❌ Failed: {stats['failed_files']}")
    print()
    
    print(f"🧩 Total chunks generated: {stats['total_chunks']}")
    print(f"📑 Unique articles: {len(stats['articles_by_id'])}")
    print(f"🔗 Unique chunk UIDs: {len(stats['all_uids'])}")
    
    if stats["duplicate_uids"]:
        print(f"⚠️  Duplicate UIDs detected: {len(stats['duplicate_uids'])}")
    else:
        print("✅ All chunk UIDs are unique")
    
    print()
    
    # Token statistics
    if stats["token_stats"]["total"] > 0:
        avg_tokens = stats["token_stats"]["total"] / stats["total_chunks"]
        print(f"📏 Token statistics:")
        print(f"   Min: {stats['token_stats']['min']} tokens")
        print(f"   Max: {stats['token_stats']['max']} tokens") 
        print(f"   Avg: {avg_tokens:.1f} tokens")
        print(f"   Target: {params.target_tokens} tokens")
        print()
    
    # Block type distribution
    if stats["chunks_by_type"]:
        print("📊 Block type distribution:")
        for block_type, count in sorted(stats["chunks_by_type"].items()):
            percentage = (count / stats["total_chunks"]) * 100
            print(f"   {block_type}: {count} ({percentage:.1f}%)")
        print()
    
    # Show articles by ID
    if stats["articles_by_id"]:
        print("📋 Article ID mapping:")
        for article_id, title in sorted(stats["articles_by_id"].items()):
            print(f"   {article_id}: {title}")
        print()
    
    # Failed files details
    if failed_files:
        print("❌ Failed files:")
        for filename, error in failed_files:
            print(f"   {filename}: {error}")
        print()
    
    # Validation summary
    if stats["failed_files"] == 0 and not stats["duplicate_uids"]:
        print("🎉 All tests passed! Chunker is working correctly.")
        print("✅ Ready for database integration testing")
        return 0
    else:
        print("⚠️  Issues detected that should be addressed")
        return 1


if __name__ == "__main__":
    sys.exit(test_chunker())