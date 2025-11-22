#!/usr/bin/env python3
"""
Test script for parsing decrypted Saber BSON data
"""

import sys
from pathlib import Path
import bson

def main():
    # Path to the decrypted file
    decrypted_file = Path("/home/sysop/Downloads/ba11646cfae1992948a4ae7d88078d56c18058bfc1736dd5a003c60ae8c7286b.decrypted")

    if not decrypted_file.exists():
        print(f"Error: Decrypted file not found at {decrypted_file}")
        print("Run test_saber_decrypt.py first to create the decrypted file")
        return 1

    print("=" * 70)
    print("Parsing Saber BSON Note Structure")
    print("=" * 70)

    # Read the decrypted BSON data
    with open(decrypted_file, 'rb') as f:
        bson_data = f.read()

    print(f"\n1. File size: {len(bson_data):,} bytes")
    print(f"   First 32 bytes (hex): {bson_data[:32].hex()}")

    # Get the BSON document length from first 4 bytes
    import struct
    doc_length = struct.unpack('<i', bson_data[0:4])[0]
    print(f"   BSON document length: {doc_length:,} bytes")

    # Try to decode as BSON (only the first document)
    print(f"\n2. Attempting to decode BSON...")
    try:
        # BSON decode expects a single document - use only the documented length
        decoded = bson.decode(bson_data[:doc_length])
        print(f"   ✓ Successfully decoded BSON document")

        # Show top-level keys
        print(f"\n3. Top-level keys in document:")
        for key in decoded.keys():
            value = decoded[key]
            value_type = type(value).__name__

            if isinstance(value, (list, dict)):
                length = len(value)
                print(f"   - '{key}': {value_type} (length: {length})")
            else:
                print(f"   - '{key}': {value_type} = {value}")

        # Based on Saber source, we expect these fields:
        # 'v': sbnVersion (int)
        # 'ni': nextImageId (int)
        # 'b': backgroundColor (int - ARGB32)
        # 'p': backgroundPattern (string)
        # 'l': lineHeight (int)
        # 'lt': lineThickness (int)
        # 'z': array of pages
        # 'c': initialPageIndex (int)

        print(f"\n4. Expected Saber fields:")
        print(f"   - File format version (v): {decoded.get('v', 'NOT FOUND')}")
        print(f"   - Next image ID (ni): {decoded.get('ni', 'NOT FOUND')}")
        print(f"   - Background color (b): {decoded.get('b', 'NOT FOUND')}")
        print(f"   - Background pattern (p): {decoded.get('p', 'NOT FOUND')}")
        print(f"   - Line height (l): {decoded.get('l', 'NOT FOUND')}")
        print(f"   - Line thickness (lt): {decoded.get('lt', 'NOT FOUND')}")
        print(f"   - Current page index (c): {decoded.get('c', 'NOT FOUND')}")

        # Check pages array
        if 'z' in decoded:
            pages = decoded['z']
            print(f"\n5. Pages array (z): {len(pages)} page(s)")

            # Examine first page structure
            if len(pages) > 0:
                first_page = pages[0]
                print(f"\n   First page structure:")
                for key in first_page.keys():
                    value = first_page[key]
                    value_type = type(value).__name__

                    if isinstance(value, (list, dict)):
                        length = len(value)
                        print(f"     - '{key}': {value_type} (length: {length})")
                    else:
                        print(f"     - '{key}': {value_type} = {value}")

                # Look for strokes
                if 's' in first_page:
                    strokes = first_page['s']
                    print(f"\n   Found {len(strokes)} stroke(s) on first page")

                    if len(strokes) > 0:
                        first_stroke = strokes[0]
                        print(f"\n   First stroke structure:")
                        for key in first_stroke.keys():
                            value = first_stroke[key]
                            value_type = type(value).__name__

                            if isinstance(value, (list, dict)):
                                length = len(value)
                                print(f"     - '{key}': {value_type} (length: {length})")

                                # If it's the points array, show first few points
                                if key == 'p' and isinstance(value, list) and len(value) > 0:
                                    print(f"       First 3 points: {value[:3]}")
                            else:
                                print(f"     - '{key}': {value_type} = {value}")
        else:
            print(f"\n5. ✗ No pages array found!")

        print("\n" + "=" * 70)
        print("BSON parsing successful!")
        print("=" * 70)
        return 0

    except Exception as e:
        print(f"   ✗ Failed to decode BSON: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
