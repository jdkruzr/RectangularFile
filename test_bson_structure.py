#!/usr/bin/env python3
"""
Analyze the BSON structure to understand how it's formatted
"""

import sys
from pathlib import Path
import struct

def main():
    decrypted_file = Path("/home/sysop/Downloads/ba11646cfae1992948a4ae7d88078d56c18058bfc1736dd5a003c60ae8c7286b.decrypted")

    if not decrypted_file.exists():
        print(f"Error: Decrypted file not found")
        return 1

    print("=" * 70)
    print("Analyzing BSON Structure")
    print("=" * 70)

    with open(decrypted_file, 'rb') as f:
        data = f.read()

    print(f"\n1. Total file size: {len(data):,} bytes")
    print(f"   First 64 bytes (hex):")
    for i in range(0, 64, 16):
        hex_str = data[i:i+16].hex(' ')
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data[i:i+16])
        print(f"   {i:04x}: {hex_str:48s} {ascii_str}")

    # Read first 4 bytes as little-endian int32 (BSON document length)
    if len(data) >= 4:
        doc_length = struct.unpack('<i', data[0:4])[0]
        print(f"\n2. First 4 bytes as int32 (little-endian): {doc_length}")
        print(f"   This should be the BSON document length")
        print(f"   File size: {len(data)}")
        print(f"   Expected doc length: {doc_length}")

        if doc_length > 0 and doc_length <= len(data):
            print(f"   ✓ Document length looks reasonable")

            # Check if the byte at position doc_length-1 is 0x00 (EOO marker)
            if data[doc_length - 1] == 0x00:
                print(f"   ✓ Found EOO marker (0x00) at position {doc_length - 1}")
            else:
                print(f"   ✗ Expected EOO marker (0x00) at position {doc_length - 1}")
                print(f"     Found: 0x{data[doc_length - 1]:02x}")

            # Try to decode just the first document
            print(f"\n3. Attempting to decode first {doc_length} bytes as BSON...")
            try:
                import bson
                first_doc = bson.decode(data[:doc_length])
                print(f"   ✓ Successfully decoded first BSON document!")
                print(f"\n   Keys in document: {list(first_doc.keys())}")

                # Check if there's more data after the first document
                if len(data) > doc_length:
                    remaining = len(data) - doc_length
                    print(f"\n4. Found {remaining} bytes after first document")
                    print(f"   This might be a second document or trailing data")

                    # Try to parse as second document
                    second_data = data[doc_length:]
                    if len(second_data) >= 4:
                        second_length = struct.unpack('<i', second_data[0:4])[0]
                        print(f"   Second section length: {second_length}")
                else:
                    print(f"\n4. No data after first document (perfect match)")

                return 0

            except Exception as e:
                print(f"   ✗ Failed to decode: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"   ✗ Document length seems invalid")
            print(f"   Might not be BSON or data is corrupted")

    print("\n" + "=" * 70)
    return 1

if __name__ == "__main__":
    sys.exit(main())
