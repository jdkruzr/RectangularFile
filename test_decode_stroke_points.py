#!/usr/bin/env python3
"""
Test decoding stroke point coordinates from BSON binary format
"""

import sys
import struct
from pathlib import Path
import bson

def main():
    decrypted_file = Path("/home/sysop/Downloads/ba11646cfae1992948a4ae7d88078d56c18058bfc1736dd5a003c60ae8c7286b.decrypted")

    if not decrypted_file.exists():
        print(f"Error: Decrypted file not found")
        return 1

    print("=" * 70)
    print("Decoding Saber Stroke Points")
    print("=" * 70)

    # Read and decode BSON
    with open(decrypted_file, 'rb') as f:
        bson_data = f.read()

    doc_length = struct.unpack('<i', bson_data[0:4])[0]
    decoded = bson.decode(bson_data[:doc_length])

    # Get first page
    pages = decoded['z']
    first_page = pages[0]
    strokes = first_page['s']

    print(f"\nPage dimensions: {first_page['w']:.1f} x {first_page['h']:.1f}")
    print(f"Number of strokes: {len(strokes)}")

    # Decode first stroke's points
    first_stroke = strokes[0]
    print(f"\nFirst stroke details:")
    print(f"  Tool type: {first_stroke['ty']}")
    print(f"  Color: 0x{first_stroke['c']:08x}")
    print(f"  Size: {first_stroke['s']}")
    print(f"  Number of points: {len(first_stroke['p'])}")

    # Decode point coordinates
    print(f"\n  Decoding points (binary → x, y, pressure):")
    points = first_stroke['p']

    for i, point_bytes in enumerate(points[:10]):  # Show first 10 points
        if len(point_bytes) == 12:  # 3 floats × 4 bytes
            # Unpack as 3 little-endian floats
            x, y, pressure = struct.unpack('<fff', point_bytes)
            print(f"    Point {i}: x={x:7.2f}, y={y:7.2f}, pressure={pressure:.4f}")
        else:
            print(f"    Point {i}: Unexpected size: {len(point_bytes)} bytes")

    # Try second stroke if it exists
    if len(strokes) > 1:
        second_stroke = strokes[1]
        print(f"\nSecond stroke details:")
        print(f"  Tool type: {second_stroke['ty']}")
        print(f"  Number of points: {len(second_stroke['p'])}")

        points2 = second_stroke['p']
        if len(points2) > 0 and len(points2[0]) == 12:
            x, y, pressure = struct.unpack('<fff', points2[0])
            print(f"  First point: x={x:7.2f}, y={y:7.2f}, pressure={pressure:.4f}")

    # Show page 2 if it exists
    if len(pages) > 1:
        second_page = pages[1]
        print(f"\nSecond page:")
        print(f"  Dimensions: {second_page['w']:.1f} x {second_page['h']:.1f}")
        print(f"  Number of strokes: {len(second_page.get('s', []))}")

    print("\n" + "=" * 70)
    print("Point decoding successful!")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
