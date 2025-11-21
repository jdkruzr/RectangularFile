#!/usr/bin/env python3
"""
Test script for Saber decryption
"""

import sys
from pathlib import Path
from processing.saber_decryptor import SaberDecryptor

def main():
    # Test parameters
    saber_folder = Path("/home/sysop/Downloads/test_saber")
    password = "ehh1701jqb"
    encrypted_file = Path("/home/sysop/Downloads/ba11646cfae1992948a4ae7d88078d56c18058bfc1736dd5a003c60ae8c7286b.sbe")
    encrypted_filename = "ba11646cfae1992948a4ae7d88078d56c18058bfc1736dd5a003c60ae8c7286b"

    print("=" * 70)
    print("Testing Saber Decryption")
    print("=" * 70)

    # Initialize decryptor
    print(f"\n1. Initializing decryptor...")
    decryptor = SaberDecryptor(password, saber_folder)
    print("   ✓ Decryptor initialized")

    # Test filename decryption
    print(f"\n2. Testing filename decryption...")
    print(f"   Encrypted: {encrypted_filename[:40]}...")
    try:
        decrypted_path = decryptor.decrypt_filename(encrypted_filename)
        print(f"   ✓ Decrypted path: {decrypted_path}")
    except Exception as e:
        print(f"   ✗ Filename decryption failed: {e}")
        return 1

    # Test file content decryption
    print(f"\n3. Testing file content decryption...")
    print(f"   File: {encrypted_file.name}")
    try:
        decrypted_content = decryptor.decrypt_file(encrypted_file)
        print(f"   ✓ Successfully decrypted {len(decrypted_content):,} bytes")

        # Show first bytes
        print(f"\n   First 200 bytes (hex):")
        print(f"   {decrypted_content[:200].hex()}")

        # Try to detect if it's BSON
        if decrypted_content[:4] == b'\x00\x00\x00\x00' or decrypted_content[0:1] in [b'{', b'[']:
            print(f"\n   Content type: Possibly JSON or BSON")
        else:
            print(f"\n   Content starts with: {decrypted_content[:50]}")

        # Save decrypted file
        output_path = encrypted_file.parent / f"{encrypted_file.stem}.decrypted"
        with open(output_path, 'wb') as f:
            f.write(decrypted_content)
        print(f"\n   ✓ Saved decrypted content to: {output_path}")

    except Exception as e:
        print(f"   ✗ File decryption failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 70)
    print("All tests passed! 🎉")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
