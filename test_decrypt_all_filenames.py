#!/usr/bin/env python3
"""
Test script to decrypt all Saber filenames in a directory
"""

import sys
import os
from pathlib import Path
from processing.saber_decryptor import SaberDecryptor

def main():
    saber_folder = Path("/mnt/saber")
    password = os.environ.get('SABER_PASSWORD', '')

    if not password:
        print("Error: SABER_PASSWORD environment variable not set")
        print("Usage: SABER_PASSWORD='your_password' python3 test_decrypt_all_filenames.py")
        return 1

    print("=" * 70)
    print("Decrypting all Saber filenames")
    print("=" * 70)

    # Initialize decryptor
    decryptor = SaberDecryptor(password, saber_folder)

    # Get all .sbe files
    saber_files_dir = saber_folder
    sbe_files = sorted(saber_files_dir.glob("*.sbe"))

    if not sbe_files:
        print(f"\nNo .sbe files found in {saber_files_dir}")
        return 1

    print(f"\nFound {len(sbe_files)} .sbe file(s):\n")

    for sbe_file in sbe_files:
        encrypted_filename = sbe_file.stem  # Remove .sbe extension
        file_size = sbe_file.stat().st_size
        mtime = sbe_file.stat().st_mtime

        try:
            decrypted_path = decryptor.decrypt_filename(encrypted_filename)
            print(f"File: {encrypted_filename[:40]}...")
            print(f"  → Decrypted: {decrypted_path}")
            print(f"  → Size: {file_size:,} bytes")
            print(f"  → Modified: {mtime}")
            print()
        except Exception as e:
            print(f"File: {encrypted_filename[:40]}...")
            print(f"  ✗ Failed to decrypt: {e}")
            print()

    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
