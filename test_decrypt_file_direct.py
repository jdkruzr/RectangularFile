#!/usr/bin/env python3
"""
Test decrypting file directly with password-derived key (no config.sbc key)
"""

import json
import hashlib
import base64
from pathlib import Path
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

# Test parameters
password = "ehh1701jqb"
REPRODUCIBLE_SALT = "8MnPs64@R&mF8XjWeLrD"

# Paths
config_path = Path("/home/sysop/Downloads/test_saber/Saber/config.sbc")
encrypted_file = Path("/home/sysop/Downloads/ba11646cfae1992948a4ae7d88078d56c18058bfc1736dd5a003c60ae8c7286b.sbe")

print("=" * 70)
print("Testing: Decrypt file directly with password-derived key")
print("=" * 70)

# Step 1: Derive password key
password_with_salt = password + REPRODUCIBLE_SALT
password_key = hashlib.sha256(password_with_salt.encode('utf-8')).digest()
print(f"\n1. Password key (hex): {password_key.hex()}")

# Step 2: Load IV from config
with open(config_path, 'r') as f:
    config = json.load(f)
iv = base64.b64decode(config['iv'])
print(f"\n2. IV (hex): {iv.hex()}")

# Step 3: Read encrypted file
with open(encrypted_file, 'rb') as f:
    encrypted_data = f.read()
print(f"\n3. Encrypted file size: {len(encrypted_data)} bytes")
print(f"   First 32 bytes (hex): {encrypted_data[:32].hex()}")

# Step 4: Try to decrypt with password key
print(f"\n4. Attempting to decrypt file with password-derived key...")
try:
    cipher = AES.new(password_key, AES.MODE_CBC, iv)
    decrypted_padded = cipher.decrypt(encrypted_data)

    print(f"   Decrypted size (with padding): {len(decrypted_padded)} bytes")
    print(f"   First 32 bytes (hex): {decrypted_padded[:32].hex()}")
    print(f"   Last 16 bytes (hex): {decrypted_padded[-16:].hex()}")

    # Try to unpad
    try:
        decrypted = unpad(decrypted_padded, AES.block_size)
        print(f"\n   ✓✓✓ SUCCESS! Decrypted {len(decrypted)} bytes")
        print(f"   First 100 bytes (hex): {decrypted[:100].hex()}")

        # Check if it looks like BSON
        if decrypted[:4] == b'\x00\x00\x00\x00' or decrypted[0:1] in [b'{', b'[']:
            print(f"   Content type: Possibly JSON or BSON")

        # Save it
        output_path = encrypted_file.parent / f"{encrypted_file.stem}.decrypted_direct"
        with open(output_path, 'wb') as f:
            f.write(decrypted)
        print(f"   Saved to: {output_path}")

    except ValueError as e:
        print(f"\n   ✗ Unpadding failed: {e}")
        print(f"   Last byte: 0x{decrypted_padded[-1]:02x}")

except Exception as e:
    print(f"   ✗ Decryption failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
