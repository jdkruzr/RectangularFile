#!/usr/bin/env python3
"""
Test decrypting file with CTR mode instead of CBC
"""

import json
import hashlib
import base64
from pathlib import Path
from Crypto.Cipher import AES
from Crypto.Util import Counter

# Test parameters
password = "ehh1701jqb"
REPRODUCIBLE_SALT = "8MnPs64@R&mF8XjWeLrD"

# Paths
config_path = Path("/home/sysop/Downloads/test_saber/Saber/config.sbc")
encrypted_file = Path("/home/sysop/Downloads/ba11646cfae1992948a4ae7d88078d56c18058bfc1736dd5a003c60ae8c7286b.sbe")

print("=" * 70)
print("Testing: Decrypt file with AES-CTR mode")
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
print(f"   IV length: {len(iv)} bytes")

# Step 3: Read encrypted file
with open(encrypted_file, 'rb') as f:
    encrypted_data = f.read()
print(f"\n3. Encrypted file size: {len(encrypted_data)} bytes")

# Step 4: Try CTR mode
print(f"\n4. Attempting to decrypt with AES-CTR mode...")
try:
    # CTR mode doesn't use padding - it's a stream cipher
    # Convert IV to counter (CTR mode needs a counter, not an IV)
    # The Dart encrypt package uses the IV as the initial counter value
    counter_val = int.from_bytes(iv, byteorder='big')
    ctr = Counter.new(128, initial_value=counter_val)

    cipher = AES.new(password_key, AES.MODE_CTR, counter=ctr)
    decrypted = cipher.decrypt(encrypted_data)

    print(f"   ✓✓✓ SUCCESS! Decrypted {len(decrypted)} bytes")
    print(f"   First 100 bytes (hex): {decrypted[:100].hex()}")
    print(f"   First 100 bytes (raw): {decrypted[:100]}")

    # Check if it looks like BSON
    if b'sbnVersion' in decrypted[:200] or b'"v":' in decrypted[:200]:
        print(f"\n   ✓ Looks like BSON/JSON data!")

    # Save it
    output_path = encrypted_file.parent / f"{encrypted_file.stem}.decrypted_ctr"
    with open(output_path, 'wb') as f:
        f.write(decrypted)
    print(f"\n   Saved to: {output_path}")

except Exception as e:
    print(f"   ✗ Decryption failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
