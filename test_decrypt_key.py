#!/usr/bin/env python3
"""
Debug script to test decrypting just the key from config.sbc
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

# Load config from actual file
config_path = Path("/home/sysop/Downloads/test_saber/Saber/config.sbc")
with open(config_path, 'r') as f:
    config = json.load(f)

print(f"Loaded config from: {config_path}")
print(f"Config contents: {config}")

print("=" * 70)
print("Debug: Decrypting key from config.sbc")
print("=" * 70)

# Step 1: Derive password key
password_with_salt = password + REPRODUCIBLE_SALT
password_key = hashlib.sha256(password_with_salt.encode('utf-8')).digest()
print(f"\n1. Password key (hex): {password_key.hex()}")
print(f"   Length: {len(password_key)} bytes")

# Step 2: Get IV
iv = base64.b64decode(config['iv'])
print(f"\n2. IV (hex): {iv.hex()}")
print(f"   Length: {len(iv)} bytes")

# Step 3: Get encrypted key
encrypted_key_b64 = config['key']
encrypted_key = base64.b64decode(encrypted_key_b64)
print(f"\n3. Encrypted key (base64): {encrypted_key_b64}")
print(f"   Encrypted key (hex): {encrypted_key.hex()}")
print(f"   Length: {len(encrypted_key)} bytes")

# Step 4: Try to decrypt
print(f"\n4. Attempting decryption...")
try:
    cipher = AES.new(password_key, AES.MODE_CBC, iv)
    decrypted_padded = cipher.decrypt(encrypted_key)
    print(f"   Decrypted (with padding, hex): {decrypted_padded.hex()}")
    print(f"   Decrypted (with padding, raw): {decrypted_padded}")

    # Try to unpad
    try:
        decrypted = unpad(decrypted_padded, AES.block_size)
        print(f"   ✓ Unpadded (hex): {decrypted.hex()}")
        print(f"   ✓ Unpadded (utf-8): {decrypted.decode('utf-8')}")
        print(f"   ✓ Length: {len(decrypted)} bytes")
    except ValueError as e:
        print(f"   ✗ Unpadding failed: {e}")
        print(f"   Last byte (should be padding): 0x{decrypted_padded[-1]:02x} (decimal {decrypted_padded[-1]})")

        # Maybe it's not padded? Or maybe it's base64-encoded?
        print(f"\n5. Trying alternative interpretations...")

        # Try as base64 string
        try:
            as_string = decrypted_padded.decode('utf-8').rstrip('\x00')
            print(f"   As UTF-8 string: {as_string}")
            try:
                decoded_base64 = base64.b64decode(as_string)
                print(f"   ✓ Decoded from base64 (hex): {decoded_base64.hex()}")
                print(f"   ✓ Decoded length: {len(decoded_base64)} bytes")
            except Exception as e2:
                print(f"   ✗ Not valid base64: {e2}")
        except Exception as e2:
            print(f"   ✗ Not valid UTF-8: {e2}")

        # Try treating decrypted data directly as the key (no padding)
        print(f"\n   Treating decrypted data as raw key (no unpadding):")
        print(f"   Raw key (hex): {decrypted_padded.hex()}")
        print(f"   Raw key length: {len(decrypted_padded)} bytes")

except Exception as e:
    print(f"   ✗ Decryption failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
