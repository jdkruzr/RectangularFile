"""
Saber Note Decryptor

Handles decryption of Saber notes (.sbe files) from WebDAV sync.

Encryption scheme (from Saber source):
1. Password key derivation: SHA256(password + "8MnPs64@R&mF8XjWeLrD")
2. Files are encrypted directly with the password-derived key (not a separate file key)
3. Algorithm: AES-256-CTR (counter mode - no padding needed)
4. IV stored in config.sbc is used as the initial counter value

Note: The "key" field in config.sbc appears to be for future use or verification,
but actual file encryption uses the password-derived key directly.
"""

import json
import hashlib
import base64
import logging
from pathlib import Path
from typing import Optional, Tuple
from Crypto.Cipher import AES
from Crypto.Util import Counter

logger = logging.getLogger(__name__)


class SaberDecryptor:
    """Handles decryption of Saber encrypted notes."""

    # Salt used in Saber's key derivation (from nextcloud_client_extension.dart:37)
    REPRODUCIBLE_SALT = "8MnPs64@R&mF8XjWeLrD"

    def __init__(self, encryption_password: str, saber_folder: Path):
        """
        Initialize the Saber decryptor.

        Args:
            encryption_password: User's Saber encryption password
            saber_folder: Path to Saber sync folder containing config.sbc
        """
        self.encryption_password = encryption_password
        self.saber_folder = Path(saber_folder)
        self._password_key = None  # Key derived from password
        self._iv = None            # IV from config.sbc

    def _derive_password_key(self) -> bytes:
        """
        Derive the password-based encryption key using Saber's method.

        This key is used to decrypt the file encryption key stored in config.sbc.
        Returns SHA256(password + salt) as 32-byte key.
        """
        if self._password_key is None:
            # Combine password with reproducible salt
            password_with_salt = self.encryption_password + self.REPRODUCIBLE_SALT
            # Hash to get 32-byte key for AES-256
            self._password_key = hashlib.sha256(password_with_salt.encode('utf-8')).digest()
            logger.debug("Derived password-based encryption key")
        return self._password_key

    def _load_iv_from_config(self) -> bytes:
        """
        Load IV from config.sbc file.

        Returns:
            IV as bytes (16 bytes for AES)

        Raises:
            FileNotFoundError: If config.sbc not found
            ValueError: If config.sbc is invalid
        """
        if self._iv is not None:
            return self._iv

        # Try multiple possible locations for config.sbc
        possible_paths = [
            self.saber_folder / "Saber" / "config.sbc",  # Original: parent/Saber/config.sbc
            self.saber_folder / "config.sbc",              # Direct mount: /mnt/saber/config.sbc
            self.saber_folder / "saber" / "config.sbc",    # Lowercase variant
        ]

        config_path = None
        for path in possible_paths:
            if path.exists():
                config_path = path
                break

        if not config_path:
            raise FileNotFoundError(
                f"config.sbc not found. Tried: {', '.join(str(p) for p in possible_paths)}. "
                "Make sure Saber has synced at least once."
            )

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            if 'iv' not in config:
                raise ValueError("config.sbc does not contain 'iv' field")

            # Decode base64 IV
            self._iv = base64.b64decode(config['iv'])
            logger.debug(f"Loaded IV from {config_path}")

            return self._iv

        except json.JSONDecodeError as e:
            raise ValueError(f"config.sbc is not valid JSON: {e}")

    def decrypt_file(self, encrypted_path: Path) -> bytes:
        """
        Decrypt a Saber encrypted file.

        Args:
            encrypted_path: Path to .sbe file

        Returns:
            Decrypted file contents as bytes

        Raises:
            FileNotFoundError: If encrypted file not found
            ValueError: If decryption fails
        """
        if not encrypted_path.exists():
            raise FileNotFoundError(f"Encrypted file not found: {encrypted_path}")

        # Read encrypted data
        with open(encrypted_path, 'rb') as f:
            encrypted_data = f.read()

        if not encrypted_data:
            raise ValueError(f"Encrypted file is empty: {encrypted_path}")

        # Get password-derived key and IV
        key = self._derive_password_key()
        iv = self._load_iv_from_config()

        try:
            # Create AES cipher in CTR mode using the IV as counter
            # CTR mode is a stream cipher, so no padding is needed
            counter_val = int.from_bytes(iv, byteorder='big')
            ctr = Counter.new(128, initial_value=counter_val)
            cipher = AES.new(key, AES.MODE_CTR, counter=ctr)

            # Decrypt (no unpadding needed for CTR mode)
            decrypted = cipher.decrypt(encrypted_data)

            logger.info(f"Successfully decrypted {encrypted_path.name} ({len(decrypted)} bytes)")
            return decrypted

        except Exception as e:
            raise ValueError(f"Decryption failed for {encrypted_path}: {e}")

    def decrypt_filename(self, encrypted_filename: str) -> str:
        """
        Decrypt a Saber encrypted filename.

        Filenames are encrypted as hex strings (base16) of the encrypted path.

        Args:
            encrypted_filename: Hex-encoded encrypted filename (without .sbe extension)

        Returns:
            Original decrypted path (e.g., "/25-11-11 Test.sbn2")

        Raises:
            ValueError: If decryption fails
        """
        key = self._derive_password_key()
        iv = self._load_iv_from_config()

        try:
            # Convert hex string to bytes
            encrypted_bytes = bytes.fromhex(encrypted_filename)

            # Create AES cipher in CTR mode
            counter_val = int.from_bytes(iv, byteorder='big')
            ctr = Counter.new(128, initial_value=counter_val)
            cipher = AES.new(key, AES.MODE_CTR, counter=ctr)

            # Decrypt (no unpadding needed for CTR mode)
            decrypted_bytes = cipher.decrypt(encrypted_bytes)

            # Decode to string
            decrypted_path = decrypted_bytes.decode('utf-8')

            logger.debug(f"Decrypted filename: {encrypted_filename[:16]}... -> {decrypted_path}")
            return decrypted_path

        except Exception as e:
            raise ValueError(f"Failed to decrypt filename: {e}")

    def decrypt_to_file(self, encrypted_path: Path, output_path: Optional[Path] = None) -> Path:
        """
        Decrypt a file and write to disk.

        Args:
            encrypted_path: Path to encrypted file
            output_path: Where to write decrypted file (defaults to same name without .enc)

        Returns:
            Path to decrypted file
        """
        decrypted_data = self.decrypt_file(encrypted_path)

        if output_path is None:
            # Remove .enc extension
            if encrypted_path.suffix == '.enc':
                output_path = encrypted_path.with_suffix('')
            else:
                output_path = encrypted_path.with_suffix(encrypted_path.suffix + '.decrypted')

        with open(output_path, 'wb') as f:
            f.write(decrypted_data)

        logger.info(f"Wrote decrypted file to {output_path}")
        return output_path


def test_decryption(saber_folder: str, password: str, test_file: str):
    """
    Test utility for decrypting a Saber file.

    Args:
        saber_folder: Path to Saber sync folder
        password: Encryption password
        test_file: Path to encrypted .sbn2.enc file to test
    """
    decryptor = SaberDecryptor(password, Path(saber_folder))

    try:
        decrypted = decryptor.decrypt_file(Path(test_file))
        print(f"✓ Successfully decrypted {len(decrypted)} bytes")
        print(f"  First 100 bytes: {decrypted[:100]}")
        return True
    except Exception as e:
        print(f"✗ Decryption failed: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python saber_decryptor.py <saber_folder> <password> <encrypted_file>")
        sys.exit(1)

    test_decryption(sys.argv[1], sys.argv[2], sys.argv[3])
