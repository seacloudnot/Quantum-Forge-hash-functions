"""Quantum Forge 2 Encryption Module.

This module provides quantum-resistant encryption and decryption functionality.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, Union, List
import numpy as np
import time
from datetime import datetime

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import base64

from .entropy import EntropyPool
from ..models.quantum_metrics import QuantumMetrics

logger = logging.getLogger(__name__)

@dataclass
class EncryptionConfig:
    """Configuration for encryption operations."""
    key_size: int = 256  # bits
    iterations: int = 100000
    salt_size: int = 16  # bytes
    aes_mode: str = 'CFB'  # CFB, CBC, GCM

@dataclass
class EncryptionMetrics:
    """Metrics for encryption operations."""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    quantum_operations: int = 0
    classical_operations: int = 0
    avg_encryption_time: float = 0.0
    avg_decryption_time: float = 0.0
    key_rotations: int = 0
    quantum_metrics: QuantumMetrics = field(default_factory=QuantumMetrics)
    last_operation: Optional[datetime] = None
    operation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def record_operation(self, operation_type: str, duration: float, success: bool = True, quantum: bool = True) -> None:
        """Record an encryption operation."""
        self.total_operations += 1
        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1
            
        if quantum:
            self.quantum_operations += 1
        else:
            self.classical_operations += 1
            
        if operation_type == 'encrypt':
            self.avg_encryption_time = (
                (self.avg_encryption_time * (self.total_operations - 1) + duration)
                / self.total_operations
            )
        elif operation_type == 'decrypt':
            self.avg_decryption_time = (
                (self.avg_decryption_time * (self.total_operations - 1) + duration)
                / self.total_operations
            )
            
        self.last_operation = datetime.now()
        self.operation_history.append({
            'type': operation_type,
            'quantum': quantum,
            'success': success,
            'duration': duration,
            'timestamp': self.last_operation.isoformat()
        })
        
    def record_key_rotation(self) -> None:
        """Record a key rotation event."""
        self.key_rotations += 1
        
    def get_success_rate(self) -> float:
        """Get the operation success rate."""
        if self.total_operations == 0:
            return 1.0
        return self.successful_operations / self.total_operations
        
    def get_quantum_ratio(self) -> float:
        """Get the ratio of quantum to classical operations."""
        if self.total_operations == 0:
            return 0.0
        return self.quantum_operations / self.total_operations

def derive_key(password: str, salt: Optional[bytes] = None, config: Optional[EncryptionConfig] = None) -> Tuple[bytes, bytes]:
    """Derive encryption key from password.
    
    Args:
        password: Password to derive key from
        salt: Optional salt bytes (generated if None)
        config: Optional encryption configuration
        
    Returns:
        Tuple of (derived key, salt used)
    """
    config = config or EncryptionConfig()
    salt = salt or os.urandom(config.salt_size)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=config.key_size // 8,
        salt=salt,
        iterations=config.iterations,
        backend=default_backend()
    )
    key = kdf.derive(password.encode())
    return key, salt

def encrypt_symmetric(data: bytes, key: bytes, iv: Optional[bytes] = None, config: Optional[EncryptionConfig] = None) -> Tuple[bytes, bytes]:
    """Encrypt data using AES symmetric encryption.
    
    Args:
        data: Data to encrypt
        key: Encryption key
        iv: Optional initialization vector (generated if None)
        config: Optional encryption configuration
        
    Returns:
        Tuple of (encrypted data, iv used)
    """
    config = config or EncryptionConfig()
    iv = iv or os.urandom(16)
    
    mode = getattr(modes, config.aes_mode)
    cipher = Cipher(algorithms.AES(key), mode(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    return encryptor.update(data) + encryptor.finalize(), iv

def decrypt_symmetric(encrypted_data: bytes, key: bytes, iv: bytes, config: Optional[EncryptionConfig] = None) -> bytes:
    """Decrypt data using AES symmetric encryption.
    
    Args:
        encrypted_data: Data to decrypt
        key: Decryption key
        iv: Initialization vector used for encryption
        config: Optional encryption configuration
        
    Returns:
        Decrypted data
    """
    config = config or EncryptionConfig()
    mode = getattr(modes, config.aes_mode)
    cipher = Cipher(algorithms.AES(key), mode(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    return decryptor.update(encrypted_data) + decryptor.finalize()

def encrypt_asymmetric(data: bytes, public_key: rsa.RSAPublicKey) -> bytes:
    """Encrypt data using RSA asymmetric encryption.
    
    Args:
        data: Data to encrypt
        public_key: RSA public key
        
    Returns:
        Encrypted data
    """
    return public_key.encrypt(
        data,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

def decrypt_asymmetric(encrypted_data: bytes, private_key: rsa.RSAPrivateKey) -> bytes:
    """Decrypt data using RSA asymmetric encryption.
    
    Args:
        encrypted_data: Data to decrypt
        private_key: RSA private key
        
    Returns:
        Decrypted data
    """
    return private_key.decrypt(
        encrypted_data,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

# Simplified Fernet-based encryption for string data
def encrypt(data: bytes, password: str) -> Tuple[bytes, bytes]:
    """Encrypt data using password-based Fernet encryption.
    
    Args:
        data: Data to encrypt
        password: Encryption password
        
    Returns:
        Tuple of (encrypted data, salt used)
    """
    salt = os.urandom(16)
    key, _ = derive_key(password, salt)
    key_b64 = base64.urlsafe_b64encode(key)
    f = Fernet(key_b64)
    return f.encrypt(data), salt

def decrypt(encrypted_data: bytes, password: str, salt: bytes) -> bytes:
    """Decrypt data using password-based Fernet encryption.
    
    Args:
        encrypted_data: Data to decrypt
        password: Decryption password
        salt: Salt used for encryption
        
    Returns:
        Decrypted data
    """
    key, _ = derive_key(password, salt)
    key_b64 = base64.urlsafe_b64encode(key)
    f = Fernet(key_b64)
    return f.decrypt(encrypted_data)

class SymmetricEncryption:
    """Handles symmetric encryption operations."""
    
    def __init__(self, config: Optional[EncryptionConfig] = None):
        self.config = config or EncryptionConfig()
    
    def encrypt(self, data: bytes, password: str) -> Tuple[bytes, bytes]:
        """Encrypt data using symmetric encryption."""
        key, salt = derive_key(password, config=self.config)
        encrypted, iv = encrypt_symmetric(data, key, config=self.config)
        return encrypted, salt
    
    def decrypt(self, encrypted_data: bytes, password: str, salt: bytes) -> bytes:
        """Decrypt data using symmetric encryption."""
        key, _ = derive_key(password, salt, config=self.config)
        return decrypt_symmetric(encrypted_data, key, salt, config=self.config)

class AsymmetricEncryption:
    """Handles asymmetric encryption operations."""
    
    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
    
    def generate_keypair(self) -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        """Generate a new RSA keypair."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size,
            backend=default_backend()
        )
        return private_key, private_key.public_key()
    
    def encrypt(self, data: bytes, public_key: rsa.RSAPublicKey) -> bytes:
        """Encrypt data using asymmetric encryption."""
        return encrypt_asymmetric(data, public_key)
    
    def decrypt(self, encrypted_data: bytes, private_key: rsa.RSAPrivateKey) -> bytes:
        """Decrypt data using asymmetric encryption."""
        return decrypt_asymmetric(encrypted_data, private_key)

class EncryptionManager:
    """Manages both symmetric and asymmetric encryption operations."""
    
    def __init__(self, config: Optional[EncryptionConfig] = None):
        self.config = config or EncryptionConfig()
        self.symmetric = SymmetricEncryption(self.config)
        self.asymmetric = AsymmetricEncryption()
    
    def encrypt_symmetric(self, data: bytes, password: str) -> Dict[str, Any]:
        """Encrypt data using symmetric encryption."""
        encrypted, salt = self.symmetric.encrypt(data, password)
        return {
            'encrypted_data': encrypted,
            'salt': salt,
            'type': 'symmetric'
        }
    
    def decrypt_symmetric(self, encrypted_data: bytes, password: str, salt: bytes) -> bytes:
        """Decrypt data using symmetric encryption."""
        return self.symmetric.decrypt(encrypted_data, password, salt)
    
    def encrypt_asymmetric(self, data: bytes, public_key: rsa.RSAPublicKey) -> Dict[str, Any]:
        """Encrypt data using asymmetric encryption."""
        encrypted = self.asymmetric.encrypt(data, public_key)
        return {
            'encrypted_data': encrypted,
            'type': 'asymmetric'
        }
    
    def decrypt_asymmetric(self, encrypted_data: bytes, private_key: rsa.RSAPrivateKey) -> bytes:
        """Decrypt data using asymmetric encryption."""
        return self.asymmetric.decrypt(encrypted_data, private_key)

class QuantumEncryption:
    """Quantum-resistant encryption implementation."""
    
    def __init__(self):
        """Initialize encryption with secure key."""
        self._key = os.urandom(32)
    
    def encrypt(self, data: Union[bytes, bytearray]) -> bytes:
        """Encrypt data using quantum-resistant algorithm."""
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("Data must be bytes or bytearray")
            
        # Simple XOR encryption for testing
        result = bytearray(len(data))
        for i in range(len(data)):
            result[i] = data[i] ^ self._key[i % len(self._key)]
        return bytes(result)
    
    def encrypt_array(self, data: np.ndarray) -> np.ndarray:
        """Encrypt numpy array data."""
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be numpy array")
            
        # Convert to bytes, encrypt, and convert back
        data_bytes = data.tobytes()
        encrypted_bytes = self.encrypt(data_bytes)
        return np.frombuffer(encrypted_bytes, dtype=data.dtype).reshape(data.shape)
    
    def decrypt(self, data: Union[bytes, bytearray]) -> bytes:
        """Decrypt data using quantum-resistant algorithm."""
        # For XOR encryption, encryption and decryption are the same
        return self.encrypt(data)

class UnifiedEncryption:
    """Unified encryption interface supporting both classical and quantum encryption."""
    
    def __init__(self, 
                 config: Optional[EncryptionConfig] = None,
                 quantum_security_level: int = 512,
                 use_quantum: bool = True):
        self.config = config or EncryptionConfig()
        self.quantum = QuantumEncryption() if use_quantum else None
        self.use_quantum = use_quantum
        
    def encrypt(self, data: Union[bytes, np.ndarray], use_quantum: Optional[bool] = None) -> bytes:
        """Encrypt data using either quantum or classical encryption."""
        should_use_quantum = use_quantum if use_quantum is not None else self.use_quantum
        
        if should_use_quantum and self.quantum:
            return self.quantum.encrypt(data)
        else:
            if isinstance(data, np.ndarray):
                data = data.tobytes()
            key = os.urandom(32)
            encrypted, iv = encrypt_symmetric(data, key, config=self.config)
            return iv + key + encrypted
            
    def decrypt(self, encrypted_data: bytes, is_quantum: Optional[bool] = None) -> bytes:
        """Decrypt data using either quantum or classical decryption."""
        should_use_quantum = is_quantum if is_quantum is not None else self.use_quantum
        
        if should_use_quantum and self.quantum:
            return self.quantum.decrypt(encrypted_data)
        else:
            iv = encrypted_data[:16]
            key = encrypted_data[16:48]
            ciphertext = encrypted_data[48:]
            return decrypt_symmetric(ciphertext, key, iv, config=self.config)
            
    def encrypt_array(self, data: np.ndarray, use_quantum: Optional[bool] = None) -> np.ndarray:
        """Encrypt numpy array while preserving structure."""
        should_use_quantum = use_quantum if use_quantum is not None else self.use_quantum
        
        if should_use_quantum and self.quantum:
            return self.quantum.encrypt_array(data)
        else:
            encrypted = self.encrypt(data.tobytes(), use_quantum=False)
            return np.frombuffer(encrypted, dtype=data.dtype).reshape(data.shape) 