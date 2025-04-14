"""Entropy generation module for Quantum Forge 2."""

import os
import time
import logging
import hashlib
import numpy as np
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import datetime

from ..models.quantum_metrics import QuantumMetrics

logger = logging.getLogger(__name__)

@dataclass
class EntropyMetrics:
    """Metrics for entropy generation."""
    total_bytes: int = 0
    quantum_bytes: int = 0
    classical_bytes: int = 0
    min_entropy: float = 0.0
    max_entropy: float = 0.0
    avg_entropy: float = 0.0
    generation_times: List[float] = field(default_factory=list)
    quantum_metrics: QuantumMetrics = field(default_factory=QuantumMetrics)
    last_update: Optional[datetime] = None
    
    def record_generation(self, num_bytes: int, is_quantum: bool, entropy: float, duration: float) -> None:
        """Record entropy generation metrics."""
        self.total_bytes += num_bytes
        if is_quantum:
            self.quantum_bytes += num_bytes
        else:
            self.classical_bytes += num_bytes
            
        self.min_entropy = min(self.min_entropy or entropy, entropy)
        self.max_entropy = max(self.max_entropy, entropy)
        self.avg_entropy = (
            (self.avg_entropy * (len(self.generation_times)) + entropy)
            / (len(self.generation_times) + 1)
        )
        
        self.generation_times.append(duration)
        self.last_update = datetime.now()
        
    def get_avg_generation_time(self) -> float:
        """Get average generation time."""
        if not self.generation_times:
            return 0.0
        return sum(self.generation_times) / len(self.generation_times)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            'bytes': {
                'total': self.total_bytes,
                'quantum': self.quantum_bytes,
                'classical': self.classical_bytes
            },
            'entropy': {
                'min': self.min_entropy,
                'max': self.max_entropy,
                'avg': self.avg_entropy
            },
            'performance': {
                'avg_generation_time': self.get_avg_generation_time(),
                'total_generations': len(self.generation_times)
            },
            'quantum_metrics': self.quantum_metrics.to_dict(),
            'last_update': self.last_update.isoformat() if self.last_update else None
        }

class EntropyPool:
    """Entropy pool for cryptographic operations."""
    
    def __init__(self, pool_size: int = 1024):
        """Initialize entropy pool."""
        self.pool_size = pool_size
        self._pool = bytearray(os.urandom(pool_size))
        self._last_refresh = time.time()
        self._refresh_interval = 60  # Refresh pool every minute
        
    def get_entropy(self, num_bytes: int) -> bytes:
        """Get entropy from the pool."""
        if time.time() - self._last_refresh > self._refresh_interval:
            self._refresh_pool()
            
        if num_bytes > self.pool_size:
            # For large requests, generate fresh entropy
            return os.urandom(num_bytes)
            
        # Get entropy from pool and rotate
        result = bytes(self._pool[:num_bytes])
        self._pool = self._pool[num_bytes:] + bytearray(os.urandom(num_bytes))
        return result
        
    def _refresh_pool(self) -> None:
        """Refresh the entropy pool."""
        self._pool = bytearray(os.urandom(self.pool_size))
        self._last_refresh = time.time()
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get entropy pool metrics."""
        return {
            'pool_size': self.pool_size,
            'last_refresh': self._last_refresh,
            'refresh_interval': self._refresh_interval
        }

# Global entropy pool instance
_global_pool: Optional[EntropyPool] = None

def get_entropy_pool(pool_size: int = 1024) -> EntropyPool:
    """Get global entropy pool instance."""
    global _global_pool
    if _global_pool is None:
        _global_pool = EntropyPool(pool_size)
    return _global_pool
 
 
 
