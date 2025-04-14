"""Quantum metrics module for Quantum Forge 2."""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from datetime import datetime

@dataclass
class QuantumMetrics:
    """Metrics for quantum operations."""
    
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    operation_times: List[float] = field(default_factory=list)
    last_operation: datetime = field(default_factory=datetime.now)
    
    def record_operation(self, duration: float, success: bool = True) -> None:
        """Record a quantum operation."""
        self.total_operations += 1
        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1
        self.operation_times.append(duration)
        self.last_operation = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            'operations': {
                'total': self.total_operations,
                'successful': self.successful_operations,
                'failed': self.failed_operations,
                'success_rate': (self.successful_operations / self.total_operations 
                               if self.total_operations > 0 else 1.0)
            },
            'performance': {
                'avg_operation_time': (sum(self.operation_times) / len(self.operation_times)
                                     if self.operation_times else 0.0),
                'total_operation_time': sum(self.operation_times)
            },
            'last_operation': self.last_operation.isoformat()
        } 