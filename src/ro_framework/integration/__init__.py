"""
Integration module for ML frameworks.

This module provides integrations with popular machine learning frameworks:
- PyTorch (torch.py)
- JAX (jax.py) - Coming soon
- TensorFlow (tensorflow.py) - Coming soon
"""

# PyTorch integration (only import if torch is available)
try:
    from ro_framework.integration.torch import TorchNeuralMapping, TorchObserver

    __all__ = ["TorchNeuralMapping", "TorchObserver"]
except ImportError:
    # PyTorch not installed
    __all__ = []
