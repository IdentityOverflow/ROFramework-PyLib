"""
Integration module for ML frameworks.

This module provides integrations with popular machine learning frameworks:
- PyTorch (torch.py)
- JAX (jax.py) - Coming soon
- TensorFlow (tensorflow.py) - Coming soon
"""

# PyTorch integration (only import if torch is available)
try:
    from ro_framework.integration.activation_tracker import ActivationTracker
    from ro_framework.integration.torch import TorchNeuralMapping, TorchObserver
    from ro_framework.integration.training import KnowledgeRegularizer

    __all__ = [
        "ActivationTracker",
        "KnowledgeRegularizer",
        "TorchNeuralMapping",
        "TorchObserver",
    ]
except ImportError:
    # PyTorch not installed
    __all__ = []

# SAE integration (only import if sae-lens is available)
try:
    from ro_framework.integration.sae import SAEObserver, create_multilayer_sae_observers

    __all__ += ["SAEObserver", "create_multilayer_sae_observers"]
except ImportError:
    # sae-lens/transformer-lens not installed
    pass
