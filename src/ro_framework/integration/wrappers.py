"""
Convenience wrappers for turning arbitrary callables and models into Observers.

These are the primary entry-points for users who want to wrap an existing
model without manually constructing DoFs, mappings, and observers.
"""

from typing import Callable, List, Optional, Sequence

import numpy as np

from ro_framework.core.dof import DoF, PolarDoF, PolarDoFType
from ro_framework.core.state import State
from ro_framework.observer.observer import Observer


# ---------------------------------------------------------------------------
# DoF creation helpers
# ---------------------------------------------------------------------------

def create_dofs_for_vector(
    n: int,
    prefix: str = "d",
    pole_negative: float = -np.inf,
    pole_positive: float = np.inf,
) -> List[PolarDoF]:
    """Create *n* PolarDoFs for a flat numeric vector.

    This is the simplest way to get DoFs when you already know how many
    dimensions your model expects / produces.

    Args:
        n: Number of DoFs (== dimensionality of the vector).
        prefix: Name prefix — DoFs will be named ``{prefix}_0``, etc.
        pole_negative: Lower bound for each DoF.
        pole_positive: Upper bound for each DoF.

    Returns:
        List of *n* PolarDoFs.
    """
    polar_type = (
        PolarDoFType.CONTINUOUS_BOUNDED
        if np.isfinite(pole_negative) and np.isfinite(pole_positive)
        else PolarDoFType.CONTINUOUS_REAL
    )
    return [
        PolarDoF(
            name=f"{prefix}_{i}",
            pole_negative=pole_negative,
            pole_positive=pole_positive,
            polar_type=polar_type,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Callable wrapper
# ---------------------------------------------------------------------------

class _CallableMapping:
    """Thin mapping adapter: wraps ``fn(np.ndarray) -> np.ndarray``."""

    def __init__(
        self,
        fn: Callable,
        input_dofs: List[DoF],
        output_dofs: List[DoF],
    ) -> None:
        self.fn = fn
        self.input_dofs = input_dofs
        self.output_dofs = output_dofs

    def __call__(self, state: State) -> State:
        vec_in = state.to_vector(self.input_dofs)
        vec_out = np.asarray(self.fn(vec_in), dtype=np.float64)
        return State.from_vector(vec_out, self.output_dofs)


def wrap_callable(
    fn: Callable,
    input_dofs: List[DoF],
    output_dofs: List[DoF],
    name: str = "callable_observer",
    self_model_fn: Optional[Callable] = None,
) -> Observer:
    """Wrap a plain ``fn(ndarray) -> ndarray`` callable as an Observer.

    This is the fastest way to get a working Observer for any function
    that maps a numeric vector to a numeric vector.

    Args:
        fn: Callable accepting a 1-D numpy array and returning a 1-D array.
        input_dofs: External DoFs (input dimensions).
        output_dofs: Internal DoFs (output dimensions).
        name: Observer name.
        self_model_fn: Optional callable for the self-model
            (same signature: ndarray -> ndarray, over output_dofs).

    Returns:
        A fully configured Observer.
    """
    world_model = _CallableMapping(fn, input_dofs, output_dofs)
    self_model = None
    if self_model_fn is not None:
        self_model = _CallableMapping(self_model_fn, output_dofs, output_dofs)

    return Observer(
        name=name,
        internal_dofs=output_dofs,
        external_dofs=input_dofs,
        world_model=world_model,
        self_model=self_model,
    )


# ---------------------------------------------------------------------------
# PyTorch wrapper
# ---------------------------------------------------------------------------

def wrap_torch_model(
    model,
    input_dofs: List[DoF],
    output_dofs: List[DoF],
    name: str = "torch_observer",
    self_model=None,
    device: str = "cpu",
    use_dropout_uncertainty: bool = False,
) -> "Observer":
    """Wrap a PyTorch ``nn.Module`` as a TorchObserver.

    Args:
        model: PyTorch nn.Module (world model).
        input_dofs: External DoFs.
        output_dofs: Internal DoFs.
        name: Observer name.
        self_model: Optional PyTorch nn.Module for the self-model.
        device: Device string ('cpu', 'cuda', etc.).
        use_dropout_uncertainty: Whether to use MC Dropout for uncertainty.

    Returns:
        A TorchObserver ready for use.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    from ro_framework.integration.torch import TorchNeuralMapping, TorchObserver

    world_mapping = TorchNeuralMapping(
        name=f"{name}_world",
        input_dofs=input_dofs,
        output_dofs=output_dofs,
        model=model,
        device=device,
        use_dropout_uncertainty=use_dropout_uncertainty,
    )

    self_mapping = None
    if self_model is not None:
        self_mapping = TorchNeuralMapping(
            name=f"{name}_self",
            input_dofs=output_dofs,
            output_dofs=output_dofs,
            model=self_model,
            device=device,
            use_dropout_uncertainty=use_dropout_uncertainty,
        )

    return TorchObserver(
        name=name,
        internal_dofs=output_dofs,
        external_dofs=input_dofs,
        world_model=world_mapping,
        self_model=self_mapping,
        device=device,
    )
