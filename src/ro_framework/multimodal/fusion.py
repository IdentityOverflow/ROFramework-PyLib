"""
Cross-modal fusion mechanisms for combining multiple modalities.

Fusion strategies allow an observer to integrate information from different
sensory modalities into a unified representation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ro_framework.core.dof import DoF, PolarDoF
from ro_framework.core.state import State
from ro_framework.observer.observer import Observer
from ro_framework.observer.mapping import MappingFunction
from ro_framework.multimodal.encoders import ModalityEncoder


class FusionStrategy(ABC):
    """
    Abstract base class for fusion strategies.

    A fusion strategy combines States from multiple modalities into a
    unified representation.
    """

    def __init__(self, modality_names: List[str], output_dofs: List[DoF]):
        """
        Initialize fusion strategy.

        Args:
            modality_names: Names of modalities to fuse
            output_dofs: Output DoFs of the fused representation
        """
        self.modality_names = modality_names
        self.output_dofs = output_dofs

    @abstractmethod
    def fuse(self, modality_states: Dict[str, State]) -> State:
        """
        Fuse states from multiple modalities.

        Args:
            modality_states: Dictionary mapping modality names to their States

        Returns:
            Fused State
        """
        pass

    @abstractmethod
    def fuse_uncertainties(
        self,
        modality_uncertainties: Dict[str, Dict[DoF, float]]
    ) -> Dict[DoF, float]:
        """
        Fuse uncertainty estimates from multiple modalities.

        Args:
            modality_uncertainties: Dictionary mapping modality names to their uncertainties

        Returns:
            Fused uncertainties for output DoFs
        """
        pass


class ConcatenationFusion(FusionStrategy):
    """
    Simple concatenation-based fusion.

    Concatenates representations from all modalities, optionally applying
    learned weights or projection.
    """

    def __init__(
        self,
        modality_names: List[str],
        modality_dofs: Dict[str, List[DoF]],
        output_dofs: Optional[List[DoF]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize concatenation fusion.

        Args:
            modality_names: Names of modalities to fuse
            modality_dofs: Dictionary mapping modality names to their DoFs
            output_dofs: Output DoFs (if None, concatenates all modality DoFs)
            weights: Optional weights for each modality
        """
        if output_dofs is None:
            # Concatenate all DoFs
            all_dofs = []
            for name in modality_names:
                all_dofs.extend(modality_dofs[name])
            output_dofs = all_dofs

        super().__init__(modality_names, output_dofs)
        self.modality_dofs = modality_dofs
        self.weights = weights or {name: 1.0 for name in modality_names}

    def fuse(self, modality_states: Dict[str, State]) -> State:
        """
        Concatenate states from multiple modalities.

        Args:
            modality_states: Dictionary mapping modality names to States

        Returns:
            Concatenated State
        """
        fused_dict = {}

        for name in self.modality_names:
            if name not in modality_states:
                continue

            state = modality_states[name]
            weight = self.weights[name]

            # Add weighted values from this modality
            for dof in self.modality_dofs[name]:
                value = state.get_value(dof)
                if value is not None:
                    if dof in fused_dict:
                        fused_dict[dof] = (fused_dict[dof] + value * weight) / 2
                    else:
                        fused_dict[dof] = value * weight

        return State(values=fused_dict)

    def fuse_uncertainties(
        self,
        modality_uncertainties: Dict[str, Dict[DoF, float]]
    ) -> Dict[DoF, float]:
        """
        Average uncertainties across modalities.

        Args:
            modality_uncertainties: Dictionary mapping modality names to uncertainties

        Returns:
            Averaged uncertainties
        """
        fused_uncertainties = {}

        for name in self.modality_names:
            if name not in modality_uncertainties:
                continue

            uncertainties = modality_uncertainties[name]
            for dof, unc in uncertainties.items():
                if dof in fused_uncertainties:
                    fused_uncertainties[dof] = (fused_uncertainties[dof] + unc) / 2
                else:
                    fused_uncertainties[dof] = unc

        return fused_uncertainties


if TORCH_AVAILABLE:

    class AttentionFusion(FusionStrategy):
        """
        Attention-based fusion using learned attention weights.

        Learns to dynamically weight different modalities based on their
        relevance for the current input.
        """

        def __init__(
            self,
            modality_names: List[str],
            modality_dims: Dict[str, int],
            output_dim: int,
            hidden_dim: int = 128,
            device: str = "cpu",
        ):
            """
            Initialize attention fusion.

            Args:
                modality_names: Names of modalities to fuse
                modality_dims: Dimensionality of each modality
                output_dim: Dimension of fused representation
                hidden_dim: Hidden dimension for attention network
                device: Device for computation
            """
            output_dofs = [
                PolarDoF(name=f"fused_feature_{i}", description=f"Fused feature {i}")
                for i in range(output_dim)
            ]
            super().__init__(modality_names, output_dofs)

            self.modality_dims = modality_dims
            self.output_dim = output_dim
            self.device = device

            # Attention network
            self.attention_network = nn.ModuleDict({
                name: nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
                for name, dim in modality_dims.items()
            }).to(device)

            # Projection networks for each modality
            self.projection_networks = nn.ModuleDict({
                name: nn.Linear(dim, output_dim)
                for name, dim in modality_dims.items()
            }).to(device)

        def fuse(self, modality_states: Dict[str, State]) -> State:
            """
            Fuse states using attention mechanism.

            Args:
                modality_states: Dictionary mapping modality names to States

            Returns:
                Attention-weighted fused State
            """
            # Convert states to tensors
            modality_tensors = {}
            for name in self.modality_names:
                if name not in modality_states:
                    continue

                state = modality_states[name]
                # Extract values as vector
                values = []
                for dof in state.values.keys():
                    value = state.get_value(dof)
                    if value is not None:
                        values.append(float(value))

                if values:
                    tensor = torch.tensor(values, device=self.device).unsqueeze(0)
                    # Pad or truncate to expected dimension
                    expected_dim = self.modality_dims[name]
                    if len(values) < expected_dim:
                        padding = torch.zeros(1, expected_dim - len(values), device=self.device)
                        tensor = torch.cat([tensor, padding], dim=1)
                    elif len(values) > expected_dim:
                        tensor = tensor[:, :expected_dim]

                    modality_tensors[name] = tensor

            if not modality_tensors:
                return State(values={})

            # Compute attention weights
            attention_scores = {}
            for name, tensor in modality_tensors.items():
                score = self.attention_network[name](tensor)
                attention_scores[name] = score

            # Softmax over modalities
            all_scores = torch.cat(list(attention_scores.values()), dim=1)
            attention_weights = F.softmax(all_scores, dim=1)

            # Apply attention and project
            weighted_sum = None
            for i, (name, tensor) in enumerate(modality_tensors.items()):
                projected = self.projection_networks[name](tensor)
                weighted = projected * attention_weights[:, i].unsqueeze(1)

                if weighted_sum is None:
                    weighted_sum = weighted
                else:
                    weighted_sum = weighted_sum + weighted

            # Convert to State
            fused_values = weighted_sum.detach().cpu().numpy().flatten()
            fused_dict = {
                dof: float(fused_values[i])
                for i, dof in enumerate(self.output_dofs)
                if i < len(fused_values)
            }

            return State(values=fused_dict)

        def fuse_uncertainties(
            self,
            modality_uncertainties: Dict[str, Dict[DoF, float]]
        ) -> Dict[DoF, float]:
            """
            Fuse uncertainties with attention weighting.

            Args:
                modality_uncertainties: Dictionary mapping modality names to uncertainties

            Returns:
                Attention-weighted uncertainties
            """
            # For simplicity, average uncertainties
            # In practice, could use attention weights
            fused_uncertainties = {}
            count = {}

            for name in self.modality_names:
                if name not in modality_uncertainties:
                    continue

                uncertainties = modality_uncertainties[name]
                avg_unc = np.mean(list(uncertainties.values()))

                for dof in self.output_dofs:
                    if dof in fused_uncertainties:
                        fused_uncertainties[dof] += avg_unc
                        count[dof] += 1
                    else:
                        fused_uncertainties[dof] = avg_unc
                        count[dof] = 1

            # Average
            for dof in fused_uncertainties:
                fused_uncertainties[dof] /= count[dof]

            return fused_uncertainties


    class GatedFusion(FusionStrategy):
        """
        Gated fusion using learned gating mechanism.

        Learns to gate information from different modalities based on
        the input and task requirements.
        """

        def __init__(
            self,
            modality_names: List[str],
            modality_dims: Dict[str, int],
            output_dim: int,
            device: str = "cpu",
        ):
            """
            Initialize gated fusion.

            Args:
                modality_names: Names of modalities to fuse
                modality_dims: Dimensionality of each modality
                output_dim: Dimension of fused representation
                device: Device for computation
            """
            output_dofs = [
                PolarDoF(name=f"fused_feature_{i}", description=f"Fused feature {i}")
                for i in range(output_dim)
            ]
            super().__init__(modality_names, output_dofs)

            self.modality_dims = modality_dims
            self.output_dim = output_dim
            self.device = device

            # Gate networks for each modality
            self.gate_networks = nn.ModuleDict({
                name: nn.Sequential(
                    nn.Linear(dim, output_dim),
                    nn.Sigmoid()
                )
                for name, dim in modality_dims.items()
            }).to(device)

            # Projection networks
            self.projection_networks = nn.ModuleDict({
                name: nn.Linear(dim, output_dim)
                for name, dim in modality_dims.items()
            }).to(device)

        def fuse(self, modality_states: Dict[str, State]) -> State:
            """
            Fuse states using gating mechanism.

            Args:
                modality_states: Dictionary mapping modality names to States

            Returns:
                Gated fused State
            """
            # Convert states to tensors
            modality_tensors = {}
            for name in self.modality_names:
                if name not in modality_states:
                    continue

                state = modality_states[name]
                values = []
                for dof in state.values.keys():
                    value = state.get_value(dof)
                    if value is not None:
                        values.append(float(value))

                if values:
                    tensor = torch.tensor(values, device=self.device).unsqueeze(0)
                    expected_dim = self.modality_dims[name]
                    if len(values) < expected_dim:
                        padding = torch.zeros(1, expected_dim - len(values), device=self.device)
                        tensor = torch.cat([tensor, padding], dim=1)
                    elif len(values) > expected_dim:
                        tensor = tensor[:, :expected_dim]

                    modality_tensors[name] = tensor

            if not modality_tensors:
                return State(values={})

            # Apply gating and projection
            gated_sum = None
            for name, tensor in modality_tensors.items():
                gate = self.gate_networks[name](tensor)
                projected = self.projection_networks[name](tensor)
                gated = gate * projected

                if gated_sum is None:
                    gated_sum = gated
                else:
                    gated_sum = gated_sum + gated

            # Normalize by number of modalities
            gated_sum = gated_sum / len(modality_tensors)

            # Convert to State
            fused_values = gated_sum.detach().cpu().numpy().flatten()
            fused_dict = {
                dof: float(fused_values[i])
                for i, dof in enumerate(self.output_dofs)
                if i < len(fused_values)
            }

            return State(values=fused_dict)

        def fuse_uncertainties(
            self,
            modality_uncertainties: Dict[str, Dict[DoF, float]]
        ) -> Dict[DoF, float]:
            """
            Fuse uncertainties with gating weighting.

            Args:
                modality_uncertainties: Dictionary mapping modality names to uncertainties

            Returns:
                Gated uncertainties
            """
            # Average uncertainties across modalities
            fused_uncertainties = {}
            count = {}

            for name in self.modality_names:
                if name not in modality_uncertainties:
                    continue

                uncertainties = modality_uncertainties[name]
                avg_unc = np.mean(list(uncertainties.values()))

                for dof in self.output_dofs:
                    if dof in fused_uncertainties:
                        fused_uncertainties[dof] += avg_unc
                        count[dof] += 1
                    else:
                        fused_uncertainties[dof] = avg_unc
                        count[dof] = 1

            for dof in fused_uncertainties:
                fused_uncertainties[dof] /= count[dof]

            return fused_uncertainties


class MultimodalObserver(Observer):
    """
    Observer that processes multiple sensory modalities.

    Combines modality-specific encoders with fusion strategies to create
    a unified world model from diverse sensory inputs.
    """

    def __init__(
        self,
        name: str,
        encoders: Dict[str, ModalityEncoder],
        fusion_strategy: FusionStrategy,
        world_model: Optional[MappingFunction] = None,
        self_model: Optional[MappingFunction] = None,
        temporal_dof: Optional[DoF] = None,
        memory_capacity: int = 1000,
    ):
        """
        Initialize multimodal observer.

        Args:
            name: Name of the observer
            encoders: Dictionary mapping modality names to encoders
            fusion_strategy: Strategy for fusing modality representations
            world_model: Optional world model mapping (external→internal)
            self_model: Optional self model (internal→internal)
            temporal_dof: Optional temporal DoF for memory tracking
            memory_capacity: Maximum memory buffer size
        """
        # Get all input DoFs from encoders
        external_dofs = []
        for encoder in encoders.values():
            external_dofs.extend(encoder.output_dofs)

        # Output DoFs are the fusion output
        internal_dofs = fusion_strategy.output_dofs

        # Resolution from fusion strategy
        resolution = {}

        # Need to provide a dummy world_model if None
        if world_model is None:
            from ro_framework.observer.mapping import IdentityMapping
            world_model = IdentityMapping(
                input_dofs=external_dofs,
                output_dofs=internal_dofs
            )

        super().__init__(
            name=name,
            external_dofs=external_dofs,
            internal_dofs=internal_dofs,
            world_model=world_model,
            self_model=self_model,
            resolution=resolution,
            temporal_dof=temporal_dof,
            memory_capacity=memory_capacity,
        )

        self.encoders = encoders
        self.fusion_strategy = fusion_strategy

    def process_multimodal_input(
        self,
        raw_inputs: Dict[str, any]
    ) -> tuple[State, Dict[DoF, float]]:
        """
        Process raw inputs from multiple modalities.

        Args:
            raw_inputs: Dictionary mapping modality names to raw inputs

        Returns:
            Tuple of (fused_state, fused_uncertainties)
        """
        # Encode each modality
        modality_states = {}
        modality_uncertainties = {}

        for name, raw_input in raw_inputs.items():
            if name not in self.encoders:
                continue

            encoder = self.encoders[name]
            state = encoder.encode(raw_input)
            uncertainty = encoder.get_uncertainty(raw_input)

            modality_states[name] = state
            modality_uncertainties[name] = uncertainty

        # Fuse modalities
        fused_state = self.fusion_strategy.fuse(modality_states)
        fused_uncertainty = self.fusion_strategy.fuse_uncertainties(modality_uncertainties)

        return fused_state, fused_uncertainty

    def observe_multimodal(
        self,
        raw_inputs: Dict[str, any]
    ) -> State:
        """
        Observe from multiple modalities and apply world model.

        Args:
            raw_inputs: Dictionary mapping modality names to raw inputs

        Returns:
            Internal State after world model mapping
        """
        # Process multimodal input
        fused_state, _ = self.process_multimodal_input(raw_inputs)

        # Apply world model if available
        if self.world_model is not None:
            internal_state = self.world_model(fused_state)
            return internal_state

        return fused_state
