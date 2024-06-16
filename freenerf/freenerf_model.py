"""
Template Model File
da modificare per freenerf

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.

NerfStudio Methods typically only change the model code.
A model, at a high level, takes in regions of space described by RayBundle objects, 
samples points along these rays, and returns rendered values for each ray.
"""

from dataclasses import dataclass, field
from typing import Type

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model


@dataclass
class FreenerfModelConfig(NerfactoModelConfig): # TODO: modificare con custom
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: NerfactoModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["background", "last_sample"] = "last_sample"
    """Whether to randomize the background color."""
    num_proposal_samples_per_ray: Tuple[int] = (64,)
    """Number of samples per ray for the proposal network."""
    num_nerf_samples_per_ray: int = 64
    """Number of samples per ray for the nerf network."""
    num_proposal_network_iterations: int = 1
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""

    _target: Type = field(default_factory=lambda: FreenerfModelModel)


class FreenerfModelModel(NerfactoModel): # TODO: modificare con custom
    """Template Model."""

    config: FreenerfModelConfig

    def populate_modules(self):
        super().populate_modules()

    # TODO: Override any potential functions/methods to implement your own method
    # or subclass from "Model" and define all mandatory fields.