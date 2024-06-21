"""
Template Model File
da modificare per freenerf

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.

NerfStudio Methods typically only change the model code.
A model, at a high level, takes in regions of space described by RayBundle objects, 
samples points along these rays, and returns rendered values for each ray.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Type

from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from .freenerf_encoding import FreeNeRFEncoding

@dataclass
class FreeNeRFModelConfig(VanillaModelConfig): # TODO: modificare con custom
    """Vanilla Model Config"""

    _target: Type = field(default_factory=lambda: FreeNeRFModel)
    
    position_encoding_num_frequencies: int = 10
    """Number of frequencies for positional encoding"""

    direction_encoding_num_frequencies: int = 4
    """Number of frequencies for directional encoding"""

    T: int = 30000
    """Number of training steps (must equal to max-num-iterations)"""


class FreeNeRFModel(NeRFModel): # TODO: modificare con custom
    """Template Model."""

    config: FreeNeRFModelConfig

    def __init__( # costruttore
        self,
        config: FreeNeRFModelConfig,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            **kwargs,
        )
        self.step: int = 0
    
    def populate_modules(self):
        super().populate_modules() 

        # elimina fields per modificarli
        del self.field_coarse
        del self.field_fine
        
        # Fields
        position_encoding = FreeNeRFEncoding(
            in_dim=3, num_frequencies=self.config.position_encoding_num_frequencies, min_freq_exp=0.0, include_input=True, implementation="tcnn"
        )
        direction_encoding = FreeNeRFEncoding(
            in_dim=3, num_frequencies=self.config.direction_encoding_num_frequencies, min_freq_exp=0.0, include_input=True, implementation="tcnn"
        )

        self.field_coarse = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )

        self.field_fine = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )

        # Ray Samplers

        # Colliders

        # Renderers

        # TODO: Losses

        # Metrics

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Returns the parameter groups needed to optimizer your model components."""

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks, such as updating a density grid for Instant NGP."""
        def set_step(step: int) -> None:
            self.step = step
            self.field_coarse.position_encoding.set_ratio_x(step / self.config.T)
            self.field_coarse.direction_encoding.set_ratio_x(step / self.config.T)

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=set_step,
            ),
        ]

    def get_outputs(self, ray_bundle: RayBundle):
        """Process a RayBundle object and return RayOutputs describing quanties for each ray."""

    def get_metrics_dict(self, outputs, batch):
        """Returns metrics dictionary which will be plotted with comet, wandb or tensorboard."""

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Returns a dictionary of losses to be summed which will be your loss."""

    # prende quello di vanilla nerf
    '''def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor] #se usati bisogna dichiarare le dipendenze
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Returns a dictionary of images and metrics to plot. Here you can apply your colormaps.""" '''

    # TODO: Override any potential functions/methods to implement your own method
    # or subclass from "Model" and define all mandatory fields.