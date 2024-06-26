"""
Modello FreeNeRF
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
from nerfstudio.model_components.losses import MSELoss, scale_gradients_by_distance_squared
from nerfstudio.utils import colormaps, misc
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.configs.config_utils import to_immutable_dict
from .freenerf_loss import occ_reg_loss_fn
from .freenerf_encoding import FreeNeRFEncoding

@dataclass
class FreeNeRFModelConfig(VanillaModelConfig):
    """Vanilla Model Config"""

    _target: Type = field(default_factory=lambda: FreeNeRFModel)
    
    position_encoding_num_frequencies: int = 16 #10
    """Number of frequencies for positional encoding"""
    direction_encoding_num_frequencies: int = 4
    """Number of frequencies for directional encoding"""
    T: int = 30000
    """Number of training steps (must equal to max-num-iterations)"""
    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0, "occ_reg_loss": 0.01})
    """loss coefficient and Occlusion reg loss molt"""
    reg_range = 10
    """Number of initial intervals to include in the regularization mask (occ reg loss)"""
    wb_prior = False
    """If True, a prior based on the assumption of white or black backgrounds is used (occ reg loss)"""
    wb_range=20
    """Range of RGB values considered to be a white or black background (occ reg loss)"""


class FreeNeRFModel(NeRFModel):

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

        # sono ricreati i fields modificati con encoding freenerf
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

        # Losses
        self.rgb_loss = MSELoss()

        # Metrics
    
    # get_param_groups -> prende quello di vanilla nerf
    '''
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Returns the parameter groups needed to optimizer your model components."""
     '''
    
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """callbacks eseguiti prima di ogni iterazione di training"""

        # imposta lo step e i ratio
        def set_step(step: int) -> None:
            self.step = step # imposta lo step attuale
            self.field_coarse.position_encoding.set_ratio_x(step / self.config.T) # set del ratio con t/T che nell'encoding verrà moltiplicato per L come da paper freenerf
            self.field_coarse.direction_encoding.set_ratio_x(step / self.config.T) # set del ratio come sopra

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=set_step, # richiama funzione PRIMA di ogni iterazione (l'argomento sarà il numero di step come da classe TrainingCallback)
            ),
        ]

    # get_outputs per aggiungere gli output per occ reg loss
    def get_outputs(self, ray_bundle: RayBundle):
        """Process a RayBundle object and return RayOutputs describing quanties for each ray."""
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        if self.temporal_distortion is not None:
            offsets = None
            if ray_samples_uniform.times is not None:
                offsets = self.temporal_distortion(
                    ray_samples_uniform.frustums.get_positions(), ray_samples_uniform.times
                )
            ray_samples_uniform.frustums.set_offsets(offsets)

        # coarse field:
        field_outputs_coarse = self.field_coarse.forward(ray_samples_uniform)
        if self.config.use_gradient_scaling:
            field_outputs_coarse = scale_gradients_by_distance_squared(field_outputs_coarse, ray_samples_uniform)
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)
        if self.temporal_distortion is not None:
            offsets = None
            if ray_samples_pdf.times is not None:
                offsets = self.temporal_distortion(ray_samples_pdf.frustums.get_positions(), ray_samples_pdf.times)
            ray_samples_pdf.frustums.set_offsets(offsets)

        # fine field:
        field_outputs_fine = self.field_fine.forward(ray_samples_pdf)
        if self.config.use_gradient_scaling:
            field_outputs_fine = scale_gradients_by_distance_squared(field_outputs_fine, ray_samples_pdf)
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

        # per occlusion reg loss
        rgb = field_outputs_fine[FieldHeadNames.RGB]
        density = field_outputs_fine[FieldHeadNames.DENSITY]

        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
            "rgb": rgb,
            "density": density
        }
        return outputs

    # get_metrics_dict -> vanilla nerf
    '''def get_metrics_dict(self, outputs, batch):
        """Returns metrics dictionary which will be plotted with comet, wandb or tensorboard."""
    '''

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Returns a dictionary of losses to be summed which will be your loss."""
        # Impostazione device e immagine di input
        device = outputs["rgb_coarse"].device
        image = batch["image"].to(device)
        # preparazione delle immagini per la loss
        coarse_pred, coarse_image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_coarse"],
            pred_accumulation=outputs["accumulation_coarse"],
            gt_image=image,
        )
        fine_pred, fine_image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_fine"],
            pred_accumulation=outputs["accumulation_fine"],
            gt_image=image,
        )
        # calcolo loss 
        rgb_loss_coarse = self.rgb_loss(coarse_image, coarse_pred) # MSELoss (ground truth e predizione)
        rgb_loss_fine = self.rgb_loss(fine_image, fine_pred)
        # occlusion regulation loss
        occ_reg_loss = occ_reg_loss_fn(outputs["rgb"], outputs["density"], reg_range=self.config.reg_range, wb_prior=self.config.wb_prior, wb_range=self.config.wb_range)
        # creazione dict
        loss_dict = {"rgb_loss_coarse": rgb_loss_coarse, "rgb_loss_fine": rgb_loss_fine, "occ_reg_loss": occ_reg_loss}
        # scalatura loss
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    # get_image_mertrics_and_images -> prende quello di vanilla nerf
    '''def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor] #se usati bisogna dichiarare le dipendenze
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Returns a dictionary of images and metrics to plot. Here you can apply your colormaps.""" '''

    # TODO: Override any potential functions/methods to implement your own method
    # or subclass from "Model" and define all mandatory fields.