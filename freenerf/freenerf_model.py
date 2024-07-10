"""
Modello FreeNeRF
"""

from __future__ import annotations

import torch

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
    T: int = 39550
    """Number of training steps (max-num-iterations*0.9)"""
    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0, "occ_reg_loss": 0.01})
    """loss coefficient and Occlusion reg loss molt"""
    reg_range = 15 #10
    """Number of initial intervals to include in the regularization mask (occ reg loss)"""
    wb_prior = True
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
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
    
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
            "depth": depth_fine,
            "rgb": rgb,
            "density": density
        }
        return outputs

    # get_metrics_dict -> vanilla nerf
    def get_metrics_dict(self, outputs, batch):
        """Returns metrics dictionary which will be plotted with comet, wandb or tensorboard."""
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]
        predicted_rgb_fine = outputs["rgb_fine"]
        mask = batch["mask"] #Maschera binaria che indica le regioni foreground.
        mask_bin = (mask == 1.) #Maschera binaria con valori booleani.

        # Assicurati che il tensore di maschera sia su cuda:0
        mask = mask.to('cuda:0')
        # Sposta anche gt_rgb su cuda:0
        gt_rgb = gt_rgb.to('cuda:0')

        #Aggiunta metrica psnr senza aggiunta delle maschere
        metrics_dict["psnr_fine"] = self.psnr(gt_rgb, predicted_rgb_fine)
        metrics_dict["psnr"] = self.psnr( gt_rgb, predicted_rgb)
        metrics_dict["ssim"] = self.ssim( gt_rgb, predicted_rgb)
        metrics_dict["lpips"] = self.lpips( gt_rgb, predicted_rgb)

        #Aggiunta maschera
        inverted_mask = ~mask
        gt_rgb_masked = gt_rgb * mask + inverted_mask #Immagine originale mascherata
        predicted_rgb_resized = predicted_rgb[:, 0, :] #Rimuoviamo la dimensione densità che non ci serve per il calcolo delle metriche
        predicted_rgb_masked = predicted_rgb_resized * mask + inverted_mask #Immagine renderizzata mascherata

        metrics_dict["psnr_masked"] = self.psnr(predicted_rgb_masked, gt_rgb_masked)

         
        """gt_rgb_masked_r = torch.moveaxis(gt_rgb_masked, -1, 0)[None, ...]

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb_masked_r = torch.moveaxis(gt_rgb_masked, -1, 0)[None, ...]
        #questi passaggi servono per rendere il tensore predicted_rgb delle dimensioni desiderate dalle funzioni per le metriche ssim e lpips
        
        density = predicted_rgb.size(dim=1)#Prendo il valore contenuto nella seconda dimensione density, che abbiamo tolto prima 
        predicted_rgb_masked_prep = torch.unsqueeze(predicted_rgb_masked, 1)#Aggiungo una dimensione di size 1 in seconda posizione
        predicted_rgb_masked_prep = predicted_rgb_masked_prep.expand(-1, density, -1)#Espande la dimensione appena aggiunta a quella contenuta in density

        predicted_rgb_masked_r = torch.moveaxis(predicted_rgb_masked_prep, -1, 0)[None, ...]

        metrics_dict["ssim_masked"] = self.ssim(predicted_rgb_masked_r, gt_rgb_masked_r)
        metrics_dict["lpips_masked"] = self.lpips(predicted_rgb_masked_r, gt_rgb_masked_r)"""
        
        #TODO controllare cosa è camera optimizer e in caso come aggiungerlo, usato da nerfacto ma non in vanillanerf
        #self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict
    

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
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor] #se usati bisogna dichiarare le dipendenze
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Returns a dictionary of images and metrics to plot. Here you can apply your colormaps.""" 
        assert self.config.collider_params is not None, "mip-NeRF requires collider parameters to be set."
        image = batch["image"].to(outputs["rgb_coarse"].device)
        image = self.renderer_rgb.blend_background(image)
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        acc_coarse = colormaps.apply_colormap(outputs["accumulation_coarse"])
        acc_fine = colormaps.apply_colormap(outputs["accumulation_fine"])

        #Parte aggiunta maschera
        mask = batch["mask"]
        inverted_mask = ~mask
        image_masked = image * mask + inverted_mask #Immagine originale mascherata
        rgb_coarse_masked = rgb_coarse * mask + inverted_mask
        rgb_fine_masked = rgb_fine * mask + inverted_mask
        


        assert self.config.collider_params is not None
        depth_coarse = colormaps.apply_depth_colormap(
            outputs["depth_coarse"],
            accumulation=outputs["accumulation_coarse"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )
        depth_fine = colormaps.apply_depth_colormap(
            outputs["depth_fine"],
            accumulation=outputs["accumulation_fine"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
        combined_acc = torch.cat([acc_coarse, acc_fine], dim=1)
        combined_depth = torch.cat([depth_coarse, depth_fine], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image_r = torch.moveaxis(image_masked, -1, 0)[None, ...]
        rgb_coarse_r = torch.moveaxis(rgb_coarse_masked, -1, 0)[None, ...]
        rgb_fine_r = torch.moveaxis(rgb_fine_masked, -1, 0)[None, ...]
        rgb_coarse_r = torch.clip(rgb_coarse_r, min=0, max=1)
        rgb_fine_r = torch.clip(rgb_fine_r, min=0, max=1)

        coarse_psnr = self.psnr(image_r, rgb_coarse_r)
        fine_psnr = self.psnr(image_r, rgb_fine_r)
        fine_ssim = self.ssim(image_r, rgb_fine_r)
        fine_lpips = self.lpips(image_r, rgb_fine_r)

        assert isinstance(fine_ssim, torch.Tensor)
        metrics_dict = {
            "psnr": float(fine_psnr.item()),
            "coarse_psnr": float(coarse_psnr.item()),
            "fine_psnr": float(fine_psnr.item()),
            "fine_ssim": float(fine_ssim.item()),
            "fine_lpips": float(fine_lpips.item()),
        }
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
        return metrics_dict, images_dict
    #plot loss

    # TODO: Override any potential functions/methods to implement your own method
    # or subclass from "Model" and define all mandatory fields.