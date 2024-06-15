"""
Template DataManager
da modificare per freenerf

The DataManager returns RayBundle and RayGT objects.
abstract methods required by the DataManager:
@abstractmethod
    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        Returns the next batch of data for train.

    @abstractmethod
    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        Returns the next batch of data for eval.

    @abstractmethod
    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        Returns the next eval image.
        Returns:
            The image index from the eval dataset, the CameraRayBundle, and the RayGT dictionary.
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)


@dataclass
class TemplateDataManagerConfig(VanillaDataManagerConfig): # TODO: modificare con custom
    """Template DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: TemplateDataManager)


class TemplateDataManager(VanillaDataManager): # TODO: modificare con custom
    """Template DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: TemplateDataManagerConfig

    def __init__(
        self,
        config: TemplateDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch