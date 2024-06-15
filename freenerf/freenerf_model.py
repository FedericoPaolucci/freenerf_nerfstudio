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
class TemplateModelConfig(NerfactoModelConfig): # TODO: modificare con custom
    """Template Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: TemplateModel)


class TemplateModel(NerfactoModel): # TODO: modificare con custom
    """Template Model."""

    config: TemplateModelConfig

    def populate_modules(self):
        super().populate_modules()

    # TODO: Override any potential functions/methods to implement your own method
    # or subclass from "Model" and define all mandatory fields.