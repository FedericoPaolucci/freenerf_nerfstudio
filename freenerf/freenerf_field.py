"""
Template Nerfstudio Field
da modificare per freenerf

Currently this subclasses the NerfactoField. Consider subclassing the base Field.

A Field is a model component that associates a region of space with some sort of quantity. 
In the most typical case, the input to a field is a 3D location and viewing direction, 
and the output is density and color.
"""

from typing import Literal, Optional

from torch import Tensor

from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.nerfacto_field import NerfactoField  # for subclassing NerfactoField
from nerfstudio.fields.base_field import Field  # for custom Field


class FreenerfField(NerfactoField): # TODO: modificare con custom
    """Template Field

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
    ) -> None:
        super().__init__(aabb=aabb, num_images=num_images)

    # TODO: Override any potential methods to implement your own field.
    # or subclass from base Field and define all mandatory methods.