"""
Encoding per freenerf
"""

from typing import Literal, Optional, Sequence
import torch
from jaxtyping import Float, Int, Shaped
from torch import Tensor, nn
from nerfstudio.field_components.encodings import NeRFEncoding


class FreeNeRFEncoding(NeRFEncoding):
    """Multi-scale sinusoidal encodings. Support ``integrated positional encodings`` if covariances are provided.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoded frequencies per axis
        min_freq_exp: Minimum frequency exponent
        max_freq_exp: Maximum frequency exponent
        include_input: Append the input coordinate to the encoding
    """

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        min_freq_exp: float,
        include_input: bool = False,
        max_freq_exp: float = None,
        implementation: Literal["tcnn", "torch"] = "torch",
    ) -> None:
        assert include_input is True, "FreeNeRF must include input"

        if max_freq_exp is None:
            max_freq_exp = num_frequencies - 1
        else:
            assert max_freq_exp == num_frequencies - 1, "FreeNeRF must max_freq_exp == num_frequencies - 1"

        super().__init__(in_dim, num_frequencies, min_freq_exp, max_freq_exp, include_input, implementation)
        self.ratio_x: float = 1.0

    def forward(
        self, 
        in_tensor: Float[Tensor, "*bs input_dim"], 
        covs: Optional[Float[Tensor, "*bs input_dim input_dim"]] = None,
    ) -> Float[Tensor, "*bs output_dim"]:
        if self.tcnn_encoding is not None:
            shape = in_tensor.shape
            if len(shape) > 2:
                in_tensor = in_tensor.reshape(-1, shape[-1])
            encodings = self.tcnn_encoding(in_tensor) 
            weights = self.get_weights().to(encodings.device)
            encodings = encodings * weights
            encodings = torch.cat((encodings, in_tensor), dim=-1) 
            if len(shape) > 2:
                encodings = encodings.reshape(*shape[:-1], -1)
        else:
            encodings = self.pytorch_fwd(in_tensor, covs) 
            weights = self.get_weights().to(encodings.device)
            encodings[:, :-self.in_dim] = encodings[:,:-self.in_dim]* weights

        return encodings

    @property
    def L(self) -> int: #parametro L nella formula
        return self.num_frequencies
    
    @torch.no_grad()
    def get_weights(self) -> Float[Tensor, "1 output_dim"]:
        weights = torch.ones((self.L,), dtype=torch.float32) 
        weights[int(self.ratio_x * self.L) : int(self.ratio_x * self.L) + 1] = self.ratio_x * self.L - int(self.ratio_x * self.L)
        weights[int(self.ratio_x * self.L) + 1 : ] = 0.0
        if self.tcnn_encoding is not None:
            weights = weights.unsqueeze(-1)
            weights = weights.repeat((self.in_dim, 2)).reshape(-1) 
        else:
            weights = weights.unsqueeze(0)
            weights = weights.repeat((2, self.in_dim)).reshape(-1)
        return weights
    
    # set del ratio
    def set_ratio_x(self, ratio_x: float) -> None:
        ratio_x = max(0.0, min(1.0, ratio_x))
        self.ratio_x = ratio_x