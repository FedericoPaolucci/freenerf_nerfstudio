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
    # init
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
        #controlli
        if max_freq_exp is None:
            max_freq_exp = num_frequencies - 1
        else:
            assert max_freq_exp == num_frequencies - 1, "FreeNeRF must max_freq_exp == num_frequencies - 1"
        # costruttore superclasse
        super().__init__(in_dim, num_frequencies, min_freq_exp, max_freq_exp, include_input, implementation)
        self.ratio_x: float = 1.0
 
    # Call forward and returns and processed tensor
    # Args:
    #    in_tensor: the input tensor to process  
    def forward(
        self, 
        in_tensor: Float[Tensor, "*bs input_dim"], #tensore con una forma che include batch size e lunghezza vettore caratteristiche
        covs: Optional[Float[Tensor, "*bs input_dim input_dim"]] = None,
    ) -> Float[Tensor, "*bs output_dim"]:
        # selezione tcnn implementation
        if self.tcnn_encoding is not None:
            shape = in_tensor.shape # salva la shape (dimensione) del tensore
            if len(shape) > 2: # se la dimensione è maggiore di 2 
                in_tensor = in_tensor.reshape(-1, shape[-1]) # lo ridimensiona in una matrice bs x input_dim (con shape[-1] l'ultima dim non viene toccata)
            encodings = self.tcnn_encoding(in_tensor) # uso funzioni sinusoidali del paper freenerf  
            weights = self.get_weights().to(encodings.device) # richiama funzione get_weights, ottiene i pesi e li sposta sul dispositivo di encoding
            encodings = encodings * weights # applica i pesi a encodings
            encodings = torch.cat((encodings, in_tensor), dim=-1) # concatena encodings e in_tensor lungo l'ultima dimensione (dim=-1)
            if len(shape) > 2:
                encodings = encodings.reshape(*shape[:-1], -1) # lo riporta a una dimensione coerente a quella iniziale (se shape era > 2)
        # selezione torch implementation
        else:
            encodings = self.pytorch_fwd(in_tensor, covs) 
            weights = self.get_weights().to(encodings.device)
            encodings[:, :-self.in_dim] = encodings[:,:-self.in_dim]* weights
        # ritorna il risultato della codifica 
        return encodings 

    @property
    def L(self) -> int: #parametro L nella formula
        return self.num_frequencies
    
    # il metodo restituisce il tensore weights (1D) modificato in base al ratio e ad L
    @torch.no_grad()
    def get_weights(self) -> Float[Tensor, "1 output_dim"]: # tensore di dimensione 1 con lunghezza output_dim
        weights = torch.ones((self.L,), dtype=torch.float32) # tensore di lunghezza L di 1
        # setta il valore nella posizione di weights specificata alla parte decimale del prodotto ratio_x * L
        weights[int(self.ratio_x * self.L) : int(self.ratio_x * self.L) + 1] = self.ratio_x * self.L - int(self.ratio_x * self.L)
        weights[int(self.ratio_x * self.L) + 1 : ] = 0.0 # imposta gli elementi del tensore dopo quella posizione (come da paper) a 0
        # selezione tcnn implementation
        if self.tcnn_encoding is not None:
            weights = weights.unsqueeze(-1) # aggiunge una dim a weights
            weights = weights.repeat((self.in_dim, 2)).reshape(-1) # con repeat nuova forma diventa (L*in_dim, 2), con reshape lo riporta a una dim
        # selezione torch implementation
        else:
            weights = weights.unsqueeze(0)
            weights = weights.repeat((2, self.in_dim)).reshape(-1) # come sopra ma (2, L*in_dim)
        return weights
    
    # set del ratio 
    def set_ratio_x(self, ratio_x: float) -> None:
        ratio_x = max(0.0, min(1.0, ratio_x)) 
        self.ratio_x = ratio_x