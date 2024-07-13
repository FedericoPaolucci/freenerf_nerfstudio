"""
config per FreeNeRF

contiene il metodo che verrà richiamato da nerfstudio
"""

from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig #configura il viewer
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig #configura il parser
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig #configura il datamanager
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig #configura l'ottimizzatore
from nerfstudio.engine.schedulers import ( 
    ExponentialDecaySchedulerConfig, #configura lo scheduler
)
from nerfstudio.engine.trainer import TrainerConfig #configura il trainer
from nerfstudio.plugins.types import MethodSpecification #configura il metodo
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from .freenerf_model import FreeNeRFModelConfig

'''
metodo freenerf_method che verrà richiamato da nerfstudio
'''
freenerf_method = MethodSpecification(
    config=TrainerConfig(
        method_name="freenerf-method", #nome del metodo
        steps_per_eval_batch=500, #numero passi tra ogni batch di valutazione
        steps_per_save=2000, #numero di passi tra ogni save
        max_num_iterations=43945, #numero massimo di iterazioni di training (varia in base alle view)
        mixed_precision=True, #precisione mista (riduce utilizzo di memoria)
        pipeline=VanillaPipelineConfig( 
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig( eval_mode="filename"), # datamanager e dataparser di nerfstudio
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=FreeNeRFModelConfig(),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-08, max_norm=0.1), # originale lr=5e-4 (learning rate), eps=1e-08 (epsilon) -> lr usare da 2e-3 a 2e-5; max_norm per il clip del gradiente (no nan)
                "scheduler": ExponentialDecaySchedulerConfig(warmup_steps=512, lr_final=2e-5, lr_init=2e-3, max_steps=43945, ramp = "linear"), #warmup per far partire il training a un lr piu basso
            },
            "temporal_distortion": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-08), # originale lr=5e-4, eps=1e-08
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15), #viewer config
        vis="viewer",
    ),
    description="metodo freenerf",
)