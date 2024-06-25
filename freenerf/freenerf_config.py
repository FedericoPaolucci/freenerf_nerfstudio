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
from .freenerf_model import FreeNeRFModel

'''
metodo freenerf_method che verrà richiamato da nerfstudio
'''
freenerf_method = MethodSpecification(
    config=TrainerConfig(
        method_name="freenerf-method", #nome del metodo
        steps_per_eval_batch=500, #numero passi tra ogni batch di valutazione
        steps_per_save=2000, #numero di passi tra ogni save
        max_num_iterations=30000, #numero massimo di iterazioni di training
        mixed_precision=True, #precisione mista (riduce utilizzo di memoria)
        pipeline=VanillaPipelineConfig( 
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(), # datamanager e dataparser di nerfstudio
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=FreeNeRFModel(
                eval_num_rays_per_chunk=1 << 15,
            ),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": None,
            },
            "temporal_distortion": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15), #viewer config
        vis="viewer",
    ),
    description="metodo freenerf",
)