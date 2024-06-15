"""
Nerfstudio Template Config
modificato per freenerf

contiene il metodo che verrà richiamato da nerfstudio
"""

from __future__ import annotations

# TODO: da modificare con quanto serve per freenerf
'''
from method_template.template_datamanager import (
    TemplateDataManagerConfig,
)
from method_template.template_model import TemplateModelConfig
from method_template.template_pipeline import (
    TemplatePipelineConfig,
)
'''

from nerfstudio.configs.base_config import ViewerConfig #configura il viewer
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig #configura il parser
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig #configura l'ottimizzatore
from nerfstudio.engine.schedulers import ( 
    ExponentialDecaySchedulerConfig, #configura lo scheduler
)
from nerfstudio.engine.trainer import TrainerConfig #configura il trainer
from nerfstudio.plugins.types import MethodSpecification #configura il metodo

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
        pipeline=TemplatePipelineConfig( # TODO: CONFIGURAZIONE PIPELINE DA MODIFICARE
            datamanager=TemplateDataManagerConfig( # TODO: datamanager custom
                dataparser=NerfstudioDataParserConfig(), # TODO: dataparser di nerfstudio?
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=TemplateModelConfig( # TODO: model custom
                eval_num_rays_per_chunk=1 << 15,
            ),
        ),
        optimizers={
            # TODO: consider changing optimizers depending on your custom method
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15), #ottimizzatore Adam con learning rate e epsilon
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000), #decadimento learning rate
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15), #RAdam
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15), #ottimizzatore camera
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15), #viewer config
        vis="viewer",
    ),
    description="metodo freenerf",
)