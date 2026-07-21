"""Declarative built-in baseline registry; values are lazy import paths."""

from baseline.abstract.factory import ModelRegistry

_MODEL_SPECS = {
    "eegpt": ("baseline.eegpt.eegpt_config.EegptConfig", "baseline.eegpt.eegpt_adapter.EegptDataLoaderFactory", "baseline.eegpt.eegpt_trainer.EegptTrainer", None),
    "labram": ("baseline.labram.labram_config.LabramConfig", "baseline.labram.labram_adapter.LabramDataLoaderFactory", "baseline.labram.labram_trainer.LabramTrainer", None),
    "bendr": ("baseline.bendr.bendr_config.BendrConfig", None, "baseline.bendr.bendr_trainer.BendrTrainer", "pip install -r requirements/bendr.txt"),
    "biot": ("baseline.biot.biot_config.BiotConfig", None, "baseline.biot.biot_trainer.BiotTrainer", None),
    "brainomni": ("baseline.brainomni.brainomni_config.BrainOmniConfig", "baseline.brainomni.brainomni_adapter.BrainOmniDataLoaderFactory", "baseline.brainomni.brainomni_trainer.BrainOmniTrainer", "pip install -r requirements/brainomni.txt"),
    "cbramod": ("baseline.cbramod.cbramod_config.CBraModConfig", "baseline.cbramod.cbramod_adapter.CBraModDataLoaderFactory", "baseline.cbramod.cbramod_trainer.CBraModTrainer", None),
    "reve": ("baseline.reve.reve_config.ReveConfig", "baseline.reve.reve_adapter.ReveDataLoaderFactory", "baseline.reve.reve_trainer.ReveTrainer", "pip install -r requirements/reve.txt"),
    "csbrain": ("baseline.csbrain.csbrain_config.CSBrainConfig", "baseline.csbrain.csbrain_adapter.CSBrainDataLoaderFactory", "baseline.csbrain.csbrain_trainer.CSBrainTrainer", None),
    "eegnet": ("baseline.eegnet.eegnet_config.EegNetConfig", None, "baseline.eegnet.eegnet_trainer.EegNetTrainer", "pip install -r requirements/supervised.txt"),
    "conformer": ("baseline.conformer.conformer_config.ConformerConfig", None, "baseline.conformer.conformer_trainer.ConformerTrainer", "pip install -r requirements/supervised.txt"),
    "mantis": ("baseline.mantis.mantis_config.MantisConfig", "baseline.mantis.mantis_adapter.MantisDataLoaderFactory", "baseline.mantis.mantis_trainer.MantisTrainer", None),
    "moment": ("baseline.moment.moment_config.MomentConfig", "baseline.moment.moment_adapter.MomentDataLoaderFactory", "baseline.moment.moment_trainer.MomentTrainer", None),
    "catch22": (
        "baseline.catch22.catch22_config.Catch22Config",
        None,
        "baseline.catch22.catch22_trainer.Catch22Trainer",
        "pip install -r requirements/feature_extractors.txt",
    ),
    "minirocket": (
        "baseline.minirocket.minirocket_config.MiniRocketConfig",
        None,
        "baseline.minirocket.minirocket_trainer.MiniRocketTrainer",
        "pip install -r requirements/feature_extractors.txt",
    ),
}


def register_builtin_models() -> None:
    if ModelRegistry._builtins_initialized:
        return
    for model_type, (config, adapter, trainer, dependency_hint) in _MODEL_SPECS.items():
        ModelRegistry.register_model(model_type, config, adapter, trainer, dependency_hint=dependency_hint)
    ModelRegistry._builtins_initialized = True
