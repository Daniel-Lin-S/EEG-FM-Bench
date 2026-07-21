import importlib
import sys

import pytest

from baseline.abstract.factory import ModelRegistry, OptionalModelDependencyError


def test_listing_models_does_not_import_model_trainers():
    assert {"brainomni", "reve", "eegnet", "conformer"}.issubset(ModelRegistry.list_models())
    assert not any(name.endswith("_trainer") for name in sys.modules if name.startswith("baseline."))


def test_brainomni_resolution_does_not_import_unselected_models():
    ModelRegistry.get_config_class("brainomni")
    ModelRegistry.get_trainer_class("brainomni")
    assert "baseline.brainomni.brainomni_trainer" in sys.modules
    assert "baseline.reve.reve_trainer" not in sys.modules
    assert "baseline.eegnet.eegnet_trainer" not in sys.modules
    assert "baseline.conformer.conformer_trainer" not in sys.modules


def test_reve_missing_dependency_has_install_guidance(monkeypatch):
    original = importlib.import_module

    def missing_optimi(module_name, package=None):
        if module_name == "baseline.reve.reve_trainer":
            raise ModuleNotFoundError("No module named 'optimi'", name="optimi")
        return original(module_name, package)

    monkeypatch.setattr(importlib, "import_module", missing_optimi)
    ModelRegistry.trainers["reve"] = "baseline.reve.reve_trainer.ReveTrainer"
    with pytest.raises(OptionalModelDependencyError, match="requirements/reve.txt"):
        ModelRegistry.get_trainer_class("reve")
