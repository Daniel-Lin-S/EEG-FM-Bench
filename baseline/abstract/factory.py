"""Factory classes for creating baseline models, adapters, and trainers."""

import importlib
from typing import Any, Dict, Optional, Type, Union

from baseline.abstract.adapter import AbstractDataLoaderFactory
from baseline.abstract.config import AbstractConfig
from baseline.abstract.trainer import AbstractTrainer

ComponentReference = Union[Type[Any], str]


class OptionalModelDependencyError(ImportError):
    """Raised when the selected baseline is missing one of its extras."""

    def __init__(self, model_type: str, dependency: str, install_hint: str):
        super().__init__(
            f"Model '{model_type}' requires the optional dependency '{dependency}'. "
            f"Install it with: {install_hint}"
        )
        self.model_type = model_type
        self.dependency = dependency
        self.install_hint = install_hint


class ModelRegistry:
    """Registry which resolves model components only when they are requested."""

    configs: Dict[str, ComponentReference] = {}
    adapters: Dict[str, Optional[ComponentReference]] = {}
    trainers: Dict[str, ComponentReference] = {}
    dependency_hints: Dict[str, str] = {}
    _builtins_initialized = False

    @classmethod
    def _ensure_builtin_models(cls):
        if not cls._builtins_initialized:
            from baseline.registry import register_builtin_models
            register_builtin_models()

    @classmethod
    def register_model(cls, model_type: str, config_class: ComponentReference,
                       adapter_class: Optional[ComponentReference], trainer_class: ComponentReference,
                       dependency_hint: Optional[str] = None):
        cls.configs[model_type] = config_class
        cls.adapters[model_type] = adapter_class
        cls.trainers[model_type] = trainer_class
        if dependency_hint:
            cls.dependency_hints[model_type] = dependency_hint

    @classmethod
    def _resolve_component(cls, model_type: str, component_name: str,
                           component: ComponentReference) -> Type[Any]:
        if not isinstance(component, str):
            return component
        module_name, separator, attribute_name = component.rpartition(".")
        if not separator:
            raise ValueError(f"Invalid {component_name} reference for '{model_type}': {component}")
        try:
            resolved = getattr(importlib.import_module(module_name), attribute_name)
        except ModuleNotFoundError as exc:
            install_hint = cls.dependency_hints.get(model_type)
            if install_hint:
                raise OptionalModelDependencyError(model_type, exc.name or module_name, install_hint) from exc
            raise
        getattr(cls, f"{component_name}s")[model_type] = resolved
        return resolved

    @classmethod
    def get_config_class(cls, model_type: str) -> Type[AbstractConfig]:
        cls._ensure_builtin_models()
        if model_type not in cls.configs:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls.configs.keys())}")
        return cls._resolve_component(model_type, "config", cls.configs[model_type])

    @classmethod
    def get_adapter_class(cls, model_type: str) -> Optional[Type[AbstractDataLoaderFactory]]:
        cls._ensure_builtin_models()
        if model_type not in cls.adapters:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls.adapters.keys())}")
        adapter_class = cls.adapters[model_type]
        return None if adapter_class is None else cls._resolve_component(model_type, "adapter", adapter_class)

    @classmethod
    def get_trainer_class(cls, model_type: str) -> Type[AbstractTrainer]:
        cls._ensure_builtin_models()
        if model_type not in cls.trainers:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls.trainers.keys())}")
        return cls._resolve_component(model_type, "trainer", cls.trainers[model_type])

    @classmethod
    def list_models(cls) -> list[str]:
        cls._ensure_builtin_models()
        return list(cls.configs.keys())

    @classmethod
    def create_config(cls, model_type: str, **kwargs) -> AbstractConfig:
        return cls.get_config_class(model_type)(model_type=model_type, **kwargs)

    @classmethod
    def create_trainer(cls, config: AbstractConfig) -> AbstractTrainer:
        return cls.get_trainer_class(config.model_type)(config)


class BaselineModelFactory:
    @staticmethod
    def create_from_config(config_dict: Dict[str, Any]) -> AbstractTrainer:
        model_type = config_dict.get("model_type", "eegpt")
        config = ModelRegistry.create_config(model_type, **config_dict)
        if not config.validate_config():
            raise ValueError(f"Invalid configuration for model type: {model_type}")
        return ModelRegistry.create_trainer(config)

    @staticmethod
    def list_available_models() -> list[str]:
        return ModelRegistry.list_models()
