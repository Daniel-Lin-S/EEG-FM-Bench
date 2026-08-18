"""Test that frozen encoder identity fix works correctly."""

from baseline.utils.identity import (
    build_campaign_semantic_config,
    build_run_semantic_config,
    semantic_digest,
)


def test_frozen_encoder_filters_encoder_lr_scale_from_identity() -> None:
    """Frozen encoder should exclude encoder_lr_scale from campaign identity."""
    base_config = {
        "model_type": "brainomni",
        "multitask": False,
        "fs": 256,
        "data": {"batch_size": 128},
        "model": {"pretrained_path": "/some/path"},
        "training": {
            "max_epochs": 30,
            "freeze_encoder": True,  # Frozen
            "encoder_lr_scale": 0.5,
            "max_lr": 1e-3,
        },
        "logging": {"experiment_name": "test"},
    }

    hpo_config = {
        "enabled": True,
        "seed": 0,
        "n_trials": 10,
        "objective": {"metric": "loss", "direction": "minimize"},
        "sampler": {"type": "tpe"},
        "pruner": {"type": "median"},
        "search_space": {
            "training.encoder_lr_scale": {  # Should be filtered out
                "distribution": "float",
                "low": 0.5,
                "high": 1.0,
            },
            "training.max_lr": {
                "distribution": "float",
                "low": 1e-4,
                "high": 1e-3,
                "log": True,
            },
        },
    }

    semantic = build_campaign_semantic_config(base_config, hpo_config)

    # encoder_lr_scale should NOT be in semantic search_space
    assert "training.encoder_lr_scale" not in semantic["hpo"]["search_space"]
    # max_lr should still be there
    assert "training.max_lr" in semantic["hpo"]["search_space"]


def test_unfrozen_encoder_keeps_encoder_lr_scale_in_identity() -> None:
    """Unfrozen encoder should keep encoder_lr_scale in campaign identity."""
    base_config = {
        "model_type": "brainomni",
        "multitask": False,
        "fs": 256,
        "data": {"batch_size": 128},
        "model": {"pretrained_path": "/some/path"},
        "training": {
            "max_epochs": 30,
            "freeze_encoder": False,  # NOT frozen
            "encoder_lr_scale": 0.5,
            "max_lr": 1e-3,
        },
        "logging": {"experiment_name": "test"},
    }

    hpo_config = {
        "enabled": True,
        "seed": 0,
        "n_trials": 10,
        "objective": {"metric": "loss", "direction": "minimize"},
        "sampler": {"type": "tpe"},
        "pruner": {"type": "median"},
        "search_space": {
            "training.encoder_lr_scale": {
                "distribution": "float",
                "low": 0.5,
                "high": 1.0,
            },
            "training.max_lr": {
                "distribution": "float",
                "low": 1e-4,
                "high": 1e-3,
                "log": True,
            },
        },
    }

    semantic = build_campaign_semantic_config(base_config, hpo_config)

    # encoder_lr_scale SHOULD be in semantic search_space (unfrozen)
    assert "training.encoder_lr_scale" in semantic["hpo"]["search_space"]
    # max_lr should still be there
    assert "training.max_lr" in semantic["hpo"]["search_space"]


def test_frozen_and_unfrozen_have_different_identity() -> None:
    """Frozen vs unfrozen encoder should have different campaign identities."""
    base_config_frozen = {
        "model_type": "brainomni",
        "multitask": False,
        "fs": 256,
        "data": {"batch_size": 128},
        "model": {"pretrained_path": "/some/path"},
        "training": {
            "max_epochs": 30,
            "freeze_encoder": True,  # Frozen
            "encoder_lr_scale": 0.5,
            "max_lr": 1e-3,
        },
        "logging": {"experiment_name": "test"},
    }

    base_config_unfrozen = {
        **base_config_frozen,
        "training": {
            **base_config_frozen["training"],
            "freeze_encoder": False,  # Unfrozen
        },
    }

    hpo_config = {
        "enabled": True,
        "seed": 0,
        "n_trials": 10,
        "objective": {"metric": "loss", "direction": "minimize"},
        "sampler": {"type": "tpe"},
        "pruner": {"type": "median"},
        "search_space": {
            "training.encoder_lr_scale": {
                "distribution": "float",
                "low": 0.5,
                "high": 1.0,
            },
            "training.max_lr": {
                "distribution": "float",
                "low": 1e-4,
                "high": 1e-3,
                "log": True,
            },
        },
    }

    frozen_semantic = build_campaign_semantic_config(
        base_config_frozen,
        hpo_config,
    )
    unfrozen_semantic = build_campaign_semantic_config(
        base_config_unfrozen,
        hpo_config,
    )

    frozen_identity = semantic_digest(frozen_semantic)
    unfrozen_identity = semantic_digest(unfrozen_semantic)

    # Different freeze states should yield different identities
    assert frozen_identity != unfrozen_identity

    # Frozen should NOT have encoder_lr_scale in search
    assert (
        "training.encoder_lr_scale"
        not in frozen_semantic["hpo"]["search_space"]
    )
    # Unfrozen SHOULD have encoder_lr_scale in search
    assert (
        "training.encoder_lr_scale"
        in unfrozen_semantic["hpo"]["search_space"]
    )


def test_frozen_run_identity_ignores_encoder_lr_scale() -> None:
    """A frozen final run is unchanged by an inactive encoder LR scale."""
    first = {
        "model_type": "brainomni",
        "multitask": False,
        "seeds": [42],
        "data": {"datasets": {"alpha": "finetune"}},
        "model": {},
        "training": {
            "freeze_encoder": True,
            "encoder_lr_scale": 0.5,
        },
        "logging": {},
    }
    second = {
        **first,
        "training": {
            **first["training"],
            "encoder_lr_scale": 0.9,
        },
    }

    assert build_run_semantic_config(
        first,
        False,
    ) == build_run_semantic_config(second, False)
