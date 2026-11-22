from __future__ import annotations

from copy import deepcopy
from unittest.mock import Mock

import pytest
from beyondbench.tasks.hard.knapsack_task import KnapsackTask

from llm_eval.runner import (
    ConfigError,
    bounded_base_task_retries,
    load_config,
    plan_config,
    validate_config,
)


def test_pretest_plan_and_validation(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config("configs/pretest.yaml")
    plan = plan_config(config)
    assert plan["planned_requests_per_model"] == 50

    configured = deepcopy(config)
    configured["models"] = [configured["models"][0]]
    configured["models"][0].update(
        {
            "enabled": True,
            "model": "actual-provider-model-id",
            "api_key_env": "TEST_DOMESTIC_KEY",
        }
    )
    monkeypatch.setenv("TEST_DOMESTIC_KEY", "secret")
    result = validate_config(configured, require_keys=True)
    assert result["planned_total_requests"] == 50
    assert result["enabled_models"] == ["deepseek"]


def test_budget_rejects_oversized_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config("configs/pretest.yaml")
    config["models"] = [config["models"][0]]
    config["models"][0].update(
        {
            "enabled": True,
            "model": "actual-provider-model-id",
            "api_key_env": "TEST_DOMESTIC_KEY",
        }
    )
    config["budget"]["max_requests_per_model"] = 10
    monkeypatch.setenv("TEST_DOMESTIC_KEY", "secret")
    with pytest.raises(ConfigError, match="exceeds budget"):
        validate_config(config, require_keys=True)


def test_single_task_targeted_stress_config_is_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config("configs/sat_stress.yaml")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    result = validate_config(config, require_keys=True)
    assert result["n_tasks"] == 1
    assert result["planned_requests_per_model"] == 6


def test_run_level_retry_bound_applies_to_hard_tasks(tmp_path) -> None:
    model = Mock()
    model.get_model_info.return_value = {"model_name": "fake"}
    with bounded_base_task_retries(1):
        task = KnapsackTask(
            model_handler=model,
            output_dir=str(tmp_path),
            num_folds=1,
            num_samples=1,
            store_details=False,
            seed=1,
        )
        assert task.max_retries == 1
