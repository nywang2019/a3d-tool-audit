from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import networkx as nx
import pytest

from llm_eval.funcbench import _gated_tools, prepare_condition, run_trial
from llm_eval.funcbench_errors import (
    _mcnemar_exact_p,
    _wilson_interval,
    classify_trial,
)


class ScriptedToolAdapter:
    def __init__(self, condition):
        self.condition = condition
        self.order = list(nx.topological_sort(condition.function_tree.graph))
        self.index = 0
        self.histories = []

    def chat(self, messages, *, tools, **_kwargs):
        self.histories.append(messages)
        if self.index >= len(self.order):
            return {
                "message": {
                    "role": "assistant",
                    "content": str(self.condition.target_value),
                },
                "content": str(self.condition.target_value),
                "tool_calls": [],
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }
        node_name = self.order[self.index]
        function = self.condition.function_tree.graph.nodes[node_name]["function"]
        arguments = {
            variable: self.condition.variable_values[variable]
            for variable in function.inputs
        }
        self.index += 1
        tool_call = {
            "id": f"call_{self.index}",
            "type": "function",
            "function": {
                "name": function.name,
                "arguments": json.dumps(arguments),
            },
        }
        return {
            "message": {
                "role": "assistant",
                "content": None,
                "reasoning_content": "I should call the next required function.",
                "tool_calls": [tool_call],
            },
            "content": "",
            "tool_calls": [tool_call],
            "finish_reason": "tool_calls",
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }


class SealedThenCoreAdapter:
    def __init__(self, condition):
        self.condition = condition
        self.sealed_name = min(condition.sealed_functions)
        target_node = condition.function_tree.desired_target_node
        required = set(nx.ancestors(condition.function_tree.graph, target_node))
        required.add(target_node)
        self.order = list(
            nx.topological_sort(condition.function_tree.graph.subgraph(required))
        )
        self.index = -1

    def chat(self, messages, *, tools, **_kwargs):
        if self.index == -1:
            self.index += 1
            tool_call = {
                "id": "sealed_call",
                "type": "function",
                "function": {
                    "name": self.sealed_name,
                    "arguments": "{}",
                },
            }
        elif self.index < len(self.order):
            node_name = self.order[self.index]
            function = self.condition.function_tree.graph.nodes[node_name][
                "function"
            ]
            arguments = {
                variable: self.condition.variable_values[variable]
                for variable in function.inputs
            }
            self.index += 1
            tool_call = {
                "id": f"core_call_{self.index}",
                "type": "function",
                "function": {
                    "name": function.name,
                    "arguments": json.dumps(arguments),
                },
            }
        else:
            return {
                "message": {
                    "role": "assistant",
                    "content": str(self.condition.target_value),
                },
                "content": str(self.condition.target_value),
                "tool_calls": [],
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }
        return {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call],
            },
            "content": "",
            "tool_calls": [tool_call],
            "finish_reason": "tool_calls",
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }


def test_prepare_condition_is_deterministic_and_has_truth() -> None:
    kwargs = {
        "depth": 2,
        "distractor_count": 5,
        "graph_seed": 123,
        "core_nodes": 4,
        "trial_seed": 456,
        "use_types": True,
    }
    first = prepare_condition(**kwargs)
    second = prepare_condition(**kwargs)
    assert first.prompt == second.prompt
    assert first.target_value == second.target_value
    assert len(first.tools) == 9
    assert first.max_calls == first.minimum_calls * 2


def test_reminder_modes_preserve_graph_and_prompt() -> None:
    kwargs = {
        "depth": 3,
        "distractor_count": 5,
        "graph_seed": 20263029,
        "core_nodes": 4,
        "trial_seed": 20281029,
        "use_types": True,
    }
    baseline = prepare_condition(
        **kwargs,
        repeat_known_variable_values=False,
    )
    reminder = prepare_condition(
        **kwargs,
        repeat_known_variable_values=True,
    )
    assert baseline.graph_fingerprint == reminder.graph_fingerprint
    assert baseline.prompt == reminder.prompt
    assert baseline.target_value == reminder.target_value


def test_paired_conditions_preserve_required_graph_and_truth() -> None:
    kwargs = {
        "depth": 3,
        "graph_seed": 20264029,
        "core_nodes": 4,
        "trial_seed": 20282029,
        "use_types": True,
        "paired_core_graph": True,
        "stable_tool_order": True,
    }
    baseline = prepare_condition(distractor_count=0, **kwargs)
    distractor = prepare_condition(distractor_count=5, **kwargs)

    baseline_graph = baseline.function_tree.graph
    distractor_graph = distractor.function_tree.graph
    assert set(baseline_graph.nodes) < set(distractor_graph.nodes)
    for name in baseline_graph.nodes:
        baseline_function = baseline_graph.nodes[name]["function"]
        distractor_function = distractor_graph.nodes[name]["function"]
        assert baseline_function.inputs == distractor_function.inputs
        assert baseline_function.output == distractor_function.output
        assert baseline_function.description == distractor_function.description

    assert baseline.required_graph_fingerprint == distractor.required_graph_fingerprint
    assert baseline.graph_fingerprint != distractor.graph_fingerprint
    assert baseline.prompt == distractor.prompt
    assert baseline.target_value == distractor.target_value
    assert len(baseline.tools) == 4
    assert len(distractor.tools) == 9

    baseline_names = {tool["function"]["name"] for tool in baseline.tools}
    distractor_names = {tool["function"]["name"] for tool in distractor.tools}
    baseline_order = [
        tool["function"]["name"]
        for tool in distractor.tools
        if tool["function"]["name"] in baseline_names
    ]
    assert baseline_order == [tool["function"]["name"] for tool in baseline.tools]
    assert baseline_names < distractor_names


def test_distractor_controls_preserve_core_but_change_affordance() -> None:
    kwargs = {
        "depth": 3,
        "graph_seed": 20264029,
        "core_nodes": 4,
        "trial_seed": 20282029,
        "use_types": True,
        "paired_core_graph": True,
        "stable_tool_order": True,
    }
    baseline = prepare_condition(distractor_count=0, **kwargs)
    conditions = {
        mode: prepare_condition(
            distractor_count=5,
            distractor_mode=mode,
            **kwargs,
        )
        for mode in (
            "connected_compatible",
            "disconnected",
            "connected_incompatible",
        )
    }
    for condition in conditions.values():
        assert (
            condition.required_graph_fingerprint
            == baseline.required_graph_fingerprint
        )
        assert condition.prompt == baseline.prompt
        assert condition.target_value == baseline.target_value
        assert len(condition.tools) == 9
    assert len({condition.graph_fingerprint for condition in conditions.values()}) == 3
    assert not conditions["connected_compatible"].unavailable_distractor_types
    assert conditions["disconnected"].unavailable_distractor_types
    assert conditions["connected_incompatible"].unavailable_distractor_types


def test_sealed_compatible_keeps_schema_but_blocks_distractor_execution() -> None:
    kwargs = {
        "depth": 3,
        "graph_seed": 20264029,
        "core_nodes": 4,
        "trial_seed": 20282029,
        "use_types": True,
        "paired_core_graph": True,
        "stable_tool_order": True,
        "distractor_count": 5,
    }
    executable = prepare_condition(
        distractor_mode="connected_compatible",
        **kwargs,
    )
    sealed = prepare_condition(
        distractor_mode="sealed_compatible",
        **kwargs,
    )
    assert sealed.tools == executable.tools
    assert sealed.prompt == executable.prompt
    assert sealed.graph_fingerprint == executable.graph_fingerprint
    assert sealed.required_graph_fingerprint == executable.required_graph_fingerprint
    assert sealed.sealed_functions
    assert not executable.sealed_functions

    row = run_trial(
        sealed,
        SealedThenCoreAdapter(sealed),
        max_tokens=128,
        temperature=1.0,
        top_p=0.95,
        use_types=True,
        trial_id="sealed",
    )
    assert row["calls"][0]["failure"] == "Sealed tool unavailable"
    assert "Sealed tool unavailable" in row["failures"]


@pytest.mark.parametrize("graph_index", [0, 7, 14])
def test_paired_main_seed_scheme_is_invariant(graph_index: int) -> None:
    base_seed = 20261029
    depth = 3
    kwargs = {
        "depth": depth,
        "graph_seed": base_seed + depth * 1000 + graph_index,
        "core_nodes": 4,
        "trial_seed": base_seed + depth * 10000 + graph_index,
        "use_types": True,
        "paired_core_graph": True,
        "stable_tool_order": True,
    }
    baseline = prepare_condition(distractor_count=0, **kwargs)
    for distractor_count in (2, 5):
        distractor = prepare_condition(
            distractor_count=distractor_count,
            **kwargs,
        )
        assert (
            baseline.required_graph_fingerprint
            == distractor.required_graph_fingerprint
        )
        assert baseline.prompt == distractor.prompt
        assert baseline.target_variable == distractor.target_variable
        assert baseline.target_value == distractor.target_value


def test_paired_fingerprint_is_stable_across_hash_seeds() -> None:
    script = (
        "from llm_eval.funcbench import prepare_condition;"
        "c=prepare_condition(depth=3,distractor_count=5,"
        "graph_seed=20264029,core_nodes=4,trial_seed=20282029,"
        "paired_core_graph=True,stable_tool_order=True);"
        "print(c.required_graph_fingerprint,c.graph_fingerprint,"
        "c.target_value,repr(c.prompt))"
    )
    project_root = Path(__file__).resolve().parents[1]
    fingerprints = []
    for hash_seed in ("1", "2"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=project_root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        fingerprints.append(result.stdout.strip().split())
    assert fingerprints[0] == fingerprints[1]


def test_run_trial_executes_tools_and_scores_exact_target() -> None:
    condition = prepare_condition(
        depth=1,
        distractor_count=0,
        graph_seed=123,
        core_nodes=2,
        trial_seed=456,
        use_types=False,
    )
    adapter = ScriptedToolAdapter(condition)
    row = run_trial(
        condition,
        adapter,
        max_tokens=128,
        temperature=1.0,
        top_p=0.95,
        use_types=False,
        trial_id="test",
    )
    assert row["outcome"] == "correct"
    assert row["num_calls"] == condition.minimum_calls
    assert row["input_tokens"] > 0
    assert any(
        message.get("reasoning_content")
        == "I should call the next required function."
        for message in adapter.histories[-1]
    )


def test_tool_gating_preserves_execution_on_scripted_call_order() -> None:
    condition = prepare_condition(
        depth=1,
        distractor_count=5,
        graph_seed=123,
        core_nodes=2,
        trial_seed=456,
        use_types=False,
        paired_core_graph=True,
        stable_tool_order=True,
    )
    initially_available = _gated_tools(
        condition,
        condition.known,
        use_types=False,
    )
    assert 0 < len(initially_available) < len(condition.tools)
    target_node = condition.function_tree.desired_target_node
    required = set(nx.ancestors(condition.function_tree.graph, target_node))
    required.add(target_node)
    adapter = ScriptedToolAdapter(condition)
    adapter.order = list(nx.topological_sort(condition.function_tree.graph.subgraph(required)))
    row = run_trial(
        condition,
        adapter,
        max_tokens=128,
        temperature=1.0,
        top_p=0.95,
        use_types=False,
        trial_id="test",
        tool_gating=True,
    )
    assert row["outcome"] == "correct"
    assert row["tool_gating"]
    assert row["available_tool_counts"]


def test_dependency_trace_metrics_from_saved_calls() -> None:
    condition = prepare_condition(
        depth=1,
        distractor_count=0,
        graph_seed=123,
        core_nodes=2,
        trial_seed=456,
        use_types=False,
    )
    order = list(nx.topological_sort(condition.function_tree.graph))
    trial = {
        "outcome": "correct",
        "correct": True,
        "calls": [
            {
                "function": name,
                "arguments": "{}",
                "failure": None,
            }
            for name in order
        ],
        "failures": [],
    }
    classification = classify_trial(trial, condition)
    assert classification["primary_category"] == "correct"
    assert classification["required_recall"] == 1.0
    assert classification["selection_precision"] == 1.0
    assert classification["dependency_order_score"] == 1.0
    assert classification["dependency_trace_fidelity"] == 1.0


def test_wrong_argument_is_reflected_in_trace_metrics() -> None:
    condition = prepare_condition(
        depth=1,
        distractor_count=0,
        graph_seed=123,
        core_nodes=2,
        trial_seed=456,
        use_types=False,
    )
    name = next(iter(condition.function_tree.graph.nodes))
    trial = {
        "outcome": "model_error",
        "correct": False,
        "calls": [{"function": name, "arguments": "{}", "failure": None}],
        "failures": ["Value of variable was incorrect"],
    }
    classification = classify_trial(trial, condition)
    assert classification["primary_category"] == "wrong_argument_value"
    assert classification["invalid_call_count"] == 1
    assert classification["argument_score"] == 0.0


def test_paired_statistics_helpers() -> None:
    low, high = _wilson_interval(9, 15)
    assert 0 < low < high < 1
    assert _mcnemar_exact_p(6, 0) == pytest.approx(0.03125)
