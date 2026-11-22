"""Low-cost API-only calibration runner built on FuncBenchGen primitives."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from .adapters import BudgetExceededError, OpenAICompatibleAdapter
from .io_utils import atomic_write_json, write_csv


def _load_vendor_class(module_name: str, filename: str, class_name: str):
    source = Path(__file__).resolve().parents[2] / "vendor" / "FuncBenchGen" / "src"
    spec = importlib.util.spec_from_file_location(module_name, source / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load FuncBenchGen module: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


FunctionDependencyTree = _load_vendor_class(
    "_funcbench_function_tree",
    "function_tree.py",
    "FunctionDependencyTree",
)
ToolCallingEvaluator = _load_vendor_class(
    "_funcbench_evaluator",
    "evaluator.py",
    "ToolCallingEvaluator",
)


class DeterministicFunctionDependencyTree(FunctionDependencyTree):
    """Sort set-derived node order before random choices for reproducibility."""

    def _build_core_graph(
        self,
        min_calls: int,
        max_critical_path_length: int,
    ) -> str | None:
        if min_calls == 0:
            return None
        first_node = self._create_node(
            inputs=[self._generate_random_name()],
            prefix="func",
            node_type="core",
        )
        last_node_in_chain = first_node
        guaranteed_core_nodes = {first_node}
        if max_critical_path_length > 0:
            previous_output = self.graph.nodes[first_node]["function"].output
            for _ in range(max_critical_path_length):
                node_name = self._create_node(
                    inputs=[previous_output],
                    prefix="func",
                    node_type="core",
                )
                self.graph.add_edge(last_node_in_chain, node_name)
                guaranteed_core_nodes.add(node_name)
                last_node_in_chain = node_name
                previous_output = self.graph.nodes[node_name]["function"].output
        target_node = last_node_in_chain
        confirmed_side_nodes: set[str] = set()
        num_side_nodes_required = max(
            min_calls - len(guaranteed_core_nodes),
            0,
        )
        candidate_pool = [
            self._create_node(
                inputs=[self._generate_random_name()],
                prefix="func",
                node_type="core",
            )
            for _ in range(num_side_nodes_required)
        ]
        while len(confirmed_side_nodes) < num_side_nodes_required:
            if not candidate_pool:
                candidate = self._create_node(
                    inputs=[self._generate_random_name()],
                    prefix="func",
                    node_type="core",
                )
            else:
                candidate = candidate_pool.pop(0)
            possible_consumers = sorted(guaranteed_core_nodes | confirmed_side_nodes)
            random.shuffle(possible_consumers)
            has_connected = False
            for consumer in possible_consumers:
                if self._add_constrained_edge(
                    candidate,
                    consumer,
                    target_node,
                    max_critical_path_length,
                ):
                    has_connected = True
                    break
            if has_connected:
                confirmed_side_nodes.add(candidate)
            else:
                function = self.graph.nodes[candidate]["function"]
                self.graph.remove_node(candidate)
                if function.name in self.used_names:
                    self.used_names.remove(function.name)
                if function.output in self.used_names:
                    self.used_names.remove(function.output)
        all_core_nodes = sorted(guaranteed_core_nodes | confirmed_side_nodes)
        for _ in range(int(len(all_core_nodes) * 2.0)):
            producer = random.choice(all_core_nodes)
            consumer = random.choice(all_core_nodes)
            self._add_constrained_edge(
                producer,
                consumer,
                target_node,
                max_critical_path_length,
            )
        return target_node


@dataclass(frozen=True)
class Condition:
    """One depth/distractor combination with one deterministic hidden graph."""

    depth: int
    distractor_count: int
    distractor_mode: str
    unavailable_distractor_types: frozenset[str]
    sealed_functions: frozenset[str]
    graph_seed: int
    core_nodes: int
    function_tree: Any
    tools: list[dict[str, Any]]
    prompt: str
    input_vars: list[str]
    known: set[str]
    variable_values: dict[str, int]
    variable_to_type: dict[str, str]
    target_variable: str
    target_value: int
    minimum_calls: int
    max_calls: int
    condition_id: str
    repeat_known_variable_values: bool
    graph_fingerprint: str
    required_graph_fingerprint: str


def _graph_fingerprint(tree: Any) -> str:
    payload = {
        "nodes": sorted(
            [
                {
                    "name": name,
                    "inputs": function.inputs,
                    "output": function.output,
                    "description": function.description,
                    "node_type": data.get("node_type"),
                }
                for name, data in tree.graph.nodes(data=True)
                if (function := data.get("function")) is not None
            ],
            key=lambda item: item["name"],
        ),
        "edges": sorted([list(edge) for edge in tree.graph.edges()]),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _required_graph_fingerprint(tree: Any) -> str:
    """Fingerprint only the target's dependency closure."""
    target_node = getattr(tree, "desired_target_node", None)
    if not target_node:
        return _graph_fingerprint(tree)
    required = set(nx.ancestors(tree.graph, target_node)) | {target_node}
    payload = {
        "nodes": sorted(
            [
                {
                    "name": name,
                    "inputs": function.inputs,
                    "output": function.output,
                    "description": function.description,
                    "node_type": data.get("node_type"),
                }
                for name, data in tree.graph.nodes(data=True)
                if name in required
                and (function := data.get("function")) is not None
            ],
            key=lambda item: item["name"],
        ),
        "edges": sorted(
            [list(edge) for edge in tree.graph.edges() if set(edge) <= required]
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _type_appended_connected_nodes(tree: Any, *, show_variable_names: bool) -> None:
    """Assign typed variables and descriptions to nodes added after core reformat."""
    variable_to_type = tree.variable_to_type_map
    supertypes = {
        value.split(" with ", 1)[0]
        for value in variable_to_type.values()
    } or {tree._generate_random_name("type")}

    def new_type() -> str:
        supertype = random.choice(sorted(supertypes))
        return f"{supertype} with {tree._generate_random_name('subtype')}"

    for node, data in tree.graph.nodes(data=True):
        if data.get("node_type") not in {"extra", "disconnected"}:
            continue
        function = data["function"]
        for variable in function.inputs:
            variable_to_type.setdefault(variable, new_type())
        variable_to_type.setdefault(function.output, new_type())

        if show_variable_names:
            input_descs = [
                f"{variable} ({variable_to_type[variable]})"
                for variable in function.inputs
            ]
            output_desc = (
                f"{function.output} ({variable_to_type[function.output]})"
            )
            if len(function.inputs) == 1:
                function.description = (
                    f"Processes variable {input_descs[0]} "
                    f"to produce variable {output_desc}"
                )
            else:
                function.description = (
                    f"Processes variables {', '.join(input_descs)} "
                    f"to produce variable {output_desc}"
                )
        else:
            input_descs = [
                f"({variable_to_type[variable]})"
                for variable in function.inputs
            ]
            output_desc = f"({variable_to_type[function.output]})"
            if len(function.inputs) == 1:
                function.description = (
                    f"Processes inputs of types {input_descs[0]} "
                    f"to produce {output_desc}"
                )
            else:
                function.description = (
                    "Processes inputs of types "
                    f"{', '.join(input_descs)} to produce {output_desc}"
                )


def _add_disconnected_distractor_nodes(
    tree: Any,
    count: int,
) -> list[str]:
    """Add visible tools with private inputs and no graph edges."""
    private_inputs = []
    for _ in range(count):
        private_input = tree._generate_random_name("unavailable")
        tree._create_node(
            inputs=[private_input],
            prefix="func",
            node_type="disconnected",
        )
        private_inputs.append(private_input)
    return private_inputs


def _add_incompatible_connected_nodes(
    tree: Any,
    count: int,
    *,
    target_node: str | None,
    depth: int,
) -> list[str]:
    """Add connected tools that also require one unavailable private input."""
    before = set(tree.graph.nodes)
    tree._add_extra_connected_nodes(count, target_node, depth)
    private_inputs = []
    for node in [name for name in tree.graph.nodes if name not in before]:
        private_input = tree._generate_random_name("unavailable")
        tree.graph.nodes[node]["function"].inputs.append(private_input)
        private_inputs.append(private_input)
    return private_inputs


def _exact_number_match(text: str, target: int) -> bool:
    return bool(re.search(rf"(?<!\d){re.escape(str(target))}(?!\d)", text or ""))


def prepare_condition(
    *,
    depth: int,
    distractor_count: int,
    graph_seed: int,
    core_nodes: int,
    trial_seed: int,
    use_types: bool = True,
    use_subtypes: bool = True,
    num_supertypes: int = 5,
    show_variable_names: bool = False,
    rename_variables: bool = False,
    num_noise_inputs: int = 0,
    repeat_known_variable_values: bool = False,
    condition_id: str | None = None,
    paired_core_graph: bool = False,
    stable_tool_order: bool = False,
    distractor_mode: str = "connected_compatible",
) -> Condition:
    """Generate one deterministic graph, typed tool schemas, prompt, and truth."""
    random.seed(graph_seed)
    np.random.seed(graph_seed % (2**32 - 1))
    minimum_calls = max(core_nodes, depth + 1)
    valid_distractor_modes = {
        "connected_compatible",
        "disconnected",
        "connected_incompatible",
        "sealed_compatible",
    }
    if distractor_mode not in valid_distractor_modes:
        raise ValueError(f"Unknown distractor mode: {distractor_mode}")
    if not paired_core_graph and distractor_mode != "connected_compatible":
        raise ValueError("Distractor controls require paired_core_graph=True")
    tree_class = (
        DeterministicFunctionDependencyTree
        if paired_core_graph
        else FunctionDependencyTree
    )
    tree = tree_class(seed=graph_seed)
    # Legacy behavior couples distractor count to graph construction. The strict
    # paired path builds the same core graph first and appends distractors later.
    total_nodes = core_nodes if paired_core_graph else core_nodes + distractor_count
    tree.build_graph_with_constraints(
        num_total_nodes=total_nodes,
        min_calls=minimum_calls,
        max_critical_path_length=depth,
        num_disconnected_nodes=0,
    )
    if paired_core_graph and use_types:
        # Reformat before appending extras so the core names and subtypes are
        # byte-for-byte identical across distractor counts.
        tree.reformat_tree_with_shared_types(
            show_variable_names_in_description=show_variable_names,
            rename_variables=rename_variables,
            use_subtypes=use_subtypes,
            num_supertypes=num_supertypes,
        )
    unavailable_variables: list[str] = []
    if paired_core_graph and distractor_count > 0:
        target_node = getattr(tree, "desired_target_node", None)
        if distractor_mode in {"connected_compatible", "sealed_compatible"}:
            tree._add_extra_connected_nodes(distractor_count, target_node, depth)
        elif distractor_mode == "disconnected":
            unavailable_variables = _add_disconnected_distractor_nodes(
                tree,
                distractor_count,
            )
        else:
            unavailable_variables = _add_incompatible_connected_nodes(
                tree,
                distractor_count,
                target_node=target_node,
                depth=depth,
            )
        if use_types:
            _type_appended_connected_nodes(
                tree,
                show_variable_names=show_variable_names,
            )
        else:
            tree._populate_function_descriptions()
        tree.total_nodes = len(tree.graph.nodes)
    elif use_types and not paired_core_graph:
        tree.reformat_tree_with_shared_types(
            show_variable_names_in_description=show_variable_names,
            rename_variables=rename_variables,
            use_subtypes=use_subtypes,
            num_supertypes=num_supertypes,
        )

    evaluator = ToolCallingEvaluator(
        tree,
        num_noise_inputs=num_noise_inputs,
        function_shuffle_seed=graph_seed,
        error_detail_level="None",
        on_wrong_inputs="Execute",
        repeat_known_variable_values=repeat_known_variable_values,
    )
    random.seed(trial_seed)
    np.random.seed(trial_seed % (2**32 - 1))
    if use_types:
        variable_to_type = dict(tree.variable_to_type_map)
        values_by_type: dict[str, int] = {}
        variable_values: dict[str, int] = {}
        for variable, variable_type in variable_to_type.items():
            if variable_type not in values_by_type:
                values_by_type[variable_type] = random.randrange(1000)
            variable_values[variable] = values_by_type[variable_type]
        type_to_vars: dict[str, list[str]] = {}
        for variable, variable_type in variable_to_type.items():
            type_to_vars.setdefault(variable_type, []).append(variable)
        input_vars, known = evaluator._get_input_variables_with_types(
            variable_to_type,
            type_to_vars,
        )
    else:
        variable_to_type = {}
        variable_values = {}
        for node_name in tree.graph.nodes:
            function = tree.graph.nodes[node_name]["function"]
            for variable in function.inputs + [function.output]:
                if variable not in variable_values:
                    variable_values[variable] = random.randrange(1000)
        input_vars = tree.get_input_vars()
        known = set(input_vars)
    evaluator.variable_values = variable_values
    target_node = getattr(tree, "desired_target_node", None)
    if target_node:
        target_variable = str(tree.graph.nodes[target_node]["function"].output)
    else:
        target_variable = str(tree.desired_output_variable)
    prompt = evaluator._get_input_prompt(target_variable, input_vars)
    displayed_input_vars = list(evaluator.revealed_vars)
    if use_types:
        unavailable_types = {
            variable_to_type[variable]
            for variable in unavailable_variables
            if variable in variable_to_type
        }
        known = {
            variable_to_type[variable]
            for variable in displayed_input_vars
            if variable in variable_to_type
        } - unavailable_types
    else:
        unavailable_types = frozenset(unavailable_variables)
        known = set(displayed_input_vars) - unavailable_types
    tools = evaluator._get_function_schemas()
    if stable_tool_order:
        # Sorting preserves the relative order of core tools when distractors are
        # inserted, preventing array-order changes from confounding the contrast.
        tools.sort(key=lambda tool: tool["function"]["name"])
    sealed_functions = frozenset(
        str(name)
        for name, data in tree.graph.nodes(data=True)
        if distractor_mode == "sealed_compatible"
        and data.get("node_type") == "extra"
    )
    return Condition(
        depth=depth,
        distractor_count=distractor_count,
        distractor_mode=(
            distractor_mode if distractor_count else "none"
        ),
        unavailable_distractor_types=frozenset(unavailable_types),
        sealed_functions=sealed_functions,
        graph_seed=graph_seed,
        core_nodes=core_nodes,
        function_tree=tree,
        tools=tools,
        prompt=prompt,
        input_vars=displayed_input_vars,
        known=set(known),
        variable_values=variable_values,
        variable_to_type=variable_to_type,
        target_variable=target_variable,
        target_value=variable_values[target_variable],
        minimum_calls=minimum_calls,
        max_calls=minimum_calls * 2,
        condition_id=condition_id or f"d{depth}-x{distractor_count}",
        repeat_known_variable_values=repeat_known_variable_values,
        graph_fingerprint=_graph_fingerprint(tree),
        required_graph_fingerprint=_required_graph_fingerprint(tree),
    )


def _execute_tool_call(
    evaluator: Any,
    call: dict[str, Any],
    known: set[str],
    *,
    use_types: bool,
    variable_to_type: dict[str, str],
) -> tuple[str, str | None]:
    function = call["function"]
    tool_call = SimpleNamespace(
        id=call["id"],
        function=SimpleNamespace(
            name=function["name"],
            arguments=function["arguments"],
        ),
    )
    try:
        return evaluator._execute_tool(
            tool_call,
            known,
            use_types=use_types,
            var_to_type=variable_to_type,
        )
    except Exception as exc:  # noqa: BLE001 - third-party evaluator boundary
        return "Error occurred.", f"{type(exc).__name__}: {exc}"


def _gated_tools(
    condition: Condition,
    known: set[str],
    *,
    use_types: bool,
) -> list[dict[str, Any]]:
    """Expose only tools whose input variables are already available."""
    available = []
    for tool in condition.tools:
        function_name = str(tool["function"]["name"])
        function = condition.function_tree.graph.nodes[function_name]["function"]
        if use_types:
            requirements = {
                condition.variable_to_type[variable]
                for variable in function.inputs
                if variable in condition.variable_to_type
            }
        else:
            requirements = set(function.inputs)
        if requirements <= known:
            available.append(tool)
    return available


def run_trial(
    condition: Condition,
    adapter: OpenAICompatibleAdapter,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    use_types: bool = True,
    trial_id: str,
    tool_gating: bool = False,
) -> dict[str, Any]:
    """Run one model trial through repeated tool calls and score exact success."""
    tree = copy.deepcopy(condition.function_tree)
    evaluator = ToolCallingEvaluator(
        tree,
        num_noise_inputs=0,
        function_shuffle_seed=condition.graph_seed,
        error_detail_level="None",
        on_wrong_inputs="Execute",
        repeat_known_variable_values=condition.repeat_known_variable_values,
    )
    evaluator.variable_values = dict(condition.variable_values)
    evaluator.revealed_vars = {
        variable: condition.variable_values[variable]
        for variable in condition.input_vars
    }
    known = set(condition.known)
    messages: list[dict[str, Any]] = [{"role": "user", "content": condition.prompt}]
    calls: list[dict[str, Any]] = []
    failures: list[str] = []
    final_content = ""
    technical_failure = False
    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    available_tool_counts: list[int] = []

    for step in range(1, condition.max_calls + 1):
        tools = (
            _gated_tools(condition, known, use_types=use_types)
            if tool_gating
            else condition.tools
        )
        available_tool_counts.append(len(tools))
        try:
            response = adapter.chat(
                messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                sequence=step,
            )
        except BudgetExceededError:
            technical_failure = True
            failures.append("budget")
            break
        except Exception as exc:  # noqa: BLE001 - each failed trial must be logged
            technical_failure = True
            failures.append(f"{type(exc).__name__}: {exc}")
            break

        usage = response["usage"]
        input_tokens += int(usage.get("prompt_tokens", 0) or 0)
        output_tokens += int(usage.get("completion_tokens", 0) or 0)
        detail = usage.get("completion_tokens_details")
        if isinstance(detail, dict):
            reasoning_tokens += int(detail.get("reasoning_tokens", 0) or 0)

        tool_calls = response["tool_calls"]
        if not tool_calls:
            final_content = response["content"]
            break

        assistant_message = {
            "role": "assistant",
            "content": response["message"].get("content"),
            "tool_calls": tool_calls,
        }
        reasoning_content = response["message"].get("reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content:
            assistant_message["reasoning_content"] = reasoning_content
        messages.append(assistant_message)
        for call in tool_calls:
            function_name = call["function"]["name"]
            function_node = tree.graph.nodes.get(function_name, {})
            if function_name in condition.sealed_functions:
                result = "Tool invocation failed: this tool is currently sealed."
                failure = "Sealed tool unavailable"
            else:
                result, failure = _execute_tool_call(
                    evaluator,
                    call,
                    known,
                    use_types=use_types,
                    variable_to_type=condition.variable_to_type,
                )
            if failure:
                failures.append(failure)
            calls.append(
                {
                    "order": len(calls) + 1,
                    "function": function_name,
                    "node_type": function_node.get("node_type", "unknown"),
                    "arguments": call["function"]["arguments"],
                    "result": result,
                    "failure": failure,
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["function"]["name"],
                    "content": json.dumps({"result": result}, ensure_ascii=False),
                }
            )
        if len(calls) >= condition.max_calls:
            failures.append("max_calls")
            break

    correct = (
        not technical_failure
        and bool(final_content)
        and len(calls) >= condition.minimum_calls
        and _exact_number_match(final_content, condition.target_value)
    )
    if technical_failure:
        outcome = "technical_failure"
    elif len(calls) < condition.minimum_calls:
        outcome = "shortcut"
    elif _exact_number_match(final_content, condition.target_value):
        outcome = "correct"
    else:
        outcome = "model_error"
    return {
        "trial_id": trial_id,
        "depth": condition.depth,
        "distractor_count": condition.distractor_count,
        "distractor_mode": condition.distractor_mode,
        "graph_seed": condition.graph_seed,
        "graph_fingerprint": condition.graph_fingerprint,
        "required_graph_fingerprint": condition.required_graph_fingerprint,
        "repeat_known_variable_values": condition.repeat_known_variable_values,
        "target_variable": condition.target_variable,
        "target_value": condition.target_value,
        "prompt": condition.prompt,
        "final_content": final_content,
        "num_calls": len(calls),
        "minimum_calls": condition.minimum_calls,
        "max_calls": condition.max_calls,
        "correct": correct,
        "outcome": outcome,
        "failures": failures,
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "tool_gating": tool_gating,
        "available_tool_counts": available_tool_counts,
    }


def _load_trial_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"Trial log must contain a list: {path}")
    return [row for row in payload if isinstance(row, dict)]


def _validate_resume_trials(
    rows: list[dict[str, Any]],
    conditions: list[Condition],
) -> None:
    expected = {condition.condition_id: condition for condition in conditions}
    fields = {
        "depth": "depth",
        "distractor_count": "distractor_count",
        "distractor_mode": "distractor_mode",
        "graph_seed": "graph_seed",
        "graph_fingerprint": "graph_fingerprint",
        "target_variable": "target_variable",
        "target_value": "target_value",
        "prompt": "prompt",
        "minimum_calls": "minimum_calls",
        "max_calls": "max_calls",
        "repeat_known_variable_values": "repeat_known_variable_values",
    }
    for row in rows:
        trial_id = str(row.get("trial_id"))
        condition = expected.get(trial_id)
        if condition is None:
            raise ValueError(f"Existing trial is not part of this configuration: {trial_id}")
        for row_field, condition_field in fields.items():
            if row_field in row:
                expected_value = getattr(condition, condition_field)
                if row[row_field] != expected_value:
                    raise ValueError(
                        "Resume task mismatch for "
                        f"{trial_id}: {row_field}={row[row_field]!r}, "
                        f"expected {expected_value!r}"
                    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generation_identity() -> dict[str, Any]:
    vendor_source = Path(__file__).resolve().parents[2] / "vendor" / "FuncBenchGen" / "src"
    return {
        "version": "paired-core-v3-controls",
        "funcbench_module_sha256": _sha256_file(Path(__file__)),
        "vendor_function_tree_sha256": _sha256_file(vendor_source / "function_tree.py"),
        "vendor_evaluator_sha256": _sha256_file(vendor_source / "evaluator.py"),
    }


def _summarize_model_rows(
    *,
    alias: str,
    rows: list[dict[str, Any]],
    raw_log_dir: Path,
    adapter_statistics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "model_alias": alias,
        "trials": len(rows),
        "correct": sum(bool(row.get("correct")) for row in rows),
        "technical_failures": sum(
            row.get("outcome") == "technical_failure" for row in rows
        ),
        "model_errors": sum(row.get("outcome") == "model_error" for row in rows),
        "shortcuts": sum(row.get("outcome") == "shortcut" for row in rows),
        "input_tokens": sum(int(row.get("input_tokens", 0)) for row in rows),
        "output_tokens": sum(int(row.get("output_tokens", 0)) for row in rows),
        "reasoning_tokens": sum(
            int(row.get("reasoning_tokens", 0)) for row in rows
        ),
        "raw_log_files": len(list(raw_log_dir.glob("*.json")))
        if raw_log_dir.exists()
        else 0,
        "last_session_adapter_statistics": adapter_statistics or {},
    }


def run_funcbench_calibration(
    config: dict[str, Any],
    *,
    results_dir: Path | str = "results",
    model_aliases: set[str] | None = None,
    resume_from: Path | str | None = None,
) -> dict[str, Any]:
    """Run a small condition grid sequentially with hard per-model budgets."""
    experiment = config["experiment"]
    models = [
        model
        for model in config.get("models", [])
        if model.get("enabled")
        and (not model_aliases or model["alias"] in model_aliases)
    ]
    if not models:
        raise ValueError("No enabled models matched the calibration request")
    generation_identity = _generation_identity()
    if resume_from is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_root = Path(results_dir) / f"{experiment['name']}-{stamp}"
        run_root.mkdir(parents=True, exist_ok=False)
        atomic_write_json(run_root / "config.snapshot.json", config)
        atomic_write_json(
            run_root / "generation_identity.json",
            generation_identity,
        )
    else:
        run_root = Path(resume_from)
        snapshot_path = run_root / "config.snapshot.json"
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Missing configuration snapshot: {snapshot_path}")
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if snapshot != config:
            raise ValueError("Resume configuration does not match the run snapshot")
        identity_path = run_root / "generation_identity.json"
        if identity_path.exists():
            existing_identity = json.loads(identity_path.read_text(encoding="utf-8"))
            if existing_identity != generation_identity:
                raise ValueError("Resume generation code does not match the run snapshot")

    conditions: list[Condition] = []
    condition_index = 0
    configured_indices = experiment.get("graph_indices")
    include_conditions = set(experiment.get("include_conditions", []))
    graph_indices = (
        [int(value) for value in configured_indices]
        if configured_indices is not None
        else list(range(int(experiment.get("graphs_per_condition", 1))))
    )
    reminder_modes = experiment.get("repeat_known_variable_values_modes")
    paired_core_graph = bool(experiment.get("paired_core_graph", False))
    stable_tool_order = bool(experiment.get("stable_tool_order", paired_core_graph))
    distractor_mode = str(
        experiment.get("distractor_mode", "connected_compatible")
    )
    tool_gating = bool(experiment.get("tool_gating", False))
    repeat_modes = (
        [bool(value) for value in reminder_modes]
        if reminder_modes is not None
        else [bool(experiment.get("repeat_known_variable_values", False))]
    )
    fingerprints: dict[tuple[int, int, int], str] = {}
    for depth in experiment["depths"]:
        for distractors in experiment["distractor_counts"]:
            condition_index += 1
            base_condition_id = f"d{depth}-x{distractors}"
            if include_conditions and base_condition_id not in include_conditions:
                continue
            for graph_index in graph_indices:
                if paired_core_graph:
                    graph_seed = (
                        int(experiment["seed"]) + int(depth) * 1000 + graph_index
                    )
                    trial_seed = (
                        int(experiment["seed"]) + int(depth) * 10000 + graph_index
                    )
                else:
                    graph_seed = (
                        int(experiment["seed"]) + condition_index * 1000 + graph_index
                    )
                    trial_seed = (
                        int(experiment["seed"]) + condition_index * 10000 + graph_index
                    )
                for repeat_known_variable_values in repeat_modes:
                    condition = prepare_condition(
                        depth=int(depth),
                        distractor_count=int(distractors),
                        graph_seed=graph_seed,
                        core_nodes=int(experiment["core_nodes"]),
                        trial_seed=trial_seed,
                        use_types=bool(experiment.get("use_types", True)),
                        use_subtypes=bool(experiment.get("use_subtypes", True)),
                        num_supertypes=int(experiment.get("num_supertypes", 5)),
                        show_variable_names=bool(
                            experiment.get("show_variable_names", False)
                        ),
                        rename_variables=bool(
                            experiment.get("rename_variables", False)
                        ),
                        num_noise_inputs=int(experiment.get("num_noise_inputs", 0)),
                        repeat_known_variable_values=repeat_known_variable_values,
                        condition_id=(
                            f"d{depth}-x{distractors}-g{graph_index}"
                            f"-r{int(repeat_known_variable_values)}"
                        ),
                        paired_core_graph=paired_core_graph,
                        stable_tool_order=stable_tool_order,
                        distractor_mode=distractor_mode,
                    )
                    fingerprint_key = (
                        (int(depth), graph_index)
                        if paired_core_graph
                        else (int(depth), int(distractors), graph_index)
                    )
                    fingerprint = (
                        condition.required_graph_fingerprint
                        if paired_core_graph
                        else condition.graph_fingerprint
                    )
                    previous = fingerprints.setdefault(fingerprint_key, fingerprint)
                    if previous != fingerprint:
                        raise RuntimeError(
                            "Core graph fingerprint changed across paired conditions: "
                            f"{fingerprint_key}"
                        )
                    conditions.append(condition)

    all_rows: list[dict[str, Any]] = []
    model_summaries: dict[str, Any] = {}
    session_adapter_statistics: dict[str, dict[str, Any]] = {}
    for model_config in models:
        alias = str(model_config["alias"])
        model_dir = run_root / alias
        model_dir.mkdir(parents=True, exist_ok=resume_from is not None)
        trials_path = model_dir / "trials.json"
        existing_rows = _load_trial_rows(trials_path)
        if resume_from is not None:
            _validate_resume_trials(existing_rows, conditions)
        completed_ids = {
            str(row.get("trial_id")) for row in existing_rows if row.get("trial_id")
        }
        raw_log_dir = model_dir / "raw"
        configured_cap = int(
            config.get("budget", {}).get("max_requests_per_model", 0)
        ) or None
        attempts_used = (
            len(list(raw_log_dir.glob("*.json"))) if raw_log_dir.exists() else 0
        )
        remaining_cap = (
            max(0, configured_cap - attempts_used) if configured_cap else None
        )
        missing_conditions = [
            condition
            for condition in conditions
            if condition.condition_id not in completed_ids
        ]
        if missing_conditions and remaining_cap == 0:
            raise RuntimeError(
                f"Request budget exhausted for {alias}: {attempts_used} attempts used"
            )
        adapter: OpenAICompatibleAdapter | None = None
        new_rows: list[dict[str, Any]] = []
        if missing_conditions:
            adapter = OpenAICompatibleAdapter(
                model_id=str(model_config["model"]),
                base_url=str(model_config["base_url"]),
                api_key_env=str(model_config["api_key_env"]),
                timeout_seconds=float(model_config.get("timeout_seconds", 300)),
                max_retries=int(model_config.get("max_retries", 2)),
                temperature=float(experiment["temperature"]),
                max_tokens=int(experiment["max_tokens"]),
                extra_body=dict(model_config.get("extra_body", {})),
                max_requests=remaining_cap,
                raw_log_dir=raw_log_dir,
                trust_env=bool(model_config.get("trust_env", False)),
            )
            for condition in missing_conditions:
                trial_id = condition.condition_id
                try:
                    row = run_trial(
                        condition,
                        adapter,
                        max_tokens=int(experiment["max_tokens"]),
                        temperature=float(experiment["temperature"]),
                        top_p=float(experiment["top_p"]),
                        use_types=bool(experiment.get("use_types", True)),
                        trial_id=trial_id,
                        tool_gating=tool_gating,
                    )
                except BudgetExceededError as exc:
                    row = {
                        "trial_id": trial_id,
                        "depth": condition.depth,
                        "distractor_count": condition.distractor_count,
                        "distractor_mode": condition.distractor_mode,
                        "correct": False,
                        "outcome": "budget_exhausted",
                        "failures": [str(exc)],
                        "num_calls": 0,
                        "tool_gating": tool_gating,
                    }
                except Exception as exc:  # noqa: BLE001 - isolate trial failures
                    row = {
                        "trial_id": trial_id,
                        "depth": condition.depth,
                        "distractor_count": condition.distractor_count,
                        "distractor_mode": condition.distractor_mode,
                        "correct": False,
                        "outcome": "technical_failure",
                        "failures": [f"{type(exc).__name__}: {exc}"],
                        "num_calls": 0,
                        "tool_gating": tool_gating,
                    }
                row["model_alias"] = alias
                new_rows.append(row)
                atomic_write_json(trials_path, existing_rows + new_rows)
                if row.get("outcome") == "budget_exhausted":
                    break
        model_rows_by_id = {
            str(row.get("trial_id")): row
            for row in existing_rows + new_rows
            if row.get("trial_id")
        }
        model_rows = [
            model_rows_by_id[condition.condition_id]
            for condition in conditions
            if condition.condition_id in model_rows_by_id
        ]
        atomic_write_json(trials_path, model_rows)
        summary = _summarize_model_rows(
            alias=alias,
            rows=model_rows,
            raw_log_dir=raw_log_dir,
            adapter_statistics=adapter.get_statistics() if adapter else {},
        )
        model_summaries[alias] = summary
        if adapter is not None:
            session_adapter_statistics[alias] = adapter.get_statistics()
        all_rows.extend(model_rows)
        atomic_write_json(model_dir / "summary.json", summary)

    # A filtered resume may touch only one model. Build the aggregate from every
    # completed model directory so the canonical CSV remains complete.
    all_rows = []
    model_summaries = {}
    enabled_aliases = {str(model["alias"]) for model in config.get("models", [])}
    for model_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        if model_dir.name not in enabled_aliases:
            continue
        trials_path = model_dir / "trials.json"
        if not trials_path.exists():
            continue
        rows = _load_trial_rows(trials_path)
        model_rows_by_id = {
            str(row.get("trial_id")): row for row in rows if row.get("trial_id")
        }
        ordered_rows = [
            model_rows_by_id[condition.condition_id]
            for condition in conditions
            if condition.condition_id in model_rows_by_id
        ]
        alias = model_dir.name
        summary = _summarize_model_rows(
            alias=alias,
            rows=ordered_rows,
            raw_log_dir=model_dir / "raw",
            adapter_statistics=session_adapter_statistics.get(alias),
        )
        model_summaries[alias] = summary
        all_rows.extend(ordered_rows)
        atomic_write_json(model_dir / "summary.json", summary)

    rows_frame = pd.DataFrame(
        [
            {
                key: row.get(key)
                for key in (
                    "model_alias",
                    "trial_id",
                    "depth",
                    "distractor_count",
                    "distractor_mode",
                    "graph_seed",
                    "correct",
                    "outcome",
                    "num_calls",
                    "minimum_calls",
                    "input_tokens",
                    "output_tokens",
                    "reasoning_tokens",
                    "target_value",
                    "final_content",
                    "graph_fingerprint",
                    "required_graph_fingerprint",
                    "repeat_known_variable_values",
                    "tool_gating",
                )
            }
            for row in all_rows
        ]
    )
    write_csv(rows_frame, run_root / "trials.csv")
    manifest = {
        "run_root": str(run_root),
        "generation_protocol": {
            **generation_identity,
            "paired_core_graph": paired_core_graph,
            "stable_tool_order": stable_tool_order,
        },
        "models": model_summaries,
        "conditions": [
            {
                "trial_id": condition.condition_id,
                "depth": condition.depth,
                "distractor_count": condition.distractor_count,
                "distractor_mode": condition.distractor_mode,
                "target_value": condition.target_value,
                "graph_fingerprint": condition.graph_fingerprint,
                "required_graph_fingerprint": condition.required_graph_fingerprint,
                "graph_seed": condition.graph_seed,
                "target_variable": condition.target_variable,
                "prompt": condition.prompt,
                "repeat_known_variable_values": condition.repeat_known_variable_values,
                "tool_names": [
                    str(tool["function"]["name"]) for tool in condition.tools
                ],
                "num_tools": len(condition.tools),
                "minimum_calls": condition.minimum_calls,
                "max_calls": condition.max_calls,
            }
            for condition in conditions
        ],
    }
    atomic_write_json(run_root / "run_manifest.json", manifest)
    return manifest


def plan_funcbench_calibration(config: dict[str, Any]) -> dict[str, Any]:
    """Estimate the maximum number of model turns without making API calls."""
    experiment = config["experiment"]
    depths = [int(value) for value in experiment["depths"]]
    distractor_counts = [int(value) for value in experiment["distractor_counts"]]
    core_nodes = int(experiment["core_nodes"])
    configured_indices = experiment.get("graph_indices")
    include_conditions = set(experiment.get("include_conditions", []))
    reminder_modes = experiment.get("repeat_known_variable_values_modes")
    repeat_modes = (
        [bool(value) for value in reminder_modes]
        if reminder_modes is not None
        else [bool(experiment.get("repeat_known_variable_values", False))]
    )
    graph_indices = (
        [int(value) for value in configured_indices]
        if configured_indices is not None
        else list(range(int(experiment.get("graphs_per_condition", 1))))
    )
    conditions = [
        {
            "depth": depth,
            "distractor_count": distractors,
            "minimum_calls": max(core_nodes, depth + 1),
            "max_calls": max(core_nodes, depth + 1) * 2,
            "graph_indices": graph_indices,
        }
        for depth in depths
        for distractors in distractor_counts
        if not include_conditions or f"d{depth}-x{distractors}" in include_conditions
    ]
    max_calls_per_model = (
        sum(item["max_calls"] for item in conditions)
        * len(graph_indices)
        * len(repeat_modes)
    )
    enabled_models = [
        model["alias"] for model in config.get("models", []) if model.get("enabled")
    ]
    return {
        "experiment": experiment["name"],
        "models": enabled_models,
        "conditions": conditions,
        "trials_per_model": len(conditions) * len(graph_indices) * len(repeat_modes),
        "max_model_calls_per_model": max_calls_per_model,
        "max_model_calls_total": max_calls_per_model * len(enabled_models),
        "hard_request_cap_per_model": config.get("budget", {}).get(
            "max_requests_per_model"
        ),
    }
