# Dataset and benchmark documentation

**Working title:** Strict-Paired FuncBenchGen Tool-Usability Set  
**Version:** 1.0.0  
**Release date:** 2026-11-22  
**Maintainer:** Nianyi Wang (`livingsailor@gmail.com`)  
**License:** CC BY 4.0 for the sanitized data and derived tables; MIT for the
project code; BSD-3-Clause for the vendored FuncBenchGen source  
**Artifact URL:** https://github.com/nywang2019/a3d-tool-audit

## Motivation

The artifact supports controlled comparisons of multi-step tool-use behavior
when the required function-dependency task is held constant but the tool
usability of added tools changes. It is intended for diagnostic evaluation of
function selection, dependency use, argument construction, and termination.

It is not intended as a general leaderboard, a safety certification, or a direct
test of training-data contamination.

## Provenance

- Upstream generator: FuncBenchGen.
- Locked commit:
  `0729e2567dfaee1e53262ba9bfd5ba6a4b4e4343`.
- Upstream license: BSD-3-Clause.
- Generation dates: 2026-10-29 and 2026-10-30.
- Generator patches: deterministic sorting of set-derived structure and
  required-graph fingerprint auditing, implemented in the local project.
- Legacy configuration and field names retain historical terms such as
  `distractor`, `connected_compatible`, and `affordance` for reproducibility;
  the reader-facing documentation uses the standardized terminology below.
- Source configurations:
  `configs/funcbench_paired_main_v2.yaml`,
  `configs/funcbench_paired_extension20.yaml`,
  `configs/funcbench_paired_dose2.yaml`,
  `configs/funcbench_paired_dose2_extension.yaml`,
  `configs/funcbench_control_disconnected.yaml`,
  `configs/funcbench_control_incompatible.yaml`,
  `configs/funcbench_prerequisite_gating.yaml`, and
  `configs/funcbench_paired_intervention.yaml`.

## Composition

The main graph-condition set contains 20 independently generated core graphs at
dependency depth three. Each graph is instantiated with zero and five
executable goal-irrelevant tools. Five model aliases produce 200 end-to-end
attempts.

Additional condition sets:

- 15 shared graphs with 0, 2, and 5 extra tools;
- 15 shared graphs under none, disconnected, connected-incompatible, and
  executable goal-irrelevant conditions;
- eight graph-matched unhidden/hidden pairs;
- four graph-matched baseline/state-reminder pairs.

Baseline cells are reused across some analyses. The run families therefore
contain many evaluation attempts but only 20 independent main graph clusters.

Each graph-condition record should expose:

- graph seed and graph index;
- full graph fingerprint;
- required-graph fingerprint;
- target variable and program-computed truth;
- tool schemas and deterministic core-tool order;
- dependency edges and required-call set;
- condition mode and extra-tool count;
- prompt and intervention flags;
- model alias and configuration snapshot.

Each evaluation attempt should expose:

- sanitized tool-call trace;
- final answer;
- exact success rate label;
- technical/budget-failure label;
- required recall, selection precision, dependency-order score, and argument
  score;
- primary error category;
- token and call counts;
- source run and trial identifiers.

## Intended uses

- Reproduce the reported paired effects.
- Develop graph- or trace-aware evaluation methods.
- Compare interventions on graph-matched instances.
- Audit how model failures change when executable goal-irrelevant tools are
  added.
- Construct additional paired suites using the same deterministic generator.

## Out-of-scope uses

- ranking models outside the studied model/configuration panel;
- inferring national or provider-wide model quality;
- claiming that generated instances are absent from model training data;
- using benchmark success as evidence of production safety;
- treating the trace-fidelity composite as a substitute for exact correctness.

## Sensitive content and privacy

The task data are synthetic function schemas and scalar values. They contain no
human-subject data, personal information, or user logs. Provider credentials must
never be included. Account identifiers and request metadata should be removed
from released traces.

## Collection and annotation

Graphs and truths are program-generated. Error categories and trace scores are
computed by deterministic analysis code. Independent human annotation and
inter-rater agreement are not included in the first submission; released
documentation must describe the categories as rule-derived diagnostics.

## Distribution plan

Release requirements:

1. retain the upstream FuncBenchGen BSD-3-Clause notice;
2. publish the versioned graph-condition bundle without credentials;
3. publish the schema and checksum manifest.
4. publish the sanitized traces or a regeneration script if raw traces cannot be
   redistributed;
5. provide a stable URL and archival DOI;
6. name a maintenance owner and retention policy;
7. validate all table-generation scripts against the released version.

## Known limitations

- Single task family.
- Fixed dependency depth three in the main study.
- Twenty main independent graphs and 15 tool-usability control graphs.
- One temperature-1 sample per main cell.
- Five model aliases in every main and auxiliary analysis panel.
- No completed double annotation.
- Public release contains sanitized trial records and excludes raw provider
  responses and credentials.

