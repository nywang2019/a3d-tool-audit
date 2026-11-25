# Applied Intelligence supplementary material

This package accompanies:

> **The Wrong Tool Looks Right: Auditing Executable Goal-Irrelevant Tools
> in Multi-Step Tool Use**

**A3D:** Affordance-Aware Audit and Diagnosis.

- **Journal:** Applied Intelligence
- **Authors:** Nianyi Wang, Tiejun Wang, Weilan Wang
- **Affiliations:** School of Mathematics and Computer Science; Key Laboratory
  of Linguistic and Cultural Computing of Ministry of Education; Gansu
  Provincial Engineering Research Center of Multi-modal Artificial
  Intelligence, Northwest Minzu University, Lanzhou, China
- **Corresponding author:** Nianyi Wang, livingsailor@gmail.com
- **Supplementary item:** Online Resource 1

## Contents

- `dataset_card.md`: task family, graph construction, intended use, and limits.
- `control15_results.md`: protocol and results for the expanded tool-usability
  controls on 15 shared graphs.
- `minimax_main20_results.md`: the fifth-model smoke test, formal 20-graph
  comparison, retry accounting, and paired statistics.
- `diagnostic_protocol.md`: rule-derived trace metrics and failure categories.
- `derived_analysis.md`: generated numerical summary for the manuscript.
- `asp_method.md`: definition, inference, and limits of the ASP paired loss.
- `sealed_probe.md`: schema-matched executable-versus-sealed design and EAP
  results.
- `repeatability_summary.md` and `repeatability_summary.json`: fresh-sample
  agreement audit.
- `tables/`: machine-readable, Markdown, and LaTeX versions of the paper
  tables, including the detailed tables moved out of the main manuscript.
- `figures/figure2_dose_response.pdf`: the 0/2/5 response curve referenced in
  the stability results.
- `case_studies/`: saved examples of diagnostic reports.
- `generated_facts.json`: compact values used to build the manuscript assets.
- `model_coverage_audit.md`: explicit audit that every analysis panel contains
  all five models.
- `reproducibility.md`: commands and scope of the local artifact build.

## Important limits

The first submission does not include independent human review of the
rule-derived error labels. The manuscript therefore treats the labels as
diagnostic evidence rather than validated ground truth. The study does not
claim a general model ranking, a stable complexity threshold, causal mediation,
or immunity to training-data contamination.

Provider API behavior may drift after the recorded run date. Raw provider
responses and credentials are not included in this package.
