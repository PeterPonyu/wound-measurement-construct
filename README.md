# Neural topic measurements in diabetic foot ulcer research: analysis software

Author: Zeyu Fu. Software version: 0.4.2.

Analysis software for a patient-level audit of fibroblast measurements from a frozen 15-topic logistic-normal expression model in diabetic foot ulcer research. It evaluates score decomposition, whole-count-pipeline sensitivity, author-mapped patient contrasts, paired anatomy and external tissue-context projections. Healing inference involves 11 patients; outcome intervals include zero and the external cohorts do not supply independent healing validation. This is a measurement audit, not a validated prognostic tool.

The discovery analysis contains 25 specimens mapped to 20 patients. Healing inference uses 14 specimens from 11 patients (7 healed and 4 not healed). Bootstrap draws and cells are not additional independent patients.

[Source code](https://github.com/PeterPonyu/wound-measurement-construct) · [Versioned downloads](https://github.com/PeterPonyu/wound-measurement-construct/releases/tag/v0.4.2) · [Software DOI](https://doi.org/10.5281/zenodo.22962314)

Read the [manuscript](output/pdf/measurement_construct_validation.pdf) and the [figure collection](manuscripts/figures/figures.pdf). The package contains 9 editable R/TikZ vector figures, numerical tables, completed reports, and its own copy of the frozen expression model. RESULTS_INDEX.json maps numerical reports to manuscript use. A copy of the model is included here so the project runs independently.

![Study design](manuscripts/figures/figure1_workflow.png)

## Rebuild from saved numerical inputs

First download and extract the complete reproduction ZIP from the versioned Releases page. The commands below run from that extracted root, not the lightweight Git checkout.

```sh
python3 verify_archive.py --smoke
python3 manuscripts/latex/export_tables.py
python3 manuscripts/build_main_figures.py
python3 manuscripts/latex/build.py --render
python3 -m pytest -p no:cacheprovider tests -q
```

The figure and PDF commands require R with ggplot2, patchwork, tikzDevice, jsonlite and digest; XeLaTeX/BibTeX/latexmk; Poppler; and installed Arial and TeX Gyre fonts. Fonts are not redistributed. Python package versions are recorded in requirements.txt and environment.json. CPU CLI smoke checks and saved-input rendering do not train models. Some optional analysis commands below fit models; they are not part of the release verification workflow.

## Rights

The MIT License applies to software, including analysis and rendering programs. Manuscripts, figures, tables, saved models, numerical results and source-study data retain their respective rights as set out in NOTICE. The full reproduction ZIP and the software-only Zenodo ZIP are different artifacts with separate manifests and checksums.

## Citation and software archiving

Use CITATION.cff for software attribution: [10.5281/zenodo.22962314](https://doi.org/10.5281/zenodo.22962314). This software DOI does not identify the manuscript. ARCHIVING.md describes artifact scopes and checksum verification.

## Data and research-material availability

Public processed count matrices and author annotations are available from these source records:

- [GSE165816](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE165816)
- [GSE231643](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE231643)
- [GSE241132](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE241132)
- [GSE223964](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE223964)
- [GSE245703](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE245703)
- [GSE268834](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE268834)
- [GSE248247](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE248247)

REPRODUCIBILITY.md describes downloads, input locations and analysis order. Public study materials require no request to the author. Source-study permissions remain in force: processed public matrices are not the same as unrestricted human sequencing reads. The complete reproduction package includes derived numerical reports, frozen weights, editable figures and the manuscript; these materials are openly downloadable from this repository's versioned Releases page without author approval or an access request. They retain the rights stated in NOTICE; public availability does not relicense source-study data.

## Mathematical verification

Version 0.4.2 corrects method descriptions and adds explicit estimands, analytic unit tests and a frozen-input sensitivity report. Read METHODS_CONTRACT.md and CHANGELOG.md for the interpretation and provenance boundaries. Recompute the added analysis with `python3 scripts/audit_measurement_math.py --output-dir outputs/reruns/mathematical_audit`. This uses saved inputs and does not refit the original representation or temporal fields.

## Observed-cell and experimental extension

Version 0.4.2 adds observed-cell maps, raw-count expression context and explicitly bounded experiments. Three patient-level contrasts retain the appropriate cell denominators and enumerate all 330 outcome allocations. The display reconstructs the original raw rows and lineage labels; UMAP is descriptive. The frozen original reports remain unchanged.

## Complete reproduction download

Download [wound-measurement-construct-0.4.2.zip](https://github.com/PeterPonyu/wound-measurement-construct/releases/tag/v0.4.2) for the exact manuscript, figures, frozen models, completed numerical reports and checksum manifest. The Git source checkout excludes large research-output directories; extract the complete ZIP before running saved-input figure/PDF rebuilds. The software-only ZIP is sufficient for code inspection and CPU import/unit tests, but is not a pretrained-result bundle.
