# Neural topic modelling of fibroblast measurements and diabetic foot ulcer healing: analysis software and reproduction archive

## Current manuscript documents

The reviewed main Article, Supplementary Information and editable document sources are available in the [submission-2026-10-12 document release](https://github.com/PeterPonyu/wound-measurement-construct/releases/tag/submission-2026-10-12) and in [submission-2026-10-12/](submission-2026-10-12/README.md).

The software and saved-analysis release remains version 0.5.0, DOI [10.5281/zenodo.23308073](https://doi.org/10.5281/zenodo.23308073). It archives the frozen computational evidence and its earlier document sources. The later editorial revision has a separate GitHub document tag; it adds no model fits or biological samples. Earlier document files remain available for provenance.

**Earlier frozen-release documents: [inline main manuscript](journal/scientific-reports-manuscript-inline.pdf).**
The [submission-layout main manuscript](journal/scientific-reports-manuscript.pdf)
and [Supplementary Information](journal/scientific-reports-supplement.pdf) are also included.

Software/reproduction version: **0.5.0**. DOI **10.5281/zenodo.23308073** was reserved at archive
assembly and is now published; its complete public payload has been downloaded and hash-verified. That identifier identifies
the software/reproduction release, not a manuscript or preprint. The immutable
prior software-only snapshot has DOI 10.5281/zenodo.22884868 and does not archive this revision.

A patient-level audit of frozen neural topic measurements in diabetic foot ulcer research. The release includes the producing analysis programs, public processed inputs, numerical evidence, fixed model states, mappings, and corrected journal documents. Healing inference uses eleven discovery patients. The analysis evaluates measurement decomposition, technical sensitivity, topic selection, refits and tissue context; it does not establish prospective healing prediction or independent clinical validation.

The full archive includes exact producing analysis code, recursive local imports,
public processed inputs, saved states, numerical tables, gene/mapping lists and
protocols. Current native paths are preserved beneath `analysis_revision/`.
Start with [the reproduction guide](REPRODUCIBILITY.md) and
[the current analysis guide](analysis_revision/REPRODUCIBILITY.md).

After downloading and extracting the complete version archive, run:

```sh
python3 verify_archive.py --smoke
```

The checksum/import/CLI smoke performs no fitting or downloads. Run the saved-input
checks documented in REPRODUCIBILITY.md using fresh output directories. The exact
journal source, figures, font contracts and build configuration are in
`journal/source/measurement/`; journal/README.md describes a fresh-copy rebuild.

The frozen representation and source data retain their original shared provenance.
Independent packaging does not add statistical independence or biological donors.
All nine fixed-grid saved model states reproduced their CPU projections and topic matches exactly. The six diagnostic refits are preserved; their full replay was not repeated in archive verification. Checkpoint replay and checksum checks do not certify full retraining or independent biological validation.

MIT applies to authored software. Scholarly documents, figures, saved research
outputs, source-study materials and third-party fonts retain the rights stated in
[NOTICE](NOTICE). Public processed count files are distinct from controlled human
sequencing reads; the latter were not used or redistributed. Font binaries and
third-party assets are documented with their own license/availability contracts.

Historical manuscript/analysis snapshots remain as provenance. The corrected
documents linked above are the default reading products for this release.


## Complete versioned reproduction package

This Git checkout and the source-and-documents asset provide current producing
source, corrected manuscripts and compact saved results. Large numerical arrays,
model states and inputs are in the complete version archive at
[software DOI 10.5281/zenodo.23308073](https://doi.org/10.5281/zenodo.23308073).
The public record determines publication status; the sealed package was assembled
while its DOI was reserved. Download every numbered ZIP part and follow
REASSEMBLE.md and SHA256SUMS. The reconstructed ZIP has SHA256
`c0561a58d9cf70e0fa386134b581f28a2c0929063c330f4e8049f3af5709343a`.

Extract the full archive before running full-manifest checks or scripts that use
the large inputs. The full archive's release-manifest.json describes that archive;
SOURCE_DOCUMENTS_MANIFEST.json describes the current default-branch compact selection, including separately versioned documents; the immutable source-and-documents asset retains its original manifest.
The DOI identifies software and research sources, not a manuscript publication.

The current default-branch compact selection is recorded in `SOURCE_DOCUMENTS_MANIFEST.json`; historical full-archive manifests apply to the immutable software archive. `PUBLICATION_STATUS.json` records the publication and separate document version.
