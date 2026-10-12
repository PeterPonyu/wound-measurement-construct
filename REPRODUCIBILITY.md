# Reproduction guide

Version 0.5.0 software/reproduction DOI 10.5281/zenodo.23308073 was reserved at archive assembly.
It does not identify an article or preprint. Check the DOI record for publication status.

The corrected main reading PDF is `journal/scientific-reports-manuscript-inline.pdf`.
Editable journal TeX, figures, bibliography, fonts/contracts and configuration are
under `journal/source/measurement/`; see journal/README.md for a fresh-copy rebuild.

Verify all included bytes and safe imports/CLI entry points from the archive root:

```sh
python3 verify_archive.py --smoke
```

Recompute bounded saved-input results from the native closure:

```sh
cd analysis_revision
python3 verify_revision.py --smoke --saved-checks --output-dir tmp/archive-reproduction-01
```

Reload and reproject the nine fixed-grid measurement checkpoints with
`python3 replay_measurement_checkpoints.py --output-dir tmp/archive-checkpoints-01`
from `analysis_revision/`. The frozen nine-state CPU replay passed with zero
projection error and exact topic assignments. The six diagnostic refits remain
available as saved evidence; their full replay was not repeated here.

Preserve every reference file and choose a new
directory for another rerun. Verification does not fit neural models or download data.

The native REPRODUCTION_CLOSURE.json binds source programs, public processed inputs,
weights, gene identities, cell order, protocols, tables and explicit historical
input hashes. The existing environment is recorded there. Earlier checkpoint,
resume and approval receipts describe historical executions and must not be reused
as a relocated resume contract or current authorization.

Current saved-input checks retain all adverse results and biological-unit limits.
Reproduction does not supply new biological validation or certify full retraining.

Processed source data retain their GEO/source-study terms. Data acquisition helpers
remain in `scripts/fetch_*.sh`, and the native registry records source access and
controlled-read restrictions. Public raw-named count archives contain processed
matrices, not controlled human sequencing reads. Those controlled reads are neither
required for these saved-input commands nor redistributed.

MIT covers authored software only. NOTICE describes source-data and scholarly-output
rights. TeX Gyre fonts carry their included GUST license. A separately licensed Arial
installation is needed for matching source recompilation/restyling; the delivered
PDFs contain their embedded fonts.
