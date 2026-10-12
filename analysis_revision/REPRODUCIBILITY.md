# Neural topic measurement audit: current analysis reproduction

This is the version 0.5.0 current-analysis closure. Software DOI 10.5281/zenodo.23308073
was reserved at archive assembly; it identifies this software/reproduction release,
not a manuscript or preprint. Publication is handled separately.
Native producing bytes and paths are preserved in this directory; run commands
from this directory. REPRODUCTION_CLOSURE.json binds every included native input,
program, checkpoint, protocol and result. Historical paths in unmodified reports
describe the producing environment; do not resume relocated historical runs.

```sh
python3 verify_revision.py --smoke
python3 verify_revision.py --saved-checks --output-dir tmp/reproduction-check-01
```

Use a new output directory for every verification. The first command checks all
hashes, parses Python sources, imports the model/analysis helpers and checks safe
CLI help entry points. The second additionally recomputes available saved-input
results. Verification does not fit neural models, create a release, or download
source data. Existing optional trainers must be run only after creating a fresh
run contract and fresh output path; original approval/resume receipts are
historical provenance, not authorization to reuse a relocated run.

The native measurement tree contains the complete selection-null, anatomical decomposition,
prevalence/intensity, refit propagation, covariate, lineage/doublet and bulk-context
follow-ups. It includes the nine fixed capacity/seed fits, six diagnostic refits,
their inference states, readouts and topic mappings, the frozen 7,002-gene panel
and decoder, 76,461-cell counts/identity cache, public GSE165816 processed-count
archive, author patient map, public GSE80178 array inputs, and the saved Ensembl
probe-to-gene mapping used for the quantitative extension. The full mapped-array
and DE-filtered sensitivity universes remain distinct. There are 11 outcome
patients; refits, allocations and cell draws do not add biological replication.

The verification wrapper recomputes all 330 selection allocations and the clinical
balance/full-array bulk summaries, using the saved mapping without network access.
It sends outputs to a fresh requested directory and compares numerical values
with the exact saved reports. It does not fit a neural model or mixture.

To reload and reproject all nine fixed capacity/seed checkpoints without training,
run `python3 replay_measurement_checkpoints.py --output-dir tmp/checkpoint-replay-01`.
This uses the original single-thread CPU inference setting and unchanged projection
and matching tolerances. It checks model state, panel, cell order, proportions and
the Hungarian assignments, retaining weak and capacity-limited matches.


The existing runtime is recorded in environment.observed.json and
requirements.observed.txt; those files observe installed packages and perform no
installation. Producing package/device information remains in native protocols,
receipts and reports. The software license does not relicense source data, model
weights, numerical results or manuscript material; see the enclosing NOTICE.

Source-data acquisition and access instructions remain in the enclosing
REPRODUCIBILITY.md, scripts/fetch_*.sh and the native expansion/registry/datasets.json.
GEO processed count matrices and array signals retain source-study terms. Included
files named RAW.tar contain public processed count matrices, not human sequencing
reads. Controlled human sequencing reads were not used or redistributed; the
source registry records EGA-controlled raw reads for relevant public processed
cohorts. Access to such reads must follow the source study's stated EGA procedure.
No controlled-access data are needed for the saved-input verification commands.

The frozen representation and discovery source data retain their shared provenance.
A separately runnable copy does not make them statistically independent or supply
new biological validation. Every native_input_bindings entry records either an
exact included match or a specific provenance gap; a missing historical binding
is not silently replaced by a current digest.
