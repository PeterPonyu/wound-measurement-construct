# Archive scope and verification

`wound-measurement-construct-0.5.0-full-source.zip` is the complete standalone reproduction release.
It contains authored software, corrected journal documents and editable sources,
public processed inputs, saved research states, tables and provenance. It is a
mixed-rights archive governed by NOTICE; it is not a software-only payload.

Reserved software/reproduction DOI: 10.5281/zenodo.23308073. The DOI was reserved at local
archive assembly; publication and uploads are handled separately. The prior
software-only DOI 10.5281/zenodo.22884868 identifies its immutable historical snapshot.
Neither identifier is a manuscript or preprint DOI.

MANIFEST.sha256.json binds every archived file except itself. The external ZIP
SHA-256 binds that manifest, and the final receipt records ZIP CRC, manifest
verification and fresh-extraction smoke/saved-input checks. Python caches,
working temporary builds, local Git state and obsolete outer manifests are excluded.
Native producing manifests and historical source fingerprints remain evidence.

After extraction, run `python3 verify_archive.py --smoke`, then the saved-input
commands in REPRODUCIBILITY.md. These checks do not certify full retraining,
universal numerical portability or biological/clinical validation.
