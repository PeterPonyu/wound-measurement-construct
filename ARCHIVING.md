# Independent software archiving

This project uses GitHub for source history and releases, and a separate Zenodo record for its MIT software. No repository-integration webhook is required. Do not reuse another study's deposition ID or DOI.

The full reproduction release contains manuscript and research outputs governed by NOTICE. Deposit the separate `wound-measurement-construct-code-0.2.0.zip` to Zenodo; its MIT-only scope matches .zenodo.json. Both ZIPs have independent checksum manifests. The software archive runs from public GEO inputs and does not require another GitHub repository.

Offline check, from this project root:

```sh
python3 archive_draft.py ../wound-measurement-construct-code-0.2.0.zip
```

Create or resume this study's draft after configuring ZENODO_ACCESS_TOKEN locally (never put credentials in source files, command arguments or issue text):

```sh
python3 archive_draft.py ../wound-measurement-construct-code-0.2.0.zip --execute
```

The program validates the archive's embedded metadata, uploads with checksum verification, and records a study-specific state under .archive-state/. It refuses state from another study and never publishes automatically. Alternatively, upload the same ZIP and metadata directly at https://zenodo.org/uploads/new. Review the exact record in Zenodo and publish it; copy the DOI into citation metadata only after the published landing page and uploaded file have been verified. A draft's reserved DOI is not a published record.
