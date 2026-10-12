# Editable submission documents

Neural topic modelling of fibroblast measurements and diabetic foot ulcer healing: a patient-level audit

This document revision presents the same frozen computational results in a shorter main paper. It moves detailed methods and comparisons to a separate supplement, uses six main display items, adds line numbers, and states the AI assistance and scope consistently. It does not add model fits, patients or biological validation.

The software and saved-analysis archive is version 0.5.0, DOI https://doi.org/10.5281/zenodo.23308073. That software DOI identifies the frozen computational evidence and earlier archived document sources. The present editorial revision has its own GitHub document tag: https://github.com/PeterPonyu/wound-measurement-construct/releases/tag/submission-2026-10-12.

With an existing TeX Live installation, latexmk, xelatex and the declared document packages, run from this directory:

    latexmk -xelatex -interaction=nonstopmode -halt-on-error paper_srep_inline.tex
    latexmk -xelatex -interaction=nonstopmode -halt-on-error supporting_srep.tex

XeLaTeX documents require a licensed Arial installation visible to the operating-system font manager; TeX Gyre Termes and the mathematics fonts come from TeX Live. Arial font files are not redistributed in this source archive. The supplied PDFs embed the fonts needed for viewing. The pdfLaTeX document uses the font packages already declared in its template.

The .bbl files and bibliography sources are supplied. Input paths are relative. SOURCE_MANIFEST.json records every packaged dependency and the hashes of the reviewed PDFs. Build logs, local machine paths, private correspondence and obsolete cover-letter drafts are excluded.

Plots and data-derived images retain the source-specific attribution in the manuscript and supplement. The original software license does not override third-party data or image terms. Refer to the frozen software release for its full research reproduction guide and third-party provenance. These files provide document reproduction; the full saved-input analyses require the software archive.

The author remains responsible for the text and interpretation. Automated preparation checks are not independent human peer review. The final journal reading and portal declarations remain author tasks.
