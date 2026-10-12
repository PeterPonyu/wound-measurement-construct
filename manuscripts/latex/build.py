#!/usr/bin/env python3
"""Compile and verify the native manuscript.

LaTeX and BibTeX are the production inputs. No Markdown draft is read.
Figures are built separately by the R/TikZ workflow.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import fitz

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BUILD = ROOT / "tmp/pdfs"
DEST = ROOT / "output/pdf"
NAMES = {1: "measurement_construct_validation"}
sys.path.insert(0, str(ROOT))
from manuscripts.r_tikz.build import figure_sources, inspect_tissue_panel, sha
from manuscripts.latex.geometry import inspect_appendix


def validate_figure_inputs():
    """Reject stale artwork before compiling a manuscript that embeds it."""
    manifest_path = ROOT / "manuscripts/figures/render_manifest.json"
    verification_path = ROOT / "outputs/figure_r_tikz/verification.json"
    manifest = json.loads(manifest_path.read_text())
    verification = json.loads(verification_path.read_text())
    expected = {re.match(r'\\paperfigure\{([^}]+)\}', p.read_text())[1]
                for n in NAMES for p in figure_sources(n)}
    if set(verification['figures']) != expected or len(verification["collections"]) != 1:
        raise RuntimeError("A complete native-caption-matched figure build is required before manuscript delivery")
    if sha(manifest_path) != verification["render_manifest_sha256"]:
        raise RuntimeError("Figure manifest changed after verification; rebuild figures")
    for group in ("sources", "auxiliary_sources", "design_sources"):
        for name, digest in manifest.get(group, {}).items():
            if sha(ROOT/name) != digest:
                raise RuntimeError(f"Figure input changed: {name}; rebuild figures")
    if sha(ROOT/"manuscripts/r_tikz/build_figures.R") != manifest["script_sha256"]:
        raise RuntimeError("R figure source changed; rebuild figures")
    for name, digest in verification.get("caption_sha256", {}).items():
        if sha(ROOT/name) != digest:
            raise RuntimeError(f"Figure caption changed: {name}; rebuild figure collections")
    for name, check in {**verification["figures"], **verification["tissue_panels"]}.items():
        if sha(ROOT/"manuscripts/figures"/f"{name}.pdf") != check["sha256"]:
            raise RuntimeError(f"Figure PDF changed after verification: {name}")


def execute(command, log=None, cwd=HERE):
    if log:
        with log.open("w") as handle:
            result = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(log.read_text(errors="replace")[-5000:])
    else:
        subprocess.run(command, cwd=cwd, check=True)


def validate_population_results_whitespace(whitespace, first_page, last_page):
    """Prevent a stranded Results float, including one shifted to a later page."""
    for item in whitespace[first_page - 1:last_page]:
        limit = 36 if item["page"] == 14 else 60
        if item["bottom_gap_pt"] > limit:
            raise RuntimeError(
                f"Population Results page {item['page']} has excessive bottom "
                f"whitespace: {item['bottom_gap_pt']} pt (limit {limit} pt)"
            )


def validate_measurement_results_whitespace(whitespace, first_page, last_page):
    """Reject the measurement opening's original or pagination-shifted gap."""
    for item in whitespace[first_page - 1:last_page]:
        limit = 36 if item["page"] == first_page else 60
        if item["bottom_gap_pt"] > limit:
            raise RuntimeError(
                f"Measurement Results page {item['page']} has excessive bottom "
                f"whitespace: {item['bottom_gap_pt']} pt (limit {limit} pt)"
            )


def require_clean_typesetting(log, n):
    for pattern in [r"Overfull \\[hv]box", r"Missing character:",
                    r"(?:Citation|Reference) .+ undefined", r"multiply defined",
                    r"ignored error:", r"(?:LaTeX|Package .+) Error:"]:
        if re.search(pattern, log):
            raise RuntimeError(f"Paper {n}: unresolved typesetting issue: {pattern}")


def validate(n, build_dir=None, *, write_extracted_text=True):
    folder = (BUILD if build_dir is None else Path(build_dir)) / "manuscript"
    stem = folder / "manuscript"
    log = stem.with_suffix(".log").read_text(errors="replace")
    require_clean_typesetting(log, n)
    if "already defined" in (folder/"build.log").read_text(errors="replace"):
        raise RuntimeError(f"Paper {n}: duplicate PDF navigation destination")
    aux = stem.with_suffix(".aux").read_text()
    citation_order = list(dict.fromkeys(key.strip() for group in re.findall(r"\\citation\{([^}]+)\}", aux)
                                        for key in group.split(",")))
    bibliography = stem.with_suffix(".bbl").read_text()
    bibliography_order = re.findall(r"\\bibitem(?:\[[\s\S]*?\])?\{([^}]+)\}", bibliography)
    if citation_order != bibliography_order:
        raise RuntimeError(f"Paper {n}: bibliography does not follow first citation order")
    pages = {m[1]: int(m[2]) for m in re.finditer(r"\\newlabel\{((?:fig|call):[^}]+)\}\{\{[^}]*\}\{(\d+)\}", aux)}
    placement = []
    for number in range(1, len(figure_sources(n)) + 1):
        key = f"p{n}f{number}"
        ref, fig = pages[f"call:{key}"], pages[f"fig:{key}"]
        if not 0 <= fig - ref <= 1:
            raise RuntimeError(f"Paper {n}, figure {number}: first citation page {ref}, figure page {fig}")
        placement.append({"figure": number, "first_citation_page": ref, "figure_page": fig})
    with fitz.open(stem.with_suffix(".pdf")) as document:
        text = "\n".join(page.get_text() for page in document)
        for word in ["outputs/", "scripts/", "SHA256", "preflight", "report.json", "BLOCKED_pending", "??"]:
            if word.lower() in text.lower():
                raise RuntimeError(f"Paper {n}: unresolved/internal text {word}")
        if re.search(r"\b(?:TODO|T[123]|connectivity audit|population dynamics)\b|Topic[- ]Simplex|/home/", text):
            raise RuntimeError(f"Paper {n}: unresolved task marker or internal project name")
        if any(page.get_images(full=True) for page in document):
            raise RuntimeError(f"Paper {n}: raster figure found")
        empty = [page.number + 1 for page in document if len(page.get_text().strip()) < 80]
        if empty:
            raise RuntimeError(f"Paper {n}: nearly empty pages {empty}")
        # A continued table with a single row can exceed the character check.
        # The manuscript end in numerical supplements, so also reject a
        # nearly blank trailing page, excluding the running header and folio.
        last = document[-1]
        body = [block for block in last.get_text("blocks")
                if block[1] > 40 and block[3] < last.rect.height - 40]
        occupied = (max(block[3] for block in body) - min(block[1] for block in body)) if body else 0
        if occupied < .20 * (last.rect.height - 80):
            raise RuntimeError(f"Paper {n}: isolated content on final page {len(document)}")
        clipped = []
        for page in document:
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        if span["text"].strip() and not (page.rect + (-1,-1,1,1)).contains(fitz.Rect(span["bbox"])):
                            clipped.append([page.number + 1, span["text"]])
        if clipped:
            raise RuntimeError(f"Paper {n}: content outside page: {clipped}")
        fonts = sorted({font[3] for page in document for font in page.get_fonts(full=True)})
        for page in document:
            for font in page.get_fonts(full=True):
                if not document.extract_font(font[0])[3]:
                    raise RuntimeError(f"Unembedded font {font[3]}")
        whitespace=[]
        for page in document:
            blocks=[b for b in page.get_text("blocks") if b[1]>40 and b[3]<page.rect.height-40]
            end=max((b[3] for b in blocks),default=40)
            whitespace.append({"page":page.number+1,"bottom_gap_pt":round(max(0,page.rect.height-49-end),1)})
        # Regression guard for the author-reported blank below the opening
        # population Results: a float barrier previously stranded 222.6 pt.
        if n == 2 and whitespace[pages["call:p2f1"] - 1]["bottom_gap_pt"] > 36:
            raise RuntimeError("Population Results opening has excessive bottom whitespace")
        if "1 Introduction" not in text or "2 Materials and methods" not in text or "3 Results" not in text:
            raise RuntimeError(f"Paper {n}: section order changed")
        results_first = next(page.number + 1 for page in document
                             if "3 Results" in page.get_text())
        discussion_first = next(page.number + 1 for page in document
                                if "4 Discussion" in page.get_text())
        if n == 2:
            validate_population_results_whitespace(whitespace, results_first, discussion_first)
        else:
            validate_measurement_results_whitespace(whitespace, results_first, discussion_first)
        tissue=inspect_tissue_panel(stem.with_suffix(".pdf"),n,pages[f"fig:p{n}f1"]-1)
        internal_links=0
        for page in document:
            for link in page.get_links():
                if link.get("kind") in (fitz.LINK_GOTO, fitz.LINK_NAMED):
                    if not 0 <= link.get("page", -1) < len(document):
                        raise RuntimeError(f"Paper {n}: unresolved PDF navigation link: {link}")
                    internal_links+=1
        result = {"pages": len(document), "figures": placement, "raster_objects": 0, "fonts": fonts,
                  "figure1_tissue":tissue,"page_bottom_gaps":whitespace,"resolved_internal_links":internal_links,
                  "results_page_range": {"first": results_first, "discussion_begins": discussion_first},
                  "population_results_whitespace_checked": n == 2,
                  "measurement_results_whitespace_checked": n == 1,
                  "references": len(re.findall(r"\\bibitem", stem.with_suffix(".bbl").read_text())),
                  "first_citation_order_verified": True, "reference_keys": bibliography_order,
                  "equations": len(re.findall(r"\\newlabel\{eq:",aux)),
                  "tables": len(re.findall(r"\\newlabel\{tab:",aux)),
                  "appendix_geometry": inspect_appendix(document)}
        if write_extracted_text:
            (folder / "extracted_text.txt").write_text(text)
    return result


def main():
    global BUILD, DEST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="Render every final page for visual inspection")
    parser.add_argument("--build-dir", default=str(BUILD), help="Fresh LaTeX build directory")
    parser.add_argument("--dest-dir", default=str(DEST), help="Destination directory for compiled PDFs")
    args = parser.parse_args()
    BUILD = Path(args.build_dir).resolve()
    DEST = Path(args.dest_dir).resolve()
    for command in ("latexmk", "xelatex", "bibtex", "pdftoppm"):
        if shutil.which(command) is None:
            raise RuntimeError(f"Required installed tool is absent: {command}")
    BUILD.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    validate_figure_inputs()
    # Resolve the canonical bibliography independently of local symlinks.
    os.environ["BIBINPUTS"] = str(ROOT / "manuscripts") + os.pathsep + os.environ.get("BIBINPUTS", "")
    execute([sys.executable, str(HERE / "export_tables.py")])
    execute([sys.executable, str(ROOT / "manuscripts/quality_tables.py"), "--check"])
    execute([sys.executable, str(ROOT / "manuscripts/followup_tables.py"), "--check"])
    if (ROOT / "manuscripts/p1_preparation_tables.py").is_file():
        execute([sys.executable, str(ROOT / "manuscripts/p1_preparation_tables.py"), "--check",
                 str(HERE / "paper1_preparation_tables.tex")])
    execute([sys.executable, str(ROOT / "scripts/verify_manuscript.py")])
    results = {}
    for n, name in NAMES.items():
        folder = BUILD / "manuscript"
        folder.mkdir(exist_ok=True)
        execute(["latexmk", "-norc", "-xelatex", "-bibtex", "-interaction=nonstopmode", "-halt-on-error",
                 "-file-line-error", "-no-shell-escape", f"-outdir={folder}", "manuscript.tex"],
                log=folder / "build.log")
        results[name] = validate(n)
    # Publish only after the document pass; scientific reports are never edited.
    for n, name in NAMES.items():
        pdf = DEST / f"{name}.pdf"
        shutil.copy2(BUILD / "manuscript/manuscript.pdf", pdf)
        results[name]["sha256"] = hashlib.sha256(pdf.read_bytes()).hexdigest()
        if args.render:
            review = BUILD / "manuscript/render"
            review.mkdir(exist_ok=True)
            for stale in review.glob("page-*.png"):
                stale.unlink()
            execute(["pdftoppm", "-r", "120", "-png", str(pdf), str(review / "page")],
                    log=review / "render.log")
            if len(list(review.glob("page-*.png"))) != results[name]["pages"]:
                raise RuntimeError(f"Paper {n}: rendered page count does not match the PDF")
    (BUILD / "manuscript_verification.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
