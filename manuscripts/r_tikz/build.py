#!/usr/bin/env python3
"""Compile R/TikZ vector figures and render the manuscript previews from them.

Python only orchestrates R, XeLaTeX and Poppler and verifies their artifacts.
It does not calculate statistics or draw any scientific panel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIG = ROOT / "manuscripts/figures"
BUILD = ROOT / "outputs/figure_r_tikz"
STAGING = BUILD / "compiled"
PAPERS = {1: ROOT / "manuscripts/latex/figure_captions"}


def run(command: list[str], log: Path | None = None, cwd: Path = ROOT) -> str:
    if log:
        with log.open("w") as handle:
            process = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f"Command failed: {command[0]}\n{log.read_text()[-5000:]}")
        return log.read_text()
    return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.STDOUT)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_environment() -> None:
    for tool in ("Rscript", "xelatex", "kpsewhich", "fc-match", "pdftoppm"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"Required tool unavailable: {tool}; no automatic installation.")
    for package in ("standalone.cls", "fontspec.sty", "tikz.sty"):
        if not run(["kpsewhich", package]).strip():
            raise RuntimeError(f"Required TeX package unavailable: {package}")
    family = run(["fc-match", "-f", "%{family}", "Arial"]).strip()
    if family.split(",")[0] != "Arial":
        raise RuntimeError(f"Arial resolved to {family!r}; refusing a silent font substitution.")
    run(["Rscript", "-e", 'p <- c("ggplot2","tikzDevice","jsonlite","digest"); '
         'if(!all(vapply(p,requireNamespace,logical(1),quietly=TRUE))) stop("Missing R packages")'])


def compile_tex(path: Path) -> Path:
    log = BUILD / f"{path.stem}.compile.txt"
    run(["xelatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error",
         "-file-line-error", f"-output-directory={STAGING}", str(path)], log=log, cwd=path.parent)
    transcript = (STAGING / f"{path.stem}.log").read_text(errors="replace")
    if "Missing character:" in transcript or re.search(r"Overfull \\[hv]box", transcript):
        raise RuntimeError(f"Missing glyph or overfull box in {path.name}; inspect {log}")
    return STAGING / f"{path.stem}.pdf"


def inspect_pdf(path: Path, pages: int = 1, width_mm: float | None = None,
                height_mm: float | None = None, figure_text: bool = True) -> dict:
    fonts, sizes = set(), []
    with fitz.open(path) as document:
        if len(document) != pages:
            raise RuntimeError(f"Unexpected page count in {path.name}: {len(document)} != {pages}")
        dimensions = []
        for page in document:
            if page.get_images(full=True):
                raise RuntimeError(f"Raster object found inside vector figure: {path.name}")
            dimensions.append([page.rect.width * 25.4 / 72, page.rect.height * 25.4 / 72])
            if width_mm and abs(dimensions[-1][0] - width_mm) > .2:
                raise RuntimeError(f"Incorrect physical width: {path.name}")
            if height_mm and abs(dimensions[-1][1] - height_mm) > .2:
                raise RuntimeError(f"Incorrect physical height: {path.name}")
            for font in page.get_fonts():
                if "Arial" not in font[3] or not document.extract_font(font[0])[3]:
                    raise RuntimeError(f"Unexpected or unembedded font in {path.name}: {font[3]}")
                fonts.add(font[3])
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        if not span["text"].strip():
                            continue
                        if figure_text and span["color"] != 0:
                            raise RuntimeError(f"Figure text must be black in {path.name}: {span['text']}")
                        if figure_text and "Bold" in span["font"] and not (
                                re.fullmatch(r"[A-Z]", span["text"].strip()) and span["size"] >= 10.8):
                            raise RuntimeError(f"Only separate panel letters may be bold in {path.name}: {span['text']}")
                        bounds = fitz.Rect(span["bbox"])
                        if not (page.rect + (-.8, -.8, .8, .8)).contains(bounds):
                            raise RuntimeError(f"Text crosses page boundary in {path.name}: {span['text']}")
                        sizes.append(span["size"])
    if not sizes or min(sizes) < 6.9:
        raise RuntimeError(f"Missing or undersized text in {path.name}")
    return {"pages": pages, "dimensions_mm": dimensions, "fonts": sorted(fonts),
            "minimum_font_pt": min(sizes), "raster_objects": 0,
            "black_text_bold_labels_only": figure_text, "sha256": sha(path)}


def latex_escape(text: str) -> str:
    # Captions are single paragraphs in the manuscript source.
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
               "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mapping.get(char, char) for char in text)


def inspect_headings(path: Path, specification: dict) -> list[dict]:
    """Measure title centring and independent bold letters in the actual PDF."""
    rows = []
    titles = specification["headings"]
    centres = specification["heading_centres_mm"]
    if isinstance(titles, str):
        titles = [titles]
    if not isinstance(centres, list):
        centres = [centres]
    if len(titles) != len(centres):
        raise RuntimeError(f"Incomplete panel-heading specification: {path.name}")
    labels = specification["panel_labels"]
    lefts = specification["label_left_mm"]
    tops = specification["heading_top_mm"]
    widths = specification["panel_width_mm"]
    if not isinstance(lefts, list):
        lefts = [lefts]
    if not isinstance(tops, list):
        tops = [tops]
    if not isinstance(widths, list):
        widths = [widths]
    if (specification.get("heading_alignment") != "panel-centred" or
            not len(titles) == len(lefts) == len(tops) == len(widths)):
        raise RuntimeError(f"Incomplete centred-heading geometry: {path.name}")
    with fitz.open(path) as document:
        page = document[0]
        spans = [s for b in page.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"]]
        bold = [s for s in spans if "Bold" in s["font"] and s["text"].strip()]
        if len(bold) != len(labels):
            raise RuntimeError(f"Incorrect number of bold panel letters: {path.name}")
        for i, (title, centre) in enumerate(zip(titles, centres)):
            if abs(centre - (lefts[i] + widths[i] / 2)) > .01:
                raise RuntimeError(f"Heading target differs from panel centre: {path.name}: {title}")
            matches = [box for box in page.search_for(title)
                       if abs(box.y0 * 25.4 / 72 - tops[i] - .5) < 1]
            if len(matches) != 1:
                raise RuntimeError(f"Panel heading must be one complete text block: {path.name}: {title}")
            box = matches[0]
            actual = (box.x0 + box.x1) / 2 * 25.4 / 72
            if abs(actual - centre) > .2:
                raise RuntimeError(f"Off-centre heading: {path.name}: {title}: {actual} vs {centre} mm")
            if (box.x0 * 25.4 / 72 < lefts[i] - .2 or
                    box.x1 * 25.4 / 72 > lefts[i] + widths[i] + .2):
                raise RuntimeError(f"Title extends outside its panel: {path.name}: {title}")
            title_spans = [s for s in spans if fitz.Rect(s["bbox"]).intersects(box)]
            if any("Bold" in s["font"] for s in title_spans):
                raise RuntimeError(f"Title must be regular and separate from label: {path.name}: {title}")
            row = {"heading": title, "centre_mm": actual, "expected_centre_mm": centre,
                   "centre_error_mm": actual - centre, "panel_width_mm": widths[i],
                   "alignment": "panel-centred"}
            if labels:
                matches = [s for s in bold if s["text"].strip() == labels[i]]
                if len(matches) != 1:
                    raise RuntimeError(f"Missing independent panel letter {labels[i]}: {path.name}")
                tag = matches[0]
                tagbox = fitz.Rect(tag["bbox"])
                overlaps = [s["text"] for s in spans if s is not tag and s["text"].strip()
                            and fitz.Rect(s["bbox"]).intersects(tagbox + (-.5, -.5, .5, .5))]
                if overlaps:
                    raise RuntimeError(f"Text overlaps panel letter {labels[i]} in {path.name}: {overlaps}")
                if (abs(tagbox.x0 * 25.4 / 72 - lefts[i]) > .6 or
                        abs(tagbox.y0 * 25.4 / 72 - tops[i]) > 1.5 or
                        tagbox.x1 + 1 >= box.x0 or tag["size"] <= max(s["size"] for s in title_spans)):
                    raise RuntimeError(f"Panel letter/title layout or hierarchy failed: {path.name}: {labels[i]}")
                row.update(label=labels[i], label_left_mm=tagbox.x0 * 25.4 / 72,
                           label_font_pt=tag["size"], separate_text_objects=True)
            rows.append(row)
    return rows


def inspect_tissue_panel(path: Path, paper: int, page_number: int = 0) -> dict:
    """Require visible filled dermis as well as labels; text alone passed before."""
    with fitz.open(path) as document:
        page = document[page_number]
        dermis = [d for d in page.get_drawings() if d.get("fill") and
                  max(abs(a-b) for a,b in zip(d["fill"], (245/255,231/255,216/255))) < .012
                  and d["rect"].get_area() > 350]
        expected = 1 if paper == 1 else 4
        if len(dermis) != expected:
            raise RuntimeError(f"Missing dermal section(s) in {path.name}: {len(dermis)} != {expected}")
        words = (["Ulcer bed", "Dermis", "Vessel", "Immune cells", "7,002-gene",
                  "Frozen MLP", "15-topic", "Fibroblast gate", "All-cell mean",
                  "Patient-unit", "sampled z", "β decoder", "Gaussian KL"]
                 if paper == 1 else
                 ["Intact skin", "Day 1", "Day 7", "Day 30", "Donor A",
                  "Donor B", "Donor C", "7,002-gene input", "5,383 covered",
                  "Frozen", "Train-only", "Shared CFM", "Product endpoints",
                  "RK4 flow", "marginal test", "held from fitting"])
        for word in words:
            if not page.search_for(word):
                raise RuntimeError(f"Missing tissue/model label {word} in {path.name}")
        return {"filled_dermal_sections": len(dermis), "required_labels": words}


def build_tissue_panels() -> dict:
    checks = {}
    for paper in PAPERS:
        source = HERE / "panel_a.tex"
        pdf = compile_tex(source)
        checks[source.stem] = inspect_pdf(pdf, width_mm=84.5, height_mm=57.8667)
        checks[source.stem]["tissue"] = inspect_tissue_panel(pdf, paper)
        with fitz.open(pdf) as doc:
            (STAGING / f"{source.stem}.svg").write_text(doc[0].get_svg_image(text_as_path=True))
        run(["pdftoppm", "-r", "300", "-singlefile", "-png", str(pdf), str(STAGING/source.stem)])
    for paper in PAPERS:
        for suffix in (".pdf", ".png", ".svg"):
            name = f"panel_a{suffix}"
            shutil.copy2(STAGING/name, FIG/name)
    return checks


def build_collection(paper: int) -> tuple[Path, int]:
    expected = 9 if paper == 1 else 7
    matches = []
    for number in range(1, expected + 1):
        source = PAPERS[paper] / f"f{number}.tex"
        match = re.fullmatch(r"\\paperfigure\{([^}]+)\}\{([^}]+)\}\{(.*)\}\s*", source.read_text(), re.S)
        if not match or match[2] != f"fig:p{paper}f{number}":
            raise RuntimeError(f"Missing native figure/caption: {source}")
        matches.append((str(number), match[1], match[3]))
    lines = [r"\documentclass[10pt]{article}",
             r"\usepackage[a4paper,left=15mm,right=15mm,top=16mm,bottom=16mm]{geometry}",
             r"\usepackage{fontspec,graphicx,xcolor}", r"\setmainfont{Arial}",
             r"\setsansfont{Arial}", r"\pagestyle{empty}", r"\setlength{\parindent}{0pt}",
             r"\setlength{\parskip}{0pt}", r"\setlength{\emergencystretch}{1em}", r"\begin{document}"]
    for index, (number, image, caption) in enumerate(matches):
        if index:
            lines.append(r"\newpage")
        lines += [r"{\fontsize{13}{16}\selectfont\bfseries Figure " + number + r"}\par\vspace{3mm}",
                  r"\includegraphics[width=180mm]{../" + image + r".pdf}\par\vspace{3mm}",
                  r"{\fontsize{9}{11.5}\selectfont " + caption + r"\par}"]
    lines.append(r"\end{document}")
    source = FIG / "tex" / "figures.tex"
    source.write_text("\n".join(lines) + "\n")
    return compile_tex(source), expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", action="append", default=[], help="Rebuild one named figure; repeatable")
    parser.add_argument("--compile-only", action="store_true", help="Compile the current R-generated sources")
    args = parser.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    STAGING.mkdir(parents=True, exist_ok=True)
    check_environment()
    tissue_checks = build_tissue_panels()
    if not args.compile_only:
        print("Rendering R panels to TikZ…", flush=True)
        run(["Rscript", str(HERE / "build_figures.R"), *args.figure], log=BUILD / "R-build.log")
    manifest = json.loads((BUILD / "manifest.json").read_text())
    if manifest["script_sha256"] != sha(HERE / "build_figures.R"):
        raise RuntimeError("R source changed since generation; rerun without --compile-only")
    for path, fingerprint in manifest.get("auxiliary_sources", {}).items():
        if sha(ROOT / path) != fingerprint:
            raise RuntimeError("Biological figure source changed since generation; rerun without --compile-only")
    for path, fingerprint in manifest.get("design_sources", {}).items():
        if sha(ROOT / path) != fingerprint:
            raise RuntimeError("Diagram source changed since generation; rerun without --compile-only")
    names = args.figure or list(manifest["figures"])
    if not set(names) <= set(manifest["figures"]):
        raise RuntimeError("Requested figure missing from the current manifest")
    verification = {"tissue_panels": tissue_checks, "figures": {}, "collections": {}, "source_reports_unchanged": False}
    for name in names:
        dimensions = manifest["figures"][name]
        pdf = compile_tex(FIG / "tex" / f"{name}.tex")
        verification["figures"][name] = inspect_pdf(pdf, width_mm=dimensions["width_mm"],
                                                   height_mm=dimensions["height_mm"])
        verification["figures"][name]["independent_panel_labels_and_titles"] = inspect_headings(pdf, dimensions)
        if name.endswith("figure1_workflow"):
            verification["figures"][name]["tissue"] = inspect_tissue_panel(pdf, 1)
        run(["pdftoppm", "-r", "300", "-singlefile", "-png", str(pdf), str(STAGING / name)])
        print(f"Verified vector + 300 dpi preview: {name}", flush=True)
    for path, fingerprint in dict(manifest["sources"]).items():
        if sha(ROOT / path) != fingerprint:
            raise RuntimeError(f"Source report changed during figure build: {path}")
    verification["source_reports_unchanged"] = True
    # Only replace the public artifacts after every requested figure has passed.
    for name in names:
        for suffix in (".pdf", ".png"):
            shutil.copy2(STAGING / f"{name}{suffix}", FIG / f"{name}{suffix}")
    if len(names) == 9:
        for paper in PAPERS:
            pdf, pages = build_collection(paper)
            verification["collections"][pdf.stem] = inspect_pdf(pdf, pages=pages, width_mm=210, figure_text=False)
            shutil.copy2(pdf, FIG / pdf.name)
            run(["pdftoppm", "-r", "110", "-png", str(pdf), str(BUILD / f"paper{paper}-review")])
    verification["render_manifest_sha256"] = sha(BUILD / "manifest.json")
    verification["build_script_sha256"] = sha(Path(__file__))
    verification["caption_sha256"] = {str(path.relative_to(ROOT)): sha(path)
                                      for folder in PAPERS.values() for path in sorted(folder.glob("f*.tex"))}
    (BUILD / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    shutil.copy2(BUILD / "manifest.json", FIG / "render_manifest.json")
    print(f"DONE: {len(names)} figures. Review: {FIG}", flush=True)


if __name__ == "__main__":
    main()
