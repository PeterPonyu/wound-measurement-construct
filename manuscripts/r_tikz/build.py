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
                        if figure_text and (span["color"] != 0 or "Bold" not in span["font"]):
                            raise RuntimeError(f"Figure text must be black and bold in {path.name}: {span['text']} ({span['font']}, {span['color']})")
                        bounds = fitz.Rect(span["bbox"])
                        if not (page.rect + (-.8, -.8, .8, .8)).contains(bounds):
                            raise RuntimeError(f"Text crosses page boundary in {path.name}: {span['text']}")
                        sizes.append(span["size"])
    if not sizes or min(sizes) < 6.9:
        raise RuntimeError(f"Missing or undersized text in {path.name}")
    return {"pages": pages, "dimensions_mm": dimensions, "fonts": sorted(fonts),
            "minimum_font_pt": min(sizes), "raster_objects": 0,
            "black_bold_figure_text": figure_text, "sha256": sha(path)}


def latex_escape(text: str) -> str:
    # Captions are single paragraphs in the manuscript source.
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
               "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mapping.get(char, char) for char in text)


def build_collection(paper: int) -> tuple[Path, int]:
    expected = 8 if paper == 1 else 6
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
        lines += [r"{\fontsize{13}{16}\selectfont\bfseries Figure " + number + r"}\par\vspace{5mm}",
                  r"\includegraphics[width=180mm]{../" + image + r".pdf}\par\vspace{5mm}",
                  r"{\fontsize{10}{13}\selectfont " + caption + r"\par}"]
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
    if not args.compile_only:
        print("Rendering R panels to TikZ…", flush=True)
        run(["Rscript", str(HERE / "build_figures.R"), *args.figure], log=BUILD / "R-build.log")
    manifest = json.loads((BUILD / "manifest.json").read_text())
    if manifest["script_sha256"] != sha(HERE / "build_figures.R"):
        raise RuntimeError("R source changed since generation; rerun without --compile-only")
    for path, fingerprint in manifest.get("design_sources", {}).items():
        if sha(ROOT / path) != fingerprint:
            raise RuntimeError("Diagram source changed since generation; rerun without --compile-only")
    names = args.figure or list(manifest["figures"])
    if not set(names) <= set(manifest["figures"]):
        raise RuntimeError("Requested figure missing from the current manifest")
    verification = {"figures": {}, "collections": {}, "source_reports_unchanged": False}
    for name in names:
        dimensions = manifest["figures"][name]
        pdf = compile_tex(FIG / "tex" / f"{name}.tex")
        verification["figures"][name] = inspect_pdf(pdf, width_mm=dimensions["width_mm"],
                                                   height_mm=dimensions["height_mm"])
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
    if len(names) == 8:
        for paper in PAPERS:
            pdf, pages = build_collection(paper)
            verification["collections"][pdf.stem] = inspect_pdf(pdf, pages=pages, width_mm=210, figure_text=False)
            shutil.copy2(pdf, FIG / pdf.name)
            run(["pdftoppm", "-r", "110", "-png", str(pdf), str(BUILD / f"paper{paper}-review")])
    verification["render_manifest_sha256"] = sha(BUILD / "manifest.json")
    verification["build_script_sha256"] = sha(Path(__file__))
    (BUILD / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    shutil.copy2(BUILD / "manifest.json", FIG / "render_manifest.json")
    print(f"DONE: {len(names)} figures. Review: {FIG}", flush=True)


if __name__ == "__main__":
    main()
