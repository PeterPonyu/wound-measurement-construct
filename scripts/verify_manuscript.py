"""Verify manuscript prose, equations and numerical anchors."""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPTS = ROOT / 'manuscripts'
PAPER_FILES = {'measurement_construct': MANUSCRIPTS / 'latex/manuscript.tex'}
ANCHORS = {'measurement_construct': ['76,461', '243', '0.933', '0.841', '24.9%', '0.0280', '0.2719', '−0.0204', '561', '358', '0.909', '0.948', '0.714', '0.786', '0.937', '+0.0476', '0.0107', '0.0402', '0.302', '0.166', '1.66', '+0.210', '14.43/15', '0.0015', '5.0%']}
AFFIRMATIVE_PATTERNS = [re.compile('(?i)\\b(?:is|was|constitutes|establishes|supports)\\b[^.\\n]{0,60}\\bvalidated\\s+(?:healing|prognostic|DFU)\\s+biomarker\\b'), re.compile('(?i)\\bTopic[- ]Simplex\\b[^.\\n]{0,45}\\b(?:is|was)\\s+(?:clearly\\s+)?superior\\b'), re.compile('(?i)\\bconditioning\\b[^.\\n]{0,35}\\b(?:is|was)\\s+the\\s+correct\\s+model\\s+class\\b'), re.compile('(?i)\\bdonor\\s+(?:number|count)\\b[^.\\n]{0,35}\\b(?:is|was|remains)\\s+(?:the|a|the\\s+main|the\\s+binding)\\s+bottleneck\\b'), re.compile('(?i)\\bGSE248247\\b[^.\\n]{0,45}\\b(?:validates|confirms|establishes)\\b[^.\\n]{0,35}\\b(?:healing|prognosis|prognostic)\\b'), re.compile('(?i)\\b(?:demonstrates|establishes|proves|validates|enables)\\b[^.\\n]{0,35}\\b(?:DFU|diabetic[- ]foot[- ]ulcer)\\b[^.\\n]{0,20}\\btrajectory\\s+prediction\\b')]

def fail(message: str) -> None:
    print(f'FAIL  {message}')
    raise SystemExit(1)

def read_native(path: Path, stack: tuple[Path, ...]=()) -> str:
    """Read every included scientific section, caption and numerical table."""
    path = path.resolve()
    if path in stack:
        fail(f'cyclic LaTeX include: {path.name}')
    text = path.read_text(encoding='utf-8')

    def include(match: re.Match) -> str:
        child = path.parent / match[1]
        if not child.suffix:
            child = child.with_suffix('.tex')
        return read_native(child, stack + (path,))
    text = re.sub('\\\\input\\{([^}]+)\\}', include, text)
    return text.replace('\\%', '%').replace('{[}', '[').replace('{]}', ']')

def check_author_style() -> None:
    """Check the author's prose, equation and cross-reference requirements."""
    totals = {'equations': 0, 'tables': 0, 'panels': 0}
    for n in (1,):
        folder = MANUSCRIPTS / 'latex'
        main = (folder / 'manuscript.tex').read_text()
        protected = {'abstract': main.split('\\section*{Abstract}', 1)[1].split('\\section{Introduction}', 1)[0], 'introduction': main.split('\\section{Introduction}', 1)[1].split('\\section{Results}', 1)[0], 'discussion': (folder / 'discussion.tex').read_text()}
        for name, text in protected.items():
            if re.search('\\\\(?:ref|eqref|autoref|cref|Cref|pageref|input|includegraphics)\\b|\\b(?:Figure|Table)s?\\s*(?:~|\\d)|(?:see|in|above|below)\\s+(?:the\\s+)?Results\\b', text):
                fail(f'Paper {n} {name}: internal result/figure/table reference')
            if re.search('\\\\(?:textbf|textit|emph|textcolor|colorbox|fcolorbox)\\b|\\\\begin\\{(?:itemize|enumerate|description)\\}', text):
                fail(f'Paper {n} {name}: decorative formatting in narrative prose')
            if re.search('\\b(?:supplementary|supplemental|appendix|appendices)\\b|\\$(?!\\$)', text, re.I):
                fail(f'Paper {n} {name}: supplementary pointer or mathematical styling in plain narrative')
        for section in ('methods', 'supplement'):
            text = (folder / f'{section}.tex').read_text()
            for label in re.finditer('\\\\label\\{(eq:[^}]+)\\}', text):
                tail = text[label.end():]
                end = re.search('\\\\end\\{(?:equation|align)\\}', tail)
                if end is None or '\\eqref{' + label[1] + '}' not in tail[end.end():end.end() + 950]:
                    fail(f'Paper {n}: equation has no adjacent explanatory citation: {label[1]}')
                totals['equations'] += 1
        narrative = '\n'.join(((folder / f'{section}.tex').read_text() for section in ('results', 'methods'))) + main
        panel_counts = {1: [3, 3, 3, 3, 6, 2, 2, 3], 2: [3, 3, 1, 2, 2, 2]}[n]
        for number, count in enumerate(panel_counts, 1):
            key = f'fig:p{n}f{number}'
            expected = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:count])
            used = set()
            for reference in re.finditer('\\\\ref\\{' + re.escape(key) + '\\}([A-Z])?(?:--([A-Z]))?', narrative):
                if not reference[1]:
                    fail(f'Paper {n}: figure reference lacks a panel letter: {key}')
                used.update((chr(i) for i in range(ord(reference[1]), ord(reference[2] or reference[1]) + 1)))
            caption = (folder / f'figure_captions/f{number}.tex').read_text()
            described = set(re.findall('\\(([A-Z])\\)', caption))
            if used != expected or described != expected:
                fail(f'Paper {n} figure {number}: panel references {used}, caption descriptions {described}, expected {expected}')
            totals['panels'] += count
        tables = (folder / 'tables.tex').read_text() + (folder / 'supplement.tex').read_text()
        for label in re.findall('\\\\label\\{(tab:[^}]+)\\}', tables):
            if '\\ref{' + label + '}' not in narrative:
                fail(f'Paper {n}: numerical table lacks a main-text citation: {label}')
            totals['tables'] += 1
    print(f"PASS  author style: protected prose sections=3, adjacent equation citations={totals['equations']}, main-text table citations={totals['tables']}, individually described/cited panels={totals['panels']}")

def main():
    import argparse
    argparse.ArgumentParser(description='Verify manuscript prose, equations and numerical anchors').parse_args()
    check_author_style()
    for name, path in PAPER_FILES.items():
        text = read_native(path)
        missing = [value for value in ANCHORS[name] if value not in text]
        if missing:
            fail(f'Missing numerical anchors: {missing}')
        for pattern in AFFIRMATIVE_PATTERNS:
            if pattern.search(text):
                fail('Unsupported affirmative interpretation')
    print('Manuscript scientific-text checks passed')
if __name__ == '__main__':
    main()
