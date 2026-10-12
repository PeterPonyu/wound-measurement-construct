#!/usr/bin/env python3
"""Build journal derivatives from the audited full manuscript without refitting.

The explicit display map is in scientific-reports.json beside the scripts folder.
Python's standard library, latexmk, TeXcount, pdftotext and pdfinfo are required.
No network, dataset acquisition or research-output mutation is performed.
"""
from __future__ import annotations
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
CONFIG = BASE / 'scientific-reports.json'
FLOAT = re.compile(r'\\begin\{(figure|table)\}.*?\\end\{\1\}', re.S)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def floats(text):
    entries = []
    for m in FLOAT.finditer(text):
        label = re.search(r'\\label\{([^}]+)\}', m[0])
        entries.append({'kind': m[1], 'label': label[1] if label else None,
                        'source': m[0]})
    return entries


def rewrite_refs(text, moved):
    for label, (kind, number) in moved.items():
        word = 'Fig.' if kind == 'figure' else 'Table'
        replacement = f'Supplementary {word}~S{number}'
        for macro in ('panelref', 'figpanel'):
            text = re.sub(r'\\' + macro + r'\{' + re.escape(label) + r'\}\{[^}]+\}',
                          lambda m: replacement, text)
        text = re.sub(r'(?:Supplementary\s+)?(?:Fig\.?|Figure|Tables?)~?\s*\\ref\{' + re.escape(label) + r'\}'
                      + (r'[a-zA-Z]?(?![A-Za-z])' if kind == 'figure' else ''),
                      lambda m: replacement, text)
        text = text.replace(r'\ref{' + label + '}', 'S' + str(number))
    # Journal policy cites the whole supplementary item in the main text.
    text = re.sub(r'\\suppanelref\{(S\d+)\}\{[^}]+\}',
                  lambda m: 'Supplementary Fig.~' + m[1], text)
    return text


def take_author_contributions(text):
    """Keep the contribution paragraph, placing it after the references."""
    heading = (r'(?:\\section\*\{Author contributions\}|'
               r'\\noindent\\textbf\{Author contributions\}\\par\\nobreak)')
    match = re.search(heading + r'\s*([^\n]+)', text)
    if not match:
        raise ValueError('Missing author contributions paragraph')
    paragraph = match[1].strip()
    return text[:match.start()] + text[match.end():], (
        r'\section*{Author contributions}' + '\n' + paragraph + '\n\n')


def apply_overrides(text, overrides):
    """Sentence-level edits that automatic whole-item rewriting cannot express.

    Each override must match exactly once, so a changed canonical sentence fails loudly.
    """
    for item in overrides:
        found = text.count(item['find'])
        if found != 1:
            raise ValueError(f'text override matched {found} times: {item["find"][:80]!r}')
        text = text.replace(item['find'], item['replace'])
    return text


def journal_equation_style(text):
    """The journal asks for 'equation (n)' in running text."""
    text = re.sub(r'(?<=[.!?] )Eq\.~(?=\\eqref)', 'Equation~', text)
    return re.sub(r'Eq\.~(?=\\eqref)', 'equation~', text)


def citation_pattern(label):
    return re.compile(r'\\(?:panelref|figpanel)\{' + re.escape(label) + r'\}|\\ref\{' + re.escape(label) + r'\}')


def place_inline(text, kept, spec='tbp'):
    """Float every kept display right after the paragraph that first cites it.

    Display numbers follow source order, so the first-citation order must equal the configured order
    (otherwise the numbers quoted in the supplement would no longer match).
    """
    limit = min(i for i in (text.find(r'\bibliographystyle'), text.find(r'\bibliography{'),
                            text.find(r'\printbibliography')) if i != -1)
    slots = []
    for item in kept:
        match = citation_pattern(item['label']).search(text, 0, limit)
        if not match:
            raise ValueError(f"{item['label']} is never cited before the references")
        end = text.find('\n\n', match.end())
        if end == -1 or end > limit:
            raise ValueError(f"no paragraph end after the first citation of {item['label']}")
        slots.append((match.start(), end, item))
    for kind in ('figure', 'table'):
        cited = [i['label'] for _, _, i in sorted(slots, key=lambda s: s[0]) if i['kind'] == kind]
        configured = [i['label'] for i in kept if i['kind'] == kind]
        if cited != configured:
            raise ValueError(f'first-citation order of {kind}s {cited} differs from display order {configured}')
    for start, end, item in sorted(slots, key=lambda s: (s[1], s[0]), reverse=True):
        source = re.sub(r'\\begin\{(figure|table)\}(?:\[[^]]*\])?',
                        lambda m: r'\begin{' + m[1] + '}[' + spec + ']', item['source'], count=1)
        text = text[:end] + '\n\n' + source + text[end:]
    return text


def audit_inline_layout(pdf, kept, max_pages_after=2):
    """Page of first citation versus page of each displayed figure/table in the compiled PDF.

    A display may sit on the page of its first citation or up to max_pages_after pages later (tall figures queue),
    never earlier, and displays must appear in numerical order.
    """
    pages = subprocess.run(['pdftotext', '-layout', str(pdf), '-'], text=True, capture_output=True,
                           check=True).stdout.split('\f')
    pages = [p for p in pages if p.strip()]
    record, counters = {}, {'figure': 0, 'table': 0}
    for item in kept:
        counters[item['kind']] += 1
        number = counters[item['kind']]
        word = 'Figure' if item['kind'] == 'figure' else 'Table'
        words = r'(?:Fig\.|Figure)' if item['kind'] == 'figure' else 'Table'
        caption = re.compile(r'(?m)^[ \t]*' + word + r'\s+' + str(number) + r'\s*[:.|]')
        cite = re.compile(r'(?<![A-Za-z])' + words + r'\s*' + str(number) + r'(?![0-9])')
        caption_pages = [i + 1 for i, p in enumerate(pages) if caption.search(p)]
        cite_pages = [i + 1 for i, p in enumerate(pages) if cite.search(caption.sub('', p))]
        if len(caption_pages) != 1 or not cite_pages:
            raise ValueError(f"{item['label']}: caption pages {caption_pages}, citation pages {cite_pages}")
        record[item['label']] = {'display': f'{word} {number}', 'first_citation_page': cite_pages[0],
                                 'display_page': caption_pages[0],
                                 'pages_after_first_citation': caption_pages[0] - cite_pages[0]}
        if not 0 <= caption_pages[0] - cite_pages[0] <= max_pages_after:
            raise ValueError(f"{item['label']} is displayed {caption_pages[0] - cite_pages[0]} pages after its first "
                             f"citation (page {cite_pages[0]}); the limit is {max_pages_after}")
    for kind, word in (('figure', 'Figure'), ('table', 'Table')):
        pages_in_order = [v['display_page'] for k, v in record.items() if v['display'].startswith(word)]
        if pages_in_order != sorted(pages_in_order):
            raise ValueError(f'{kind}s are not displayed in numerical order: {pages_in_order}')
    return record


def main_text_word_counts(text, qa):
    """Count only Introduction, Results and Discussion, excluding Methods."""
    counts = {}
    for section in ('Introduction', 'Results', 'Discussion'):
        match = re.search(r'\\section\{' + section + r'\}(.*?)'
                          r'(?=\\section\*?\{|\\bibliograph|\\end\{document\})', text, re.S)
        if not match:
            raise ValueError('Missing main-text section: ' + section)
        source = qa / ('word-count-' + section.lower() + '.tex')
        source.write_text(match[1])
        run = subprocess.run(['texcount', '-sum=1,0,0,0,0,0,0', '-brief', str(source)],
                             text=True, capture_output=True, check=True)
        (qa / ('word-count-' + section.lower() + '.txt')).write_text(run.stdout + run.stderr)
        counts[section.lower()] = int(re.match(r'\s*(\d+)', run.stdout)[1])
    return counts


def journal_display_style(text, cfg):
    if cfg.get('journal_figure_directory'):
        # Explicit paths also route figures whose original graphicspath lives in an input preamble.
        text = re.sub(r'(\\includegraphics(?:\[[^\]]*\])?\{)([^}]+)(\})',
                      lambda m: m[1] + 'journal-figures/' + Path(m[2]).name + m[3], text)
        text = text.replace('{figures/', '{journal-figures/')
        text = text.replace('\\graphicspath{{figures/}', '\\graphicspath{{journal-figures/}{figures/}')
    if cfg.get('journal_lowercase_panels'):
        text = re.sub(r'(\\ref\{fig:[^}]+\})([A-Z](?:--[A-Z])?)', lambda m: m[1]+m[2].lower(), text)
        text = re.sub(r'((?:Figures?|Figs?\.?)~?\s*\d+)([A-H](?:--[A-H])?)',
                      lambda m: m[1] + m[2].lower(), text)
        text = re.sub(r'(\\ref\{fig:[^}]+\}[a-h](?:--[a-h])?)([\s~]+and[\s~]+)([A-H])',
                      lambda m: m[1] + m[2] + m[3].lower(), text)
        text = re.sub(r'((?:Figures?|Figs?\.?)~?\s*\d+[a-h](?:--[a-h])?)([\s~]+and[\s~]+)([A-H])',
                      lambda m: m[1] + m[2] + m[3].lower(), text)
        text = re.sub(r'(colors match )([A-H])\b', lambda m: m[1]+m[2].lower(), text)
        # These grouped panel phrases are distinct from biological B/plasma
        # terminology and donor labels such as Other A/B.
        for original, replacement in {
            'D/E retain': 'd/e retain', 'A and D have': 'a and d have',
            'A--C show': 'a--c show', 'E, F and H describe': 'e, f and h describe',
            'B and D are': 'b and d are', 'from A--C': 'from a--c',
            'E/F distinguish': 'e/f distinguish',
        }.items():
            text = text.replace(original, replacement)
        def caption(match):
            return re.sub(r'\(([A-Z](?:--[A-Z])?)\)', lambda x: '('+x[1].lower()+')', match[0])
        text = re.sub(r'\\caption\{[^\n]+', caption, text)
    if cfg.get('journal_ledger_directory'):
        text = text.replace('{ledgers/', '{journal-ledgers/')
    return text


def main():
    cfg = json.loads(CONFIG.read_text())
    main_path = BASE / cfg['main_source']
    original = main_path.read_text()
    supp_path = BASE / cfg['supplement_source'] if cfg['supplement_source'] else None
    original_supp = supp_path.read_text() if supp_path else ''
    main_items, supp_items = floats(original), floats(original_supp)
    kept = [x for x in main_items if x['label'] in cfg['main_' + x['kind'] + 's']]
    moved_items = [x for x in main_items if x not in kept]
    assert {x['label'] for x in kept} == set(cfg['main_figures'] + cfg['main_tables'])
    assert len(kept) <= 8, 'Scientific Reports permits eight main display items'
    nfig = sum(x['kind'] == 'figure' for x in supp_items)
    ntab = sum(x['kind'] == 'table' for x in supp_items)
    moved = {}
    for item in moved_items:
        if item['kind'] == 'figure':
            nfig += 1
            number = nfig
        else:
            ntab += 1
            number = ntab
        moved[item['label']] = (item['kind'], number)
    supp_refs = {}
    counters = {'figure': 0, 'table': 0}
    for block in re.finditer(r'\\begin\{(figure|table|longtable)\}.*?\\end\{\1\}', original_supp, re.S):
        kind = 'figure' if block[1] == 'figure' else 'table'
        counters[kind] += 1
        label = re.search(r'\\label\{([^}]+)\}', block[0])
        if label:
            supp_refs[label[1]] = (kind, counters[kind])
    # Tables spread across pages are valid supplementary tables.
    ntab = counters['table']
    if ntab != sum(x['kind'] == 'table' for x in supp_items):
        for item in moved_items:
            if item['kind'] == 'table':
                ntab += 1
                moved[item['label']] = ('table', ntab)
    else:
        ntab += sum(x['kind'] == 'table' for x in moved_items)
    text = rewrite_refs(FLOAT.sub('', original), {**supp_refs, **moved})
    text = apply_overrides(text, cfg.get('text_overrides', []))
    if cfg.get('equation_style'):
        text = journal_equation_style(text)
    extended_discussion = ''
    if cfg.get('discussion_sentence_limit'):
        discussion = re.search(r'\\section\{Discussion\}(.*?)(?=\\section\*|\\section\{|\\end\{document\})', text, re.S)
        if discussion:
            extended_discussion = (r'\section*{Supplementary Note: Extended discussion}' + discussion[1])
            paragraphs = re.split(r'\n\s*\n', discussion[1].strip())
            shortened = '\n\n'.join(' '.join(re.split(r'(?<=[.!?])\s+(?=[A-Z\\])', p)[:cfg['discussion_sentence_limit']])
                                      for p in paragraphs)
            text = text[:discussion.start(1)] + '\n' + shortened + '\n\n' + text[discussion.end(1):]
    title = cfg['title']
    assert len(title.split()) <= 20 or cfg.get('preserve_author_title'), 'Title exceeds 20 words'
    text = re.sub(r'\\title\{[^\n]+\}', lambda m: r'\title{' + title + '}', text)
    text = re.sub(r'pdftitle=\{[^}]+\}', lambda m: 'pdftitle={' + title + '}', text)
    if r'\section*{Data and code availability}' in text:
        text = text.replace(r'\section*{Data and code availability}', r'\section*{Data Availability}')
        text = text.replace(r'\section*{Competing interests and funding}', r'\section*{Additional Information}')
        # The original author footnote becomes a title-page affiliation.
        text = re.sub(r'\\author\{Zeyu~Fu.*?\}\n\\date',
                      lambda m: r'\author{Zeyu~Fu$^{1,*}$\,\orcidlink{0009-0001-8329-0108}\\[3pt]'
                      '\n' + r'\small $^1$State Key Laboratory of Trauma and Chemical Poisoning,\\'
                      '\n' + r'\small Institute of Combined Injury, Chongqing Engineering Research Center for Nanomedicine,\\'
                      '\n' + r'\small College of Preventive Medicine, Army Medical University, Chongqing, China\\'
                      '\n' + r'\small $^*$Correspondence: \texttt{fuzeyu99@126.com}}' + '\n' + r'\date', text, flags=re.S)
        ethics = re.search(r'\\section\*\{Ethics statement\}(.*?)(?=\\section)', text, re.S)
        if ethics:
            text = text[:ethics.start()] + text[ethics.end():]
            text = text.replace(r'\section{Results}', r'\subsection{Ethics and computational assistance}'
                                + ethics[1] + '\nAI-assisted tools supported code development, computational checks and manuscript editing. '
                                'The author is responsible for the scientific interpretation and final text.\n\n'
                                + r'\section{Results}', 1)
    else:
        text = text.replace(r'\noindent\textbf{Data and code availability}\par\nobreak', r'\section*{Data Availability}')
        text = text.replace(r'\noindent\textbf{Competing interests}\par\nobreak', r'\section*{Additional Information}')
    text, contributions = take_author_contributions(text)
    # Two layouts of the same prose. The checklist layout supplies contributions and main legends
    # after the references; the reading layout floats each display at its first citation.
    end_displays = [re.sub(r'\\begin\{(figure|table)\}(?:\[[^]]*\])?',
                          lambda m: r'\begin{' + m[1] + '}[!htbp]', x['source']) for x in kept]
    inline_text = place_inline(text.replace(r'\end{document}', '\n' + contributions + r'\end{document}'),
                               kept, cfg.get('inline_float_spec', 'tbp'))
    if cfg.get('inline_vfuzz_pt'):   # sub-point overfull vboxes in the output routine are not visible defects
        inline_text = inline_text.replace(r'\begin{document}', r'\vfuzz=' + str(cfg['inline_vfuzz_pt']) + 'pt\n'
                                          + r'\begin{document}', 1)
    text = text.replace(r'\end{document}', '\n' + contributions + r'\clearpage' + '\n' +
                        r'\section*{Figures and tables}' + '\n' +
                        '\n\n'.join(end_displays) + '\n' + r'\end{document}')
    if original_supp:
        supplementary = original_supp.replace(r'\end{document}', '\n' +
                         r'\clearpage\section*{Additional supporting displays}' + '\n' +
                         '\n\n'.join(x['source'] for x in moved_items) + '\n' + r'\end{document}')
        supplementary = re.sub(r'\\title\{[^\n]+\}', lambda m: r'\title{' + title + '}', supplementary)
        supplementary = supplementary.replace(r'\begin{document}', r'\begin{document}' + '\n' +
                                               r'\renewcommand{\thefigure}{S\arabic{figure}}' + '\n', 1)
    else:
        preamble = original.split(r'\begin{document}')[0]
        supplementary = (preamble + r'\begin{document}\maketitle' + '\n' +
            r'\section*{Supplementary Information}' + '\n' +
            r'\renewcommand{\thefigure}{S\arabic{figure}}\renewcommand{\thetable}{S\arabic{table}}' + '\n' +
            '\n\n'.join(x['source'] for x in moved_items) + '\n' + r'\end{document}')
    supplementary = re.sub(r'\\title\{[^\n]+\}', lambda m: r'\title{' + title + '}', supplementary)
    supplementary = re.sub(r'pdftitle=\{[^}]+\}', lambda m: 'pdftitle={' + title + '}', supplementary)
    # Cross references inside moved captions are uncommon but must remain local.
    supplementary = rewrite_refs(supplementary, moved)
    if extended_discussion:
        bib = re.search(r'\\bibliographystyle\{[^}]+\}\s*\\bibliography\{[^}]+\}', original)
        supplementary = supplementary.replace(r'\end{document}', '\n' + extended_discussion + '\n' +
                                                (bib[0] if bib else '') + '\n' + r'\end{document}')
    main_numbers = {}
    for kind in ('figure', 'table'):
        main_numbers.update({x['label']: str(i + 1) for i, x in enumerate(y for y in kept if y['kind'] == kind)})
    for label, number in main_numbers.items():
        supplementary = supplementary.replace(r'\ref{' + label + '}', number)
    if re.search(r'\\cite\w*\{', supplementary) and r'\bibliography{' not in supplementary:
        bib = re.search(r'\\bibliographystyle\{[^}]+\}\s*\\bibliography\{[^}]+\}', original)
        if bib:
            supplementary = supplementary.replace(r'\end{document}', '\n' + bib[0] + '\n' + r'\end{document}')
    supplementary = supplementary.replace(r'\bibliographystyle{',
                                           r'\FloatBarrier' + '\n' + r'\bibliographystyle{')
    if original_supp:
        supplementary = re.sub(r'\\bibliographystyle\{[^}]+\}\s*\\bibliography\{[^}]+\}',
            lambda m: r'\begingroup\setlength{\bibsep}{0pt}' + '\n' + m[0] + '\n' + r'\endgroup',
            supplementary)
    folder = main_path.parent
    sources = {'manuscript': folder / 'paper_srep.tex', 'supplement': folder / 'supporting_srep.tex',
               'manuscript-inline': folder / 'paper_srep_inline.tex'}
    text = journal_display_style(text, cfg)
    supplementary = journal_display_style(supplementary, cfg)
    inline_text = journal_display_style(inline_text, cfg)
    sources['manuscript'].write_text(text)
    sources['supplement'].write_text(supplementary)
    sources['manuscript-inline'].write_text(inline_text)
    output = (BASE / cfg['output']).resolve()
    qa = (BASE / cfg['qa']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    qa.mkdir(parents=True, exist_ok=True)
    section_counts = main_text_word_counts(text, qa)
    assert text.index(r'\section*{Data Availability}') < text.index(r'\bibliography{')
    assert text.index(r'\section*{Author contributions}') > text.index(r'\bibliography{')
    assert text.index(r'\section*{Figures and tables}') > text.index(r'\section*{Author contributions}')
    assert text.index(r'\section*{Data Availability}') < inline_text.index(r'\bibliography{') < inline_text.index(r'\section*{Author contributions}')
    assert r'\section*{Figures and tables}' not in inline_text
    for item in kept:
        label = r'\label{' + item['label'] + '}'
        assert text.count(label) == 1 and inline_text.count(label) == 1, item['label']
        assert inline_text.index(label) > citation_pattern(item['label']).search(inline_text).start()
    assert not re.search(r'\\(?:footnote|thanks)\b', text)
    assert re.search(r'\\subsection\{Ethics and computational assistance\}', text)
    assert 'AI-assisted tools' in text
    assert re.search(r'\\section\*\{Additional Information\}', text)
    for source in (text, supplementary):
        assert r'\title{' + title + '}' in source
    for match in re.finditer(r'(?:Fig(?:ure)?\.?|Tables?)~?\s*S\d+\b', text):
        assert 'Supplementary' in text[max(0, match.start() - 18):match.start()], match[0]
    assert 'Supplementary Supplementary' not in text
    documents = {}
    for name, src in sources.items():
        if name == 'supplement':
            # Extended supporting prose can refer to numbered main equations.
            # Resolve their values from the just-built main aux, without a
            # machine-specific external-document path in the distributed TeX.
            aux = (qa / 'build-manuscript' / 'paper_srep.aux').read_text()
            refs = dict(re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}', aux))
            for label, value in refs.items():
                if re.search(r'\\label\{' + re.escape(label) + r'\}', supplementary):
                    continue
                supplementary = supplementary.replace(r'\eqref{' + label + '}', '(' + value + ')')
                supplementary = supplementary.replace(r'\ref{' + label + '}', value)
            src.write_text(supplementary)
        build = qa / ('build-' + name)
        build.mkdir(exist_ok=True)
        command = ['latexmk', '-' + cfg.get('compiler', 'pdf'), '-interaction=nonstopmode', '-halt-on-error',
                   '-outdir=' + str(build), src.name]
        run = subprocess.run(command, cwd=folder, text=True, capture_output=True)
        (qa / (name + '-build.txt')).write_text(run.stdout + run.stderr)
        if run.returncode:
            raise RuntimeError(f'{name} compilation failed; see {qa}')
        log = (build / (src.stem + '.log')).read_text(errors='replace')
        defects = re.findall(r'Overfull[^\n]*|LaTeX Warning: (?:Float too large|.*undefined)[^\n]*|There were undefined[^\n]*', log)
        if defects:
            raise RuntimeError(f'{name}: {defects}')
        target = output / ('scientific-reports-' + name + '.pdf')
        shutil.copy2(build / (src.stem + '.pdf'), target)
        extracted = subprocess.run(['pdftotext', '-layout', str(target), '-'], text=True, capture_output=True, check=True).stdout
        if name.startswith('manuscript'):
            abstract = extracted.split('Abstract', 1)[1].split('Keywords:', 1)[0] if 'Keywords:' in extracted else extracted.split('Abstract', 1)[1].split('Introduction', 1)[0]
            abstract = re.sub(r'(?m)^\s*\d+\s*$', '', abstract).strip()
            abstract_words = len(abstract.split())
            assert 100 <= abstract_words <= 200, f'Abstract has {abstract_words} words'
            assert not re.search(r'\\(?:footnote|thanks)\b', text), 'Journal main text must not contain footnotes'
        info = subprocess.run(['pdfinfo', str(target)], text=True, capture_output=True, check=True).stdout
        pages = int(re.search(r'Pages:\s*(\d+)', info)[1])
        documents[name] = {'path': str(target), 'sha256': sha(target), 'pages': pages,
                           'source_sha256': sha(src), 'bytes': target.stat().st_size}
    inline_layout = audit_inline_layout(Path(documents['manuscript-inline']['path']), kept, cfg.get('inline_max_pages_after_citation', 2))
    report = {'status': 'PASS', 'title': title, 'title_words': len(title.split()),
              'abstract_words': abstract_words, 'main_figures': len(cfg['main_figures']),
              'main_tables': len(cfg['main_tables']), 'supplementary_figures': nfig,
              'supplementary_tables': ntab, 'moved_displays': moved,
              'main_text_words': sum(section_counts.values()),
              'main_text_section_words': section_counts,
              'main_text_count_method': 'TeXcount prose words in Introduction, Results and Discussion; excludes Abstract, Methods, references, headings, equations and figure legends.',
              'main_text_recommendation_met': sum(section_counts.values()) <= 4500,
              'title_recommendation_met': len(title.split()) <= 20,
              'title_page_and_declaration_order': 'PASS',
              'supplement_title_matches': True,
              'canonical_source_sha256': sha(main_path), 'canonical_supplement_sha256': sha(supp_path) if supp_path else None,
              'configuration_sha256': sha(CONFIG), 'generator_sha256': sha(Path(__file__)),
              'documents': documents, 'inline_layout': inline_layout,
              'layouts': {'manuscript': 'journal-checklist layout: legends, figures and the table follow the references',
                          'manuscript-inline': 'reading layout: each figure and table floats after the paragraph that first cites it'},
              'scope': 'Journal document conversion and mechanical checks; no new experiments or independent peer review.',
              'guidelines': 'https://www.nature.com/srep/author-instructions/submission-guidelines',
              'checklist': 'https://www.nature.com/documents/srep-checklist-for-initial-submissions.pdf'}
    (qa / 'format-check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:report[k] for k in ['status','title_words','abstract_words','main_figures','main_tables','supplementary_figures','supplementary_tables']}))


if __name__ == '__main__':
    main()
