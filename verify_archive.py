#!/usr/bin/env python3
"""Verify the full-source archive and optional safe native-analysis smoke."""
import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    entries=json.loads((ROOT/'MANIFEST.sha256.json').read_text())
    for name,digest in entries.items():
        rel=PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or rel.as_posix()!=name:
            raise ValueError('Unsafe manifest entry')
        target=ROOT/name
        if not target.is_file() or target.is_symlink() or sha(target)!=digest:
            raise ValueError('Missing or changed archive file: '+name)
        if target.suffix=='.py':ast.parse(target.read_text(),filename=name)
    cff=yaml.safe_load((ROOT/'CITATION.cff').read_text())
    zenodo=json.loads((ROOT/'.zenodo.json').read_text())
    status=json.loads((ROOT/'RELEASE_STATUS.json').read_text())
    assert cff['title']==zenodo['title']
    assert str(cff['version'])==str(zenodo['version'])==status['version']=='0.5.0'
    assert cff['doi']==zenodo['doi']==status['doi']
    assert status['doi_status']=='reserved_at_archive_assembly'
    assert status['doi_scope']=='Software and reproduction release; not a manuscript or preprint DOI'
    journal=json.loads((ROOT/'journal/JOURNAL_SOURCE_PROVENANCE.json').read_text())
    for row in journal['files']:
        target=ROOT/'journal'/row['destination']
        if not target.is_file() or sha(target)!=row['sha256']:
            raise ValueError('Corrected journal source binding differs: '+row['destination'])
    if args.smoke:
        subprocess.run([sys.executable,'analysis_revision/verify_revision.py','--smoke'],cwd=ROOT,check=True)
    print(json.dumps({'status':'passed','manifest_files':len(entries),
          'corrected_journal_source_files':len(journal['files']),
          'reserved_software_doi':status['doi'],'smoke_requested':args.smoke,
          'model_training_performed':False,'remote_publication_performed':False},indent=2))


if __name__=='__main__':main()
