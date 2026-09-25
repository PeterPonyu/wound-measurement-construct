#!/usr/bin/env python3
"""
Data Adapter Layer: real GEO archives -> engine-standard tensors.

This is the layer the ported engines were missing. The engines take plain
float tensors; GEO ships per-sample archives in at least two incompatible
shapes, neither of which is 10x-standard:

- GSE165816: 54 *dense* CSVs (genes x barcodes), one per GSM. Naive
  pd.concat of these is ~24 GB int64 in RAM (452 MB x 54, measured) against
  ~44 GB available, so every sample is converted to CSR int32 on read and the
  dense frame is dropped immediately (~1.8 GB for the full cohort at the
  measured 92.6% sparsity).
- GSE326622: standard 10x mtx/barcodes/features triplets.

Phenotype comes from the series matrix, which is the only authoritative
source for the sample labels; GSE165816 carries `disease` with
DFU-healer / DFU-nonhealer / Non-DFU Diabetic / Non-diabetic, which is the
cohort's only real clinical outcome variable.
"""
import gzip
import io
import os
import re
import tarfile
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread

def _shape(m: Any) -> Tuple[int, int]:
    """scipy's stubs type `.shape` as Optional; every matrix here is 2-D."""
    rows, cols = m.shape
    return (int(rows), int(cols))

@dataclass
class CohortMatrix:
    """Cells x genes counts plus aligned per-cell metadata."""
    X: sp.csr_matrix
    genes: np.ndarray
    cells: np.ndarray
    obs: pd.DataFrame
    notes: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        n_cells, n_genes = _shape(self.X)
        nz = self.X.nnz / (n_cells * n_genes)
        return f"CohortMatrix(cells={n_cells}, genes={n_genes}, density={nz:.3%}, samples={self.obs['gsm'].nunique()})"

def parse_series_matrix(path: str) -> pd.DataFrame:
    """
    Extracts per-sample phenotype from a GEO series_matrix.txt.gz.

    Returns one row per GSM with every `!Sample_characteristics_ch1` key that
    the series actually declares. Callers must check which columns exist
    rather than assuming clinical covariates are present: across GSE165816,
    GSE231643, GSE80178 and GSE255786 the only available keys are
    tissue / disease / cell type / race. There is no age, sex, BMI, HbA1c or
    wound duration anywhere in this cohort family.
    """
    titles: List[str] = []
    gsms: List[str] = []
    chars: List[List[str]] = []
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', errors='ignore') as fh:
        for line in fh:
            if line.startswith('!series_matrix_table_begin'):
                break
            if not line.startswith('!Sample_'):
                continue
            parts = [p.strip().strip('"') for p in line.rstrip('\n').split('\t')]
            tag, values = (parts[0], parts[1:])
            if tag == '!Sample_geo_accession':
                gsms = values
            elif tag == '!Sample_title':
                titles = values
            elif tag == '!Sample_characteristics_ch1':
                chars.append(values)
    if not gsms:
        raise ValueError(f'no !Sample_geo_accession found in {path}')
    df = pd.DataFrame({'gsm': gsms})
    if titles and len(titles) == len(gsms):
        df['title'] = titles
    cols: Dict[str, List[Optional[str]]] = {}
    for row in chars:
        if len(row) != len(gsms):
            continue
        for i, value in enumerate(row):
            if ':' not in value:
                continue
            key, val = value.split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            cols.setdefault(key, [None] * len(gsms))[i] = val.strip()
    for key, values in cols.items():
        df[key] = values
    return df
_GSM_RE = re.compile('(GSM\\d+)')

def _read_dense_csv_member(fobj) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Dense genes x barcodes CSV -> (CSR cells x genes int32, genes, barcodes)."""
    df = pd.read_csv(fobj, index_col=0)
    genes = df.index.to_numpy(dtype=object)
    barcodes = df.columns.to_numpy(dtype=object)
    mat = sp.csr_matrix(df.to_numpy(dtype=np.int32).T)
    del df
    return (mat, genes, barcodes)

def load_dense_csv_tar(tar_path: str, series_matrix: Optional[str]=None, max_samples: Optional[int]=None, gene_policy: str='union', min_sample_frac: float=0.0) -> CohortMatrix:
    """
    Loads a GEO RAW.tar of per-GSM dense count CSVs into one CSR cohort matrix.

    gene_policy controls how the per-sample gene indices are reconciled:

    - "union" (default): a gene missing from a sample is filled with zeros.
      This is the right reading for GSE165816. Its CSVs are dense but their
      gene lists differ (measured: 14,364 to 21,134 genes per sample, union
      26,923, strict intersection only 10,645), which means the submitter
      pre-filtered each sample to the genes *detected in that sample*. Absence
      therefore encodes zero detection, not an unmeasured gene.
    - "intersection": keep only genes present in every sample. Conservative,
      but on this cohort it discards exactly the sample-specific inflammatory
      markers the analysis is about (MMP1, MMP3, MMP13 and IFNG all drop out).

    min_sample_frac additionally requires a gene to appear in at least that
    fraction of samples' indices; 0.0 keeps the full union.
    """
    if gene_policy not in {'union', 'intersection'}:
        raise ValueError(f"gene_policy must be 'union' or 'intersection', got {gene_policy!r}")
    notes: List[str] = []
    per_sample: List[Tuple[str, sp.csr_matrix, np.ndarray, np.ndarray]] = []
    with tarfile.open(tar_path, 'r') as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(('.csv.gz', '.csv'))]
        members.sort(key=lambda m: m.name)
        if max_samples is not None:
            members = members[:max_samples]
        if not members:
            raise ValueError(f'no CSV members in {tar_path}')
        for m in members:
            hit = _GSM_RE.search(m.name)
            gsm = hit.group(1) if hit else m.name
            raw = tar.extractfile(m)
            if raw is None:
                notes.append(f'unreadable member skipped: {m.name}')
                continue
            buf = io.BytesIO(raw.read())
            fobj = gzip.open(buf, 'rb') if m.name.endswith('.gz') else buf
            mat, genes, barcodes = _read_dense_csv_member(fobj)
            per_sample.append((gsm, mat, genes, barcodes))
    return _assemble(per_sample, series_matrix, gene_policy, min_sample_frac, notes)

def _assemble(per_sample: List[Tuple[str, sp.csr_matrix, np.ndarray, np.ndarray]], series_matrix: Optional[str], gene_policy: str, min_sample_frac: float, notes: List[str]) -> CohortMatrix:
    """Reconciles per-sample gene indices and joins phenotype. Shared by both loaders."""
    if gene_policy not in {'union', 'intersection'}:
        raise ValueError(f"gene_policy must be 'union' or 'intersection', got {gene_policy!r}")
    n_samples = len(per_sample)
    occurrences: Dict[object, int] = {}
    for _, _, genes, _ in per_sample:
        for g in genes:
            occurrences[g] = occurrences.get(g, 0) + 1
    if gene_policy == 'intersection':
        keep = {g for g, c in occurrences.items() if c == n_samples}
    else:
        min_count = int(np.ceil(min_sample_frac * n_samples))
        keep = {g for g, c in occurrences.items() if c >= max(min_count, 1)}
    seen, gene_list = (set(), [])
    for _, _, genes, _ in per_sample:
        for g in genes:
            if g in keep and g not in seen:
                seen.add(g)
                gene_list.append(g)
    gene_order = np.array(gene_list, dtype=object)
    master = pd.Index(gene_order)
    n_full = sum((1 for g in keep if occurrences[g] == n_samples))
    notes.append(f'gene_policy={gene_policy}: kept {len(gene_order)} genes (union {len(occurrences)}, present-in-all {n_full}, samples {n_samples})')
    blocks, cells, gsm_col = ([], [], [])
    n_genes = len(gene_order)
    for gsm, mat, genes, barcodes in per_sample:
        pos = master.get_indexer(genes)
        mat = mat.tocsr()
        if gene_policy == 'intersection':
            take = pd.Index(genes).get_indexer(gene_order)
            block = mat[:, take]
        else:
            coo = mat.tocoo()
            sel = pos[coo.col] >= 0
            block = sp.csr_matrix((coo.data[sel], (coo.row[sel], pos[coo.col][sel])), shape=(_shape(mat)[0], n_genes), dtype=np.int32)
            del coo
        blocks.append(block)
        cells.extend((f'{gsm}:{b}' for b in barcodes))
        gsm_col.extend([gsm] * _shape(mat)[0])
    X = sp.csr_matrix(sp.vstack(blocks, format='csr'))
    obs = pd.DataFrame({'gsm': gsm_col})
    if series_matrix:
        pheno = parse_series_matrix(series_matrix)
        obs = obs.merge(pheno, on='gsm', how='left')
        missing = int(obs[pheno.columns[1]].isna().sum()) if len(pheno.columns) > 1 else 0
        if missing:
            notes.append(f'{missing} cells had no phenotype row in the series matrix')
    return CohortMatrix(X=X, genes=gene_order, cells=np.array(cells, dtype=object), obs=obs, notes=notes)

def _read_10x_triplet(features_fh, barcodes_fh, matrix_fh, feature_column: int=1) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """CellRanger triplet -> (CSR cells x genes int32, gene symbols, barcodes)."""
    feats = pd.read_csv(features_fh, sep='\t', header=None, dtype=str)
    col = feature_column if feats.shape[1] > feature_column else 0
    symbols = feats[col].to_numpy(dtype=object)
    barcodes = pd.read_csv(barcodes_fh, sep='\t', header=None, dtype=str)[0].to_numpy(dtype=object)
    mat = sp.coo_matrix(mmread(matrix_fh))
    mat = sp.csr_matrix(mat.T).astype(np.int32)
    uniq, inverse = np.unique(symbols.astype(str), return_inverse=True)
    if uniq.size != symbols.size:
        collapse = sp.csr_matrix((np.ones(inverse.size, dtype=np.int32), (np.arange(inverse.size), inverse)), shape=(symbols.size, uniq.size))
        mat = sp.csr_matrix(mat @ collapse).astype(np.int32)
        symbols = uniq.astype(object)
    return (mat, symbols, barcodes)

def load_10x_mtx_tar(tar_path: str, series_matrix: Optional[str]=None, max_samples: Optional[int]=None, feature_column: int=1, gene_policy: str='union') -> CohortMatrix:
    """
    Loads a GEO RAW.tar of per-GSM CellRanger triplets into one CSR cohort.

    Gene identity is taken from the features file's symbol column (column 1),
    because that is what the GSE165816 panel is keyed on; Ensembl IDs would not
    join across the two cohorts.
    """
    groups: Dict[str, Dict[str, str]] = {}
    with tarfile.open(tar_path, 'r') as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            base = os.path.basename(m.name)
            for kind in ('features', 'barcodes', 'matrix'):
                if kind in base:
                    hit = _GSM_RE.search(base)
                    key = hit.group(1) if hit else base.split('_')[0]
                    groups.setdefault(key, {})[kind] = m.name
                    break
        keys = sorted((k for k, v in groups.items() if {'features', 'barcodes', 'matrix'} <= set(v)))
        if max_samples is not None:
            keys = keys[:max_samples]
        if not keys:
            raise ValueError(f'no complete 10x triplets in {tar_path}')
        per_sample = []
        for gsm in keys:
            g = groups[gsm]

            def member(kind: str):
                fh = tar.extractfile(g[kind])
                if fh is None:
                    raise ValueError(f'unreadable member {g[kind]}')
                return gzip.open(io.BytesIO(fh.read()), 'rb')
            mat, symbols, barcodes = _read_10x_triplet(member('features'), member('barcodes'), member('matrix'), feature_column=feature_column)
            per_sample.append((gsm, mat, symbols, barcodes))
    return _assemble(per_sample, series_matrix, gene_policy, 0.0, [])

def load_10x_mtx_dir(dir_path: str, series_matrix: Optional[str]=None, max_samples: Optional[int]=None, feature_column: int=1, gene_policy: str='union') -> CohortMatrix:
    """
    Same as load_10x_mtx_tar but for loose per-GSM triplets on disk.

    GSE326622 has to be fetched this way: its series-level RAW.tar is truncated
    server-side at ~780 MB of 1,132,769,280 and cannot be retrieved, while the
    per-sample files under geo/samples/ download cleanly. See
    scripts/fetch_gse326622_per_gsm.sh.
    """
    groups: Dict[str, Dict[str, str]] = {}
    for base in sorted(os.listdir(dir_path)):
        for kind in ('features', 'barcodes', 'matrix'):
            if kind in base:
                hit = _GSM_RE.search(base)
                key = hit.group(1) if hit else base.split('_')[0]
                groups.setdefault(key, {})[kind] = os.path.join(dir_path, base)
                break
    keys = sorted((k for k, v in groups.items() if {'features', 'barcodes', 'matrix'} <= set(v)))
    if max_samples is not None:
        keys = keys[:max_samples]
    if not keys:
        raise ValueError(f'no complete 10x triplets in {dir_path}')
    per_sample = []
    for gsm in keys:
        g = groups[gsm]

        def opener(kind: str):
            path = g[kind]
            return gzip.open(path, 'rb') if path.endswith('.gz') else open(path, 'rb')
        mat, symbols, barcodes = _read_10x_triplet(opener('features'), opener('barcodes'), opener('matrix'), feature_column=feature_column)
        per_sample.append((gsm, mat, symbols, barcodes))
    return _assemble(per_sample, series_matrix, gene_policy, 0.0, [])

def load_10x_zip_dir(dir_path: str, series_matrix: Optional[str]=None, max_samples: Optional[int]=None, feature_column: int=1, gene_policy: str='union') -> CohortMatrix:
    """
    Loads per-GSM .zip archives that each wrap one CellRanger triplet.

    GSE241132 ships this way: GSM7717079_PWH26D0.zip contains
    PWH26D0/{matrix.mtx.gz,features.tsv.gz,barcodes.tsv.gz}. The zips were
    built on macOS, so every real entry is shadowed by a `__MACOSX/._name`
    resource fork that must be skipped or it will be mistaken for the data.
    """
    per_sample = []
    zips = sorted((f for f in os.listdir(dir_path) if f.endswith('.zip')))
    if max_samples is not None:
        zips = zips[:max_samples]
    if not zips:
        raise ValueError(f'no .zip archives in {dir_path}')
    for fname in zips:
        hit = _GSM_RE.search(fname)
        gsm = hit.group(1) if hit else fname.split('_')[0]
        with zipfile.ZipFile(os.path.join(dir_path, fname)) as zf:
            members = [n for n in zf.namelist() if not os.path.basename(n).startswith('._') and '__MACOSX' not in n]
            picked = {}
            for kind in ('features', 'barcodes', 'matrix'):
                match = [n for n in members if kind in os.path.basename(n)]
                if not match:
                    raise ValueError(f'{fname}: no {kind} member found')
                picked[kind] = match[0]

            def member(kind: str):
                return gzip.open(io.BytesIO(zf.read(picked[kind])), 'rb')
            mat, symbols, barcodes = _read_10x_triplet(member('features'), member('barcodes'), member('matrix'), feature_column=feature_column)
        per_sample.append((gsm, mat, symbols, barcodes))
    return _assemble(per_sample, series_matrix, gene_policy, 0.0, [])

def library_normalize(X: sp.csr_matrix) -> sp.csr_matrix:
    """Row-normalizes counts to per-cell proportions (the simplex decoder's x)."""
    X = X.astype(np.float32).tocsr()
    totals = np.asarray(X.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    inv = sp.diags((1.0 / totals).astype(np.float32))
    return sp.csr_matrix(inv @ X)
TECHNICAL_GENE_PATTERNS = ('^MT-', '^mt-', '^RP[LS]\\d', '^Rp[ls]\\d', '^MALAT1$', '^Malat1$', '^MT[12][A-Z]$', '^HSP(A1|90A)', '^Hsp(a1|90a)')

def technical_gene_mask(genes: np.ndarray, patterns: Tuple[str, ...]=TECHNICAL_GENE_PATTERNS) -> np.ndarray:
    """Boolean mask marking technical-signal genes in `genes`."""
    rx = re.compile('|'.join(patterns))
    return np.array([bool(rx.match(str(g))) for g in genes], dtype=bool)

def mito_fraction(X: sp.csr_matrix, genes: np.ndarray) -> np.ndarray:
    """Per-cell fraction of counts on mitochondrial genes."""
    rx = re.compile('^(MT-|mt-)')
    mt = np.flatnonzero([bool(rx.match(str(g))) for g in genes])
    totals = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    totals[totals == 0] = 1.0
    if mt.size == 0:
        return np.zeros(_shape(X)[0])
    return np.asarray(X[:, mt].sum(axis=1)).ravel() / totals

def select_target_genes(X: sp.csr_matrix, genes: np.ndarray, n_top: int=7000, pinned: Optional[List[str]]=None, exclude_technical: bool=True, log_normalize: bool=True) -> Tuple[np.ndarray, np.ndarray]:
    Xf = X.astype(np.float32).tocsr()
    if log_normalize:
        totals = np.asarray(Xf.sum(axis=1)).ravel()
        totals[totals == 0] = 1.0
        Xf = sp.csr_matrix(sp.diags((10000.0 / totals).astype(np.float32)) @ Xf)
        Xf.data = np.log1p(Xf.data)
    Xf = Xf.tocsc()
    n = _shape(Xf)[0]
    mean = np.asarray(Xf.mean(axis=0)).ravel()
    sq = np.asarray(Xf.multiply(Xf).sum(axis=0)).ravel() / n
    var = np.maximum(sq - mean ** 2, 0.0)
    disp = var / np.maximum(mean, 1e-08)
    if exclude_technical:
        disp = np.where(technical_gene_mask(genes), -np.inf, disp)
    order = np.argsort(-disp)
    keep = [int(i) for i in order[:n_top] if np.isfinite(disp[i])]
    if pinned:
        pinned_idx = pd.Index(genes).get_indexer(np.array(pinned, dtype=object))
        for gi in pinned_idx:
            if gi >= 0 and int(gi) not in keep:
                keep.append(int(gi))
    idx = np.array(sorted(set(keep)), dtype=int)
    return (idx, genes[idx])
if __name__ == '__main__':
    print('=' * 60)
    print('Self-Testing Data Adapter...')
    import tempfile
    from scipy.io import mmwrite
    rng = np.random.default_rng(0)
    gene_sets = [pd.Index([f'GENE{i}' for i in range(300)]), pd.Index([f'GENE{i}' for i in range(50, 350)])]
    with tempfile.TemporaryDirectory() as td:
        tar_path = os.path.join(td, 'MOCK_RAW.tar')
        with tarfile.open(tar_path, 'w') as tar:
            for s, gsm in enumerate(['GSM000001', 'GSM000002']):
                g = gene_sets[s]
                cols = pd.Index([f'CELL{s}_{j}' for j in range(40)])
                d = pd.DataFrame(rng.poisson(0.4, size=(300, 40)), index=g, columns=cols)
                b = io.BytesIO()
                with gzip.open(b, 'wt') as gz:
                    d.to_csv(gz)
                info = tarfile.TarInfo(f'{gsm}_counts.csv.gz')
                info.size = b.tell()
                b.seek(0)
                tar.addfile(info, b)
        coh = load_dense_csv_tar(tar_path, gene_policy='union')
        print(f'  union      : {coh!r}')
        assert _shape(coh.X) == (80, 350), _shape(coh.X)
        assert sp.issparse(coh.X) and coh.X.dtype == np.int32
        assert coh.obs['gsm'].nunique() == 2
        only_s1 = np.flatnonzero(pd.Index(coh.genes).isin([f'GENE{i}' for i in range(300, 350)]))
        s0_rows = np.flatnonzero((coh.obs['gsm'] == 'GSM000001').to_numpy())
        assert coh.X[s0_rows][:, only_s1].nnz == 0, 'zero-fill leaked nonzeros'
        raw_tot = coh.X.sum()
        inter = load_dense_csv_tar(tar_path, gene_policy='intersection')
        print(f'  intersection: {inter!r}')
        assert _shape(inter.X) == (80, 250), _shape(inter.X)
        assert inter.X.sum() < raw_tot, 'intersection should discard counts'
        print(f'  counts kept: union={raw_tot} intersection={inter.X.sum()}')
        norm = library_normalize(coh.X)
        sums = np.asarray(norm.sum(axis=1)).ravel()
        assert np.allclose(sums, 1.0, atol=1e-05), sums[:3]
        print(f'  library_normalize: row sums = {sums[:3]} (all 1.0)')
        idx, panel = select_target_genes(coh.X, coh.genes, n_top=50, pinned=['GENE299', 'GENE0'])
        assert 'GENE299' in set(panel) and 'GENE0' in set(panel)
        assert len(idx) == len(set(idx.tolist()))
        print(f'  select_target_genes: {len(panel)} genes, pinned present')
        tech_genes = np.array(['MT-CO1', 'RPL10', 'MALAT1', 'IL1B', 'KRT14'], dtype=object)
        mask = technical_gene_mask(tech_genes)
        assert mask.tolist() == [True, True, True, False, False], mask
        tiny = sp.csr_matrix(np.array([[10, 0, 0, 90], [0, 0, 50, 50]], dtype=np.int32))
        mf = mito_fraction(tiny, np.array(['MT-CO1', 'MT-ND1', 'ACTB', 'GAPDH'], dtype=object))
        assert np.allclose(mf, [0.1, 0.0]), mf
        print(f'  technical_gene_mask + mito_fraction: ok (mito {mf})')
    with tempfile.TemporaryDirectory() as td:
        tar_path = os.path.join(td, 'MOCK10X_RAW.tar')
        n_g, n_c = (6, 5)
        symbols = ['AAA', 'BBB', 'DUP', 'DUP', 'CCC', 'MT-CO1']
        dense = np.arange(1, n_g * n_c + 1, dtype=np.int32).reshape(n_g, n_c)
        with tarfile.open(tar_path, 'w') as tar:
            for gsm in ['GSM111111', 'GSM222222']:

                def add(suffix: str, payload: bytes):
                    info = tarfile.TarInfo(f'{gsm}_S_{suffix}')
                    info.size = len(payload)
                    tar.addfile(info, io.BytesIO(payload))
                feats = '\n'.join((f'ENSG{i:05d}\t{s}\tGene Expression' for i, s in enumerate(symbols))) + '\n'
                b = io.BytesIO()
                with gzip.open(b, 'wt') as gz:
                    gz.write(feats)
                add('features.tsv.gz', b.getvalue())
                b = io.BytesIO()
                with gzip.open(b, 'wt') as gz:
                    gz.write('\n'.join((f'BC{j}-1' for j in range(n_c))) + '\n')
                add('barcodes.tsv.gz', b.getvalue())
                raw = io.BytesIO()
                mmwrite(raw, sp.coo_matrix(dense), field='integer')
                b = io.BytesIO()
                with gzip.open(b, 'wb') as gz:
                    gz.write(raw.getvalue())
                add('matrix.mtx.gz', b.getvalue())
        tenx = load_10x_mtx_tar(tar_path)
        print(f'  10x loader : {tenx!r}')
        assert _shape(tenx.X) == (2 * n_c, 5), _shape(tenx.X)
        assert tenx.X.sum() == 2 * dense.sum(), (tenx.X.sum(), 2 * dense.sum())
        dup_col = int(np.flatnonzero(tenx.genes == 'DUP')[0])
        expected_dup = dense[2] + dense[3]
        got = np.asarray(tenx.X[:n_c, dup_col].todense()).ravel()
        assert np.array_equal(got, expected_dup), (got, expected_dup)
        print(f'  duplicate symbols summed, not dropped: DUP = {got.tolist()}')
        zdir = os.path.join(td, 'zips')
        os.makedirs(zdir, exist_ok=True)
        for gsm in ['GSM111111', 'GSM222222']:
            with zipfile.ZipFile(os.path.join(zdir, f'{gsm}_S.zip'), 'w') as zf:
                for suffix in ('features.tsv.gz', 'barcodes.tsv.gz', 'matrix.mtx.gz'):
                    with tarfile.open(tar_path) as t:
                        fh_member = t.extractfile(f'{gsm}_S_{suffix}')
                        assert fh_member is not None
                        payload = fh_member.read()
                    zf.writestr(f'S/S_{suffix}', payload)
                    zf.writestr(f'__MACOSX/S/._S_{suffix}', b'resource-fork-junk')
        zcoh = load_10x_zip_dir(zdir)
        print(f'  zip loader : {zcoh!r}')
        assert _shape(zcoh.X) == _shape(tenx.X), (_shape(zcoh.X), _shape(tenx.X))
        assert zcoh.X.sum() == tenx.X.sum(), (zcoh.X.sum(), tenx.X.sum())
        print('  zip path matches tar path exactly; __MACOSX forks ignored')
    print('Data Adapter test PASSED!')
    print('=' * 60)
