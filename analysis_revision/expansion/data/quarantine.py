"""Bounded, allowlist-based archive inspection and extraction.

Archives are inspected completely before any selected member is written.  No
``extractall`` or TAR ``getmembers`` call is used.  An explicit file allowlist
and a new, non-symlinked destination are required.  Only selected regular files
are copied, with byte caps and SHA-256 receipts.  Inventory is a metadata check,
not a content/CRC certification of unselected ZIP members.
This is a staging boundary only; it does not certify a dataset's scientific
identity or Stage-A contract.
"""
from __future__ import annotations

import bz2
import gzip
import hashlib
import lzma
import math
import os
import re
import shutil
import stat
import struct
import tarfile
import zipfile
import zlib
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class ArchiveSafetyError(ValueError):
    """An archive or extraction request violates a quarantine invariant."""


@dataclass(frozen=True)
class ArchiveLimits:
    max_members: int = 100_000
    max_member_bytes: int = 2 * 1024 * 1024 * 1024
    max_total_bytes: int = 8 * 1024 * 1024 * 1024
    max_compression_ratio: float = 200.0
    chunk_bytes: int = 1 << 20
    reject_nested_archives: bool = True
    max_metadata_bytes: int = 16 << 20

    def __post_init__(self):
        for field in ("max_members", "max_member_bytes", "max_total_bytes", "chunk_bytes",
                      "max_metadata_bytes"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 < value < (1 << 63):
                raise ArchiveSafetyError(f"{field} must be a positive bounded integer")
        if self.chunk_bytes > 16 << 20 or self.max_metadata_bytes > 64 << 20:
            raise ArchiveSafetyError("chunk/metadata read limits exceed the bounded allocation cap")
        ratio = self.max_compression_ratio
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise ArchiveSafetyError("max_compression_ratio must be finite and positive")
        try:
            valid_ratio = math.isfinite(ratio) and ratio > 0
        except OverflowError:
            valid_ratio = False
        if not valid_ratio:
            raise ArchiveSafetyError("max_compression_ratio must be finite and positive")
        if not isinstance(self.reject_nested_archives, bool):
            raise ArchiveSafetyError("reject_nested_archives must be boolean")


_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")
_DRIVE = re.compile(r"^[A-Za-z]:")


def _member_path(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ArchiveSafetyError(f"unsafe archive member name: {name!r}")
    if name.startswith("/") or _DRIVE.match(name):
        raise ArchiveSafetyError(f"absolute archive member name: {name!r}")
    normalized = name[:-1] if name.endswith("/") else name
    parts = normalized.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ArchiveSafetyError(f"non-normalized archive member name: {name!r}")
    if any(":" in part for part in parts):
        raise ArchiveSafetyError(f"drive or alternate-stream archive member name: {name!r}")
    try:
        name_bytes = normalized.encode("utf-8")
    except UnicodeError as error:
        raise ArchiveSafetyError(f"invalid archive member encoding: {name!r}") from error
    if len(name_bytes) > 4096 or len(parts) > 128:
        raise ArchiveSafetyError("archive member path exceeds length/depth cap")
    return normalized


def _is_nested(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(_ARCHIVE_SUFFIXES)


class _MemberIndex:
    """Track explicit and implicit directories without a quadratic prefix scan."""

    def __init__(self, limits: ArchiveLimits):
        self.limits = limits
        self.seen: dict[str, bool] = {}
        self.parents: set[str] = set()
        self.name_bytes = 0
        self.total = 0

    def add(self, record: dict) -> None:
        name, is_dir = record["name"], record["is_dir"]
        if len(self.seen) >= self.limits.max_members:
            raise ArchiveSafetyError("archive member count exceeds cap")
        if name in self.seen:
            raise ArchiveSafetyError(f"duplicate normalized archive member: {name}")
        if not is_dir and name in self.parents:
            raise ArchiveSafetyError(f"file/member-prefix collision: {name}")
        parents = [str(parent) for parent in PurePosixPath(name).parents if str(parent) != "."]
        for parent in parents:
            if self.seen.get(parent) is False:
                raise ArchiveSafetyError(f"file/member-prefix collision: {parent} and {name}")
        self.name_bytes += len(name.encode("utf-8")) + sum(
            len(parent.encode("utf-8")) for parent in parents if parent not in self.parents
        )
        if self.name_bytes > self.limits.max_metadata_bytes:
            raise ArchiveSafetyError("archive name metadata exceeds cap")
        size = record["size"]
        if size < 0 or size > self.limits.max_member_bytes:
            raise ArchiveSafetyError(f"member exceeds size cap or has negative size: {name}")
        self.total += size
        if self.total > self.limits.max_total_bytes:
            raise ArchiveSafetyError("archive uncompressed total exceeds cap")
        if self.limits.reject_nested_archives and not is_dir and _is_nested(name):
            raise ArchiveSafetyError(f"nested archive member is not allowed: {name}")
        self.seen[name] = is_dir
        self.parents.update(parents)


def _safe_mode_zip(info: zipfile.ZipInfo) -> None:
    mode = (info.external_attr >> 16) & 0xFFFF
    if mode and (mode & 0o170000) not in (0, 0o100000, 0o040000):
        raise ArchiveSafetyError(f"special or link ZIP member is not allowed: {info.filename}")
    if (mode & 0o170000) == 0o040000 and not info.is_dir():
        raise ArchiveSafetyError(f"ZIP directory mode/name mismatch: {info.filename}")
    if (mode & 0o170000) == 0o100000 and info.is_dir():
        raise ArchiveSafetyError(f"ZIP regular-file mode/name mismatch: {info.filename}")
    if info.flag_bits & (0x01 | 0x40):
        raise ArchiveSafetyError(f"encrypted ZIP member is not allowed: {info.filename}")


def _safe_tar_member(member: tarfile.TarInfo) -> None:
    if member.type == tarfile.GNUTYPE_SPARSE or member.sparse is not None or any(
        key.startswith("GNU.sparse.") or key in ("SCHILY.realsize", "SCHILY.holesdata")
        or (key == "SCHILY.filetype" and value == "sparse")
        for key, value in member.pax_headers.items()
    ):
        raise ArchiveSafetyError(f"sparse TAR member is not allowed: {member.name}")
    if member.issym() or member.islnk() or member.isdev() or member.isfifo() or member.ischr() or member.isblk():
        raise ArchiveSafetyError(f"link or special TAR member is not allowed: {member.name}")
    if not (member.isfile() or member.isdir()):
        raise ArchiveSafetyError(f"unknown TAR member type is not allowed: {member.name}")


def _guarded_tarinfo(limits: ArchiveLimits):
    """Reject oversized extension headers *before* tarfile allocates their body.

    TarInfo's processing hooks also reject sparse maps before the stdlib sparse
    parser can expand them.  A separate metadata/depth cap bounds hidden PAX/GNU
    headers, which are not returned as inventory members.
    """
    metadata_bytes = member_count = metadata_depth = 0

    class GuardedTarInfo(tarfile.TarInfo):
        @classmethod
        def frombuf(cls, buf, encoding, errors):
            try:
                result = super().frombuf(buf, encoding, errors)
                # The stdlib strips *all* trailing slashes from a TAR directory.
                # Do not let that hide non-normalized paths such as rna//.
                if result.isdir() and buf[:100].split(b"\x00", 1)[0].endswith(b"//"):
                    raise ArchiveSafetyError("non-normalized TAR directory member name")
                return result
            except tarfile.EOFHeaderError:
                raise
            except tarfile.HeaderError as error:
                # TarFile.next otherwise tolerates a damaged header after the
                # first member as EOF, hiding truncated/malformed members.
                raise ArchiveSafetyError(f"invalid or truncated TAR header: {error}") from error

        def _proc_member(self, archive):
            nonlocal metadata_bytes, member_count, metadata_depth
            if self.size < 0:
                raise ArchiveSafetyError(f"negative TAR member size: {self.name}")
            if self.type in (tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.SOLARIS_XHDTYPE,
                             tarfile.GNUTYPE_LONGNAME):
                metadata_depth += 1
                metadata_bytes += tarfile.BLOCKSIZE + self._block(self.size)
                if metadata_bytes > limits.max_metadata_bytes or metadata_depth > 32:
                    raise ArchiveSafetyError("TAR extension metadata exceeds size/depth cap")
            else:
                metadata_depth = 0
                member_count += 1
                if member_count > limits.max_members:
                    raise ArchiveSafetyError("archive member count exceeds cap")
                _safe_tar_member(self)
                _member_path(self.name)
                if self.size > limits.max_member_bytes:
                    raise ArchiveSafetyError(f"member exceeds size cap: {self.name}")
                if self.isdir() and self.size:
                    raise ArchiveSafetyError(f"TAR directory contains payload bytes: {self.name}")
            result = super()._proc_member(archive)
            _safe_tar_member(result)
            return result

        def _apply_pax_info(self, headers, encoding, errors):
            if any(key.startswith("GNU.sparse.") or key in ("SCHILY.realsize", "SCHILY.holesdata")
                   or (key == "SCHILY.filetype" and value == "sparse")
                   for key, value in headers.items()):
                raise ArchiveSafetyError("sparse TAR extension is not allowed")
            return super()._apply_pax_info(headers, encoding, errors)

        def _proc_gnusparse_00(self, *args):
            raise ArchiveSafetyError("sparse TAR extension is not allowed")

        _proc_gnusparse_01 = _proc_gnusparse_00
        _proc_gnusparse_10 = _proc_gnusparse_00

    return GuardedTarInfo


class _BoundedTarReader:
    def __init__(self, source, limits: ArchiveLimits, archive_bytes: int, compressed: bool):
        self.source = source
        self.limits = limits
        self.bytes_read = 0
        # Include header/padding overhead; payload, member count and extension
        # metadata still have their own independent, stricter caps.
        self.cap = (limits.max_total_bytes + limits.max_members * 1024
                    + limits.max_metadata_bytes + tarfile.RECORDSIZE)
        self.cap_reason = "decoded TAR stream exceeds byte cap"
        if compressed and limits.max_compression_ratio < self.cap / max(archive_bytes, 1):
            self.cap = int(archive_bytes * limits.max_compression_ratio)
            self.cap_reason = "TAR total compression ratio exceeds cap"

    def read(self, size):
        if size < 0:
            raise ArchiveSafetyError("unbounded TAR read is not allowed")
        size = min(size, self.limits.chunk_bytes, self.cap - self.bytes_read + 1)
        block = self.source.read(size)
        self.bytes_read += len(block)
        if self.bytes_read > self.cap:
            raise ArchiveSafetyError(self.cap_reason)
        return block


@contextmanager
def _open_tar(raw, limits: ArchiveLimits, archive_bytes: int):
    raw.seek(0)
    magic = raw.read(6)
    raw.seek(0)
    with ExitStack() as stack:
        if magic.startswith(b"\x1f\x8b"):
            source = stack.enter_context(gzip.GzipFile(fileobj=raw, mode="rb"))
            compression = "gzip"
        elif magic.startswith(b"BZh"):
            source = stack.enter_context(bz2.BZ2File(raw, mode="rb"))
            compression = "bzip2"
        elif magic.startswith(b"\xfd7zXZ\x00"):
            source = stack.enter_context(lzma.LZMAFile(raw, mode="rb"))
            compression = "xz"
        else:
            source, compression = raw, None
        reader = _BoundedTarReader(source, limits, archive_bytes, compression is not None)
        archive = stack.enter_context(tarfile.open(fileobj=reader, mode="r|",
            bufsize=tarfile.BLOCKSIZE, tarinfo=_guarded_tarinfo(limits)))
        yield archive, reader, compression


def _check_zip_directory(raw, limits: ArchiveLimits) -> None:
    """Bound the central directory before ZipFile eagerly loads it.

    Use the stdlib EOCD/ZIP64 locator, but count fixed central headers ourselves;
    a forged EOCD member count must not bypass the cap.  No payload is read here.
    """
    end = zipfile._EndRecData(raw)
    if end is None:
        raise ArchiveSafetyError("missing ZIP end-of-directory record")
    if end[zipfile._ECD_DISK_NUMBER] or end[zipfile._ECD_DISK_START]:
        raise ArchiveSafetyError("multi-disk ZIP archive is not allowed")
    size = end[zipfile._ECD_SIZE]
    if size > limits.max_metadata_bytes:
        raise ArchiveSafetyError("ZIP directory metadata exceeds cap")
    expected = end[zipfile._ECD_ENTRIES_TOTAL]
    if expected > limits.max_members:
        raise ArchiveSafetyError("archive member count exceeds cap")
    start = end[zipfile._ECD_LOCATION] - size
    if end[zipfile._ECD_SIGNATURE] == zipfile.stringEndArchive64:
        start -= zipfile.sizeEndCentDir64 + zipfile.sizeEndCentDir64Locator
    if start < 0:
        raise ArchiveSafetyError("invalid ZIP directory offset")
    raw.seek(start)
    consumed = count = 0
    while consumed < size:
        count += 1
        if count > limits.max_members:
            raise ArchiveSafetyError("archive member count exceeds cap")
        header = raw.read(zipfile.sizeCentralDir)
        if len(header) != zipfile.sizeCentralDir or not header.startswith(b"PK\x01\x02"):
            raise ArchiveSafetyError("invalid or truncated ZIP directory header")
        lengths = struct.unpack_from("<HHH", header, 28)
        consumed += zipfile.sizeCentralDir + sum(lengths)
        if consumed > size:
            raise ArchiveSafetyError("truncated ZIP directory metadata")
        raw.seek(sum(lengths), os.SEEK_CUR)
    if count != expected or end[zipfile._ECD_ENTRIES_THIS_DISK] != expected:
        raise ArchiveSafetyError("ZIP directory member count mismatch")


def _zip_records(archive: zipfile.ZipFile, limits: ArchiveLimits, archive_bytes: int) -> list[dict]:
    records, index = [], _MemberIndex(limits)
    for info in archive.infolist():
        _safe_mode_zip(info)
        # ZipInfo.filename can be silently truncated at NUL by the stdlib.
        name = _member_path(info.orig_filename)
        if info.filename != info.orig_filename:
            raise ArchiveSafetyError(f"ZIP member name was truncated: {info.orig_filename!r}")
        is_dir = info.is_dir()
        size, compressed = int(info.file_size), int(info.compress_size)
        if size < 0 or compressed < 0 or (not compressed and size):
            raise ArchiveSafetyError(f"member has invalid compressed size: {name}")
        if is_dir and size:
            raise ArchiveSafetyError(f"ZIP directory contains payload bytes: {name}")
        if compressed and size / compressed > limits.max_compression_ratio:
            raise ArchiveSafetyError(f"member compression ratio exceeds cap: {name}")
        record = {"name": name, "size": size, "compressed_size": compressed,
                  "is_dir": is_dir, "crc": int(info.CRC)}
        index.add(record)
        records.append(record)
    if index.total / max(archive_bytes, 1) > limits.max_compression_ratio:
        raise ArchiveSafetyError("ZIP total compression ratio exceeds cap")
    return records


def _read_tar(raw, limits: ArchiveLimits, archive_bytes: int, *, expected=None,
              selected=None, destination_fd=None, copied=None) -> dict:
    records, index = [], _MemberIndex(limits)
    with _open_tar(raw, limits, archive_bytes) as (archive, reader, compression):
        while (member := archive.next()) is not None:
            # Older Python versions cache stream members too.  Keep at most
            # one TarInfo, rather than using version-specific stream=True.
            archive.members.clear()
            _safe_tar_member(member)
            name = _member_path(member.name)
            if member.isdir() and member.size:
                raise ArchiveSafetyError(f"TAR directory contains payload bytes: {name}")
            record = {"name": name, "size": int(member.size), "compressed_size": None,
                      "is_dir": member.isdir(), "crc": None}
            index.add(record)
            if expected is not None and (len(records) >= len(expected) or record != expected[len(records)]):
                raise ArchiveSafetyError("TAR inventory changed before extraction")
            records.append(record)
            if selected is not None and name in selected:
                source = archive.extractfile(member)
                if source is None:
                    raise ArchiveSafetyError(f"cannot read selected TAR member: {name}")
                with source:
                    copied.append(_copy_member(source, record, destination_fd, limits))
        # Drain the decoder, including gzip's CRC/ISIZE trailer.  TarFile stops
        # at the first zero block, which would otherwise hide gzip corruption
        # or an appended second TAR.  Require the second EOF block as well.
        trailer_bytes = 0
        while block := archive.fileobj.read(limits.chunk_bytes):
            trailer_bytes += len(block)
            if any(block):
                raise ArchiveSafetyError("non-zero data after TAR end marker")
        if trailer_bytes < tarfile.BLOCKSIZE or trailer_bytes % tarfile.BLOCKSIZE:
            raise ArchiveSafetyError("truncated TAR end marker/padding")
        if expected is not None and records != expected:
            raise ArchiveSafetyError("TAR inventory changed before extraction")
        return {"members": records, "decoded_stream_bytes": reader.bytes_read,
                "compression": compression}


@contextmanager
def _open_archive(path: Path):
    # O_NONBLOCK avoids hanging if a concurrently replaced input becomes a
    # FIFO; fstat rejects everything except a regular file before reading it.
    if path.is_symlink():
        raise ArchiveSafetyError(f"archive is missing or symlinked: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as raw:
        identity = os.fstat(raw.fileno())
        if not stat.S_ISREG(identity.st_mode):
            raise ArchiveSafetyError(f"archive is not a regular file: {path}")
        yield raw, identity


def _unchanged(raw, identity) -> None:
    current = os.fstat(raw.fileno())
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(current, key) != getattr(identity, key) for key in keys):
        raise ArchiveSafetyError("archive changed during inventory/extraction")


def _inspect_raw(path: Path, raw, identity, limits: ArchiveLimits) -> dict:
    if path.name.lower().endswith(".zip"):
        _check_zip_directory(raw, limits)
        with zipfile.ZipFile(raw) as archive:
            result = {"members": _zip_records(archive, limits, identity.st_size)}
        kind = "zip"
    else:
        result = _read_tar(raw, limits, identity.st_size)
        kind = "tar"
    _unchanged(raw, identity)
    return {"path": str(path), "kind": kind, "archive_bytes": identity.st_size,
            **result, "member_count": len(result["members"]),
            "uncompressed_bytes": sum(record["size"] for record in result["members"])}


_ARCHIVE_ERRORS = (tarfile.TarError, zipfile.BadZipFile, EOFError, OSError, RuntimeError,
                   ValueError, NotImplementedError, zlib.error, lzma.LZMAError)


def inspect_archive(path: str | Path, *, limits: ArchiveLimits | None = None) -> dict:
    """Return a bounded metadata inventory; compressed TAR is fully streamed."""
    limits = limits if limits is not None else ArchiveLimits()
    path = Path(path)
    try:
        with _open_archive(path) as (raw, identity):
            return _inspect_raw(path, raw, identity, limits)
    except ArchiveSafetyError:
        raise
    except _ARCHIVE_ERRORS as error:
        raise ArchiveSafetyError(f"cannot inspect archive {path}: {error}") from error


def _fresh_destination(destination: Path, archive: Path) -> None:
    _reject_symlink_ancestry(destination)
    if destination.exists():
        raise ArchiveSafetyError(f"quarantine destination must be new: {destination}")
    try:
        archive.resolve().relative_to(destination.resolve())
    except ValueError:
        return
    raise ArchiveSafetyError("archive cannot be inside its new quarantine destination")


def _reject_symlink_ancestry(path: Path) -> None:
    # Preserve lexical '..' until after checking: resolving it first could
    # conceal a symlinked ancestor (link/../new-destination).
    absolute = path.absolute()
    for parent in (absolute, *absolute.parents):
        if parent.is_symlink():
            raise ArchiveSafetyError(f"symlink in quarantine destination ancestry: {parent}")


@contextmanager
def _destination_directory(destination: Path):
    """Create, copy and clean up below pinned non-symlinked directory handles."""
    parts = destination.absolute().parts
    parent_fd = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    descriptor, identity = None, None
    try:
        for part in parts[1:-1]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=parent_fd)
            except FileExistsError:
                pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = child
        name = parts[-1]
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError as error:
            raise ArchiveSafetyError(f"quarantine destination must be new: {destination}") from error
        identity = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        try:
            descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (identity.st_dev, identity.st_ino):
                raise ArchiveSafetyError("quarantine destination changed while opening")
            yield descriptor
            _reject_symlink_ancestry(destination)
            current = destination.stat(follow_symlinks=False)
            if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
                raise ArchiveSafetyError("quarantine destination changed during extraction")
        except BaseException as error:
            try:
                current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
                    raise OSError("destination ownership changed; refusing to remove replacement")
                shutil.rmtree(name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                raise ArchiveSafetyError(f"quarantine cleanup failed: {cleanup_error}") from error
            raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _selected_names(inventory: dict, allowlist) -> set[str]:
    all_names = {record["name"] for record in inventory["members"] if not record["is_dir"]}
    if allowlist is None or isinstance(allowlist, (str, bytes)):
        raise ArchiveSafetyError("an explicit, non-empty file allowlist is required")
    selected = set()
    for count, name in enumerate(allowlist, 1):
        if count > inventory["member_count"]:
            raise ArchiveSafetyError("allowlist count exceeds archive member count")
        normalized = _member_path(name)
        if name.endswith("/"):
            raise ArchiveSafetyError("allowlist must name regular files, not directories")
        selected.add(normalized)
    if not selected:
        raise ArchiveSafetyError("an explicit, non-empty file allowlist is required")
    missing = selected - all_names
    if missing:
        raise ArchiveSafetyError(f"allowlist names are absent or directories: {sorted(missing)}")
    return selected


@contextmanager
def _target_file(destination_fd: int, name: str):
    """Walk below the owned directory fd, never following link components."""
    descriptor = os.dup(destination_fd)
    try:
        parts = name.split("/")
        for part in parts[:-1]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        target = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=descriptor)
        with os.fdopen(target, "wb") as sink:
            yield sink
    finally:
        os.close(descriptor)


def _copy_member(source, record: dict, destination_fd: int, limits: ArchiveLimits) -> dict:
    name, expected = record["name"], record["size"]
    copied_bytes, digest = 0, hashlib.sha256()
    with _target_file(destination_fd, name) as sink:
        while block := source.read(min(limits.chunk_bytes, expected - copied_bytes + 1)):
            copied_bytes += len(block)
            if copied_bytes > expected or copied_bytes > limits.max_member_bytes:
                raise ArchiveSafetyError(f"decoded member exceeds size cap/inventory: {name}")
            digest.update(block)
            sink.write(block)
    if copied_bytes != expected:
        raise ArchiveSafetyError(f"decoded member size mismatch: {name}")
    return {"name": name, "bytes": copied_bytes, "sha256": digest.hexdigest()}


def extract_quarantine(path: str | Path, destination: str | Path, *, allowlist=None,
                       limits: ArchiveLimits | None = None) -> dict:
    """Inspect then copy an explicit file allowlist; clean up on any failure."""
    limits = limits if limits is not None else ArchiveLimits()
    archive_path = Path(path)
    destination = Path(destination)
    copied = []
    try:
        with _open_archive(archive_path) as (raw, identity):
            inventory = _inspect_raw(archive_path, raw, identity, limits)
            selected = _selected_names(inventory, allowlist)
            _fresh_destination(destination, archive_path)
            with _destination_directory(destination) as descriptor:
                if inventory["kind"] == "zip":
                    _check_zip_directory(raw, limits)
                    with zipfile.ZipFile(raw) as archive:
                        records = _zip_records(archive, limits, identity.st_size)
                        if records != inventory["members"]:
                            raise ArchiveSafetyError("ZIP inventory changed before extraction")
                        by_name = {record["name"]: record for record in records if not record["is_dir"]}
                        for name in sorted(selected):
                            with archive.open(name, "r") as source:
                                copied.append(_copy_member(source, by_name[name], descriptor, limits))
                else:
                    _read_tar(raw, limits, identity.st_size, expected=inventory["members"],
                              selected=selected, destination_fd=descriptor, copied=copied)
                _unchanged(raw, identity)
    except BaseException as error:
        if isinstance(error, _ARCHIVE_ERRORS) and not isinstance(error, ArchiveSafetyError):
            raise ArchiveSafetyError(f"cannot extract archive {archive_path}: {error}") from error
        raise
    copied.sort(key=lambda record: record["name"])
    return {"status": "quarantined", "archive": str(archive_path), "destination": str(destination),
            "kind": inventory["kind"], "selected": copied, "inventory": inventory}
