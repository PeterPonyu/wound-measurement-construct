#!/usr/bin/env python3
"""Archive authored source without changing Git's index, branches or remotes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SCOPE = ["AGENTS.md", "README.md", "LICENSE", "NOTICE", "pytest.ini",
         "package.json", "package-lock.json", "reproduce.sh", "config", "core_engines",
         "directions", "docs", "expansion", "manuscripts", "publication", "scripts", "tests", "tracks"]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    status_before = git("status", "--porcelain=v1", "-z")
    names = sorted(set(git("ls-files", "-z", "--cached", "--others", "--exclude-standard",
                           "--", *SCOPE).decode().split("\0")) - {""})
    files = {name: {"sha256": sha(ROOT / name), "bytes": (ROOT / name).stat().st_size}
             for name in names if (ROOT / name).is_file() and not (ROOT / name).is_symlink()}
    evidence = {"utc": datetime.now(timezone.utc).isoformat(), "git_head": git("rev-parse", "HEAD").decode().strip(),
                "scope": SCOPE, "files": files, "file_count": len(files),
                "purpose": "Internal authored-source checkpoint, not full data reproduction or a public release",
                "excluded": ["data", "outputs", "output", "tmp", "vendored runtimes", "Git credentials"],
                "git_index_modified": False, "remote_writes": False}
    (out / "manifest.json").write_text(json.dumps(evidence, indent=2) + "\n")
    (out / "git-status.nul").write_bytes(status_before)
    (out / "staged.patch").write_bytes(git("diff", "--cached", "--binary"))
    (out / "unstaged.patch").write_bytes(git("diff", "--binary"))
    archive = out / "authored-source.tar"
    with tarfile.open(archive, "w") as stream:
        for name in files:
            stream.add(ROOT / name, arcname=name, recursive=False)
    with tarfile.open(archive) as stream:
        assert set(stream.getnames()) == set(files)
        for member in stream:
            assert hashlib.sha256(stream.extractfile(member).read()).hexdigest() == files[member.name]["sha256"]
    # No tracked/index mutation is permitted. The new output folder can appear
    # as untracked, so compare staged state rather than whole-worktree status.
    assert (out / "staged.patch").read_bytes() == git("diff", "--cached", "--binary")
    (out / "authored-source.tar.sha256").write_text(f"{sha(archive)}  authored-source.tar\n")
    print(json.dumps({"files": len(files), "archive": str(archive), "sha256": sha(archive)}, indent=2))


if __name__ == "__main__":
    main()
