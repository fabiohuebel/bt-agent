"""Read-only publication inventory. Never prints matched credential values.

Scans tracked working files, optionally all reachable Git blobs, and supplied
wheel/sdist archives without extraction. Heuristics are not a security or legal
clearance. The report is a local review artifact, not an upload command.
"""

import argparse
import base64
from collections import Counter
import csv
from email.parser import BytesParser
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile
import zipfile

RULES = {
    "private_key": (
        "credential",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"),
    ),
    "github_token": (
        "credential",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{60,})\b"),
    ),
    "provider_token": (
        "credential",
        re.compile(
            r"\b(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{24,}|xox[baprs]-[A-Za-z0-9-]{20,})\b"
        ),
    ),
    "aws_access_id": ("credential", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    "literal_secret": (
        "review",
        re.compile(
            r"""(?i)(?:api[_-]?key|client[_-]?secret|password|access[_-]?token)["']?\s*[:=]\s*["'][^"'\r\n]{16,}["']"""
        ),
    ),
    "url_credentials": ("review", re.compile(r"https?://[^\s/:@]+:[^\s/@]+@")),
    "personal_path": ("review", re.compile(r"/(?:home|Users)/[^\s/'\"`]+|/root/")),
    "private_network": (
        "review",
        re.compile(
            r"\b(?:10\.(?:\d{1,3}\.){2}\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|[a-zA-Z0-9-]+\.tail[a-zA-Z0-9]+\.ts\.net)\b"
        ),
    ),
}
MEDIA = {
    ".safetensors",
    ".pt",
    ".pth",
    ".ckpt",
    ".db3",
    ".mcap",
    ".bag",
    ".mp4",
    ".mov",
    ".jpg",
    ".jpeg",
    ".png",
    ".ply",
    ".glb",
    ".urdf",
}


def safe_label(value):
    for name, (level, pattern) in RULES.items():
        if level == "credential" or name in {"literal_secret", "url_credentials"}:
            value = pattern.sub(f"[redacted-{name}]", value)
    return value


def scan_bytes(data):
    """Return locations/counts only: no excerpts, matching values or token hashes."""
    text = data.decode("utf-8", errors="replace")
    rows = []
    for name, (level, pattern) in RULES.items():
        locations = sorted({text.count("\n", 0, m.start()) + 1 for m in pattern.finditer(text)})
        if locations:
            rows.append(dict(rule=name, level=level, lines=locations, count=len(locations)))
    return rows


def git(repo, *args, **kwargs):
    return subprocess.check_output(["git", "-C", str(repo), *args], **kwargs)


def tracked_inventory(repo, max_bytes):
    paths = git(repo, "ls-files", "-z").decode().split("\0")
    records, omitted = [], []
    for name in filter(None, paths):
        path = repo / name
        label = safe_label(name)
        if path.is_symlink():
            # Do not follow tracked symlinks out of the checkout.
            omitted.append(dict(path=label, reason="symlink-not-followed"))
            continue
        if not path.is_file():
            omitted.append(dict(path=label, reason="missing-or-not-a-file"))
            continue
        size = path.stat().st_size
        if size > max_bytes:
            omitted.append(dict(path=label, reason="size-limit", bytes=size))
            continue
        data = path.read_bytes()
        records.append(
            dict(
                path=label,
                bytes=size,
                sha256=hashlib.sha256(data).hexdigest(),
                findings=scan_bytes(data),
                media_or_model_suffix=path.suffix.lower() in MEDIA,
                lfs_pointer=data.startswith(b"version https://git-lfs.github.com/spec/v1"),
            )
        )
    return dict(
        files=records,
        omitted=omitted,
        scope="tracked working-tree files only; untracked files excluded",
    )


def history_inventory(repo, max_bytes):
    raw = git(repo, "rev-list", "--objects", "--all").decode().splitlines()
    names = {}
    for line in raw:
        oid, _, name = line.partition(" ")
        names.setdefault(oid, safe_label(name))
    types = (
        git(
            repo,
            "cat-file",
            "--batch-check=%(objectname) %(objecttype) %(objectsize)",
            input=("\n".join(names) + "\n").encode(),
        )
        .decode()
        .splitlines()
    )
    blobs = [row.split() for row in types if " blob " in row]
    findings, omitted = [], []
    count = total = 0
    process = subprocess.Popen(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    try:
        for oid, kind, raw_size in blobs:
            size = int(raw_size)
            if size > max_bytes:
                omitted.append(
                    dict(blob=oid, path_hint=names[oid], bytes=size, reason="size-limit")
                )
                continue
            process.stdin.write((oid + "\n").encode())
            process.stdin.flush()
            header = process.stdout.readline().decode().strip().split()
            if header != [oid, kind, raw_size]:
                raise RuntimeError("Unexpected Git object response")
            data = process.stdout.read(size)
            if len(data) != size or process.stdout.read(1) != b"\n":
                raise RuntimeError("Truncated Git object response")
            hits = scan_bytes(data)
            if hits:
                findings.append(dict(blob=oid, path_hint=names[oid], findings=hits))
            count += 1
            total += size
    finally:
        process.stdin.close()
        process.stdout.close()
        if process.wait(timeout=10):
            raise RuntimeError("Git object scanner failed")
    return dict(
        blobs_scanned=count,
        bytes_scanned=total,
        findings=findings,
        omitted=omitted,
        shallow=git(repo, "rev-parse", "--is-shallow-repository").decode().strip() == "true",
        scope="blobs reachable from local Git refs; not reflogs, dangling objects or unfetched refs",
    )


def archive_inventory(path, max_bytes):
    records, omitted, errors, payloads = [], [], [], {}
    wheel = path.suffix == ".whl"
    if wheel:
        archive = zipfile.ZipFile(path)
        members = [
            (i.filename, i.file_size, stat.S_ISLNK(i.external_attr >> 16), i)
            for i in archive.infolist()
            if not i.is_dir()
        ]
        read = archive.read
    else:
        archive = tarfile.open(path, "r:*")
        members = [
            (m.name, m.size, not m.isfile(), m) for m in archive.getmembers() if not m.isdir()
        ]
        read = lambda member: archive.extractfile(member).read()
    with archive:
        seen = set()
        for name, size, link, member in members:
            label = safe_label(name)
            entry = PurePosixPath(name)
            if entry.is_absolute() or ".." in entry.parts or "\\" in name or name in seen:
                errors.append(dict(path=label, reason="unsafe-or-duplicate-archive-path"))
                continue
            seen.add(name)
            if link or size > max_bytes:
                omitted.append(dict(path=label, reason="link-or-size-limit", bytes=size))
                continue
            data = read(member)
            payloads[name] = data
            if any(p in {".git", ".env", ".build", "__pycache__"} for p in entry.parts):
                errors.append(dict(path=label, reason="unexpected-private-build-content"))
            records.append(
                dict(
                    path=label,
                    bytes=size,
                    sha256=hashlib.sha256(data).hexdigest(),
                    findings=scan_bytes(data),
                    media_or_model_suffix=entry.suffix.lower() in MEDIA,
                )
            )
    metadata_paths = (
        [n for n in payloads if n.endswith(".dist-info/METADATA")]
        if wheel
        else [n for n in payloads if n.count("/") == 1 and n.endswith("/PKG-INFO")]
    )
    metadata = (
        BytesParser().parsebytes(payloads[metadata_paths[0]]) if len(metadata_paths) == 1 else None
    )
    if metadata is None:
        errors.append(dict(reason="missing-or-ambiguous-metadata"))
    record_checks = None
    if wheel:
        rec = [n for n in payloads if n.endswith(".dist-info/RECORD")]
        record_checks = len(rec) == 1
        if record_checks:
            covered = set()
            for name, digest, size in csv.reader(io.StringIO(payloads[rec[0]].decode())):
                covered.add(name)
                data = payloads.get(name)
                if data is None:
                    record_checks = False
                elif name != rec[0]:
                    expected = "sha256=" + base64.urlsafe_b64encode(
                        hashlib.sha256(data).digest()
                    ).decode().rstrip("=")
                    record_checks &= digest == expected and size == str(len(data))
            record_checks &= covered == set(payloads)
        if not record_checks:
            errors.append(dict(reason="wheel-record-incomplete-or-mismatched"))
    return dict(
        file=safe_label(path.name),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        files=records,
        omitted=omitted,
        errors=errors,
        record_verified=record_checks,
        metadata=(
            dict(
                name=safe_label(metadata.get("Name", "")),
                version=safe_label(metadata.get("Version", "")),
                classifiers=[safe_label(v) for v in metadata.get_all("Classifier", [])],
                license_files=[safe_label(v) for v in metadata.get_all("License-File", [])],
            )
            if metadata
            else None
        ),
        scope="archive members inspected without extraction; no legal clearance implied",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--history", action="store_true")
    p.add_argument(
        "--fail-on-credentials",
        action="store_true",
        help="Exit 2 after writing a report if a credential-pattern candidate was found",
    )
    p.add_argument("--archive", type=Path, action="append", default=[])
    p.add_argument("--max-bytes", type=int, default=32 * 1024**2)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.max_bytes <= 0 or args.out.exists():
        p.error("Positive size limit and a new output path are required")
    repo = args.repo.resolve(strict=True)
    working = tracked_inventory(repo, args.max_bytes)
    history = history_inventory(repo, args.max_bytes) if args.history else None
    archives = [archive_inventory(path, args.max_bytes) for path in args.archive]
    all_rows = (
        working["files"]
        + (history["findings"] if history else [])
        + [r for a in archives for r in a["files"]]
    )
    counts = Counter(hit["rule"] for row in all_rows for hit in row["findings"])
    report = dict(
        schema="release-review-inventory/v1",
        git_head=git(repo, "rev-parse", "HEAD").decode().strip(),
        working_tree=working,
        history=history,
        archives=archives,
        finding_records_by_rule=dict(counts),
        public_release_approved=False,
        limits="Heuristic candidates, not proof of no secrets. No entropy/encoded-secret/dynamic-secret detection; omissions and scan scopes are explicit. Legal and owner review remains required.",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as out:
        json.dump(report, out, indent=2)
        out.write("\n")
    print(
        json.dumps(
            dict(
                report=safe_label(str(args.out)),
                tracked_files=len(working["files"]),
                history_blobs=history["blobs_scanned"] if history else None,
                finding_records_by_rule=dict(counts),
                public_release_approved=False,
            )
        )
    )
    if args.fail_on_credentials and any(RULES[name][0] == "credential" for name in counts):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
