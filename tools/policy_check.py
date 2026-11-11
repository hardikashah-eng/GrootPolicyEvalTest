"""Repository policy check: block content that Atlassian's terms or basic hygiene do not allow.

Run before every commit or PR to the Bitbucket repository (Bitbucket Pipelines runs it on
every pull request and on pushes to Main):

    python tools/policy_check.py            # scan the working tree
    python tools/policy_check.py PATH ...   # scan only these files

Exit code 0 = clean, 1 = at least one blocking finding. Standard library only.

What it enforces, and why:

* **No secrets or credentials** (private keys, cloud/API tokens, .env files, keystores).
  Leaked secrets are a security incident, and code that exposes credentials can count
  as "scripting exploits" under the Atlassian Acceptable Use Policy.
* **No media or bulk storage** (audio, video, disk images, large archives, any file over
  MAX_FILE_MB). Atlassian's Bitbucket terms allow removal of a whole repository whose
  content consumes "an unreasonable amount of storage" unrelated to source code.
* **No malware-like or exploit content** (known test signatures, obvious payload markers).
* **A license file is present** (Bitbucket terms require public repositories to state their
  license; we require it always so making the repo public later is safe).

It cannot judge intellectual-property ownership or whether content is misleading in
general; reviewers still do that on every PR.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_FILE_MB = 5.0
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", "runs", "data", "demo-output"}
ALLOW_LARGE = {"uv.lock"}  # generated lockfile; text, legitimately big

BLOCKED_NAMES = re.compile(
    r"(^|/)(\.env(\..*)?|id_rsa|id_dsa|id_ecdsa|id_ed25519|credentials\.json|service[-_]account.*\.json|"
    r".*\.(pem|key|p12|pfx|keystore|jks|ppk|kdbx))$",
    re.IGNORECASE,
)
ALLOWED_NAMES = re.compile(r"(^|/)\.env\.(example|sample|template)$", re.IGNORECASE)
MEDIA_EXT = {
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv",
    ".iso", ".img", ".dmg", ".vmdk", ".qcow2", ".vdi", ".exe", ".msi", ".apk", ".torrent",
}
ARCHIVE_EXT = {".zip", ".tar", ".gz", ".tgz", ".7z", ".rar", ".bz2", ".xz", ".bundle"}
MAX_ARCHIVE_MB = 1.0

SECRET_PATTERNS = [
    ("private key", re.compile(r"-----BEGIN (RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----")),
    ("AWS access key", re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b|\bgithub_pat_[A-Za-z0-9_]{60,}\b")),
    ("Bitbucket/Atlassian token", re.compile(r"\bATBB[A-Za-z0-9]{24,}\b|\bATATT[A-Za-z0-9_\-=]{40,}\b")),
    ("Slack token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b")),
    ("OpenAI/Anthropic key", re.compile(r"\bsk-(ant-)?[A-Za-z0-9_\-]{32,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{34,}\b")),
    ("hard-coded credential", re.compile(
        r"(?i)\b(password|passwd|secret|token|api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret)\b\s*[:=]\s*['\"][^'\"\s]{8,}['\"]")),
    ("URL with embedded credentials", re.compile(r"\b[a-z][a-z0-9+.-]*://[^/\s:@'\"]+:[^/\s@'\"]{3,}@")),
]
MALWARE_PATTERNS = [
    ("EICAR test signature", re.compile(r"X5O!P%@AP\[4\\PZX54\(P\^\)7CC\)7\}\$EICAR")),
    ("reverse-shell payload", re.compile(r"(?i)\b(bash\s+-i\s+>&\s*/dev/tcp/|nc\s+-e\s+/bin/(ba)?sh)")),
    ("crypto-miner", re.compile(r"(?i)\b(xmrig|stratum\+tcp://|coinhive)\b")),
]
# This file contains its own patterns, so content checks skip it (name and size rules still apply).
SELF = Path(__file__).resolve()


def tracked_files(paths: list[str]) -> list[Path]:
    if paths:
        return [Path(p).resolve() for p in paths if Path(p).is_file()]
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        out += [Path(dirpath) / f for f in filenames]
    return sorted(out)


def rel(p: Path) -> str:
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def check_file(p: Path) -> list[str]:
    r = rel(p)
    problems = []
    size_mb = p.stat().st_size / 1e6
    ext = p.suffix.lower()
    if BLOCKED_NAMES.search(r) and not ALLOWED_NAMES.search(r):
        problems.append(f"{r}: credential/key file name; keep secrets out of the repository")
    if ext in MEDIA_EXT:
        problems.append(f"{r}: media/binary-image file ({ext}); not source code")
    if ext in ARCHIVE_EXT and size_mb > MAX_ARCHIVE_MB:
        problems.append(f"{r}: archive of {size_mb:.1f} MB (limit {MAX_ARCHIVE_MB} MB); commit sources, not archives")
    if size_mb > MAX_FILE_MB and p.name not in ALLOW_LARGE:
        problems.append(f"{r}: {size_mb:.1f} MB (limit {MAX_FILE_MB} MB); use external storage or Git LFS")
    if p == SELF or size_mb > MAX_FILE_MB:
        return problems
    try:
        text = p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return problems  # binary: only name/size rules apply
    for kind, rx in SECRET_PATTERNS + MALWARE_PATTERNS:
        for m in rx.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            problems.append(f"{r}:{line}: possible {kind}")
    return problems


def main(argv: list[str]) -> int:
    files = tracked_files(argv)
    problems = [msg for f in files for msg in check_file(f)]
    if not argv and not any((ROOT / n).exists() for n in ("LICENSE", "LICENSE.md", "LICENSE.txt")):
        problems.append("no LICENSE file at the repository root")
    if problems:
        print(f"Policy check FAILED: {len(problems)} finding(s) in {len(files)} files")
        for msg in problems:
            print(f"  - {msg}")
        print("Fix these, or if a finding is a false positive, adjust tools/policy_check.py in a reviewed PR.")
        return 1
    print(f"Policy check passed: {len(files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
