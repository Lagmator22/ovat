# ovat/core/ovms_installer.py
"""Fetch the right OVMS archive for this machine, into a folder OVAT searches.

Why this exists. Installing OVAT used to be four manual steps and one judgement
call: pip install, then read the README, then pick one archive out of six, then
curl/tar it, and usually then `export OVAT_OVMS=...` because the archive landed
somewhere the locator does not look. The judgement call is the dangerous part --
`python_off` cannot do tool calling, and choosing it gives an agent that answers
fluently and silently never calls a tool.

Why it is not part of `pip install ovat`. The archive is 126 MB (Windows) to
185 MB (Ubuntu 24), Linux needs three different builds that cannot be chosen at
wheel-build time, wheels have no post-install hook to run a downloader, and
macOS has no OVMS build at all -- so a bundled wheel would charge every Mac user
~180 MB for a binary that cannot run. A subcommand that fetches on demand is the
same shape as `playwright install` and `python -m spacy download`.

The install goes to ~/.ovat/ovms, which ovms_locator searches, so nothing here
ever edits PATH.
"""
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

# The build this project has actually been verified against on an AI PC.
# One constant, so the README, the installer and the tests cannot drift.
OVMS_VERSION = "2026.2.1"

_RELEASE_URL = ("https://github.com/openvinotoolkit/model_server/releases/"
                "download/v{version}/{asset}")

# Per-user, not per-clone: one download serves every venv and every checkout,
# and it survives `rm -rf` of a project folder.
DEFAULT_ROOT = os.path.join(os.path.expanduser("~"), ".ovat", "ovms")

_EXE = "ovms.exe" if sys.platform == "win32" else "ovms"


def _os_release() -> dict:
    """/etc/os-release as a dict, or {} if it cannot be read.

    Shared by _linux_flavour and linux_support_note so the two cannot disagree
    about what distro this is -- the flavour chosen and the warning about it
    must come from one reading, not two.
    """
    data = {}
    try:
        with open("/etc/os-release", encoding="utf-8") as handle:
            for line in handle:
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                data[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        return {}
    return data


def _linux_flavour() -> str:
    """Which Linux archive this distro needs: ubuntu24, ubuntu22 or redhat.

    Intel ships one build per distro family and they are not interchangeable --
    they link against that release's system libraries. /etc/os-release is the
    portable way to ask; it is a shell-style key=value file present on every
    systemd distro.

    An unknown distro falls back to ubuntu24 (the newest) but the CALLER is
    told, because a silent wrong guess produces a binary that fails to load its
    own libraries, which reads as "OVAT is broken" rather than "your distro is
    not on the list".
    """
    data = _os_release()
    if not data:
        return "ubuntu24"

    ident = (data.get("ID") or "").lower()
    like = (data.get("ID_LIKE") or "").lower()
    version = (data.get("VERSION_ID") or "")

    if ident == "ubuntu" or "ubuntu" in like:
        return "ubuntu22" if version.startswith("22") else "ubuntu24"
    if ident in {"rhel", "centos", "fedora", "rocky", "almalinux"} or \
            "rhel" in like or "fedora" in like:
        return "redhat"
    return "ubuntu24"


#: Ubuntu releases Intel actually publishes a build for. Anything else gets
#: the newest one as a guess, and the user is TOLD, because the failure mode
#: is a binary that cannot load its own libraries -- which reads as "OVAT is
#: broken", not "your distro is not on the list".
_SUPPORTED_UBUNTU = ("22", "24")


def linux_support_note() -> str | None:
    """A warning for this distro, or None when it is one Intel builds for.

    Reported from Ubuntu 26.04: it was handed the ubuntu24 archive with no
    warning at all, and that binary cannot start there -- it wants
    libpython3.12.so.1.0, libxml2.so.2 and libtbb.so.12, none of which 26.04
    carries. The old code called this "the documented fallback" and documented
    it only in a docstring the user never sees.

    A note rather than a refusal: a guess that works is common (Mint, Pop,
    Debian derivatives all resolve to ubuntu24 and are usually fine), so
    blocking would break more people than it protects. Saying so costs
    nothing and turns a baffling crash into an expected one.
    """
    if sys.platform != "linux":
        return None
    data = _os_release()
    ident = (data.get("ID") or "").lower()
    like = (data.get("ID_LIKE") or "").lower()
    version = data.get("VERSION_ID") or ""

    if ident == "ubuntu":
        if not version.startswith(_SUPPORTED_UBUNTU):
            return (f"Ubuntu {version} has no OVMS build; installing the "
                    f"ubuntu24 one. If it will not start, that is why: it "
                    f"needs this release's system libraries. Use Ubuntu "
                    f"22.04 or 24.04 for a supported install.")
        return None
    if ident in {"rhel", "centos", "fedora", "rocky", "almalinux"} or \
            "rhel" in like or "fedora" in like:
        return None
    if "ubuntu" in like or "debian" in like:
        return (f"{ident or 'this distro'} is not one Intel builds for; "
                f"using the ubuntu24 archive, which usually works on "
                f"Ubuntu/Debian derivatives.")
    return (f"{ident or 'this distro'} is not a distro Intel publishes OVMS "
            f"for; trying the ubuntu24 archive. If it will not start, that "
            f"is why.")


def asset_for_platform(version: str = OVMS_VERSION) -> tuple | None:
    """(filename, url) for this OS, or None where OVMS does not exist.

    None means macOS, and it is not an error: Intel publishes Windows and
    Linux x86-64 only. The caller turns that into an explanation, not a
    failure.
    """
    if sys.platform == "darwin":
        return None
    if sys.platform == "win32":
        asset = f"ovms_windows_{version}_python_on.zip"
    else:
        asset = f"ovms_{_linux_flavour()}_{version}_python_on.tar.gz"
    return asset, _RELEASE_URL.format(version=version, asset=asset)


def installed_binary(root: str = DEFAULT_ROOT) -> str | None:
    """The ovms binary under `root`, if a usable install is already there."""
    for candidate in (os.path.join(root, _EXE),
                      os.path.join(root, "bin", _EXE)):
        if os.path.isfile(candidate):
            return candidate
    return None


def _expected_sha256(asset_url: str) -> str | None:
    """The published checksum, or None if the release does not carry one.

    Upstream naming is inconsistent: python_off ships
    `ovms_windows_..._python_off.zip.sha256` while python_on ships
    `ovms_windows_..._python_on.sha256` with no `.zip`. Try both rather than
    guess. A missing checksum is a warning, not a hard failure -- refusing to
    install because Intel forgot a sibling file would be our bug on their
    release process.
    """
    stem, _, _ = asset_url.rpartition(".")
    if asset_url.endswith(".tar.gz"):
        stem = asset_url[:-len(".tar.gz")]
    for url in (asset_url + ".sha256", stem + ".sha256"):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                text = response.read().decode("utf-8", "replace").strip()
        except Exception:
            continue
        # Files are either a bare digest or "<digest>  <filename>".
        digest = text.split()[0] if text.split() else ""
        if len(digest) == 64:
            return digest.lower()
    return None


def _download(url: str, target: str, on_progress=None) -> None:
    """Stream `url` to `target`, reporting (done, total) bytes as it goes.

    Proxies are deliberately HONOURED here, unlike the OVMS health check which
    bypasses them: reaching github.com from inside a corporate network is
    exactly what a proxy is for, whereas a request to localhost must never go
    through one.
    """
    with urllib.request.urlopen(url, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        with open(target, "wb") as handle:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if on_progress is not None:
                    on_progress(done, total)


def _is_safe_member(name: str) -> bool:
    """Reject absolute paths and ..\\ escapes before extracting.

    tarfile grew a `filter=` argument in 3.12, but this project supports 3.10,
    so the check is done by hand rather than left to the stdlib version that
    happens to be present.
    """
    if os.path.isabs(name) or name.startswith(("/", "\\")):
        return False
    parts = name.replace("\\", "/").split("/")
    return ".." not in parts


def _make_owner_writable(root: str) -> None:
    """Add the owner's write bit everywhere under `root`.

    OVMS ships its libraries AND their directories mode 0o555, and unlinking a
    file requires write permission on the file's PARENT DIRECTORY, not on the
    file. So the shutil.move below -- which falls back to copy-then-rmtree
    whenever staging and destination are on different filesystems, and /tmp
    usually is -- died for every non-root user with

        PermissionError: [Errno 13] Permission denied: 'libovms_shared.so'

    while root sailed through, because CAP_DAC_OVERRIDE ignores permission
    bits entirely. That is why this passed in Docker, in CI and in the
    maintainer's own container -- all of them root -- and failed for the first
    real user who ran it as themselves. Verified as uid 1001 on Ubuntu 24.04.

    Directories get +wx (descend and modify), files get +w. Symlinks are
    skipped: chmod follows them, so touching one would change the mode of
    whatever it points at instead.
    """
    def relax(path: str, is_dir: bool) -> None:
        if os.path.islink(path):
            return
        try:
            mode = os.stat(path).st_mode
            os.chmod(path, mode | (0o300 if is_dir else 0o200))
        except OSError:
            # Best effort. A file we cannot chmod is one the move may fail on,
            # and that failure is far more informative than one raised here.
            pass

    relax(root, True)
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames:
            relax(os.path.join(dirpath, name), True)
        for name in filenames:
            relax(os.path.join(dirpath, name), False)


def _extract_tar(tf, staging: str, members) -> None:
    """Unpack a tar, pinning the extraction filter rather than inheriting it.

    Python's default changed under us. 3.14 makes `data` the default, and
    `data` REFUSES the absolute symlinks these archives contain, so the same
    code that worked on 3.12 raises AbsoluteLinkError on 3.14 -- reported from
    Ubuntu 26.04. Asking for `tar` explicitly keeps one behaviour across every
    version instead of tracking the interpreter's.

    Path traversal is already blocked by _is_safe_member before we get here,
    which is the protection `data` would otherwise be providing.
    """
    # Force the owner's write bit ON before a single byte is written. This is
    # the fix, and it has to happen HERE rather than afterwards: OVMS ships
    # 0o555 DIRECTORIES, tarfile creates a directory before the members inside
    # it, and writing a file into a directory with no write bit fails outright.
    # The extraction itself raises, so any repair that runs after extractall
    # never executes -- which is exactly what an earlier attempt at this got
    # wrong, and why it still failed on a real archive while passing against a
    # synthetic one whose member ordering happened to avoid the case.
    #
    # Mutating members is deliberate over a filter= callable: it behaves
    # identically on 3.10 through 3.14, whereas the filter API arrived in 3.12
    # and was only partly backported.
    for member in members:
        member.mode |= 0o700 if member.isdir() else 0o600
    try:
        tf.extractall(staging, members=members, filter="tar")
    except TypeError:
        # Python 3.10/3.11 without the backport: no filter= parameter at all.
        tf.extractall(staging, members=members)


def _extract(archive: str, into: str) -> None:
    """Unpack, then flatten the archive's own top-level ovms/ directory.

    Every published archive contains a single `ovms/` folder. Extracting it
    verbatim under ~/.ovat/ovms would give ~/.ovat/ovms/ovms/bin/ovms, which
    the locator does not search. Flattening once here keeps the on-disk layout
    the same as a manual unpack: <root>/ovms.exe on Windows, <root>/bin/ovms
    on Linux.
    """
    staging = tempfile.mkdtemp(prefix="ovat-ovms-")
    try:
        if archive.endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                members = [m for m in zf.namelist() if _is_safe_member(m)]
                zf.extractall(staging, members=members)
        else:
            with tarfile.open(archive, "r:gz") as tf:
                members = [m for m in tf.getmembers()
                           if _is_safe_member(m.name)]
                _extract_tar(tf, staging, members)

        # Before anything is moved. The archive's own 0o555 directories would
        # otherwise make the move below fail for every non-root user.
        _make_owner_writable(staging)

        entries = os.listdir(staging)
        root = staging
        if len(entries) == 1 and os.path.isdir(os.path.join(staging, entries[0])):
            root = os.path.join(staging, entries[0])

        os.makedirs(into, exist_ok=True)
        for name in os.listdir(root):
            source = os.path.join(root, name)
            destination = os.path.join(into, name)
            if os.path.exists(destination):
                # Relax first: an EARLIER install left 0o555 directories here
                # too, so rmtree(ignore_errors=True) would quietly fail to
                # clear them and the move would then land on a non-empty path.
                # Silently doing nothing is the worst of the options.
                if os.path.isdir(destination):
                    _make_owner_writable(destination)
                    shutil.rmtree(destination)
                else:
                    os.remove(destination)
            shutil.move(source, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def install(root: str = DEFAULT_ROOT, version: str = OVMS_VERSION,
            force: bool = False, on_progress=None) -> tuple:
    """Put a usable OVMS under `root`. Returns (binary_path, what_happened).

    what_happened is 'already-installed' or 'installed', so the CLI can say
    which without re-deriving it. Idempotent by default: running this twice is
    a no-op, which is what makes it safe to call from `serve` and from scripts.

    Raises RuntimeError on a checksum mismatch, and leaves nothing behind.
    """
    existing = installed_binary(root)
    if existing and not force:
        return existing, "already-installed"

    resolved = asset_for_platform(version)
    if resolved is None:
        raise RuntimeError("OVMS has no macOS build; nothing to install.")
    asset, url = resolved

    workdir = tempfile.mkdtemp(prefix="ovat-dl-")
    archive = os.path.join(workdir, asset)
    try:
        _download(url, archive, on_progress)

        expected = _expected_sha256(url)
        if expected:
            digest = hashlib.sha256()
            with open(archive, "rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            actual = digest.hexdigest()
            if actual != expected:
                raise RuntimeError(
                    f"checksum mismatch for {asset}: expected {expected}, "
                    f"got {actual}. The download was corrupt or tampered "
                    f"with; nothing was installed.")

        if force and os.path.isdir(root):
            # Same reason as in _extract: the install we are replacing is full
            # of 0o555 directories, and ignore_errors would turn "could not
            # delete the old one" into a silent no-op, so --force would appear
            # to work while leaving the previous version in place.
            _make_owner_writable(root)
            shutil.rmtree(root)
        _extract(archive, root)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    binary = installed_binary(root)
    if binary is None:
        raise RuntimeError(
            f"unpacked {asset} into {root} but found no {_EXE} afterwards; "
            f"the archive layout may have changed upstream.")
    if sys.platform != "win32":
        # tarfile preserves the executable bit, but a restrictive umask during
        # extraction can still land it non-executable.
        os.chmod(binary, os.stat(binary).st_mode | 0o111)
    return binary, "installed"
