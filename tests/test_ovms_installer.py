# tests/test_ovms_installer.py
"""Tests for `ovat setup`: which archive, which checksum, where it lands.

Nothing here touches the network. The download is either monkeypatched or
served from a local file:// URL, because a test that needs GitHub to be up is
a test that fails for reasons that have nothing to do with this code.

The decisions worth pinning are the ones a wrong answer hides: picking the
python_off build (agent answers, never calls a tool), picking the wrong Linux
flavour (binary cannot load its own .so files), and extracting one level too
deep (locator never finds it).
"""
import hashlib
import io
import os
import sys
import tarfile
import zipfile

import pytest

from ovat.core import ovms_installer as installer


# --------------------------------------------------------------------------
# Which archive does this machine need?
# --------------------------------------------------------------------------

def test_windows_gets_the_python_on_zip(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    asset, url = installer.asset_for_platform("9.9.9")
    assert asset == "ovms_windows_9.9.9_python_on.zip"
    # python_off cannot do tool calling, which is the entire point of OVAT.
    assert "python_off" not in asset
    assert url.endswith(asset)
    assert "/v9.9.9/" in url


def test_macos_has_nothing_to_install(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    # None, not an exception: an unsupported platform is a fact to report,
    # not a failure to raise.
    assert installer.asset_for_platform() is None


@pytest.mark.parametrize("os_release, expected", [
    ('ID=ubuntu\nVERSION_ID="24.04"\n', "ubuntu24"),
    ('ID=ubuntu\nVERSION_ID="22.04"\n', "ubuntu22"),
    ('ID=rhel\nVERSION_ID="9.4"\n', "redhat"),
    ('ID=rocky\nVERSION_ID="9.3"\n', "redhat"),
    ('ID=linuxmint\nID_LIKE=ubuntu\nVERSION_ID="21"\n', "ubuntu24"),
    ('ID=arch\n', "ubuntu24"),          # unknown → newest, and callers warn
])
def test_linux_flavour_from_os_release(tmp_path, monkeypatch,
                                       os_release, expected):
    release = tmp_path / "os-release"
    release.write_text(os_release, encoding="utf-8")

    real_open = open

    def fake_open(path, *args, **kwargs):
        if path == "/etc/os-release":
            return real_open(release, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    assert installer._linux_flavour() == expected


def test_missing_os_release_falls_back(monkeypatch):
    def boom(path, *args, **kwargs):
        if path == "/etc/os-release":
            raise OSError("no such file")
        raise AssertionError("unexpected open")

    monkeypatch.setattr("builtins.open", boom)
    assert installer._linux_flavour() == "ubuntu24"


# --------------------------------------------------------------------------
# Checksums
# --------------------------------------------------------------------------

def test_checksum_tries_both_upstream_namings(monkeypatch):
    """Upstream is inconsistent: `.zip.sha256` for one build, `.sha256` for
    the other. Trying only the documented one silently skips verification."""
    digest = "a" * 64
    seen = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(url, timeout=None):
        seen.append(url)
        if url.endswith("_python_on.sha256"):
            return Response(f"{digest}  ovms.zip\n".encode())
        raise OSError("404")

    monkeypatch.setattr(installer.urllib.request, "urlopen", fake_urlopen)
    got = installer._expected_sha256(
        "https://example.invalid/ovms_windows_1.0_python_on.zip")
    assert got == digest
    assert seen[0].endswith(".zip.sha256")      # documented form tried first
    assert seen[1].endswith("_python_on.sha256")


def test_missing_checksum_is_tolerated(monkeypatch):
    def always_404(url, timeout=None):
        raise OSError("404")

    monkeypatch.setattr(installer.urllib.request, "urlopen", always_404)
    # None, not a raise: refusing to install because Intel forgot a sibling
    # file would be our bug on their release process.
    assert installer._expected_sha256("https://example.invalid/x.zip") is None


# --------------------------------------------------------------------------
# Extraction and install
# --------------------------------------------------------------------------

def _zip_with_top_level_ovms(path, exe_name):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"ovms/{exe_name}", "binary")
        zf.writestr("ovms/lib/libfoo.so", "lib")


def test_extract_flattens_the_top_level_directory(tmp_path):
    archive = tmp_path / "a.zip"
    _zip_with_top_level_ovms(archive, installer._EXE)
    dest = tmp_path / "root"

    installer._extract(str(archive), str(dest))

    # <root>/ovms.exe, NOT <root>/ovms/ovms.exe -- the locator only searches
    # the former, so one extra level makes a correct download invisible.
    assert (dest / installer._EXE).is_file()
    # The archive's wrapper DIRECTORY must be gone. Checked as is_dir() and not
    # exists(): off Windows _EXE is plain "ovms", so `not (dest/"ovms").exists()`
    # asserted the very path the line above requires to be a file, and the test
    # could only pass on Windows. It has been red on Linux and macOS since it
    # was written, which is how a suite reported green on one machine and
    # nowhere else.
    assert not (dest / "ovms").is_dir()


def test_unsafe_members_are_skipped(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("ovms/ok.txt", "fine")
        zf.writestr("../escaped.txt", "should not be written")
    dest = tmp_path / "root"

    installer._extract(str(archive), str(dest))

    assert (dest / "ok.txt").is_file()
    assert not (tmp_path / "escaped.txt").exists()


def test_tar_gz_is_supported(tmp_path):
    archive = tmp_path / "a.tar.gz"
    payload = tmp_path / "ovms_stage"
    (payload / "bin").mkdir(parents=True)
    (payload / "bin" / "ovms").write_text("binary")
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(payload, arcname="ovms")
    dest = tmp_path / "root"

    installer._extract(str(archive), str(dest))

    assert (dest / "bin" / "ovms").is_file()


def test_installed_binary_finds_both_layouts(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "_EXE", "ovms")
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "ovms").write_text("x")
    assert installer.installed_binary(str(flat)) == str(flat / "ovms")

    nested = tmp_path / "nested"
    (nested / "bin").mkdir(parents=True)
    (nested / "bin" / "ovms").write_text("x")
    assert installer.installed_binary(str(nested)) == str(nested / "bin" / "ovms")

    assert installer.installed_binary(str(tmp_path / "empty")) is None


def _install_from_local_zip(tmp_path, monkeypatch, digest_override=None):
    """Wire install() to a local archive so no network is touched."""
    archive = tmp_path / "src.zip"
    _zip_with_top_level_ovms(archive, installer._EXE)
    raw = archive.read_bytes()
    real_digest = hashlib.sha256(raw).hexdigest()

    monkeypatch.setattr(installer, "asset_for_platform",
                        lambda version=None: ("src.zip", "file:///src.zip"))
    monkeypatch.setattr(installer, "_download",
                        lambda url, target, on_progress=None:
                        open(target, "wb").write(raw))
    monkeypatch.setattr(installer, "_expected_sha256",
                        lambda url: digest_override or real_digest)
    return archive


def test_install_places_the_binary_and_is_idempotent(tmp_path, monkeypatch):
    _install_from_local_zip(tmp_path, monkeypatch)
    root = tmp_path / "root"

    binary, what = installer.install(root=str(root))
    assert what == "installed"
    assert os.path.isfile(binary)

    # Second call must not re-download; that is what makes it safe to call
    # from `serve` and from scripts.
    binary_again, what_again = installer.install(root=str(root))
    assert what_again == "already-installed"
    assert binary_again == binary


def test_checksum_mismatch_installs_nothing(tmp_path, monkeypatch):
    _install_from_local_zip(tmp_path, monkeypatch, digest_override="b" * 64)
    root = tmp_path / "root"

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        installer.install(root=str(root))

    # A corrupt download must not leave a half-install behind for the locator
    # to find and try to run.
    assert installer.installed_binary(str(root)) is None


def test_install_refuses_on_macos(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "asset_for_platform",
                        lambda version=None: None)
    with pytest.raises(RuntimeError, match="macOS"):
        installer.install(root=str(tmp_path / "root"))


def _tar_with_read_only_tree(path):
    """An archive shaped like the real one: 0o555 dirs and libraries."""
    import tarfile, tempfile
    src = tempfile.mkdtemp()
    os.makedirs(os.path.join(src, "ovms", "lib"))
    os.makedirs(os.path.join(src, "ovms", "bin"))
    for rel in ("ovms/lib/libovms_shared.so", f"ovms/bin/{installer._EXE}"):
        with open(os.path.join(src, rel), "w", encoding="utf-8") as handle:
            handle.write("x")

    def read_only(info):
        info.mode = 0o555
        return info

    with tarfile.open(path, "w:gz") as tf:
        tf.add(os.path.join(src, "ovms"), arcname="ovms", filter=read_only)
    return src


@pytest.mark.skipif(sys.platform == "win32",
                    reason="Unix permission bits; Windows zip restores none")
def test_extract_survives_the_read_only_modes_the_archive_ships(tmp_path):
    """OVMS ships 0o555 directories, and unlinking needs write on the PARENT.

    shutil.move falls back to copy-then-rmtree across filesystems, so this blew
    up with EACCES on 'libovms_shared.so' for every non-root user. Root has
    CAP_DAC_OVERRIDE and never saw it, which is why Docker, CI and the
    maintainer's own container all reported success.
    """
    archive = tmp_path / "ovms.tar.gz"
    _tar_with_read_only_tree(str(archive))
    dest = tmp_path / "root"

    installer._extract(str(archive), str(dest))
    assert (dest / "bin" / installer._EXE).is_file()


@pytest.mark.skipif(sys.platform == "win32", reason="Unix permission bits")
def test_extract_can_overwrite_a_previous_read_only_install(tmp_path):
    """Re-running setup must replace an install whose dirs are also 0o555.

    rmtree(ignore_errors=True) turned "could not delete the old one" into a
    silent no-op, so a re-install would look fine and leave the old build.
    """
    archive = tmp_path / "ovms.tar.gz"
    _tar_with_read_only_tree(str(archive))
    dest = tmp_path / "root"

    installer._extract(str(archive), str(dest))
    installer._extract(str(archive), str(dest))     # the second one is the test
    assert (dest / "bin" / installer._EXE).is_file()


@pytest.mark.parametrize("os_release, expect_note", [
    ('ID=ubuntu\nVERSION_ID="24.04"\n', False),
    ('ID=ubuntu\nVERSION_ID="22.04"\n', False),
    ('ID=ubuntu\nVERSION_ID="26.04"\n', True),      # no build exists; must warn
    ('ID=rhel\nVERSION_ID="9.4"\n', False),
    ('ID=arch\n', True),
])
def test_unsupported_distros_are_warned_about(tmp_path, monkeypatch,
                                              os_release, expect_note):
    """Ubuntu 26.04 was handed the ubuntu24 archive silently, and it cannot
    start there. A guess is allowed; a SILENT guess is not."""
    monkeypatch.setattr(sys, "platform", "linux")
    release = tmp_path / "os-release"
    release.write_text(os_release, encoding="utf-8")
    real_open = open

    def fake_open(path, *args, **kwargs):
        if path == "/etc/os-release":
            return real_open(release, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    note = installer.linux_support_note()
    assert (note is not None) is expect_note
    if expect_note:
        assert "ubuntu24" in note
