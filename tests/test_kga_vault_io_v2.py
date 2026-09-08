"""Synthetic-only tests for the bounded vault byte primitive."""

from __future__ import annotations

import errno
import hashlib
import os
import socket
import stat
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest

from kga.vault_io import MAX_VAULT_OBJECT_BYTES, READ_CHUNK_BYTES, VaultDirectory, VaultIOError


@pytest.fixture(autouse=True)
def descriptor_lifetime(monkeypatch):
    """Every successful os.open must be noninheritable and closed once.

    Fault tests release ambiguous descriptors themselves when simulating the
    alternative where the kernel leaves the descriptor open after an error.
    Python file helpers manage their own descriptors independently of os.open.
    """
    real_open, real_close = os.open, os.close
    live = set()

    def tracked_open(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        assert not os.get_inheritable(fd)
        assert fd not in live
        live.add(fd)
        return fd

    def tracked_close(fd):
        real_close(fd)
        live.discard(fd)

    monkeypatch.setattr(os, "open", tracked_open)
    monkeypatch.setattr(os, "close", tracked_close)
    yield
    assert not live, f"unreleased synthetic descriptors: {live}"


def cap(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    st = root.stat()
    return root, VaultDirectory.open(str(root), expected_device=st.st_dev, expected_inode=st.st_ino)


def test_round_trip_and_no_overwrite(tmp_path):
    root, vault = cap(tmp_path)
    try:
        data = b"synthetic-public-bytes"
        digest = vault.create_bytes("public.json", data, max_bytes=1024)
        assert digest == hashlib.sha256(data).hexdigest()
        assert vault.read_bytes("public.json", expected_sha256=digest, max_bytes=1024) == data
        with pytest.raises(VaultIOError, match="ALREADY_EXISTS"):
            vault.create_bytes("public.json", b"changed", max_bytes=1024)
        assert stat.S_IMODE((root / "public.json").stat().st_mode) & 0o077 == 0
    finally:
        vault.close()


@pytest.mark.parametrize("name", ["../private.json", "./private.json", "Private.json", "x", "a/b.json", ""])
def test_rejects_leaf_path_before_open(tmp_path, name):
    _, vault = cap(tmp_path)
    try:
        with pytest.raises(VaultIOError, match="INVALID_OBJECT_NAME"):
            vault.read_bytes(name, expected_sha256="0" * 64, max_bytes=1)
    finally:
        vault.close()


def test_rejects_hardlink_before_content_read(tmp_path):
    root, vault = cap(tmp_path)
    try:
        (root / "real.json").write_bytes(b"private")
        os.link(root / "real.json", root / "linked.json")
        with pytest.raises(VaultIOError, match="NOT_SINGLE_REGULAR"):
            vault.read_bytes("linked.json", expected_sha256=hashlib.sha256(b"private").hexdigest(), max_bytes=100)
    finally:
        vault.close()


def test_rejects_limits_and_wrong_digest(tmp_path):
    _, vault = cap(tmp_path)
    try:
        with pytest.raises(VaultIOError, match="INVALID_LIMIT"):
            vault.create_bytes("x.json", b"x", max_bytes=MAX_VAULT_OBJECT_BYTES + 1)
        with pytest.raises(VaultIOError, match="DIGEST_MISMATCH"):
            vault.create_bytes("x.json", b"x", max_bytes=10)
            vault.read_bytes("x.json", expected_sha256="0" * 64, max_bytes=10)
    finally:
        vault.close()


def test_close_is_idempotent_and_use_after_close_fails(tmp_path):
    _, vault = cap(tmp_path)
    vault.close()
    vault.close()
    with pytest.raises(VaultIOError, match="CLOSED"):
        vault.read_bytes("x.json", expected_sha256="0" * 64, max_bytes=1)


def test_symlink_parent_is_rejected(tmp_path):
    real = tmp_path / "real"
    root = real / "vault"
    root.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    st = root.stat()
    with pytest.raises(VaultIOError):
        with VaultDirectory.open(str(alias / "vault"), expected_device=st.st_dev, expected_inode=st.st_ino):
            pass


@pytest.mark.parametrize("operation", ["read", "write"])
def test_object_close_failure_returns_no_success_and_poisons_without_retry(tmp_path, monkeypatch, operation):
    root, vault = cap(tmp_path)
    (root / "public.json").write_bytes(b"x")
    original_close = os.close
    object_closes = []

    def close_with_eio(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            object_closes.append(fd)
            # Simulate the valid outcome where close reports EIO after releasing
            # the descriptor. A retry must not close a reused descriptor.
            original_close(fd)
            raise OSError(errno.EIO, "synthetic close failure")
        return original_close(fd)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "close", close_with_eio)
            with pytest.raises(VaultIOError, match="CLOSE_FAILURE"):
                if operation == "read":
                    vault.read_bytes("public.json", expected_sha256=hashlib.sha256(b"x").hexdigest(), max_bytes=1)
                else:
                    vault.create_bytes("created.json", b"x", max_bytes=1)
        with pytest.raises(VaultIOError, match="CLOSED"):
            vault.create_bytes("after.json", b"x", max_bytes=1)
        assert len(object_closes) == 1
    finally:
        vault.close()


def test_context_exit_closes_on_exception_and_rejects_every_operation(tmp_path):
    _, vault = cap(tmp_path)
    with pytest.raises(RuntimeError), vault as entered:
        assert entered is vault
        raise RuntimeError("synthetic host failure")
    vault.close()
    for operation in (
        lambda: vault.__enter__(),
        lambda: vault.read_bytes("x.json", expected_sha256="0" * 64, max_bytes=1),
        lambda: vault.create_bytes("x.json", b"x", max_bytes=1),
    ):
        with pytest.raises(VaultIOError, match="CLOSED"):
            operation()


@pytest.mark.parametrize("field", ["device", "inode"])
def test_wrong_root_identity_closes_descriptors(tmp_path, field):
    st = tmp_path.stat()
    with pytest.raises(VaultIOError, match="ROOT_IDENTITY"):
        VaultDirectory.open(
            str(tmp_path),
            expected_device=st.st_dev + (field == "device"),
            expected_inode=st.st_ino + (field == "inode"),
        )


@pytest.mark.parametrize("value", [True, False, -1, 1.0, "1", None])
@pytest.mark.parametrize("field", ["device", "inode"])
def test_root_identity_validation_precedes_open(tmp_path, monkeypatch, field, value):
    def unexpected_open(*args, **kwargs):
        pytest.fail("invalid identity reached filesystem")

    monkeypatch.setattr(os, "open", unexpected_open)
    with pytest.raises(VaultIOError, match="INVALID_ROOT_IDENTITY"):
        VaultDirectory.open(
            str(tmp_path),
            expected_device=value if field == "device" else 0,
            expected_inode=value if field == "inode" else 0,
        )


@pytest.mark.parametrize("suffix", ["/", "//child", "/./child", "/../child", "\x00child"])
def test_root_lexical_alias_validation_precedes_open(tmp_path, monkeypatch, suffix):
    def unexpected_open(*args, **kwargs):
        pytest.fail("invalid root spelling reached filesystem")

    monkeypatch.setattr(os, "open", unexpected_open)
    with pytest.raises(VaultIOError, match="INVALID_ROOT"):
        VaultDirectory.open(str(tmp_path) + suffix, expected_device=0, expected_inode=0)


@pytest.mark.parametrize("value", ["relative", "./relative", "", "/", "//root", Path("/physical"), None, 1])
def test_root_requires_exact_native_absolute_string(monkeypatch, value):
    def unexpected_open(*args, **kwargs):
        pytest.fail("invalid root type reached filesystem")

    monkeypatch.setattr(os, "open", unexpected_open)
    with pytest.raises(VaultIOError, match="INVALID_ROOT"):
        VaultDirectory.open(value, expected_device=0, expected_inode=0)


@pytest.mark.parametrize("kind", ["root_symlink", "regular", "missing"])
def test_root_must_exist_as_real_directory(tmp_path, kind):
    root = tmp_path / "root"
    if kind == "root_symlink":
        root.symlink_to(tmp_path, target_is_directory=True)
    elif kind == "regular":
        root.write_bytes(b"synthetic")
    with pytest.raises(VaultIOError):
        VaultDirectory.open(str(root), expected_device=tmp_path.stat().st_dev, expected_inode=0)


@pytest.mark.parametrize("operation", ["read", "write"])
@pytest.mark.parametrize("value", [True, False, -1, 0, MAX_VAULT_OBJECT_BYTES + 1, 1.5, "1", None])
def test_all_limits_validate_before_leaf_open(tmp_path, monkeypatch, operation, value):
    _, vault = cap(tmp_path)
    with vault:

        def unexpected_open(*args, **kwargs):
            pytest.fail("invalid limit reached filesystem")

        monkeypatch.setattr(os, "open", unexpected_open)
        with pytest.raises(VaultIOError, match="INVALID_LIMIT"):
            if operation == "read":
                vault.read_bytes("x.json", expected_sha256="0" * 64, max_bytes=value)
            else:
                vault.create_bytes("x.json", b"x", max_bytes=value)


@pytest.mark.parametrize("value", ["", "0" * 63, "0" * 65, "A" * 64, "g" * 64, "0" * 64 + "\n", b"0" * 64, None])
def test_digest_validation_precedes_leaf_open(tmp_path, monkeypatch, value):
    _, vault = cap(tmp_path)
    with vault:

        def unexpected_open(*args, **kwargs):
            pytest.fail("invalid digest reached filesystem")

        monkeypatch.setattr(os, "open", unexpected_open)
        with pytest.raises(VaultIOError, match="INVALID_DIGEST"):
            vault.read_bytes("x.json", expected_sha256=value, max_bytes=1)


@pytest.mark.parametrize("value", ["x", bytearray(b"x"), memoryview(b"x"), None, 1, b"xx"])
def test_data_validation_precedes_creation(tmp_path, monkeypatch, value):
    root, vault = cap(tmp_path)
    with vault:

        def unexpected_open(*args, **kwargs):
            pytest.fail("invalid data reached filesystem")

        monkeypatch.setattr(os, "open", unexpected_open)
        with pytest.raises(VaultIOError, match="INVALID_DATA"):
            vault.create_bytes("x.json", value, max_bytes=1)
        assert not (root / "x.json").exists()


@pytest.mark.parametrize("operation", ["read", "write"])
@pytest.mark.parametrize(
    "name",
    [
        "/x.json",
        "a\\x.json",
        "x\x00.json",
        "é.json",
        "X.json",
        ".json",
        "x.json\n",
        "a" * 97 + ".json",
        b"x.json",
        None,
    ],
)
def test_basename_grammar_validates_before_leaf_open(tmp_path, monkeypatch, operation, name):
    _, vault = cap(tmp_path)
    with vault:

        def unexpected_open(*args, **kwargs):
            pytest.fail("invalid name reached filesystem")

        monkeypatch.setattr(os, "open", unexpected_open)
        with pytest.raises(VaultIOError, match="INVALID_OBJECT_NAME"):
            if operation == "read":
                vault.read_bytes(name, expected_sha256="0" * 64, max_bytes=1)
            else:
                vault.create_bytes(name, b"x", max_bytes=1)


def test_boundary_names_empty_payload_and_ceiling_limit(tmp_path):
    _, vault = cap(tmp_path)
    with vault:
        for name in ("0.json", "a_b-c.json", "a" * 96 + ".json"):
            digest = vault.create_bytes(name, b"", max_bytes=MAX_VAULT_OBJECT_BYTES)
            assert digest == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            assert vault.read_bytes(name, expected_sha256=digest, max_bytes=MAX_VAULT_OBJECT_BYTES) == b""


def test_renamed_parent_and_replaced_root_remain_pinned(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    root, vault = cap(parent)
    with vault:
        digest = vault.create_bytes("public.json", b"original", max_bytes=100)
        moved = tmp_path / "moved"
        parent.rename(moved)
        root.mkdir(parents=True)
        (root / "public.json").write_bytes(b"replacement")
        assert vault.read_bytes("public.json", expected_sha256=digest, max_bytes=100) == b"original"
        vault.create_bytes("new.json", b"pinned", max_bytes=100)
        assert (moved / "vault" / "new.json").read_bytes() == b"pinned"
        assert not (root / "new.json").exists()


def test_leaf_replacement_after_open_reads_only_original_descriptor(tmp_path, monkeypatch):
    root, vault = cap(tmp_path)
    with vault:
        digest = vault.create_bytes("public.json", b"original", max_bytes=100)
        real_open = os.open

        def replace_after_open(path, flags, *args, **kwargs):
            fd = real_open(path, flags, *args, **kwargs)
            if path == "public.json":
                (root / "public.json").rename(root / "saved.json")
                (root / "public.json").write_bytes(b"replacement")
            return fd

        monkeypatch.setattr(os, "open", replace_after_open)
        assert vault.read_bytes("public.json", expected_sha256=digest, max_bytes=100) == b"original"


@pytest.mark.parametrize("kind", ["hardlink", "directory", "fifo", "socket", "symlink"])
def test_special_file_is_rejected_before_read_and_never_overwritten(tmp_path, monkeypatch, kind):
    root, vault = cap(tmp_path)
    target = root / "target.json"
    target.write_bytes(b"sentinel synthetic bytes")
    leaf = root / "special.json"
    sock = None
    if kind == "hardlink":
        os.link(target, leaf)
    elif kind == "directory":
        leaf.mkdir()
    elif kind == "fifo":
        os.mkfifo(leaf)
    elif kind == "socket":
        sock = socket.socket(socket.AF_UNIX)
        # macOS AF_UNIX paths have a short length limit; binding via cwd keeps
        # the real socket on APFS without choosing an unrelated scratch root.
        previous = Path.cwd()
        try:
            os.chdir(root)
            sock.bind("special.json")
        finally:
            os.chdir(previous)
    else:
        leaf.symlink_to(target)
    before = leaf.lstat()
    try:
        with vault:

            def unexpected_read(*args):
                pytest.fail("special file contents were read")

            monkeypatch.setattr(os, "read", unexpected_read)
            with pytest.raises(VaultIOError):
                vault.read_bytes("special.json", expected_sha256="0" * 64, max_bytes=100)
            with pytest.raises(VaultIOError, match="ALREADY_EXISTS"):
                vault.create_bytes("special.json", b"changed", max_bytes=100)
            assert target.read_bytes() == b"sentinel synthetic bytes"
            after = leaf.lstat()
            assert (after.st_ino, after.st_mode) == (before.st_ino, before.st_mode)
    finally:
        if sock is not None:
            sock.close()


def test_read_flags_and_chunk_budget(tmp_path, monkeypatch):
    root, vault = cap(tmp_path)
    payload = b"x" * (READ_CHUNK_BYTES * 2 + 7)
    (root / "public.json").write_bytes(payload)
    real_open, real_read = os.open, os.read
    sizes = []

    def checked_open(path, flags, *args, **kwargs):
        assert kwargs.get("dir_fd") is not None
        assert flags & os.O_NOFOLLOW
        assert flags & os.O_NONBLOCK
        assert flags & os.O_CLOEXEC
        return real_open(path, flags, *args, **kwargs)

    def checked_read(fd, size):
        sizes.append(size)
        assert 0 < size <= READ_CHUNK_BYTES
        return real_read(fd, size)

    with vault:
        monkeypatch.setattr(os, "open", checked_open)
        monkeypatch.setattr(os, "read", checked_read)
        assert (
            vault.read_bytes("public.json", expected_sha256=hashlib.sha256(payload).hexdigest(), max_bytes=len(payload))
            == payload
        )
    assert len(sizes) == 4
    assert sum(sizes) == len(payload) + 2


def test_declared_oversize_is_rejected_before_read(tmp_path, monkeypatch):
    root, vault = cap(tmp_path)
    (root / "public.json").write_bytes(b"xx")
    with vault:

        def unexpected_read(*args):
            pytest.fail("declared oversize contents were read")

        monkeypatch.setattr(os, "read", unexpected_read)
        with pytest.raises(VaultIOError, match="TOO_LARGE"):
            vault.read_bytes("public.json", expected_sha256="0" * 64, max_bytes=1)


@pytest.mark.parametrize("mutation", ["growth", "truncate", "same_size", "metadata", "ctime"])
def test_observed_read_mutation_never_returns_payload(tmp_path, monkeypatch, mutation):
    root, vault = cap(tmp_path)
    leaf = root / "public.json"
    leaf.write_bytes(b"original")
    real_read = os.read
    changed = False

    def mutate_during_read(fd, size):
        nonlocal changed
        if not changed:
            changed = True
            if mutation == "growth":
                with leaf.open("ab") as stream:
                    stream.write(b"extra")
            elif mutation == "truncate":
                leaf.write_bytes(b"x")
            elif mutation == "same_size":
                leaf.write_bytes(b"modified")
            elif mutation == "ctime":
                before = leaf.stat()
                leaf.chmod(0o400)
                after = leaf.stat()
                assert before.st_mtime_ns == after.st_mtime_ns
                assert before.st_ctime_ns != after.st_ctime_ns
            else:
                st = leaf.stat()
                os.utime(leaf, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        return real_read(fd, size)

    with vault:
        monkeypatch.setattr(os, "read", mutate_during_read)
        with pytest.raises(VaultIOError, match="TOO_LARGE|MUTATED|SIZE_CHANGED"):
            vault.read_bytes("public.json", expected_sha256=hashlib.sha256(b"original").hexdigest(), max_bytes=8)


def test_short_writes_and_durability_order(tmp_path, monkeypatch):
    root, vault = cap(tmp_path)
    real_write, real_fsync, real_close = os.write, os.fsync, os.close
    events = []

    def short_write(fd, data):
        events.append("write")
        return real_write(fd, data[:2])

    def fsync(fd):
        events.append("parent_fsync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file_fsync")
        return real_fsync(fd)

    def close(fd):
        events.append("root_close" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file_close")
        return real_close(fd)

    with vault:
        monkeypatch.setattr(os, "write", short_write)
        monkeypatch.setattr(os, "fsync", fsync)
        monkeypatch.setattr(os, "close", close)
        digest = vault.create_bytes("public.json", b"abcdef", max_bytes=6)
        assert digest == hashlib.sha256(b"abcdef").hexdigest()
        assert (root / "public.json").read_bytes() == b"abcdef"
        assert events == ["write", "write", "write", "file_fsync", "file_close", "parent_fsync"]


@pytest.mark.parametrize("failure", ["zero", "enospc", "file_fsync", "parent_fsync"])
def test_write_failures_preserve_attempt_and_refuse_overwrite(tmp_path, monkeypatch, failure):
    root, vault = cap(tmp_path)
    real_write, real_fsync = os.write, os.fsync
    writes = 0

    def write(fd, data):
        nonlocal writes
        writes += 1
        if writes == 1:
            return real_write(fd, data[:1])
        if failure == "zero":
            return 0
        if failure == "enospc":
            raise OSError(errno.ENOSPC, "synthetic disk full")
        return real_write(fd, data)

    def fsync(fd):
        is_parent = stat.S_ISDIR(os.fstat(fd).st_mode)
        if failure == ("parent_fsync" if is_parent else "file_fsync"):
            raise OSError(errno.EIO, "synthetic durability failure")
        return real_fsync(fd)

    with vault:
        with monkeypatch.context() as patch:
            patch.setattr(os, "write", write)
            patch.setattr(os, "fsync", fsync)
            with pytest.raises(VaultIOError, match="WRITE_FAILURE|IO_FAILURE"):
                vault.create_bytes("public.json", b"abc", max_bytes=3)
        expected = b"a" if failure in {"zero", "enospc"} else b"abc"
        assert (root / "public.json").read_bytes() == expected
        with pytest.raises(VaultIOError, match="ALREADY_EXISTS"):
            vault.create_bytes("public.json", b"replacement", max_bytes=100)
        assert (root / "public.json").read_bytes() == expected


@pytest.mark.parametrize("mode", [0o700, 0o1600, 0o2600, 0o4600, 0o640])
def test_write_rejects_permissions_exceeding_0600(tmp_path, monkeypatch, mode):
    root, vault = cap(tmp_path)
    real_write = os.write

    def change_permissions(fd, data):
        written = real_write(fd, data)
        os.fchmod(fd, mode)
        return written

    with vault:
        monkeypatch.setattr(os, "write", change_permissions)
        with pytest.raises(VaultIOError, match="WRITE_IDENTITY"):
            vault.create_bytes("public.json", b"x", max_bytes=1)
        assert (root / "public.json").read_bytes() == b"x"


@pytest.mark.parametrize("kind", ["hardlink", "size", "device", "type"])
def test_write_final_identity_checks_before_fsync(tmp_path, monkeypatch, kind):
    root, vault = cap(tmp_path)
    real_write, real_fstat = os.write, os.fstat

    def write(fd, data):
        count = real_write(fd, data)
        if kind == "hardlink":
            os.link(root / "public.json", root / "extra.json")
        elif kind == "size":
            os.ftruncate(fd, 0)
        return count

    def fstat(fd):
        actual = real_fstat(fd)
        if kind in {"device", "type"} and stat.S_ISREG(actual.st_mode):
            fields = {name: getattr(actual, name) for name in dir(actual) if name.startswith("st_")}
            fields["st_dev" if kind == "device" else "st_mode"] = (
                actual.st_dev + 1 if kind == "device" else stat.S_IFDIR | 0o600
            )
            return SimpleNamespace(**fields)
        return actual

    def unexpected_fsync(fd):
        pytest.fail("invalid write identity reached durability success path")

    with vault:
        monkeypatch.setattr(os, "write", write)
        monkeypatch.setattr(os, "fstat", fstat)
        monkeypatch.setattr(os, "fsync", unexpected_fsync)
        with pytest.raises(VaultIOError, match="NOT_SINGLE_REGULAR|SIZE_CHANGED|WRITE_IDENTITY"):
            vault.create_bytes("public.json", b"x", max_bytes=1)
        assert (root / "public.json").exists()


@pytest.mark.parametrize("operation", ["read", "write", "failed_read", "failed_write"])
@pytest.mark.parametrize("released", [True, False])
def test_ambiguous_object_close_is_never_retried(tmp_path, monkeypatch, operation, released):
    root, vault = cap(tmp_path)
    (root / "public.json").write_bytes(b"x")
    real_close, real_open = os.close, os.open
    ambiguous = []
    replacement = []
    attempts = []

    def close(fd):
        if ambiguous and fd == ambiguous[0]:
            attempts.append(fd)
            pytest.fail("ambiguous descriptor close retried")
        if stat.S_ISREG(os.fstat(fd).st_mode):
            ambiguous.append(fd)
            attempts.append(fd)
            if released:
                real_close(fd)
                replacement.append(real_open(root / "public.json", os.O_RDONLY | os.O_CLOEXEC))
                assert replacement[0] == fd
            raise OSError(errno.EIO, "synthetic ambiguous close")
        return real_close(fd)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "close", close)
            with pytest.raises(VaultIOError, match="CLOSE_FAILURE"):
                if operation in {"read", "failed_read"}:
                    digest = hashlib.sha256(b"x").hexdigest() if operation == "read" else "0" * 64
                    vault.read_bytes("public.json", expected_sha256=digest, max_bytes=1)
                else:
                    if operation == "failed_write":
                        patch.setattr(os, "write", lambda fd, data: 0)
                    vault.create_bytes("created.json", b"x", max_bytes=1)
        assert len(attempts) == 1
        assert os.fstat(ambiguous[0]).st_size == (0 if operation == "failed_write" and not released else 1)
        for action in (
            lambda: vault.__enter__(),
            lambda: vault.read_bytes("public.json", expected_sha256="0" * 64, max_bytes=1),
            lambda: vault.create_bytes("after.json", b"x", max_bytes=1),
        ):
            with pytest.raises(VaultIOError, match="CLOSED"):
                action()
        if "write" in operation:
            assert (root / "created.json").exists()
    finally:
        vault.close()
        for fd in replacement if released else ambiguous:
            real_close(fd)


def test_short_reads_make_bounded_progress_and_hash_complete_bytes(tmp_path, monkeypatch):
    root, vault = cap(tmp_path)
    payload = b"synthetic short reads"
    (root / "public.json").write_bytes(payload)
    original_read = os.read
    count = 0

    def short_read(fd, size):
        nonlocal count
        count += 1
        assert 0 < size <= READ_CHUNK_BYTES
        assert count <= len(payload) + 1
        return original_read(fd, min(size, 1))

    with vault:
        monkeypatch.setattr(os, "read", short_read)
        assert (
            vault.read_bytes("public.json", expected_sha256=hashlib.sha256(payload).hexdigest(), max_bytes=len(payload))
            == payload
        )
    assert count == len(payload) + 1


@pytest.mark.parametrize("failure", ["pre_fstat", "read", "post_fstat"])
def test_read_os_failure_closes_file_and_preserves_capability(tmp_path, monkeypatch, failure):
    root, vault = cap(tmp_path)
    payload = b"synthetic"
    (root / "public.json").write_bytes(payload)
    real_fstat, real_read = os.fstat, os.read
    stats = 0

    def fstat(fd):
        nonlocal stats
        stats += 1
        if (failure == "pre_fstat" and stats == 1) or (failure == "post_fstat" and stats == 2):
            raise OSError(errno.EIO, "synthetic stat failure")
        return real_fstat(fd)

    def read(fd, size):
        if failure == "read":
            raise OSError(errno.EIO, "synthetic read failure")
        return real_read(fd, size)

    with vault:
        with monkeypatch.context() as patch:
            patch.setattr(os, "fstat", fstat)
            patch.setattr(os, "read", read)
            with pytest.raises(VaultIOError, match="IO_FAILURE"):
                vault.read_bytes("public.json", expected_sha256=hashlib.sha256(payload).hexdigest(), max_bytes=100)
        assert (
            vault.read_bytes("public.json", expected_sha256=hashlib.sha256(payload).hexdigest(), max_bytes=100)
            == payload
        )


def test_root_close_failure_is_terminal_and_not_retried(tmp_path, monkeypatch):
    root, vault = cap(tmp_path)
    real_close, real_open = os.close, os.open
    replacement = []
    attempts = []

    def close(fd):
        attempts.append(fd)
        real_close(fd)
        replacement.append(real_open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC))
        raise OSError(errno.EIO, "synthetic close failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "close", close)
            with pytest.raises(VaultIOError, match="CLOSE_FAILURE"):
                vault.close()
            vault.close()
            with pytest.raises(VaultIOError, match="CLOSED"):
                vault.__enter__()
        assert len(attempts) == 1
        assert stat.S_ISDIR(os.fstat(replacement[0]).st_mode)
    finally:
        for fd in replacement:
            real_close(fd)


def test_root_walk_close_failure_aborts_without_retry(tmp_path, monkeypatch):
    st = tmp_path.stat()
    real_close = os.close
    failed = []

    def close(fd):
        if fd in failed:
            pytest.fail("root walk retried ambiguous close")
        real_close(fd)
        if not failed:
            failed.append(fd)
            raise OSError(errno.EIO, "synthetic walk close failure")

    monkeypatch.setattr(os, "close", close)
    with pytest.raises(VaultIOError, match="CLOSE_FAILURE"):
        VaultDirectory.open(str(tmp_path), expected_device=st.st_dev, expected_inode=st.st_ino)


@pytest.mark.parametrize("operation", ["root", "read_open", "read", "write", "fsync", "close"])
def test_native_private_details_are_suppressed_in_errors_and_tracebacks(tmp_path, monkeypatch, operation):
    secret = "".join(("sentinel", "_private_", "path_payload_", "9472"))
    root = tmp_path / secret
    root.mkdir()
    st = root.stat()
    vault = VaultDirectory.open(str(root), expected_device=st.st_dev, expected_inode=st.st_ino)
    name = secret + ".json"
    (root / name).write_bytes(secret.encode())
    real_close = os.close

    def private_error(*args, **kwargs):
        raise OSError(errno.EIO, secret, str(root / name))

    def close(fd):
        real_close(fd)
        private_error()

    try:
        with monkeypatch.context() as patch:
            target = {
                "root": "open",
                "read_open": "open",
                "read": "read",
                "write": "write",
                "fsync": "fsync",
                "close": "close",
            }[operation]
            patch.setattr(os, target, close if operation == "close" else private_error)
            with pytest.raises(VaultIOError) as captured:
                if operation == "root":
                    VaultDirectory.open(str(root), expected_device=st.st_dev, expected_inode=st.st_ino)
                elif operation in {"read_open", "read", "close"}:
                    vault.read_bytes(name, expected_sha256=hashlib.sha256(secret.encode()).hexdigest(), max_bytes=100)
                else:
                    vault.create_bytes("attempt.json", secret.encode(), max_bytes=100)
            assert captured.value.code in {"IO_FAILURE", "CLOSE_FAILURE"}
            rendered = "".join(traceback.format_exception(captured.value))
            assert secret not in str(captured.value)
            assert secret not in rendered
            assert str(root) not in rendered
            assert captured.value.__cause__ is None
            assert captured.value.__suppress_context__
    finally:
        vault.close()
