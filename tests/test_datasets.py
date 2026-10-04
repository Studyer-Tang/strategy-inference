"""TSF format, pinned bytes, and cache transactions without network access."""

import hashlib
import io
import zipfile
from datetime import datetime

import numpy as np
import pytest

import strategy_inference.datasets as datasets


def _tsf(values="1,?,3", *, name="price", missing="true", attributes=""):
    return (
        "# Original observations; no calendar is manufactured.\n"
        "@relation sample\n"
        "@attribute series_name string\n"
        "@attribute start_timestamp date\n"
        f"{attributes}"
        "@frequency daily\n"
        f"@missing {missing}\n"
        "@equallength true\n"
        "@horizon 2\n"
        "@data\n"
        f"{name}:2010-07-17 00-00-00:{values}\n"
    )


def _zip(content, *, names=("sample.tsf",)):
    with io.BytesIO() as stream:
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in names:
                archive.writestr(name, content)
        return stream.getvalue()


def _read(tmp_path, content, *, encoding="cp1252", suffix=".tsf"):
    path = tmp_path / f"local{suffix}"
    path.write_bytes(content if isinstance(content, bytes) else content.encode(encoding))
    return datasets.read_tsf(path, encoding=encoding)


@pytest.fixture
def tiny_archive(monkeypatch, tmp_path):
    content = _zip(_tsf().encode("cp1252"))
    spec = datasets._Archive(
        "tiny_dataset.zip",
        hashlib.sha256(content).hexdigest(),
        len(content),
        "https://zenodo.org/records/example",
    )
    monkeypatch.setattr(datasets, "_ARCHIVES", {"tiny": spec})
    directory = tmp_path / "nested" / "cache"
    path = directory / f"tiny-{datasets._REVISION[:12]}.zip"
    return content, spec, directory, path


def _response(monkeypatch, content):
    calls = []

    def open_url(request, *, timeout):
        calls.append((request, timeout))
        return io.BytesIO(content)

    monkeypatch.setattr(datasets, "urlopen", open_url)
    return calls


def _no_network(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Network accessed for a cached/offline read")

    monkeypatch.setattr(datasets, "urlopen", fail)


def test_local_values_missing_positions_dates_and_headers_are_preserved(tmp_path):
    original = _tsf(values="1.123456789012345,?,0,-2e-4")
    result = _read(tmp_path, original)
    assert result.name == "sample"
    assert result.names == ("price",)
    assert result.frequency == "daily"
    item = result["price"]
    assert item.values.dtype == np.float64
    np.testing.assert_array_equal(item.values, [1.123456789012345, np.nan, 0, -0.0002])
    assert item.start_timestamp == datetime(2010, 7, 17)
    assert item.start_timestamp.tzinfo is None
    assert result.metadata["horizon"] == "2"
    assert result.sha256 == hashlib.sha256(original.encode("cp1252")).hexdigest()
    assert result.source is result.license is result.revision is None
    with pytest.raises(ValueError):
        item.values[0] = 0
    with pytest.raises(TypeError):
        item.attributes["start_timestamp"] = None
    with pytest.raises(TypeError):
        result.metadata["frequency"] = "monthly"
    with pytest.raises(KeyError):
        result["not-present"]


def test_cp1252_unicode_comments_and_local_utf8_are_explicit(tmp_path):
    cp = _read(tmp_path, _tsf(name="café £"))
    assert cp.names == ("café £",)
    utf = _read(tmp_path, _tsf(name="气温"), encoding="utf-8")
    assert utf.names == ("气温",)
    with pytest.raises(UnicodeDecodeError):
        _read(tmp_path, _tsf(name="café").encode("cp1252"), encoding="utf-8")


def test_no_timestamp_or_name_attribute_creates_no_fictional_calendar(tmp_path):
    text = "@attribute height numeric\n@data\n1.25:1,2\n-2e2:3,4,5\n"
    result = _read(tmp_path, text)
    assert result.names == ("1", "2")
    assert result.frequency is None
    assert result["1"].attributes == {"height": 1.25}
    assert result["2"].attributes == {"height": -200.0}
    assert all(item.start_timestamp is None for item in result.series)
    assert [len(item.values) for item in result.series] == [2, 3]


def test_multiple_series_remain_separate_and_ordered(tmp_path):
    text = _tsf(values="1,?,3") + "difficulty:2010-07-17 00-00-00:?,5,6\n"
    result = _read(tmp_path, text)
    assert result.names == ("price", "difficulty")
    np.testing.assert_array_equal(result["difficulty"].values, [np.nan, 5, 6])


@pytest.mark.parametrize(
    "text",
    [
        "",
        "# Only a comment\n",
        "@data\n",
        "@attribute x string\n@data\n",
        "@attribute x string\nfirst:1,2\n@data\n",
        "@attribute x string\n@data extra\nfirst:1,2\n",
        "@attribute x string\n@attribute x numeric\n@data\na:1,2\n",
        "@attribute x integer\n@data\na:1,2\n",
        "@attribute x\n@data\na:1,2\n",
        _tsf().replace("@frequency daily", "@frequency"),
        _tsf().replace("@frequency daily", "@frequency daily\n@frequency hourly"),
        _tsf().replace("@missing true", "@missing yes"),
        _tsf().replace("@equallength true", "@equallength False"),
        _tsf().replace("@horizon 2", "@horizon 0"),
        _tsf().replace("@horizon 2", "@horizon 2.5"),
        _tsf() + "@data\n",
        _tsf() + "price:2010-07-18 00-00-00:1,2,3\n",
        _tsf() + "other:2010-07-18 00-00-00:1,2\n",
        _tsf().replace("price:2010", ":2010"),
        _tsf().replace("2010-07-17", "2010-02-30"),
        _tsf().replace("price:2010-07-17 00-00-00:", "price:"),
        _tsf(missing="false"),
        "@attribute a numeric\n@data\nnan:1,2\n",
        "@attribute a numeric\n@data\ninf:1,2\n",
    ],
)
def test_malformed_headers_records_and_contradictions_raise(tmp_path, text):
    with pytest.raises(ValueError):
        _read(tmp_path, text)


@pytest.mark.parametrize(
    "values",
    [
        "",
        "1,",
        ",1",
        "1,,2",
        "1, ,2",
        "1,\t,2",
        " ,1",
        "1, ",
        "1,2bad",
        "1;2",
        "1 2",
        "1,nan",
        "NaN",
        "None",
        "inf",
        "-inf",
        "1e999",
        "[[1,2],[3,4]]",
        "??",
        "+?",
        "-?",
        "1,+?,2",
        "1,-?,2",
        "1,+ ?,2",
    ],
)
def test_truncated_nonfinite_and_multidimensional_values_are_not_silently_parsed(
    tmp_path,
    values,
):
    with pytest.raises(ValueError):
        _read(tmp_path, _tsf(values=values))


def test_zip_uses_archive_digest_and_never_extracts_paths(tmp_path):
    content = _zip(_tsf().encode("cp1252"), names=("../../outside.tsf",))
    result = _read(tmp_path, content, suffix=".zip")
    assert result.sha256 == hashlib.sha256(content).hexdigest()
    assert result.names == ("price",)
    assert not (tmp_path.parent / "outside.tsf").exists()
    assert sorted(item.name for item in tmp_path.iterdir()) == ["local.zip"]


@pytest.mark.parametrize("names", [(), ("wrong.csv",), ("a.tsf", "b.tsf")])
def test_zip_requires_exactly_one_tsf_member(tmp_path, names):
    with pytest.raises(ValueError, match="exactly one"):
        _read(tmp_path, _zip(_tsf(), names=names), suffix=".zip")


def test_local_input_and_uncompressed_zip_limits_precede_parsing(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "_MAX_BYTES", 500)
    with pytest.raises(ValueError, match="input exceeds"):
        _read(tmp_path, b"x" * 501)
    # A small compressed archive can contain an inadmissibly large member.
    zipped = _zip(("#" + "a" * 2000 + "\n" + _tsf()).encode())
    assert len(zipped) < 500
    with pytest.raises(ValueError, match="Uncompressed TSF"):
        _read(tmp_path, zipped, suffix=".zip")


def test_first_download_uses_pinned_url_then_atomic_cache_and_offline_reuse(
    tiny_archive,
    monkeypatch,
):
    content, spec, directory, path = tiny_archive
    calls = _response(monkeypatch, content)
    result = datasets.load_dataset("tiny", cache_dir=directory, timeout=2.5)
    assert len(calls) == 1
    request, timeout = calls[0]
    assert request.full_url == (
        f"{datasets._REPOSITORY}/resolve/{datasets._REVISION}/data/{spec.filename}"
    )
    assert request.get_header("User-agent") == "strategy-inference-datasets"
    assert timeout == 2.5
    assert path.read_bytes() == content
    assert list(directory.iterdir()) == [path]
    assert result.name == "tiny"
    assert result.sha256 == spec.sha256
    assert result.source == spec.source
    assert result.license == "CC-BY-4.0"
    assert result.revision == datasets._REVISION
    _no_network(monkeypatch)
    cached = datasets.load_dataset("tiny", cache_dir=directory, offline=True)
    np.testing.assert_array_equal(cached["price"].values, result["price"].values)


def test_default_cache_respects_xdg_directory(tiny_archive, monkeypatch, tmp_path):
    content, _, _, _ = tiny_archive
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _response(monkeypatch, content)
    datasets.load_dataset("tiny")
    expected = tmp_path / "strategy-inference" / "datasets"
    assert [x.name for x in expected.iterdir()] == [f"tiny-{datasets._REVISION[:12]}.zip"]


def test_offline_cache_miss_is_read_only(tiny_archive, monkeypatch):
    _, _, directory, _ = tiny_archive
    _no_network(monkeypatch)
    with pytest.raises(FileNotFoundError, match="not cached"):
        datasets.load_dataset("tiny", cache_dir=directory, offline=True)
    assert not directory.exists()


@pytest.mark.parametrize("bad", [b"wrong", b"", b"same-sized"])
def test_bad_cached_bytes_are_rejected_without_network_or_overwrite(
    tiny_archive,
    monkeypatch,
    bad,
):
    content, _, directory, path = tiny_archive
    if bad == b"same-sized":
        bad = content[:-1] + bytes([content[-1] ^ 1])
    directory.mkdir(parents=True)
    path.write_bytes(bad)
    _no_network(monkeypatch)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        datasets.load_dataset("tiny", cache_dir=directory)
    assert path.read_bytes() == bad
    assert list(directory.iterdir()) == [path]


@pytest.mark.parametrize("bad", [b"short", b"long", b"same-sized"])
def test_failed_download_does_not_create_cache(tiny_archive, monkeypatch, bad):
    content, _, directory, _ = tiny_archive
    if bad == b"long":
        bad = content + b"excess"
    elif bad == b"same-sized":
        bad = content[:-1] + bytes([content[-1] ^ 1])
    _response(monkeypatch, bad)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        datasets.load_dataset("tiny", cache_dir=directory)
    assert not directory.exists()


def test_even_hash_verified_malformed_tsf_is_not_cached(tiny_archive, monkeypatch):
    _, spec, directory, _ = tiny_archive
    content = _zip(b"@data\nbad")
    monkeypatch.setattr(
        datasets,
        "_ARCHIVES",
        {"tiny": spec._replace(size=len(content), sha256=hashlib.sha256(content).hexdigest())},
    )
    _response(monkeypatch, content)
    with pytest.raises(ValueError):
        datasets.load_dataset("tiny", cache_dir=directory)
    assert not directory.exists()


def test_network_failure_never_creates_cache(tiny_archive, monkeypatch):
    _, _, directory, _ = tiny_archive

    def fail(*args, **kwargs):
        raise OSError("connection failed")

    monkeypatch.setattr(datasets, "urlopen", fail)
    with pytest.raises(OSError, match="connection failed"):
        datasets.load_dataset("tiny", cache_dir=directory)
    assert not directory.exists()


def test_atomic_replace_failure_removes_temporary_file(tiny_archive, monkeypatch):
    content, _, directory, path = tiny_archive
    _response(monkeypatch, content)

    def fail(*args):
        assert args[0].parent == directory
        assert args[0].read_bytes() == content
        assert args[1] == path
        raise OSError("replace failed")

    monkeypatch.setattr(datasets.os, "replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        datasets.load_dataset("tiny", cache_dir=directory)
    assert directory.exists()
    assert list(directory.iterdir()) == []


def test_partial_write_failure_removes_temporary_file(tiny_archive, monkeypatch):
    content, _, directory, _ = tiny_archive
    _response(monkeypatch, content)
    original = datasets.tempfile.NamedTemporaryFile

    class FailingWrite:
        def __init__(self, *args, **kwargs):
            self.file = original(*args, **kwargs)
            self.name = self.file.name

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def write(self, value):
            self.file.write(value[:3])
            raise OSError("write failed")

    monkeypatch.setattr(datasets.tempfile, "NamedTemporaryFile", FailingWrite)
    with pytest.raises(OSError, match="write failed"):
        datasets.load_dataset("tiny", cache_dir=directory)
    assert directory.exists()
    assert list(directory.iterdir()) == []


@pytest.mark.parametrize("timeout", [True, np.bool_(True), 0, -1, np.nan, np.inf, [], "bad"])
def test_invalid_timeout_never_touches_network(tiny_archive, monkeypatch, timeout):
    _, _, directory, _ = tiny_archive
    _no_network(monkeypatch)
    with pytest.raises(ValueError, match="timeout"):
        datasets.load_dataset("tiny", cache_dir=directory, timeout=timeout)


@pytest.mark.parametrize("name", ["unknown", "", 1, None, True])
def test_unknown_dataset_is_not_downloaded(tiny_archive, monkeypatch, name):
    _, _, directory, _ = tiny_archive
    _no_network(monkeypatch)
    with pytest.raises(ValueError, match="Unknown dataset"):
        datasets.load_dataset(name, cache_dir=directory)


@pytest.mark.parametrize("offline", [1, "yes", None, np.bool_(True)])
def test_offline_requires_a_bool(tiny_archive, monkeypatch, offline):
    _, _, directory, _ = tiny_archive
    _no_network(monkeypatch)
    with pytest.raises(ValueError, match="offline"):
        datasets.load_dataset("tiny", cache_dir=directory, offline=offline)


def test_curated_catalog_has_three_small_immutable_cc_by_archives():
    assert set(datasets.available_datasets()) == {"fred_md", "bitcoin", "oikolab_weather"}
    assert datasets._REVISION == "58aafbe2712ff481c014f562e42723f2820fd5d4"
    expected = {
        "fred_md": (169107, "305c0edd2b5e97159c6339be4990ea97fdf86772c1edef2a0dfc836bf29f45c3"),
        "bitcoin": (220403, "aca43bae943dbf24617e885b5165f2493578fb195a888702ec85947e727f015f"),
        "oikolab_weather": (
            1326101,
            "6f0d2dce3a5aa17c26627ea6107cbf4e5b10050f8a1648d6d03306cd0777f735",
        ),
    }
    for name, (size, digest) in expected.items():
        assert datasets._ARCHIVES[name].size == size
        assert datasets._ARCHIVES[name].sha256 == digest
        assert datasets._ARCHIVES[name].source.startswith("https://zenodo.org/records/")
