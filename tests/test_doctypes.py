"""Document-type lookup is driven by YAML files only. Types here are made up."""

import os
import re
from importlib import resources
from pathlib import Path

import pytest
import yaml

from dococr import list_doc_types, load_doc_type
from dococr.doctypes import DOCTYPES_DIR_ENV, DocTypeError, find_doc_types

SRC = Path(__file__).resolve().parent.parent / "src" / "dococr"


def write(folder: Path, file_name: str, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / file_name
    path.write_text(text, encoding="utf-8")
    return path


def doc(name, field="f1", aliases=(), title=""):
    lines = [f"name: {name}"]
    if title:
        lines.append(f"title: {title}")
    if aliases:
        lines.append("aliases: [" + ", ".join(f'"{a}"' for a in aliases) + "]")
    lines.append(f"fields:\n  - name: {field}")
    return "\n".join(lines) + "\n"


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    monkeypatch.delenv(DOCTYPES_DIR_ENV, raising=False)


def test_by_name_file_name_and_alias(tmp_path):
    write(tmp_path, "widget_card.yaml", doc("gizmo", aliases=["Gadget Pass", "gp"]))
    for ask in ("gizmo", "Gizmo", "widget card", "widget_card", "Gadget-Pass", "GP"):
        assert load_doc_type(ask, dirs=[tmp_path]).fields[0].name == "f1", ask
    assert load_doc_type("gizmo", dirs=[tmp_path]).name == "gizmo"


def test_yml_extension_found(tmp_path):
    write(tmp_path, "thing.yml", doc("thing"))
    assert load_doc_type("thing", dirs=[tmp_path]).fields


def test_unknown_name_is_empty_doc_type(tmp_path):
    d = load_doc_type("no such thing", dirs=[tmp_path])
    assert d.name == "no_such_thing" and d.fields == []


def test_user_folder_overrides_shipped(tmp_path):
    shipped = find_doc_types()[0][0].name
    write(tmp_path, f"{shipped}.yaml", doc(shipped, field="mine"))
    assert [f.name for f in load_doc_type(shipped, dirs=[tmp_path]).fields] == ["mine"]
    assert load_doc_type(shipped).fields[0].name != "mine"
    assert list_doc_types([tmp_path]).count(shipped) == 1


def test_user_alias_can_take_a_shipped_name(tmp_path):
    shipped = find_doc_types()[0][0].name
    write(tmp_path, "x.yaml", doc("x", aliases=[shipped]))
    assert load_doc_type(shipped, dirs=[tmp_path]).name == "x"


def test_environment_variable_with_several_folders(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    write(a, "one.yaml", doc("one"))
    write(b, "two.yaml", doc("two"))
    write(b, "one.yaml", doc("one", field="other"))
    monkeypatch.setenv(DOCTYPES_DIR_ENV, os.pathsep.join([str(a), str(b)]))
    assert load_doc_type("two").fields
    assert load_doc_type("one").fields[0].name == "f1"  # first folder wins
    assert {"one", "two"} <= set(list_doc_types())


def test_env_is_read_at_call_time(tmp_path, monkeypatch):
    write(tmp_path, "late.yaml", doc("late"))
    assert not load_doc_type("late").fields
    monkeypatch.setenv(DOCTYPES_DIR_ENV, str(tmp_path))
    assert load_doc_type("late").fields


def test_dirs_argument_ahead_of_environment(tmp_path, monkeypatch):
    env, arg = tmp_path / "env", tmp_path / "arg"
    write(env, "same.yaml", doc("same", field="from_env"))
    write(arg, "same.yaml", doc("same", field="from_arg"))
    monkeypatch.setenv(DOCTYPES_DIR_ENV, str(env))
    assert load_doc_type("same").fields[0].name == "from_env"
    assert load_doc_type("same", dirs=[arg]).fields[0].name == "from_arg"


def test_listing_has_no_duplicates(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    write(a, "dup.yaml", doc("dup"))
    write(b, "dup.yaml", doc("dup"))
    write(b, "other.yaml", doc("other"))
    monkeypatch.setenv(DOCTYPES_DIR_ENV, str(b))
    names = list_doc_types([a])
    assert names == sorted(set(names)) and {"dup", "other"} <= set(names)


def test_tie_in_one_folder_name_beats_alias_then_file_name(tmp_path):
    write(tmp_path, "a.yaml", doc("aaa", field="by_alias", aliases=["target"]))
    write(tmp_path, "b.yaml", doc("bbb", field="by_alias_b", aliases=["target"]))
    assert load_doc_type("target", dirs=[tmp_path]).fields[0].name == "by_alias"  # first file name
    write(tmp_path, "z.yaml", doc("target", field="by_name"))
    assert load_doc_type("target", dirs=[tmp_path]).fields[0].name == "by_name"  # name beats alias


def test_broken_file_does_not_break_other_lookups(tmp_path):
    write(tmp_path, "good.yaml", doc("good"))
    write(tmp_path, "bad_yaml.yaml", "name: [unclosed\n")
    write(tmp_path, "bad_shape.yaml", "name: shaped\nfields: 5\n")
    write(tmp_path, "bad_top.yaml", "- just\n- a list\n")
    assert load_doc_type("good", dirs=[tmp_path]).fields
    assert not load_doc_type("unrelated", dirs=[tmp_path]).fields
    docs, skipped = find_doc_types([tmp_path])
    assert "good" in [d.name for d in docs]
    assert len(skipped) == 3
    assert all(str(tmp_path) in s for s in skipped)
    assert any("bad_shape.yaml" in s and "fields" in s for s in skipped)


def test_asking_for_a_broken_file_names_it(tmp_path):
    write(tmp_path, "bad_shape.yaml", "name: shaped\nfields: 5\n")
    for ask in ("shaped", "bad shape"):
        with pytest.raises(DocTypeError) as e:
            load_doc_type(ask, dirs=[tmp_path])
        assert "bad_shape.yaml" in str(e.value) and "fields" in str(e.value)
    write(tmp_path, "worse.yaml", "name: [unclosed\n")
    with pytest.raises(DocTypeError, match="worse.yaml"):
        load_doc_type("worse", dirs=[tmp_path])


def test_explicit_file_error_names_the_file(tmp_path):
    p = write(tmp_path, "bad.yaml", "fields: []\n")  # no name
    with pytest.raises(DocTypeError, match="bad.yaml"):
        load_doc_type(str(p))
    with pytest.raises(DocTypeError, match="missing.yaml"):
        load_doc_type(str(tmp_path / "missing.yaml"))


def test_missing_folder_is_an_error(tmp_path):
    with pytest.raises(DocTypeError, match="nowhere"):
        load_doc_type("x", dirs=[tmp_path / "nowhere"])


def test_no_document_names_in_the_code():
    """The names and aliases come from the shipped YAML files; none may appear in the code."""
    words = set()
    for p in (resources.files("dococr") / "presets").iterdir():
        if p.name.endswith(".yaml"):
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            words |= {p.name[: -len(".yaml")], data["name"], *data.get("aliases", [])}
    assert words
    for module in ("doctypes.py", "cli.py", "api.py"):
        text = (SRC / module).read_text(encoding="utf-8").lower()
        for w in words:
            pattern = r"(?<![a-z0-9])" + r"[ _\-]?".join(re.escape(x) for x in re.split(r"[\s_\-]+", w.lower())) + r"(?![a-z0-9])"
            assert not re.search(pattern, text), f"{module} mentions {w!r}"


# ---- cache ----------------------------------------------------------------------------------

import threading

from dococr import doctypes as _dt


def _write(folder, stem, name=None, label="Thing No", mtime=None):
    p = folder / f"{stem}.yaml"
    p.write_text(f"name: {name or stem}\nfields:\n  - name: f1\n    label: {label}\n", encoding="utf-8")
    if mtime is not None:  # make an edit visible even on a coarse-grained clock
        os.utime(p, (mtime, mtime))
    return p


def test_cache_hit_equals_fresh_result(tmp_path):
    _write(tmp_path, "cachedtype")
    first = load_doc_type("cachedtype", [tmp_path])
    second = load_doc_type("cachedtype", [tmp_path])
    assert first == second and first.fields[0].label == "Thing No"
    assert find_doc_types([tmp_path]) == find_doc_types([tmp_path])


def test_cache_does_not_reparse_unchanged_files(tmp_path, monkeypatch):
    _write(tmp_path, "cachedtype")
    load_doc_type("cachedtype", [tmp_path])
    calls = []
    real = _dt._read
    monkeypatch.setattr(_dt, "_read", lambda p: calls.append(p) or real(p))
    load_doc_type("cachedtype", [tmp_path])
    assert not [c for c in calls if str(tmp_path) in str(c)]


def test_edit_is_picked_up(tmp_path):
    _write(tmp_path, "cachedtype", mtime=1_000_000)
    assert load_doc_type("cachedtype", [tmp_path]).fields[0].label == "Thing No"
    _write(tmp_path, "cachedtype", label="Other No", mtime=2_000_000)
    assert load_doc_type("cachedtype", [tmp_path]).fields[0].label == "Other No"


def test_edit_with_same_size_and_new_mtime_is_picked_up(tmp_path):
    _write(tmp_path, "cachedtype", label="AAAA", mtime=1_000_000)
    load_doc_type("cachedtype", [tmp_path])
    _write(tmp_path, "cachedtype", label="BBBB", mtime=1_000_005)
    assert load_doc_type("cachedtype", [tmp_path]).fields[0].label == "BBBB"


def test_new_file_and_deleted_file_are_picked_up(tmp_path):
    assert load_doc_type("latecomer", [tmp_path]).fields == []
    p = _write(tmp_path, "latecomer")
    assert load_doc_type("latecomer", [tmp_path]).fields[0].name == "f1"
    assert "latecomer" in list_doc_types([tmp_path])
    p.unlink()
    assert load_doc_type("latecomer", [tmp_path]).fields == []
    assert "latecomer" not in list_doc_types([tmp_path])


def test_broken_file_reported_then_fixed(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [unclosed", encoding="utf-8")
    assert find_doc_types([tmp_path])[1]
    _write(tmp_path, "bad", mtime=3_000_000)
    assert not [m for m in find_doc_types([tmp_path])[1] if "bad.yaml" in m]


def test_mutating_a_returned_object_does_not_leak(tmp_path):
    _write(tmp_path, "cachedtype")
    doc = load_doc_type("cachedtype", [tmp_path])
    doc.fields[0].label = "Tampered"
    doc.fields.clear()
    doc.aliases.append("x")
    listed = find_doc_types([tmp_path])[0]
    listed[0].fields.clear()
    again = load_doc_type("cachedtype", [tmp_path])
    assert again.fields[0].label == "Thing No" and "x" not in again.aliases
    assert find_doc_types([tmp_path])[0][0].fields


def test_concurrent_lookups_are_safe(tmp_path):
    for i in range(6):
        _write(tmp_path, f"t{i}")
    (tmp_path / "scratch").mkdir()
    errors, results = [], []

    def work(i):
        try:
            for n in range(40):
                if i == 0 and n % 10 == 0:
                    # edit while others read; replace atomically so readers never see a torn file
                    tmp = _write(tmp_path / "scratch", "t0", label=f"L{n}", mtime=1_000_000 + n)
                    os.replace(tmp, tmp_path / "t0.yaml")
                results.append(load_doc_type(f"t{n % 6}", [tmp_path]).fields[0].name)
                find_doc_types([tmp_path])
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and set(results) == {"f1"}


def test_yaml_loader_is_a_safe_loader():
    assert issubclass(_dt._SafeLoader, (yaml.SafeLoader, getattr(yaml, "CSafeLoader", yaml.SafeLoader)))
    with pytest.raises(yaml.YAMLError):
        yaml.load("!!python/object/apply:os.getcwd []", Loader=_dt._SafeLoader)


def test_date_order_from_yaml(tmp_path):
    from dococr.doctypes import DocTypeError
    from dococr.normalize import normalize_value

    write(tmp_path, "us.yaml", "name: us\ndate_order: month_first\nfields:\n"
          "  - {name: signed, type: date}\n  - {name: born, type: date, date_order: day_first}\n")
    signed, born = load_doc_type("us", dirs=[tmp_path]).fields
    assert signed.date_order == "month_first" and born.date_order == "day_first"
    assert normalize_value("12/05/08", signed)[0] == "2008-12-05"
    assert normalize_value("12/05/08", born)[0] == "2008-05-12"
    write(tmp_path, "bad.yaml", "name: bad\ndate_order: MDY\nfields:\n  - {name: d, type: date}\n")
    with pytest.raises(DocTypeError):
        load_doc_type("bad", dirs=[tmp_path])
