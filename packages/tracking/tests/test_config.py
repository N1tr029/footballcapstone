"""Loading .env — and never leaking what it loaded."""

from gridiron_tracking.config import find_env, load_env


def test_values_load(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text('GEMINI_API_KEY=abc123\n# a comment\nOTHER="with spaces"\n')
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OTHER", raising=False)
    names = load_env(tmp_path / ".env")
    assert set(names) == {"GEMINI_API_KEY", "OTHER"}
    import os
    assert os.environ["GEMINI_API_KEY"] == "abc123"
    assert os.environ["OTHER"] == "with spaces"      # quotes are syntax, not value


def test_the_real_environment_wins(tmp_path, monkeypatch):
    """An explicit `KEY=... command` must override the file, not be ignored by it."""
    (tmp_path / ".env").write_text("GEMINI_API_KEY=from_file\n")
    monkeypatch.setenv("GEMINI_API_KEY", "from_shell")
    load_env(tmp_path / ".env")
    import os
    assert os.environ["GEMINI_API_KEY"] == "from_shell"


def test_it_returns_names_never_values(tmp_path, monkeypatch):
    """The loader is called in CLI paths that print; a value must never be returnable."""
    (tmp_path / ".env").write_text("SECRET_THING=supersecret\n")
    monkeypatch.delenv("SECRET_THING", raising=False)
    out = load_env(tmp_path / ".env")
    assert out == ["SECRET_THING"]
    assert "supersecret" not in str(out)


def test_export_prefix_and_blank_lines_are_tolerated(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("\n\nexport GEMINI_API_KEY=xyz\n\nnot_a_pair\n")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert load_env(tmp_path / ".env") == ["GEMINI_API_KEY"]


def test_a_missing_file_is_not_an_error(tmp_path):
    assert load_env(tmp_path / "nope.env") == []


def test_it_searches_upward(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("A=1\n")
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    assert find_env() == (tmp_path / ".env").resolve()
