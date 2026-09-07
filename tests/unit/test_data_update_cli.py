"""The one-shot refresh console entry point.

The load-bearing test here is that a bad or informational argument must NOT
cause a network refresh. `main(argv)` accepted argv and ignored it, so
`portfolio-analyzer-update --help` -- the exact command someone runs to find out
what the tool does -- fetched every registered source instead of printing help.
One of those sources is documented as holding blocks against the source IP.
"""

import pytest

from portfolioanalyzer import data_update_cli as cli


@pytest.fixture
def never_fetch(monkeypatch):
    """update_all must not be reached. If it is, the test fails loudly."""
    def boom():
        raise AssertionError("update_all() was called; this argv must not refresh")
    monkeypatch.setattr(cli, "update_all", boom)


class TestArgumentsDoNotTriggerARefresh:
    def test_help_prints_and_exits_without_fetching(self, never_fetch, capsys):
        with pytest.raises(SystemExit) as e:
            cli.main(["--help"])
        assert e.value.code == 0
        assert "refresh" in capsys.readouterr().out.lower()

    def test_an_unknown_argument_exits_without_fetching(self, never_fetch):
        with pytest.raises(SystemExit) as e:
            cli.main(["--nonsense"])
        assert e.value.code != 0


class TestNormalOperation:
    def test_no_arguments_refreshes_and_reports(self, monkeypatch, capsys):
        monkeypatch.setattr(cli, "update_all", lambda: [
            {"name": "a", "ok": True, "rows": 10, "last_date": "2026-09-07"},
            {"name": "b", "ok": False, "error": "boom"},
        ])
        assert cli.main([]) == 0
        out = capsys.readouterr()
        assert "a: 10 rows through 2026-09-07" in out.out
        assert "b" in out.err and "boom" in out.err
        assert "1/2 source(s) updated" in out.out

    def test_exit_1_only_when_every_source_failed(self, monkeypatch):
        monkeypatch.setattr(cli, "update_all", lambda: [{"name": "a", "ok": False, "error": "x"}])
        assert cli.main([]) == 1

    def test_dry_run_reports_without_fetching(self, never_fetch, capsys):
        """Lets someone see what would be refreshed without touching a host that
        bans on bursts."""
        assert cli.main(["--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "would refresh" in out.lower()
