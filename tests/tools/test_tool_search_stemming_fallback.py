"""The tool-search catalog must import and stay self-consistent without the optional stemmer.

``snowballstemmer`` ships with the 3.14 dependency set (pyproject.toml), while the updater keeps
older interpreters alive. An absent package used to fail the module import outright, which took
down anything importing ``agent.tool_call_batches`` — every tool round on such an interpreter.
"""
import os
import subprocess
import sys
import textwrap

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools import tool_search_catalog as catalog


def _blocked_import_script(body: str) -> str:
    return "import importlib.abc, sys\n" + textwrap.dedent("""
        class BlockSnowball(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == 'snowballstemmer':
                    raise ImportError('blocked for the fallback test')
                return None

        sys.meta_path.insert(0, BlockSnowball())
        """) + textwrap.dedent(body)


def _run_without_snowball(body: str) -> str:
    proc = subprocess.run([sys.executable, "-c", _blocked_import_script(body)],
                          cwd=_REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_module_imports_without_the_stemmer_package():
    out = _run_without_snowball("""
        import tools.tool_search_catalog as c
        assert c.stemmer_available() is False
        print('imported')
        """)
    assert "imported" in out


def test_fallback_catalog_still_matches_the_documented_morphology():
    """Real BM25 search, index and query built by the same fallback tokenizer."""
    out = _run_without_snowball("""
        import tools.tool_search_catalog as c
        entries = [
            c.CatalogEntry(name='create_issue', description='Open a tracked issue.',
                           schema={}, source='other', source_name='',
                           _tokens=c._tokenize('create issue open tracked')),
            c.CatalogEntry(name='send_email', description='Send mail.',
                           schema={}, source='other', source_name='',
                           _tokens=c._tokenize('send email mail')),
        ]
        hits = c.search_catalog(entries, 'issues', limit=1)
        assert [h.name for h in hits] == ['create_issue'], [h.name for h in hits]
        print('matched')
        """)
    assert "matched" in out


@pytest.fixture
def fallback(monkeypatch):
    """Force the fallback path in-process; restore the real stemmer and its cache afterwards."""
    monkeypatch.setattr(catalog, "snowballstemmer", None)
    catalog._stem.cache_clear()
    yield
    catalog._stem.cache_clear()


def test_index_and_query_share_exactly_one_tokenizer(fallback):
    assert catalog._tokenize("issues") == catalog._tokenize("issue") == ["issu"]
    assert catalog._tokenize("creating messages") == catalog._tokenize("create message")
    assert "issu" in catalog._tokenize("create_issue")
    assert catalog._tokenize("Issues") == catalog._tokenize("issues")


@pytest.mark.parametrize("raw,stemmed", [
    ("tools", "tool"),
    ("policies", "policy"),
    ("status", "status"),
    ("glass", "glass"),
    ("api", "api"),
    ("mcp", "mcp"),
])
def test_fallback_strips_plurals_without_conflating_singular_lookalikes(fallback, raw, stemmed):
    assert catalog._stem(raw) == stemmed


def test_fallback_keeps_search_self_consistent_on_a_real_query_number(fallback):
    entries = [catalog.CatalogEntry(name="create_issue", description="Open a tracked issue.",
                                    schema={}, source="other", source_name="",
                                    _tokens=catalog._tokenize("create issue open tracked"))]
    assert [e.name for e in catalog.search_catalog(entries, "create issue", limit=1)] == ["create_issue"]


@pytest.mark.skipif(not catalog.stemmer_available(), reason="snowballstemmer is not installed here")
def test_installed_stemmer_keeps_the_snowball_contract():
    catalog._stem.cache_clear()
    assert catalog._stem("issues") == catalog._stem("issue")
    assert "issu" in catalog._tokenize("create_issue")
