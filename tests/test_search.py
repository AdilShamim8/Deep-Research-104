from src.search.web_search import search_web


def test_search_web_returns_list() -> None:
    results = search_web("query")
    assert isinstance(results, list)
