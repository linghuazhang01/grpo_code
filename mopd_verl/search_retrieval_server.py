"""Local retrieval service compatible with verl.tools.search_tool.SearchTool."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    title: str
    snippet: str
    url: str

    def to_retrieval_doc(self, rank: int) -> dict[str, Any]:
        contents = f"{self.title}\n{self.snippet}\nURL: {self.url}".strip()
        return {"document": {"contents": contents}, "score": 1.0 / max(rank, 1)}


class SearchBackend(Protocol):
    def search(self, query: str, topk: int) -> list[SearchResult]:
        """Return up to topk search results for one query."""


class DDGSSearchBackend:
    def search(self, query: str, topk: int) -> list[SearchResult]:
        try:
            from ddgs import DDGS
        except ModuleNotFoundError:
            try:
                from duckduckgo_search import DDGS
            except ModuleNotFoundError as exc:
                raise RuntimeError("Install ddgs or duckduckgo-search to use the local search retrieval server.") from exc

        with DDGS() as client:
            raw_results = list(client.text(query, max_results=topk))
        results: list[SearchResult] = []
        for item in raw_results[:topk]:
            title = str(item.get("title", "")).strip()
            snippet = str(item.get("body", item.get("snippet", ""))).strip()
            url = str(item.get("href", item.get("url", ""))).strip()
            if title or snippet or url:
                results.append(SearchResult(title=title, snippet=snippet, url=url))
        return results


class JsonSearchCache:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._values: dict[str, list[dict[str, str]]] = {}
        if self.path is not None and self.path.exists():
            self._values = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, query: str, topk: int) -> list[SearchResult] | None:
        cached = self._values.get(self._key(query, topk))
        if cached is None:
            return None
        return [SearchResult(**item) for item in cached]

    def set(self, query: str, topk: int, results: Sequence[SearchResult]) -> None:
        if self.path is None:
            return
        self._values[self._key(query, topk)] = [asdict(result) for result in results]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp_path.write_text(json.dumps(self._values, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.path)

    @staticmethod
    def _key(query: str, topk: int) -> str:
        payload = json.dumps({"query": query, "topk": topk}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RetrievalService:
    def __init__(self, backend: SearchBackend, cache: JsonSearchCache | None = None) -> None:
        self.backend = backend
        self.cache = cache or JsonSearchCache(None)

    def search_batch(self, queries: Sequence[str], topk: int) -> dict[str, Any]:
        result_batches: list[list[dict[str, Any]]] = []
        for query in queries:
            cached = self.cache.get(query, topk)
            if cached is None:
                cached = self.backend.search(query, topk)
                self.cache.set(query, topk, cached)
            result_batches.append([result.to_retrieval_doc(rank=index + 1) for index, result in enumerate(cached)])
        return {"result": result_batches}


def _make_handler(service: RetrievalService):
    class SearchRequestHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/retrieve":
                self._write_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                payload = json.loads(body.decode("utf-8") or "{}")
                queries = payload.get("queries", [])
                topk = int(payload.get("topk", 3))
                if not isinstance(queries, list) or not all(isinstance(item, str) for item in queries):
                    raise ValueError("'queries' must be a list of strings.")
                self._write_json(service.search_batch(queries, topk), HTTPStatus.OK)
            except Exception as exc:
                self._write_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _write_json(self, payload: dict[str, Any], status: HTTPStatus) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return SearchRequestHandler


def run_server(host: str, port: int, backend: SearchBackend, cache_path: str | Path | None) -> None:
    service = RetrievalService(backend=backend, cache=JsonSearchCache(cache_path))
    server = ThreadingHTTPServer((host, port), _make_handler(service))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    LOGGER.info("Search retrieval service listening on http://%s:%s/retrieve", host, port)
    server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cache-path", default="temp/search_cache/ddgs_cache.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_server(args.host, args.port, DDGSSearchBackend(), args.cache_path)


if __name__ == "__main__":
    main()
