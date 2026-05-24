from typing import Any, Callable, Dict, List, Tuple

from .browsecomp_searcher import BrowseCompBM25Searcher, snippetize


def build_searcher(index_path: str) -> BrowseCompBM25Searcher:
    return BrowseCompBM25Searcher(index_path=index_path)


def retrieve_once(
    searcher: BrowseCompBM25Searcher,
    query: str,
    k: int = 5,
    snippet_max_chars: int = 1200,
) -> List[Dict[str, Any]]:
    docs = searcher.search(query, k=k)
    return [
        {
            "docid": doc["docid"],
            "score": doc["score"],
            "snippet": snippetize(doc["text"], snippet_max_chars),
            "url": doc.get("url", ""),
        }
        for doc in docs
    ]


def format_rag_context(results: List[Dict[str, Any]]) -> str:
    blocks = []
    for rank, item in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Document {rank}]",
                    f"docid: {item['docid']}",
                    f"score: {item['score']}",
                    f"url: {item.get('url', '')}",
                    item["snippet"],
                ]
            )
        )
    return "\n\n".join(blocks)


def get_search_tool_specs_and_registry(
    searcher: BrowseCompBM25Searcher,
    k: int = 5,
    snippet_max_chars: int = 1200,
) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., Any]]]:
    def search(query: str) -> List[Dict[str, Any]]:
        return retrieve_once(searcher=searcher, query=query, k=k, snippet_max_chars=snippet_max_chars)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    f"Search the BrowseComp-Plus BM25 index and return top-{k} results "
                    "with docid, score, and snippet."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        }
    ]
    return tools, {"search": search}


def get_agent_tool_specs_and_registry(
    searcher: BrowseCompBM25Searcher,
    k: int = 5,
    snippet_max_chars: int = 1200,
) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., Any]]]:
    def search(query: str) -> List[Dict[str, Any]]:
        return retrieve_once(searcher=searcher, query=query, k=k, snippet_max_chars=snippet_max_chars)

    def get_document(docid: str) -> Dict[str, Any]:
        doc = searcher.get_document(docid)
        if doc is None:
            return {"docid": docid, "error": "document not found"}
        return doc

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    f"Search the BrowseComp-Plus BM25 index and return top-{k} results "
                    "with docid, score, and snippet."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_document",
                "description": "Retrieve a full document by its docid.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "docid": {"type": "string", "description": "Document id"},
                    },
                    "required": ["docid"],
                },
            },
        },
    ]
    return tools, {"search": search, "get_document": get_document}


def _clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "... [truncated]"


def _normalize_keywords(keywords: Any) -> List[str]:
    if isinstance(keywords, str):
        pieces = [piece.strip() for piece in keywords.split(",")]
    elif isinstance(keywords, list):
        pieces = [str(piece).strip() for piece in keywords]
    else:
        pieces = []
    normalized = []
    seen = set()
    for piece in pieces:
        key = piece.lower()
        if piece and key not in seen:
            normalized.append(piece)
            seen.add(key)
    return normalized


def find_keyword_windows(
    text: str,
    keyword: str,
    window_chars: int = 900,
    max_matches: int = 5,
) -> List[Dict[str, Any]]:
    keyword = str(keyword or "").strip()
    if not keyword:
        return []

    lowered_text = text.lower()
    lowered_keyword = keyword.lower()
    matches = []
    cursor = 0
    while len(matches) < max_matches:
        pos = lowered_text.find(lowered_keyword, cursor)
        if pos < 0:
            break
        left = max(0, pos - window_chars // 2)
        right = min(len(text), pos + len(keyword) + window_chars // 2)
        matches.append(
            {
                "keyword": keyword,
                "start": pos,
                "snippet": text[left:right].strip(),
            }
        )
        cursor = pos + max(len(keyword), 1)
    return matches


def get_research_tool_specs_and_registry(
    searcher: BrowseCompBM25Searcher,
    k: int = 8,
    snippet_max_chars: int = 1400,
    document_window_chars: int = 2200,
) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., Any]]]:
    def search(query: str) -> List[Dict[str, Any]]:
        return retrieve_once(searcher=searcher, query=query, k=k, snippet_max_chars=snippet_max_chars)

    def get_document(docid: str) -> Dict[str, Any]:
        doc = searcher.get_document(docid)
        if doc is None:
            return {"docid": docid, "error": "document not found"}
        return doc

    def get_document_window(docid: str, start: int = 0, window_chars: int = document_window_chars) -> Dict[str, Any]:
        doc = searcher.get_document(docid)
        if doc is None:
            return {"docid": docid, "error": "document not found"}
        text = doc.get("text", "")
        start = max(0, int(start or 0))
        window_chars = max(1, int(window_chars or document_window_chars))
        end = min(len(text), start + window_chars)
        return {
            "docid": docid,
            "url": doc.get("url", ""),
            "start": start,
            "end": end,
            "text_length": len(text),
            "snippet": text[start:end],
        }

    def find_in_document(
        docid: str,
        keyword: str,
        window_chars: int = 900,
        max_matches: int = 5,
    ) -> Dict[str, Any]:
        doc = searcher.get_document(docid)
        if doc is None:
            return {"docid": docid, "keyword": keyword, "matches": [], "error": "document not found"}
        matches = find_keyword_windows(
            text=doc.get("text", ""),
            keyword=keyword,
            window_chars=int(window_chars or 900),
            max_matches=int(max_matches or 5),
        )
        return {
            "docid": docid,
            "url": doc.get("url", ""),
            "keyword": keyword,
            "num_matches": len(matches),
            "matches": matches,
        }

    def collect_evidence_snippets(
        docids: List[str],
        keywords: Any,
        window_chars: int = 900,
        max_snippets: int = 10,
    ) -> Dict[str, Any]:
        keyword_list = _normalize_keywords(keywords)
        snippets = []
        for docid in [str(item).strip() for item in docids][:10]:
            if not docid:
                continue
            doc = searcher.get_document(docid)
            if doc is None:
                snippets.append({"docid": docid, "error": "document not found"})
                continue
            for keyword in keyword_list:
                for match in find_keyword_windows(
                    text=doc.get("text", ""),
                    keyword=keyword,
                    window_chars=int(window_chars or 900),
                    max_matches=3,
                ):
                    snippets.append(
                        {
                            "docid": docid,
                            "url": doc.get("url", ""),
                            "keyword": keyword,
                            "start": match["start"],
                            "snippet": match["snippet"],
                        }
                    )
                    if len(snippets) >= max_snippets:
                        return {"keywords": keyword_list, "snippets": snippets}
        return {"keywords": keyword_list, "snippets": snippets}

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    f"Search the BrowseComp-Plus BM25 index and return top-{k} results "
                    "with docid, score, url, and snippet."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_document",
                "description": "Retrieve a full document by docid. Use sparingly because documents can be long.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "docid": {"type": "string", "description": "Document id"},
                    },
                    "required": ["docid"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_document_window",
                "description": "Retrieve a bounded character window from a document by docid and start offset.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "docid": {"type": "string", "description": "Document id"},
                        "start": {"type": "integer", "description": "Start character offset, default 0"},
                        "window_chars": {"type": "integer", "description": "Window length in characters"},
                    },
                    "required": ["docid"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_in_document",
                "description": "Find keyword occurrences inside a known document and return local evidence windows.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "docid": {"type": "string", "description": "Document id"},
                        "keyword": {"type": "string", "description": "Keyword or phrase to find"},
                        "window_chars": {"type": "integer", "description": "Characters around each match"},
                        "max_matches": {"type": "integer", "description": "Maximum number of matches"},
                    },
                    "required": ["docid", "keyword"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "collect_evidence_snippets",
                "description": (
                    "Given candidate docids and keywords, collect matching snippets for final evidence checking."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "docids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Candidate document ids",
                        },
                        "keywords": {
                            "type": "string",
                            "description": "Comma-separated keywords to search in those documents",
                        },
                        "window_chars": {"type": "integer", "description": "Characters around each match"},
                        "max_snippets": {"type": "integer", "description": "Maximum snippets to return"},
                    },
                    "required": ["docids", "keywords"],
                },
            },
        },
    ]
    return tools, {
        "search": search,
        "get_document": get_document,
        "get_document_window": get_document_window,
        "find_in_document": find_in_document,
        "collect_evidence_snippets": collect_evidence_snippets,
    }


def make_initial_search_queries(question: str, max_queries: int = 4, max_query_chars: int = 260) -> List[str]:
    text = " ".join(str(question or "").split())
    if not text:
        return []

    queries = []

    def add_query(query: str) -> None:
        query = " ".join(str(query or "").split()).strip(" ,.;:")
        if not query:
            return
        if len(query) > max_query_chars:
            query = query[:max_query_chars].rstrip()
        key = query.lower()
        if key not in {item.lower() for item in queries}:
            queries.append(query)

    quoted_phrases = []
    for marker in ('"', "'", "“", "”"):
        if marker in text:
            break
    for piece in text.replace("“", '"').replace("”", '"').split('"')[1::2]:
        if 2 <= len(piece.split()) <= 12:
            quoted_phrases.append(piece)

    year_tokens = []
    current = []
    for ch in text:
        if ch.isdigit():
            current.append(ch)
        elif current:
            token = "".join(current)
            if len(token) == 4:
                year_tokens.append(token)
            current = []
    if current:
        token = "".join(current)
        if len(token) == 4:
            year_tokens.append(token)

    long_words = []
    for raw in text.replace("/", " ").replace("-", " ").split():
        word = raw.strip(".,;:!?()[]{}\"'")
        if len(word) >= 7 and any(ch.isalpha() for ch in word):
            long_words.append(word)

    add_query(text)
    if quoted_phrases:
        add_query(" ".join(quoted_phrases[:4] + year_tokens[:4]))
    if long_words:
        add_query(" ".join(long_words[:18] + year_tokens[:4]))
    if len(text) > max_query_chars:
        add_query(text[:max_query_chars])

    return queries[:max_queries]


def get_v3_research_tool_specs_and_registry(
    searcher: BrowseCompBM25Searcher,
    k: int = 8,
    snippet_max_chars: int = 1400,
    document_window_chars: int = 2200,
) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., Any]]]:
    tools, registry = get_research_tool_specs_and_registry(
        searcher=searcher,
        k=k,
        snippet_max_chars=snippet_max_chars,
        document_window_chars=document_window_chars,
    )

    def search_many(queries: List[str], per_query_k: int = 5) -> List[Dict[str, Any]]:
        results = []
        seen_docids = set()
        for query in [str(item).strip() for item in queries][:6]:
            if not query:
                continue
            for item in retrieve_once(
                searcher=searcher,
                query=query,
                k=int(per_query_k or 5),
                snippet_max_chars=snippet_max_chars,
            ):
                docid = str(item.get("docid", ""))
                if not docid or docid in seen_docids:
                    continue
                seen_docids.add(docid)
                with_query = dict(item)
                with_query["query"] = query
                results.append(with_query)
        return results

    search_many_spec = {
        "type": "function",
        "function": {
            "name": "search_many",
            "description": (
                "Run several focused BrowseComp-Plus BM25 searches and return a deduplicated list "
                "of results with query, docid, score, url, and snippet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Focused search queries",
                    },
                    "per_query_k": {
                        "type": "integer",
                        "description": "Maximum results to return for each query",
                    },
                },
                "required": ["queries"],
            },
        },
    }

    return tools + [search_many_spec], {**registry, "search_many": search_many}
