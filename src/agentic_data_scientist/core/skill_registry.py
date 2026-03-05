"""Skill discovery and hybrid Top-K retrieval utilities."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import requests


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_+\-]{2,}")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "using",
    "use",
    "your",
    "you",
    "are",
    "can",
    "all",
    "any",
    "not",
    "but",
    "what",
    "when",
    "where",
    "how",
    "why",
}
_ALIAS_EXPANSIONS = {
    "rnaseq": ("rna", "seq", "sequencing", "transcriptomics"),
    "deg": ("differential", "expression"),
    "scrna": ("single", "cell", "rna", "sequencing"),
    "wgs": ("whole", "genome", "sequencing"),
    "wxs": ("whole", "exome", "sequencing"),
    "snv": ("variant", "mutation"),
    "cnv": ("copy", "number", "variation"),
    "vcf": ("variant", "genotype"),
    "ngs": ("next", "generation", "sequencing"),
    "crispr": ("gene", "editing"),
}
_WORD_SPLIT = re.compile(r"[-_]+")
_DEFAULT_VECTOR_DIM = 256
_DEFAULT_EMBED_TIMEOUT_SECONDS = 20.0
_DEFAULT_EMBED_CACHE_SIZE = 4096

logger = logging.getLogger(__name__)

_EMBEDDING_CACHE: Dict[str, List[float]] = {}


def _env_enabled(name: str, default: bool = True) -> bool:
    raw = os.getenv(name, "true" if default else "false").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except Exception:
        value = default
    return max(minimum, value)


def _env_float(name: str, default: float, minimum: float = 0.0, maximum: float | None = None) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except Exception:
        value = default
    value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _cache_key(*, provider: str, model: str, task: str, text: str) -> str:
    digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=16).hexdigest()
    return f"{provider}|{model}|{task}|{digest}"


def _normalize_dense_vector(values: Sequence[float]) -> List[float]:
    vector = [float(v) for v in values]
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 0:
        return vector
    return [v / norm for v in vector]


def _cache_get(key: str) -> List[float] | None:
    value = _EMBEDDING_CACHE.get(key)
    if value is None:
        return None
    return list(value)


def _cache_put(key: str, vector: Sequence[float]) -> None:
    cache_limit = _env_int("ADS_EMBEDDING_CACHE_SIZE", _DEFAULT_EMBED_CACHE_SIZE, minimum=128)
    _EMBEDDING_CACHE[key] = [float(v) for v in vector]
    while len(_EMBEDDING_CACHE) > cache_limit:
        try:
            oldest_key = next(iter(_EMBEDDING_CACHE))
        except StopIteration:
            break
        _EMBEDDING_CACHE.pop(oldest_key, None)


def _cosine_dense(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    if size <= 0:
        return 0.0
    dot = 0.0
    for idx in range(size):
        dot += float(left[idx]) * float(right[idx])
    return float(dot)


@dataclass(frozen=True)
class SkillCard:
    """Minimal searchable skill descriptor."""

    skill_name: str
    skill_dir: str
    summary: str
    search_text: str
    tokens: tuple[str, ...]


def _skills_scope_name() -> str:
    return os.getenv("ADS_SKILLS_SCOPE_NAME", "scientific-skills").strip() or "scientific-skills"


def _default_repo_root() -> Path:
    # skill_registry.py -> core -> agentic_data_scientist -> src -> repo_root
    return Path(__file__).resolve().parents[3]


def _ordered_dedupe_paths(paths: Sequence[Path]) -> List[Path]:
    seen: set[str] = set()
    result: List[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _candidate_skill_roots(working_dir: str | None = None) -> List[Path]:
    scope_name = _skills_scope_name()
    repo_root = _default_repo_root()
    cwd = Path.cwd()
    working_path = Path(working_dir).resolve() if working_dir else None

    candidates: List[Path] = []

    env_scoped_dir = os.getenv("ADS_SKILLS_SCOPE_DIR", "").strip()
    if env_scoped_dir:
        path = Path(env_scoped_dir)
        if not path.is_absolute():
            path = (cwd / path).resolve()
        candidates.append(path)

    for base in [working_path, cwd, repo_root]:
        if base is None:
            continue
        candidates.append((base / ".claude" / "skills" / scope_name).resolve())

    env_source = os.getenv("ADS_LOCAL_SKILLS_SOURCE", "").strip()
    if env_source:
        src = Path(env_source)
        if src.is_absolute():
            candidates.append(src)
        else:
            candidates.append((cwd / src).resolve())
            if working_path is not None:
                candidates.append((working_path / src).resolve())

    for base in [working_path, cwd, repo_root]:
        if base is None:
            continue
        candidates.append((base / "scientific-skills").resolve())

    return _ordered_dedupe_paths(candidates)


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    tokens: List[str] = []
    for token in _TOKEN_PATTERN.findall(text.lower()):
        if token in _STOPWORDS:
            continue
        tokens.append(token)
        compact = token.replace("-", "").replace("_", "")
        if compact and compact != token and compact not in _STOPWORDS:
            tokens.append(compact)

        if "-" in token or "_" in token:
            for part in _WORD_SPLIT.split(token):
                part = part.strip()
                if len(part) >= 2 and part not in _STOPWORDS:
                    tokens.append(part)

        alias_key = compact or token
        for alias_token in _ALIAS_EXPANSIONS.get(alias_key, ()):
            if alias_token not in _STOPWORDS:
                tokens.append(alias_token)
    return tokens


def _read_skill_summary(skill_md: Path) -> tuple[str, str, str]:
    try:
        content = skill_md.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        fallback = skill_md.parent.name
        return fallback, "", fallback

    title = ""
    summary = ""
    excerpt_lines: List[str] = []
    in_code_block = False
    for line in content.splitlines():
        raw = line.strip()
        if not raw:
            continue
        if raw.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if raw.startswith("#") and not title:
            title = raw.lstrip("#").strip()
            continue
        if not summary and not raw.startswith(("-", "*")):
            summary = raw
        excerpt_lines.append(raw)
        if len(excerpt_lines) >= 20:
            break

    excerpt = " ".join(excerpt_lines)
    return title or skill_md.parent.name, summary, excerpt


def discover_skills(working_dir: str | None = None) -> List[SkillCard]:
    """Discover available skills from scoped/local roots."""
    cards: List[SkillCard] = []
    seen_names: set[str] = set()

    for root in _candidate_skill_roots(working_dir=working_dir):
        if not root.exists() or not root.is_dir():
            continue
        for skill_dir in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            title, summary, excerpt = _read_skill_summary(skill_md)
            skill_name = skill_dir.name.strip()
            if not skill_name or skill_name in seen_names:
                continue
            seen_names.add(skill_name)

            search_parts = [skill_name, title, summary, excerpt]
            search_text = " ".join(part for part in search_parts if part)
            cards.append(
                SkillCard(
                    skill_name=skill_name,
                    skill_dir=str(skill_dir.resolve()),
                    summary=summary,
                    search_text=search_text,
                    tokens=tuple(_tokenize(search_text)),
                )
            )

    return cards


def _embed_with_gemini(texts: Sequence[str], *, task: str) -> List[List[float]]:
    del task
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing")

    model = os.getenv("ADS_GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip() or "gemini-embedding-001"
    timeout = _env_float(
        "ADS_EMBEDDING_TIMEOUT_SECONDS",
        _DEFAULT_EMBED_TIMEOUT_SECONDS,
        minimum=1.0,
        maximum=120.0,
    )
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={api_key}"

    vectors: List[List[float]] = []
    for text in texts:
        payload = {
            "model": f"models/{model}",
            "content": {"parts": [{"text": text}]},
        }
        response = requests.post(endpoint, json=payload, timeout=float(timeout))
        if not response.ok:
            raise RuntimeError(f"gemini embedding failed: http={response.status_code} body={response.text[:240]}")
        body = response.json()
        values = (((body.get("embedding") or {}).get("values")) or []) if isinstance(body, dict) else []
        if not isinstance(values, list) or not values:
            raise RuntimeError("gemini embedding response missing values")
        vectors.append(_normalize_dense_vector(values))
    return vectors


def _embed_with_jina(texts: Sequence[str], *, task: str) -> List[List[float]]:
    api_key = (
        os.getenv("ADS_JINA_API_KEY", "").strip()
        or os.getenv("JINA_API_KEY", "").strip()
    )
    if not api_key:
        raise RuntimeError("ADS_JINA_API_KEY/JINA_API_KEY is missing")

    model = (
        os.getenv("ADS_JINA_EMBEDDING_MODEL", "jina-embeddings-v5-text-small").strip()
        or "jina-embeddings-v5-text-small"
    )
    api_base = os.getenv("ADS_JINA_API_BASE", "https://api.jina.ai/v1").strip() or "https://api.jina.ai/v1"
    timeout = _env_float(
        "ADS_EMBEDDING_TIMEOUT_SECONDS",
        _DEFAULT_EMBED_TIMEOUT_SECONDS,
        minimum=1.0,
        maximum=120.0,
    )
    task_name = "retrieval.query" if str(task).strip().lower() == "query" else "retrieval.passage"
    payload = {
        "model": model,
        "task": task_name,
        "normalized": True,
        "input": list(texts),
    }
    response = requests.post(
        api_base.rstrip("/") + "/embeddings",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json=payload,
        timeout=float(timeout),
    )
    if not response.ok:
        raise RuntimeError(f"jina embedding failed: http={response.status_code} body={response.text[:240]}")

    body = response.json()
    raw_data = body.get("data", []) if isinstance(body, dict) else []
    if not isinstance(raw_data, list) or not raw_data:
        raise RuntimeError("jina embedding response missing data")

    by_index: Dict[int, List[float]] = {}
    for item in raw_data:
        if not isinstance(item, dict):
            continue
        idx = int(item.get("index", -1))
        embedding = item.get("embedding", [])
        if idx < 0 or not isinstance(embedding, list) or not embedding:
            continue
        by_index[idx] = _normalize_dense_vector(embedding)

    vectors: List[List[float]] = []
    for idx in range(len(texts)):
        vector = by_index.get(idx)
        if not vector:
            raise RuntimeError(f"jina embedding missing index {idx}")
        vectors.append(vector)
    return vectors


def _resolve_embedding_provider_chain() -> List[str]:
    primary = os.getenv("ADS_EMBEDDING_PRIMARY_PROVIDER", "gemini").strip().lower()
    fallback = os.getenv("ADS_EMBEDDING_FALLBACK_PROVIDER", "jina").strip().lower()
    providers = [primary, fallback]
    seen: set[str] = set()
    result: List[str] = []
    for provider in providers:
        if provider in {"", "none", "off", "false"}:
            continue
        if provider in seen:
            continue
        seen.add(provider)
        result.append(provider)
    return result


def _embed_texts_with_provider(
    *,
    provider: str,
    texts: Sequence[str],
    task: str,
) -> List[List[float]]:
    provider_name = provider.strip().lower()
    if provider_name == "gemini":
        return _embed_with_gemini(texts, task=task)
    if provider_name == "jina":
        return _embed_with_jina(texts, task=task)
    raise RuntimeError(f"unsupported embedding provider: {provider_name}")


def _get_external_embeddings(
    *,
    texts: Sequence[str],
    task: str,
    provider: str | None = None,
) -> tuple[str, List[List[float]]] | None:
    if not texts:
        return None
    if not _env_enabled("ADS_EMBEDDING_ENABLED", default=False):
        return None

    providers = [provider.strip().lower()] if provider else _resolve_embedding_provider_chain()
    if not providers:
        return None

    # Resolve model names for cache key consistency.
    model_by_provider = {
        "gemini": os.getenv("ADS_GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip() or "gemini-embedding-001",
        "jina": os.getenv("ADS_JINA_EMBEDDING_MODEL", "jina-embeddings-v5-text-small").strip()
        or "jina-embeddings-v5-text-small",
    }

    # Try providers in order, but use cache first for each provider.
    errors: List[str] = []
    for provider in providers:
        model_name = model_by_provider.get(provider, provider)
        cached_vectors: List[List[float] | None] = []
        missing_texts: List[str] = []
        missing_indices: List[int] = []

        for idx, text in enumerate(texts):
            key = _cache_key(provider=provider, model=model_name, task=task, text=text)
            cached = _cache_get(key)
            if cached is not None:
                cached_vectors.append(cached)
            else:
                cached_vectors.append(None)
                missing_texts.append(text)
                missing_indices.append(idx)

        if missing_texts:
            try:
                new_vectors = _embed_texts_with_provider(provider=provider, texts=missing_texts, task=task)
                for local_idx, vector in enumerate(new_vectors):
                    global_idx = missing_indices[local_idx]
                    cached_vectors[global_idx] = list(vector)
                    key = _cache_key(
                        provider=provider,
                        model=model_name,
                        task=task,
                        text=missing_texts[local_idx],
                    )
                    _cache_put(key, vector)
            except Exception as exc:
                errors.append(f"{provider}: {exc}")
                logger.warning(f"[SkillRegistry] embedding fetch failed ({provider}): {exc}")
                continue

        final_vectors: List[List[float]] = []
        incomplete = False
        for item in cached_vectors:
            if item is None:
                incomplete = True
                break
            final_vectors.append(list(item))
        if incomplete:
            errors.append(f"{provider}: incomplete vector set")
            continue
        return provider, final_vectors

    if errors:
        logger.warning("[SkillRegistry] external embeddings unavailable: " + "; ".join(errors))
    return None


def _stable_hash(token: str, *, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % max(1, dim)


def _build_idf(cards: Sequence[SkillCard]) -> Dict[str, float]:
    doc_count = max(1, len(cards))
    df_counter: Counter[str] = Counter()
    for card in cards:
        if not card.tokens:
            continue
        df_counter.update(set(card.tokens))

    idf: Dict[str, float] = {}
    for token, df in df_counter.items():
        numerator = doc_count - df + 0.5
        denominator = df + 0.5
        idf[token] = math.log(1.0 + (numerator / denominator))
    return idf


def _build_sparse_vector(
    token_counts: Counter[str],
    *,
    idf: Dict[str, float],
    dim: int,
) -> Dict[int, float]:
    vector: Dict[int, float] = defaultdict(float)
    for token, freq in token_counts.items():
        if freq <= 0:
            continue
        tf_weight = 1.0 + math.log1p(float(freq))
        idf_weight = idf.get(token, 1.0)
        weight = tf_weight * idf_weight

        idx_primary = _stable_hash(token, dim=dim)
        idx_secondary = _stable_hash(f"{token}#2", dim=dim)
        vector[idx_primary] += weight
        vector[idx_secondary] += 0.5 * weight

    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm <= 0:
        return {}
    return {idx: value / norm for idx, value in vector.items()}


def _cosine_sparse(left: Dict[int, float], right: Dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = 0.0
    for idx, value in left.items():
        dot += value * right.get(idx, 0.0)
    return float(dot)


def _vector_cosine(left: Any, right: Any) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        return _cosine_sparse(left, right)
    if isinstance(left, list) and isinstance(right, list):
        return _cosine_dense(left, right)
    return 0.0


def _bm25_score(
    *,
    query_terms: Counter[str],
    doc_terms: Counter[str],
    idf: Dict[str, float],
    doc_len: int,
    avg_doc_len: float,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    if not query_terms or not doc_terms:
        return 0.0
    length_norm = 1.0 - b + b * (float(doc_len) / max(1.0, avg_doc_len))
    score = 0.0
    for term in query_terms.keys():
        freq = float(doc_terms.get(term, 0))
        if freq <= 0:
            continue
        numerator = freq * (k1 + 1.0)
        denominator = freq + k1 * length_norm
        score += idf.get(term, 0.0) * (numerator / max(1e-9, denominator))
    return float(score)


def _name_bonus(query_text: str, skill_name: str) -> float:
    q = query_text.lower().strip()
    s = skill_name.lower().strip()
    if not q or not s:
        return 0.0
    if s in q:
        return 0.25
    q_compact = q.replace("-", "").replace("_", "").replace(" ", "")
    s_compact = s.replace("-", "").replace("_", "")
    if s_compact and s_compact in q_compact:
        return 0.18
    return 0.0


def _apply_mmr(
    *,
    candidates: List[Dict[str, Any]],
    top_k: int,
    mmr_lambda: float,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    remaining = list(candidates)

    while remaining and len(selected) < max(1, int(top_k)):
        best_idx = -1
        best_score = -1e9
        for idx, candidate in enumerate(remaining):
            relevance = float(candidate.get("relevance_score", 0.0))
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(
                    max(0.0, _vector_cosine(candidate.get("_vector"), chosen.get("_vector")))
                    for chosen in selected
                )
            mmr_score = mmr_lambda * relevance - (1.0 - mmr_lambda) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        if best_idx < 0:
            break
        chosen = remaining.pop(best_idx)
        chosen["mmr_score"] = round(best_score, 6)
        selected.append(chosen)

    return selected


def recommend_skills(
    *,
    query: str,
    working_dir: str | None = None,
    top_k: int = 5,
    min_score: float = 0.12,
) -> List[Dict[str, Any]]:
    """Return top-k skills for one query using hybrid retrieval + MMR diversification."""
    query_text = (query or "").strip()
    if not query_text:
        return []

    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return []

    cards = discover_skills(working_dir=working_dir)
    if not cards:
        return []

    query_counts = Counter(query_tokens)
    idf = _build_idf(cards)
    avg_doc_len = sum(len(card.tokens) for card in cards) / max(1, len(cards))

    vector_dim = _env_int("ADS_SKILL_VECTOR_DIM", _DEFAULT_VECTOR_DIM, minimum=64)
    blend_alpha = _env_float("ADS_SKILL_HYBRID_BLEND_ALPHA", 0.65, minimum=0.0, maximum=1.0)
    mmr_enabled = _env_enabled("ADS_SKILL_MMR_ENABLED", default=True)
    mmr_lambda = _env_float("ADS_SKILL_MMR_LAMBDA", 0.78, minimum=0.0, maximum=1.0)
    candidate_pool = _env_int("ADS_SKILL_CANDIDATE_POOL", max(8, int(top_k) * 4), minimum=max(1, int(top_k)))

    query_vector = _build_sparse_vector(query_counts, idf=idf, dim=vector_dim)
    raw_candidates: List[Dict[str, Any]] = []
    max_bm25 = 0.0

    for card in cards:
        doc_counts = Counter(card.tokens)
        if not doc_counts:
            continue

        bm25 = _bm25_score(
            query_terms=query_counts,
            doc_terms=doc_counts,
            idf=idf,
            doc_len=max(1, len(card.tokens)),
            avg_doc_len=max(1.0, avg_doc_len),
        )
        max_bm25 = max(max_bm25, bm25)
        card_vector = _build_sparse_vector(doc_counts, idf=idf, dim=vector_dim)
        vector_score = max(0.0, _cosine_sparse(query_vector, card_vector))

        overlap_terms = sorted(set(query_counts.keys()).intersection(set(doc_counts.keys())))
        raw_candidates.append(
            {
                "card": card,
                "bm25_raw": bm25,
                "vector_score": vector_score,
                "overlap_terms": overlap_terms,
                "_vector": card_vector,
            }
        )

    if not raw_candidates:
        return []

    # Optional external embeddings: primary/fallback provider chain.
    # We only re-rank a lexical candidate subset to limit API cost/latency.
    external_vector_by_skill: Dict[str, List[float]] = {}
    external_provider_used = ""
    external_pool_size = _env_int(
        "ADS_EMBEDDING_CANDIDATE_POOL",
        max(8, int(candidate_pool)),
        minimum=max(1, int(top_k)),
    )
    ranked_for_external = sorted(
        raw_candidates,
        key=lambda item: (
            -float(item["bm25_raw"]),
            -len(item["overlap_terms"]),
            str(item["card"].skill_name),
        ),
    )
    external_candidates = ranked_for_external[: max(1, int(external_pool_size))]
    if external_candidates and _env_enabled("ADS_EMBEDDING_ENABLED", default=False):
        query_external = _get_external_embeddings(texts=[query_text], task="query")
        if query_external is not None:
            provider_name, query_vectors = query_external
            document_external = _get_external_embeddings(
                texts=[item["card"].search_text for item in external_candidates],
                task="document",
                provider=provider_name,
            )
            if document_external is not None and document_external[0] == provider_name:
                external_provider_used = provider_name
                query_dense = query_vectors[0]
                doc_vectors = list(document_external[1])
                limit = min(len(doc_vectors), len(external_candidates))
                for idx in range(limit):
                    item = external_candidates[idx]
                    doc_vector = doc_vectors[idx]
                    card = item["card"]
                    external_vector_by_skill[card.skill_name] = list(doc_vector)
                    item["vector_score"] = max(0.0, _cosine_dense(query_dense, doc_vector))
                    item["_vector"] = list(doc_vector)

    normalized_candidates: List[Dict[str, Any]] = []
    for item in raw_candidates:
        card = item["card"]
        bm25_norm = float(item["bm25_raw"]) / max(1e-9, max_bm25)
        vector_score = float(item["vector_score"])
        bonus = _name_bonus(query_text, card.skill_name)
        relevance = blend_alpha * bm25_norm + (1.0 - blend_alpha) * vector_score + bonus
        if relevance < float(min_score):
            continue
        normalized_candidates.append(
            {
                "skill_name": card.skill_name,
                "skill_dir": card.skill_dir,
                "summary": card.summary,
                "relevance_score": relevance,
                "lexical_score": bm25_norm,
                "vector_score": vector_score,
                "name_bonus": bonus,
                "overlap_count": len(item["overlap_terms"]),
                "matched_terms": item["overlap_terms"][:8],
                "_vector": item["_vector"],
                "embedding_provider": external_provider_used if card.skill_name in external_vector_by_skill else "local",
            }
        )

    if not normalized_candidates:
        return []

    normalized_candidates.sort(
        key=lambda item: (
            -float(item["relevance_score"]),
            -int(item["overlap_count"]),
            str(item["skill_name"]),
        )
    )

    pool = normalized_candidates[: max(1, int(candidate_pool))]
    if mmr_enabled and len(pool) > 1 and int(top_k) > 1:
        selected = _apply_mmr(candidates=pool, top_k=max(1, int(top_k)), mmr_lambda=mmr_lambda)
    else:
        selected = pool[: max(1, int(top_k))]

    results: List[Dict[str, Any]] = []
    for item in selected:
        results.append(
            {
                "skill_name": item["skill_name"],
                "skill_dir": item["skill_dir"],
                "summary": item["summary"],
                "score": round(float(item["relevance_score"]), 6),
                "lexical_score": round(float(item["lexical_score"]), 6),
                "vector_score": round(float(item["vector_score"]), 6),
                "name_bonus": round(float(item["name_bonus"]), 6),
                "overlap_count": int(item["overlap_count"]),
                "matched_terms": list(item["matched_terms"]),
                "mmr_score": round(float(item.get("mmr_score", item["relevance_score"])), 6),
                "embedding_provider": str(item.get("embedding_provider", "local")),
            }
        )
    return results


def format_skill_advice(recommendations: Iterable[Dict[str, Any]], *, total_skills: int = 0) -> str:
    """Format retrieved skills for planner/executor prompt injection."""
    recs = list(recommendations)
    if not recs:
        if total_skills > 0:
            return (
                f"Skill inventory size: {total_skills}. "
                "No high-confidence skill matches found for this request."
            )
        return "No skills detected."

    lines = []
    if total_skills > 0:
        lines.append(f"Skill inventory size: {total_skills}.")
    lines.append("Top matched scientific skills:")
    for idx, rec in enumerate(recs, start=1):
        name = str(rec.get("skill_name", "")).strip()
        summary = str(rec.get("summary", "")).strip()
        matched_terms = rec.get("matched_terms", [])
        match_hint = ", ".join(str(term) for term in matched_terms[:4])
        line = f"{idx}. {name}"
        if summary:
            line += f" - {summary}"
        if match_hint:
            line += f" (matched: {match_hint})"
        lines.append(line)
    return "\n".join(lines)
