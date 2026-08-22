import hashlib
import json
import os
import pickle
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / ".cache"

# One pooled session for the whole process. A report makes on the order of a thousand
# statsapi calls; `requests.get` opens a fresh TCP + TLS connection for every one of them,
# and the handshake is a large share of the cost of a small JSON response. Keep-alive
# removes it. The pool is sized for the thread pools that fan these calls out.
_SESSION = requests.Session()
_ADAPTER = HTTPAdapter(pool_connections=16, pool_maxsize=32, max_retries=2)
_SESSION.mount("https://", _ADAPTER)
_SESSION.mount("http://", _ADAPTER)


def cache_enabled():
    return os.environ.get("SCOUTING_REPORT_DISABLE_CACHE", "").lower() not in {"1", "true", "yes"}


def _cache_key(payload):
    serialized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _path(namespace, key, suffix, create=False):
    """Cache path for a key. `create` only when about to write: this is called on every
    lookup, and an unconditional mkdir there is a syscall per cache hit."""
    safe_namespace = namespace.replace("/", "_").replace("\\", "_")
    directory = CACHE_ROOT / safe_namespace
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{key}.{suffix}"


def cached_json_request(url, params=None, namespace="api", force=False, cache_key_extra=None):
    # cache_key_extra lets date-sensitive endpoints (e.g. game logs, whose URL carries no
    # date) bust the cache by report/as-of date so stale snapshots aren't served forever.
    payload = {"url": url, "params": params or {}}
    if cache_key_extra is not None:
        payload["extra"] = cache_key_extra
    key = _cache_key(payload)
    path = _path(namespace, key, "json")

    if cache_enabled() and not force and path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    response = _SESSION.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    if cache_enabled():
        path = _path(namespace, key, "json", create=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f)

    return data


def cached_dataframe_call(namespace, func, *args, force=False, **kwargs):
    payload = {
        "func": getattr(func, "__name__", str(func)),
        "args": args,
        "kwargs": kwargs,
    }
    key = _cache_key(payload)
    path = _path(namespace, key, "pkl")

    if cache_enabled() and not force and path.exists():
        with path.open("rb") as f:
            return pickle.load(f)

    data = func(*args, **kwargs)

    if cache_enabled():
        path = _path(namespace, key, "pkl", create=True)
        with path.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return data
