import hashlib
import json
import os
import pickle
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / ".cache"


def cache_enabled():
    return os.environ.get("SCOUTING_REPORT_DISABLE_CACHE", "").lower() not in {"1", "true", "yes"}


def _cache_key(payload):
    serialized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _path(namespace, key, suffix):
    safe_namespace = namespace.replace("/", "_").replace("\\", "_")
    directory = CACHE_ROOT / safe_namespace
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

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    if cache_enabled():
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
        with path.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return data
