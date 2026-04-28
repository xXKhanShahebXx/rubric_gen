from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(payload: Any) -> str:
    serialized = stable_json_dumps(payload).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def make_cache_key(stage: str, payload: Dict[str, Any]) -> str:
    return f"{stage}::{stable_hash(payload)}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class JsonlCache:
    def __init__(self, path: Path, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._items: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    def load(self) -> None:
        with self._lock:
            if not self.enabled:
                self._loaded = True
                return
            if self._loaded:
                return
            if self.path.exists():
                with self.path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        key = record.get("key")
                        if key:
                            self._items[key] = record
            self._loaded = True

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        self.load()
        with self._lock:
            return self._items.get(key)

    def set(self, key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return {"key": key, **payload}
        self.load()
        record = {"key": key, **payload}
        with self._lock:
            self._items[key] = record
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(stable_json_dumps(record) + "\n")
        return record

    def get_or_set(self, key: str, payload_factory):
        cached = self.get(key)
        if cached is not None:
            return cached, True
        payload = payload_factory()
        return self.set(key, payload), False
