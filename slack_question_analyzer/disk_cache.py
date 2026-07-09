"""
Persistent JSON-value cache backed by a single file, keyed by hashed strings.

Used for embeddings (which never change for a given model+text) and LLM
outputs (deterministic at temperature 0 with a fixed seed), so repeat
analyses skip provider calls entirely.
"""

import os
import json
import logging
import hashlib
import tempfile
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class JsonDiskCache:
    """Maps arbitrary strings to JSON-serializable values, persisted to disk."""

    def __init__(self, provider: str, model: str, cache_dir: str,
                 enabled: bool = True, max_entries: Optional[int] = None):
        self.enabled = enabled
        self.max_entries = max_entries  # oldest entries evicted beyond this
        self._memory = {}
        self._dirty = False
        self._dirty_count = 0
        self._disk_stat = None  # (mtime_ns, size) as of our last load/write

        safe_model = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in model)
        self.cache_path = Path(cache_dir) / f"{provider}_{safe_model}.json"

        if self.enabled and self.cache_path.exists():
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                # Valid JSON of the wrong shape (a list, a string...) must
                # also mean "start fresh", not an AttributeError on first get
                self._memory = loaded if isinstance(loaded, dict) else {}
                self._disk_stat = self._stat()
            except (json.JSONDecodeError, OSError):
                # Corrupt or unreadable cache: start fresh rather than failing
                self._memory = {}

    def _stat(self):
        try:
            st = self.cache_path.stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[Any]:
        key = self._key(text)
        value = self._memory.get(key)
        if value is not None and self.max_entries:
            # Refresh recency (move to end): eviction trims the front of the
            # dict, and an entry that was only ever READ this run would
            # otherwise sit at the front and be evicted by a mid-run save() —
            # breaking read-after-save on a full cache.
            self._memory[key] = self._memory.pop(key)
        return value

    def set(self, text: str, value: Any):
        self._memory[self._key(text)] = value
        self._dirty = True
        self._dirty_count += 1

    def save(self, min_dirty: int = 0):
        """
        Persist the cache to disk (atomic write). Entries written by another
        process since we loaded are merged in rather than overwritten, so
        concurrent instances (CLI + server, parallel jobs) don't lose work.

        Profiled: this used to re-read AND re-serialize the whole file on
        every call — 33ms per LLM verdict at 5000 entries, seconds per call
        for embedding-sized caches. Two mitigations, semantics preserved:
        - the reload-merge is skipped when the on-disk file is untouched
          since our last load/write (the single-process common case; a
          concurrent writer changes mtime/size and gets merged as before)
        - callers may pass min_dirty=N to batch writes; a final save() with
          the default 0 flushes whatever remains
        """
        if not self.enabled or not self._dirty:
            return
        if min_dirty and self._dirty_count < min_dirty:
            return
        try:
            if self.cache_path.exists() and self._stat() != self._disk_stat:
                merged = {}
                try:
                    with open(self.cache_path, 'r', encoding='utf-8') as f:
                        merged = json.load(f)
                except (json.JSONDecodeError, OSError):
                    merged = {}
                if not isinstance(merged, dict):
                    merged = {}
                merged.update(self._memory)
            else:
                merged = self._memory

            # Cap growth: dicts keep insertion order, so the front is oldest
            if self.max_entries and len(merged) > self.max_entries:
                for key in list(merged.keys())[:len(merged) - self.max_entries]:
                    del merged[key]

            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=self.cache_path.parent, suffix='.tmp')
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(merged, f)
            os.replace(tmp_path, self.cache_path)
            self._memory = merged
            self._dirty = False
            self._dirty_count = 0
            self._disk_stat = self._stat()
        except OSError as e:
            # Cache persistence is best-effort; results are unaffected
            logger.warning("Could not save cache %s: %s", self.cache_path.name, e)
