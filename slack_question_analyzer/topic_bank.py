"""
Persistent topic bank: the analyzer learns across analyses.

Every analysis deposits its question groups here (topic, summary, and a
centroid embedding). Future analyses match new groups against the bank, so
recurring topics keep their established names (stable labels week over week),
skip redundant LLM labeling, and accumulate history ("seen in N analyses").
"""

import os
import json
import time
import uuid
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TopicBank:
    """JSON-backed store of known topics with centroid matching."""

    def __init__(self, path: Optional[str] = None, enabled: Optional[bool] = None,
                 model: Optional[str] = None):
        if enabled is None:
            enabled = os.getenv('TOPIC_BANK', 'on').lower() not in ('off', '0', 'false')
        self.enabled = enabled
        self.model = model  # embedding model: entries only match their own model
        self.path = Path(path or os.getenv('TOPIC_BANK_PATH', 'topic_bank.json'))
        self.entries: List[Dict] = []
        self._deleted_ids = set()  # tombstones so merge-on-save keeps deletes

        if self.enabled and self.path.exists():
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    self.entries = json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("Topic bank at %s is unreadable; starting fresh", self.path)
                self.entries = []
            for entry in self.entries:  # banks from before ids existed
                entry.setdefault('id', uuid.uuid4().hex)

    @staticmethod
    def _unit(vector) -> Optional[np.ndarray]:
        v = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(v)
        return v / norm if norm else None

    def match(self, centroid, threshold: float) -> Optional[Dict]:
        """Best bank entry whose centroid similarity clears the threshold."""
        if not self.enabled or not self.entries or centroid is None:
            return None
        c = self._unit(centroid)
        if c is None:
            return None

        best, best_sim = None, threshold
        for entry in self.entries:
            # Entries from a different embedding model can share dimensions but
            # live in a different space — never match across models
            if self.model and entry.get('model') and entry['model'] != self.model:
                continue
            v = self._unit(entry['centroid'])
            if v is None or v.shape != c.shape:
                continue  # dimension mismatch (legacy entries without a model stamp)
            sim = float(c @ v)
            if sim >= best_sim:
                best, best_sim = entry, sim
        return best

    def record(self, group: Dict, centroid, matched: Optional[Dict] = None) -> Optional[Dict]:
        """
        Update the matched entry with this group's occurrence, or add a new
        entry. Established topic names are kept (that's the point: stability).
        """
        if not self.enabled or centroid is None:
            return None
        today = time.strftime('%Y-%m-%d')

        if matched is not None:
            old_n = matched.get('question_count', 1)
            blended = (np.asarray(matched['centroid'], dtype=float) * old_n
                       + np.asarray(centroid, dtype=float) * group['count'])
            unit = self._unit(blended)
            if unit is not None:
                matched['centroid'] = [round(float(x), 6) for x in unit]
            matched['question_count'] = old_n + group['count']
            matched['analysis_count'] = matched.get('analysis_count', 1) + 1
            matched['last_seen'] = today
            return matched

        entry = {
            'id': uuid.uuid4().hex,
            'model': self.model,
            'topic': group.get('topic'),
            'summary': group.get('summary'),
            'representative_question': group['representative_question'],
            'keywords': group.get('keywords', []),
            'centroid': [round(float(x), 6) for x in centroid],
            'question_count': group['count'],
            'analysis_count': 1,
            'first_seen': today,
            'last_seen': today,
        }
        self.entries.append(entry)
        return entry

    def _find(self, topic_id: str) -> Optional[Dict]:
        return next((e for e in self.entries if e.get('id') == topic_id), None)

    def rename(self, topic_id: str, new_name: str) -> bool:
        """Rename a topic (the fix for a bad name sticking forever)."""
        entry = self._find(topic_id)
        if entry is None:
            return False
        entry['topic'] = new_name
        self.save()
        return True

    def delete(self, topic_id: str) -> bool:
        """Remove a junk topic from the bank."""
        entry = self._find(topic_id)
        if entry is None:
            return False
        self.entries.remove(entry)
        self._deleted_ids.add(topic_id)
        self.save()
        return True

    def merge(self, source_id: str, target_id: str) -> bool:
        """
        Merge one topic into another: the target keeps its name; counts add
        up and the centroid becomes the weighted blend. The source is removed.
        """
        source = self._find(source_id)
        target = self._find(target_id)
        if source is None or target is None or source is target:
            return False

        s_n = source.get('question_count', 1)
        t_n = target.get('question_count', 1)
        s_v = self._unit(source['centroid'])
        t_v = self._unit(target['centroid'])
        if s_v is not None and t_v is not None and s_v.shape == t_v.shape:
            blended = self._unit(t_v * t_n + s_v * s_n)
            if blended is not None:
                target['centroid'] = [round(float(x), 6) for x in blended]
        target['question_count'] = t_n + s_n
        target['analysis_count'] = (target.get('analysis_count', 1)
                                    + source.get('analysis_count', 0))
        target['last_seen'] = max(target.get('last_seen') or '',
                                  source.get('last_seen') or '') or target.get('last_seen')
        self.entries.remove(source)
        self._deleted_ids.add(source_id)
        self.save()
        return True

    def save(self):
        """
        Persist the bank (atomic write; best-effort). Entries added on disk by
        another instance since we loaded are merged in (by id) rather than
        overwritten; our own deletions are honored via tombstones.
        """
        if not self.enabled:
            return
        try:
            known = {e.get('id') for e in self.entries}
            merged = list(self.entries)
            if self.path.exists():
                try:
                    with open(self.path, 'r', encoding='utf-8') as f:
                        for entry in json.load(f):
                            eid = entry.get('id')
                            if eid and eid not in known and eid not in self._deleted_ids:
                                merged.append(entry)
                                known.add(eid)
                except (json.JSONDecodeError, OSError):
                    pass

            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=self.path.parent, suffix='.tmp')
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(merged, f, ensure_ascii=False)
            os.replace(tmp_path, self.path)
            self.entries = merged
        except OSError as e:
            logger.warning("Could not save topic bank: %s", e)
