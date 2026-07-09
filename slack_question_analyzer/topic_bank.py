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
import hashlib
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
                    loaded = json.load(f)
                # Wrong shape (valid JSON that isn't a list of dicts) also
                # means "start fresh" — same contract as unreadable
                self.entries = [e for e in loaded if isinstance(e, dict)] \
                    if isinstance(loaded, list) else []
                if not isinstance(loaded, list):
                    logger.warning("Topic bank at %s has an unexpected shape; "
                                   "starting fresh", self.path)
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

    # Per-entry cap on stored occurrence fingerprints (oldest evicted).
    # 400 fingerprints @ 16 hex chars is ~7KB per very-hot topic — cheap
    # insurance against unbounded growth on long-lived banks.
    SEEN_FP_MAX = 400

    @staticmethod
    def _fingerprints(group: Dict) -> List[str]:
        """Stable identity per occurrence: normalized text + date. The same
        message re-uploaded in an overlapping export (the natural 'export
        last 90 days every month' workflow) produces the same fingerprint."""
        fps = []
        for q in group.get('questions', []):
            text = ' '.join((q.get('normalized_text')
                             or q.get('text') or '').split()).lower()
            if not text:
                continue
            key = f"{text}|{q.get('date') or 'Unknown'}"
            fps.append(hashlib.sha1(key.encode('utf-8')).hexdigest()[:16])
        # One fingerprint per occurrence IDENTITY: two people asking the
        # identical question on the same date collapse to one, matching how
        # the topic-history endpoint counts — badge and chart must agree
        return list(dict.fromkeys(fps))

    def record(self, group: Dict, centroid, matched: Optional[Dict] = None) -> Optional[Dict]:
        """
        Update the matched entry with this group's occurrence, or add a new
        entry. Established topic names are kept (that's the point: stability).

        Occurrences are FINGERPRINTED (text+date): re-uploading a transcript
        that overlaps a previous upload must not inflate question_count or
        the 'recurring xN' badge — only genuinely new occurrences count.
        """
        if not self.enabled or centroid is None:
            return None
        today = time.strftime('%Y-%m-%d')
        fps = self._fingerprints(group)

        if matched is not None:
            seen = matched.setdefault('seen', [])
            seen_set = set(seen)
            new_fps = [fp for fp in fps if fp not in seen_set]
            new_count = len(new_fps) if fps else group['count']
            if not new_fps and fps:
                # Every occurrence was already recorded (an overlapping
                # re-upload): the topic recurred in the DATA before, not
                # again now — no count inflation, no centroid re-blend, and
                # no last_seen bump (re-analyzing old data is not a sighting)
                return matched
            matched['last_seen'] = today
            old_n = matched.get('question_count', 1)
            # Seeds are recorded with count 0 — a zero BLEND weight would
            # let the first real sighting completely replace the curated
            # centroid instead of blending with it (the count stays honest)
            blended = (np.asarray(matched['centroid'], dtype=float) * max(old_n, 1)
                       + np.asarray(centroid, dtype=float) * new_count)
            unit = self._unit(blended)
            if unit is not None:
                matched['centroid'] = [round(float(x), 6) for x in unit]
            matched['question_count'] = old_n + new_count
            matched['analysis_count'] = matched.get('analysis_count', 1) + 1
            seen.extend(new_fps)
            del seen[:-self.SEEN_FP_MAX]
            return matched

        entry = {
            'id': uuid.uuid4().hex,
            'model': self.model,
            'topic': group.get('topic'),
            'summary': group.get('summary'),
            'representative_question': group['representative_question'],
            'keywords': group.get('keywords', []),
            'centroid': [round(float(x), 6) for x in centroid],
            # Fingerprints are the occurrence identity — count them, not raw
            # members (two identical text+date members are one occurrence)
            'question_count': len(fps) if fps else group['count'],
            'analysis_count': 1,
            'first_seen': today,
            'last_seen': today,
            'seen': fps[-self.SEEN_FP_MAX:],
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

    def set_published(self, topic_id: str, published: bool) -> Optional[str]:
        """
        Mark a topic's FAQ as published (or clear the mark). Returns the
        stamped date, '' when cleared, or None for an unknown topic. The
        date matters: the topic-history chart draws a marker at it, so the
        user can watch ask-volume fall (or not) after their doc went live.
        """
        entry = self._find(topic_id)
        if entry is None:
            return None
        if published:
            entry['faq_published'] = time.strftime('%Y-%m-%d')
        else:
            entry.pop('faq_published', None)
        self.save()
        return entry.get('faq_published', '')

    def set_answer(self, topic_id: str, answer: str) -> Optional[str]:
        """
        Save (or clear, with an empty string) the CURATED answer for a
        topic. Once a human has approved wording, it becomes the canonical
        answer: every future FAQ export uses it instead of re-drafting from
        thread replies, turning the export into a living document rather
        than a per-analysis snapshot. Returns the stored answer, '' when
        cleared, or None for an unknown topic.
        """
        entry = self._find(topic_id)
        if entry is None:
            return None
        answer = (answer or '').strip()
        if answer:
            entry['curated_answer'] = answer
            entry['answer_updated'] = time.strftime('%Y-%m-%d')
        else:
            entry.pop('curated_answer', None)
            entry.pop('answer_updated', None)
        self.save()
        return answer

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
        # Union the occurrence fingerprints so a post-merge re-upload of
        # either side's transcript still counts as already-seen
        merged_seen = list(dict.fromkeys(
            (target.get('seen') or []) + (source.get('seen') or [])))
        target['seen'] = merged_seen[-self.SEEN_FP_MAX:]
        # A published FAQ survives the merge (latest date wins, so the
        # history marker keeps pointing at the real publish moment)
        published = max(target.get('faq_published') or '',
                        source.get('faq_published') or '')
        if published:
            target['faq_published'] = published
        # A curated answer survives too — the target's own wins (it is the
        # topic being kept), the source's fills the gap otherwise
        if not target.get('curated_answer') and source.get('curated_answer'):
            target['curated_answer'] = source['curated_answer']
            if source.get('answer_updated'):
                target['answer_updated'] = source['answer_updated']
        # The source's id (and any ids IT absorbed) become aliases of the
        # target, so saved analyses that recorded the old topic_id still
        # count toward the merged topic's history
        aliases = target.setdefault('merged_ids', [])
        for alias in [source_id] + (source.get('merged_ids') or []):
            if alias and alias not in aliases and alias != target_id:
                aliases.append(alias)
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
                        on_disk = json.load(f)
                    # Tolerate a wrong-shape file the same way load does
                    for entry in (on_disk if isinstance(on_disk, list) else []):
                        if not isinstance(entry, dict):
                            continue
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
