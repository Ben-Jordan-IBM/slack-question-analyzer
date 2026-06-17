"""
Loading transcript content from files, including zipped Slack exports.

Shared by the CLI (file paths) and the API server (uploaded bytes).
"""

import io
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence

TEXT_EXTENSIONS = ('.json', '.txt', '.csv')


def contents_from_zip_bytes(raw_bytes: bytes,
                            max_bytes: Optional[int] = None) -> List[str]:
    """
    Extract text contents from a zipped Slack export.

    Skips directories, __MACOSX entries, hidden files, and non-text files.
    Raises ValueError when the (uncompressed) contents exceed max_bytes or
    no usable files are found; zipfile.BadZipFile when it isn't a zip.
    """
    total = 0
    contents = []
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename
            base = name.rsplit('/', 1)[-1]
            if name.startswith('__MACOSX') or base.startswith('.'):
                continue
            if not base.lower().endswith(TEXT_EXTENSIONS):
                continue
            total += info.file_size
            if max_bytes is not None and total > max_bytes:
                raise ValueError(
                    f'Zip contents exceed the {max_bytes // (1024 * 1024)}MB limit')
            contents.append(archive.read(info).decode('utf-8', errors='replace'))
    if not contents:
        raise ValueError('Zip contains no .json, .txt, or .csv files')
    return contents


def load_input_files(paths: Sequence[str],
                     max_bytes: Optional[int] = None) -> List[str]:
    """Read transcript contents from file paths; .zip archives are unpacked."""
    contents = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.suffix.lower() == '.zip':
            contents.extend(contents_from_zip_bytes(path.read_bytes(),
                                                    max_bytes=max_bytes))
        else:
            contents.append(path.read_text(encoding='utf-8', errors='replace'))
    return contents
