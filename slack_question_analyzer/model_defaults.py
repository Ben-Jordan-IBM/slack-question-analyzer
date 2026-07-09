"""
Hardware-aware default for the Ollama chat model.

The 8B model gives noticeably better topic names, verification verdicts and
summaries, but needs roughly 8GB of RAM to run comfortably alongside the
embedding model, the OS and a browser. So the default is:

- machine with >= MIN_RAM_GB_FOR_PREFERRED total RAM -> llama3.1:8b
- smaller machine (or RAM unknown)                   -> llama3.2 (3B, ~2GB)

OLLAMA_GENERATION_MODEL in .env always wins. Separately, GroupLabeler drops
down to the small model at runtime if the preferred one isn't downloaded —
so existing installs that only have llama3.2 keep working untouched.
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)

PREFERRED_GENERATION_MODEL = 'llama3.1:8b'   # ~4.9GB download
FALLBACK_GENERATION_MODEL = 'llama3.2'       # 3B, ~2GB download
MIN_RAM_GB_FOR_PREFERRED = 12


def total_ram_gb():
    """Total physical RAM in GB, or None if it can't be determined."""
    try:
        if sys.platform == 'win32':
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return None
            return stat.ullTotalPhys / 2**30
        # Linux and macOS
        return os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / 2**30
    except (OSError, ValueError, AttributeError):
        return None


def default_generation_model() -> str:
    """The Ollama chat model to use when OLLAMA_GENERATION_MODEL isn't set."""
    pinned = os.getenv('OLLAMA_GENERATION_MODEL')
    if pinned:
        return pinned
    ram = total_ram_gb()
    if ram is not None and ram >= MIN_RAM_GB_FOR_PREFERRED:
        return PREFERRED_GENERATION_MODEL
    return FALLBACK_GENERATION_MODEL
