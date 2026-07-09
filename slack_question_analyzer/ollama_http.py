"""
Shared helpers for turning Ollama HTTP failures into messages a
non-technical user can act on.

Ollama's error responses carry the one sentence that matters (e.g.
"model requires more system memory (9.3 GiB) than is available (7.1
GiB)") in the JSON body — `raise_for_status()` alone throws it away and
surfaces "500 Server Error: Internal Server Error" instead. Connection
errors similarly surface urllib3 pool internals. Both used to reach the
dashboard verbatim.
"""

import requests


def http_error_detail(response) -> str:
    """The actionable sentence from an Ollama error response, if any."""
    try:
        detail = (response.json() or {}).get('error') or ''
    except ValueError:
        detail = ''
    return str(detail).strip()


def raise_with_detail(response):
    """raise_for_status(), but the raised HTTPError carries Ollama's own
    error sentence instead of just the bare status line."""
    # Duck-typed: response-like objects without .ok (test fakes) are judged
    # by status code, defaulting to success like a bodyless 200
    ok = getattr(response, 'ok', None)
    if ok is None:
        ok = int(getattr(response, 'status_code', 200) or 200) < 400
    if ok:
        return
    detail = http_error_detail(response)
    if detail:
        raise requests.HTTPError(
            f"Ollama error (HTTP {response.status_code}): {detail}",
            response=response)
    response.raise_for_status()


def friendly_failure(error: Exception, ollama_url: str) -> str:
    """One plain-English line for a terminal Ollama failure."""
    if isinstance(error, requests.ConnectionError):
        return (f"Ollama at {ollama_url} stopped responding or is not "
                f"running. Start it with: ollama serve (if it was running, "
                f"it may have crashed — a machine out of memory is the "
                f"usual cause).")
    if isinstance(error, requests.Timeout):
        return (f"Ollama at {ollama_url} took too long to answer. The "
                f"machine may be overloaded — close heavy applications and "
                f"try again.")
    return str(error)
