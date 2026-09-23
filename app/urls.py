"""Working out the URL participants should actually visit.

This is less trivial than it looks. In a Codespace the app is reached through
a forwarded-port domain that the process itself has no direct knowledge of, and
printing a localhost URL on the join screen would be a silent, total failure —
everyone scans the QR and nothing happens.

So: three sources, most explicit first, and the join page always shows the
resulting URL as text so a wrong guess is visible rather than silent.
"""

import os
from typing import Optional

_DEFAULT_PORT = 8000


def app_port() -> int:
    """Return the port this app is configured to bind to."""
    value = os.environ.get("PORT") or os.environ.get("SCANPATH_PORT") or _DEFAULT_PORT
    try:
        return int(value)
    except (TypeError, ValueError):
        return _DEFAULT_PORT


def _request_port(request_base_url: str) -> Optional[int]:
    """Extract a port number from a request URL when present."""
    from urllib.parse import urlparse

    parsed = urlparse(str(request_base_url))
    return parsed.port


def codespace_url(port: Optional[int] = None) -> Optional[str]:
    """Reconstruct the public forwarded-port URL inside a Codespace."""
    port = app_port() if port is None else port
    name = os.environ.get("CODESPACE_NAME")
    domain = os.environ.get("GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN")
    if name and domain:
        return f"https://{name}-{port}.{domain}"
    return None


def public_base_url(request_base_url: str, port: Optional[int] = None) -> dict:
    """Return the participant-facing base URL and where it came from.

    Order: an explicit override, then Codespaces environment, then whatever
    host the request arrived on.
    """
    override = os.environ.get("SCANPATH_PUBLIC_URL")
    if override:
        return {"url": override.rstrip("/"), "source": "SCANPATH_PUBLIC_URL"}

    if os.environ.get("CODESPACE_NAME") and os.environ.get("GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN"):
        port = port or _request_port(request_base_url) or app_port()
        cs = codespace_url(port)
        if cs:
            return {"url": cs, "source": "codespace"}

    return {"url": str(request_base_url).rstrip("/"), "source": "request"}


# Hosts that resolve only on the machine running the app.
_LOOPBACK = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"}


def is_reachable_by_others(url: str) -> bool:
    """Could a phone elsewhere open this URL?

    Two ways to fail. A loopback host is obviously local. Less obviously, a
    bare hostname with no dot (`testserver`, `mymachine`) only resolves on a
    network that already knows that name, so it is no good on a QR code
    either. A LAN address like 192.168.1.5 *is* fine — everyone on the venue
    wifi can reach it — so the check must not simply reject private IPs.
    """
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").lower()
    if not host or host in _LOOPBACK:
        return False
    if host.startswith("127."):
        return False
    return "." in host or ":" in host  # FQDN/IPv4, or an IPv6 literal
