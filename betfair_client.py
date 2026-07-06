"""
betfair_client.py - Minimal Betfair Exchange API client (Delayed App Key).

Uses non-interactive (bot) certificate login, since this is designed to run
unattended in a scheduled job. Only the read-only endpoints needed to find
settled Australian thoroughbred races and their winners are implemented -
this does not place bets.

Required environment variables:
    BETFAIR_USERNAME   - Betfair account username
    BETFAIR_PASSWORD   - Betfair account password
    BETFAIR_APP_KEY    - Delayed (or Live) application key
    BETFAIR_CERT_PEM   - client certificate, PEM text (or path via BETFAIR_CERT_PATH)
    BETFAIR_KEY_PEM    - client private key, PEM text (or path via BETFAIR_KEY_PATH)

See: https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687915/Non-Interactive+bot+login
"""

import os
import tempfile
from datetime import datetime, timezone

import requests

CERT_LOGIN_URL = "https://identitysso-cert.betfair.com/api/certlogin"
RPC_URL = "https://api.betfair.com/exchange/betting/json-rpc/v1"
HORSE_RACING_EVENT_TYPE_ID = "7"


class BetfairAuthError(RuntimeError):
    pass


def _write_temp_pem(content: str, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _resolve_cert_paths() -> tuple[str, str]:
    """Return (cert_path, key_path), writing temp files if PEM text was
    supplied directly via env vars (as is convenient in CI secrets)."""
    cert_path = os.environ.get("BETFAIR_CERT_PATH")
    key_path = os.environ.get("BETFAIR_KEY_PATH")

    if not cert_path:
        cert_pem = os.environ.get("BETFAIR_CERT_PEM")
        if not cert_pem:
            raise BetfairAuthError(
                "Set BETFAIR_CERT_PATH or BETFAIR_CERT_PEM (client certificate)."
            )
        cert_path = _write_temp_pem(cert_pem, ".crt")

    if not key_path:
        key_pem = os.environ.get("BETFAIR_KEY_PEM")
        if not key_pem:
            raise BetfairAuthError(
                "Set BETFAIR_KEY_PATH or BETFAIR_KEY_PEM (client private key)."
            )
        key_path = _write_temp_pem(key_pem, ".key")

    return cert_path, key_path


class BetfairClient:
    def __init__(self):
        self.username = os.environ.get("BETFAIR_USERNAME")
        self.password = os.environ.get("BETFAIR_PASSWORD")
        self.app_key = os.environ.get("BETFAIR_APP_KEY")
        if not (self.username and self.password and self.app_key):
            raise BetfairAuthError(
                "Set BETFAIR_USERNAME, BETFAIR_PASSWORD, and BETFAIR_APP_KEY."
            )
        self.session_token: str | None = None

    def login(self) -> None:
        """Non-interactive certificate login. Populates self.session_token."""
        cert_path, key_path = _resolve_cert_paths()

        resp = requests.post(
            CERT_LOGIN_URL,
            data={"username": self.username, "password": self.password},
            headers={
                "X-Application": self.app_key,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            cert=(cert_path, key_path),
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("loginStatus") != "SUCCESS":
            raise BetfairAuthError(f"Betfair login failed: {body}")
        self.session_token = body["sessionToken"]

    def _rpc(self, method: str, params: dict) -> dict:
        if not self.session_token:
            self.login()
        resp = requests.post(
            RPC_URL,
            json={
                "jsonrpc": "2.0",
                "method": f"SportsAPING/v1.0/{method}",
                "params": params,
                "id": 1,
            },
            headers={
                "X-Application": self.app_key,
                "X-Authentication": self.session_token,
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        if "error" in body:
            raise RuntimeError(f"Betfair API error calling {method}: {body['error']}")
        return body["result"]

    def list_recently_started_au_thoroughbred_markets(
        self, started_after: datetime, started_before: datetime
    ) -> list[dict]:
        """Return WIN markets for AU thoroughbred races with a scheduled off
        time in [started_after, started_before] (both UTC-aware datetimes).

        Each item includes marketId, marketName, venue, raceNumber (best
        effort parse), marketStartTime, and runners (selectionId -> name).
        """
        market_filter = {
            "eventTypeIds": [HORSE_RACING_EVENT_TYPE_ID],
            "marketCountries": ["AU"],
            "marketTypeCodes": ["WIN"],
            "marketStartTime": {
                "from": started_after.isoformat().replace("+00:00", "Z"),
                "to": started_before.isoformat().replace("+00:00", "Z"),
            },
        }

        result = self._rpc(
            "listMarketCatalogue",
            {
                "filter": market_filter,
                "marketProjection": ["EVENT", "MARKET_START_TIME", "RUNNER_METADATA"],
                "maxResults": 200,
            },
        )

        markets = []
        for m in result:
            event = m.get("event", {})
            markets.append({
                "market_id": m["marketId"],
                "market_name": m.get("marketName", ""),
                "venue": event.get("venue", ""),
                "market_start_time": m.get("marketStartTime"),
                "runners": [
                    {
                        "selection_id": r["selectionId"],
                        "runner_name": r.get("runnerName", ""),
                    }
                    for r in m.get("runners", [])
                ],
            })
        return markets

    def get_settled_results(self, market_ids: list[str]) -> dict[str, dict]:
        """For each market, return {marketId: {status, runners: {selectionId: status}}}.

        Only markets whose status is CLOSED have final WINNER/LOSER runner
        statuses; markets still open/suspended are returned with their
        current status so callers can skip them and retry later.
        """
        if not market_ids:
            return {}

        out: dict[str, dict] = {}
        # listMarketBook allows up to 25 markets per call
        for i in range(0, len(market_ids), 25):
            chunk = market_ids[i:i + 25]
            result = self._rpc(
                "listMarketBook",
                {
                    "marketIds": chunk,
                    "priceProjection": {"priceData": []},
                },
            )
            for m in result:
                out[m["marketId"]] = {
                    "status": m.get("status"),
                    "runners": {
                        r["selectionId"]: r.get("status")
                        for r in m.get("runners", [])
                    },
                }
        return out


def parse_runner_number(runner_name: str) -> tuple[int | None, str]:
    """Betfair AU racing runner names are formatted 'N. Horse Name'.
    Returns (runner_number, horse_name); falls back to (None, full string)."""
    if ". " in runner_name:
        prefix, _, name = runner_name.partition(". ")
        if prefix.strip().isdigit():
            return int(prefix.strip()), name.strip()
    return None, runner_name.strip()
