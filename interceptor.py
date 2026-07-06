"""
interceptor.py - Intercepts ALL network responses from TAB.com.au via Playwright.

Uses stealth + system Chrome channel to bypass bot detection in headless mode.
Captures racing API data and WebSocket frames, saves to raw_race_data.json.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from playwright.async_api import (
    async_playwright,
    Response,
    WebSocket,
    TimeoutError as PlaywrightTimeout,
)
from playwright_stealth import Stealth

# -- Configuration -----------------------------------------------------------
TARGET_URL = "https://www.tab.com.au/racing"
RACE_PAGE = "https://www.tab.com.au/racing/meetings/today/R"
OUTPUT_FILE = Path(__file__).parent / "raw_race_data.json"
NAV_TIMEOUT_MS = 60_000
IDLE_TIMEOUT_S = 20
MAX_CF_RETRIES = 3

# Optional: point at a real installed Chrome for better fingerprint evasion
# (set CHROME_EXECUTABLE_PATH locally, e.g. to the Windows Chrome binary).
# Left unset, Playwright's bundled Chromium is used - this is what CI runners
# use, since they don't have a real Chrome install at a fixed path.
CHROME_EXECUTABLE_PATH = os.environ.get("CHROME_EXECUTABLE_PATH")


def _launch_kwargs() -> dict:
    kwargs = {
        "headless": True,
        "args": [
            "--disable-blink-features=AutomationControlled",
            "--no-first-run",
            "--no-default-browser-check",
        ],
    }
    if CHROME_EXECUTABLE_PATH:
        kwargs["executable_path"] = CHROME_EXECUTABLE_PATH
    return kwargs

# Domains that are just tracking/analytics noise - ignore them
NOISE_DOMAINS = [
    "launchdarkly.com", "demdex.net", "adobedc.", "adobe.com",
    "branch.io", "go-mpulse.net", "adsrvr.org", "clickagy.com",
    "google", "facebook", "doubleclick", "insight.", "stt.tab.com.au",
]


def is_noise(url: str) -> bool:
    return any(d in url for d in NOISE_DOMAINS)


def is_racing_data(url: str) -> bool:
    """Match URLs that carry actual racing/betting data."""
    keywords = [
        "form-guide", "markets", "races", "meetings",
        "tab-info-service", "racing", "runners", "odds",
        "results", "bet-", "price", "pool", "exotic",
        "recommendation-service",
    ]
    return "api.beta.tab.com.au" in url and any(kw in url.lower() for kw in keywords)


async def handle_response(
    response: Response,
    racing_data: list[dict],
    all_responses: list[dict],
) -> None:
    """Capture every non-noise JSON response; flag racing data separately."""
    url = response.url

    if is_noise(url):
        return

    content_type = response.headers.get("content-type", "")
    is_json = "json" in content_type or "javascript" in content_type

    # Try to grab JSON body from anything that looks like an API call
    body = None
    if is_json or "api.beta.tab.com.au" in url or "cmsapi.tab.com.au" in url:
        try:
            body = await response.json()
        except Exception:
            try:
                body = await response.text()
            except Exception:
                body = None

    if body is None:
        return

    entry = {
        "url": url,
        "status": response.status,
        "content_type": content_type,
        "data": body,
    }

    all_responses.append(entry)
    tag = "    "

    if is_racing_data(url):
        racing_data.append(entry)
        tag = "[+] RACE"

    # Summarise what we got
    if isinstance(body, dict):
        keys = list(body.keys())[:6]
        print(f"  {tag} {response.status} {url[:130]}  keys={keys}")
    elif isinstance(body, list):
        print(f"  {tag} {response.status} {url[:130]}  list[{len(body)}]")
    else:
        preview = str(body)[:80]
        print(f"  {tag} {response.status} {url[:130]}  {preview}")


def handle_ws_open(ws: WebSocket, ws_messages: list[dict]) -> None:
    """Track WebSocket connections - TAB may push race data over WS."""
    print(f"  [WS] Connected: {ws.url[:120]}")

    def on_frame(payload) -> None:
        # payload can be a string or a dict with 'data' key depending on version
        raw = payload if isinstance(payload, str) else payload.get("data", str(payload))
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            data = raw
        ws_messages.append({"url": ws.url, "data": data})
        if isinstance(data, dict):
            preview = str(list(data.keys()))[:100]
        elif isinstance(data, list):
            preview = f"list[{len(data)}]"
        else:
            preview = str(data)[:100]
        print(f"  [WS] {ws.url[:80]}: {preview}")

    ws.on("framereceived", on_frame)


async def run(target_url: str | None = None, log_callback=None) -> None:
    """Run the interceptor.

    Args:
        target_url: A specific TAB race URL to navigate to. If the URL
                    points directly to a race (contains '/racing/20'),
                    the interceptor skips the meetings page and goes
                    straight there. Otherwise it follows the default
                    landing -> meetings -> first-race flow.
        log_callback: Optional callable(str) for GUI log streaming.
    """

    def log(msg: str) -> None:
        print(msg)
        if log_callback:
            log_callback(msg)

    # Decide navigation targets from the provided URL
    landing = TARGET_URL
    direct_race = None
    if target_url:
        target_url = target_url.strip()
        if "/racing/" in target_url and any(c.isdigit() for c in target_url.split("/racing/")[-1]):
            # Looks like a direct race link e.g. .../racing/2026-06-08/TRACK/X/R/1
            direct_race = target_url
            landing = target_url
        else:
            landing = target_url

    racing_data: list[dict] = []
    all_responses: list[dict] = []
    ws_messages: list[dict] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(**_launch_kwargs())

        stealth = Stealth()
        context = await browser.new_context(
            ignore_https_errors=True,
            viewport={"width": 1920, "height": 1080},
            locale="en-AU",
            timezone_id="Australia/Sydney",
            geolocation={"latitude": -33.8688, "longitude": 151.2093},
            permissions=["geolocation"],
        )
        await stealth.apply_stealth_async(context)

        page = await context.new_page()

        # Attach listeners
        page.on("response", lambda r: asyncio.ensure_future(
            handle_response(r, racing_data, all_responses)
        ))
        page.on("websocket", lambda ws: handle_ws_open(ws, ws_messages))

        # -- Navigate to landing / direct race page --------------------------
        for attempt in range(1, MAX_CF_RETRIES + 1):
            try:
                log(f"[*] Attempt {attempt} -> {landing}")
                resp = await page.goto(landing, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
            except PlaywrightTimeout:
                log(f"[!] Timeout on attempt {attempt}")
                if attempt == MAX_CF_RETRIES:
                    log("[X] Max retries. Exiting.")
                    await browser.close()
                    return
                continue

            title = await page.title()
            status = resp.status if resp else 0
            log(f"    Status={status}, Title={title!r}")

            if status in (403, 503) or "just a moment" in title.lower() or "access denied" in title.lower():
                log("[!] Blocked. Waiting 8s before retry...")
                await page.wait_for_timeout(8_000)
                continue
            else:
                log("[OK] Page loaded.")
                break

        # Let the SPA fully hydrate
        await page.wait_for_timeout(4_000)

        # -- If we went to a direct race URL, skip the meetings flow ---------
        if not direct_race:
            log(f"\n[*] Navigating to meetings page -> {RACE_PAGE}")
            try:
                await page.goto(RACE_PAGE, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
            except PlaywrightTimeout:
                log("[!] Meetings page timed out, continuing anyway...")

            await page.wait_for_timeout(6_000)

            # Try clicking into a specific race
            log("[*] Looking for individual race links...")
            race_links = page.locator('a[href*="/racing/"][href*="/R"]')
            count = await race_links.count()
            if count == 0:
                race_links = page.locator('a[href*="/racing/"]')
                count = await race_links.count()

            log(f"    Found {count} links")
            for i in range(count):
                href = await race_links.nth(i).get_attribute("href") or ""
                segments = [s for s in href.split("/") if s]
                skip_words = ["jackpot", "calendar", "futures", "meetings/today", "meetings/results"]
                if len(segments) >= 4 and not any(sw in href for sw in skip_words):
                    log(f"[*] Clicking into race: {href}")
                    await race_links.nth(i).click()
                    await page.wait_for_timeout(8_000)
                    log(f"    Now at: {page.url}")
                    break

        # -- Collect data for a fixed window ---------------------------------
        log(f"[*] Collecting data for {IDLE_TIMEOUT_S}s...")
        await asyncio.sleep(IDLE_TIMEOUT_S)

        await browser.close()

    # -- Save results --------------------------------------------------------
    log(f"\nAPI responses: {len(racing_data)} | WS messages: {len(ws_messages)}")

    output = {
        "racing_data": racing_data,
        "websocket_messages": ws_messages,
    }

    if racing_data or ws_messages:
        OUTPUT_FILE.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        log(f"[OK] Saved to {OUTPUT_FILE}")
    else:
        log("[!] No racing data captured.")


async def run_multi(minutes: int = 30, log_callback=None) -> None:
    """Scan all races starting within *minutes* from now.

    Navigates to the TAB meetings page, reads the meetings API response
    to find races with start times in the upcoming window, then visits
    each race page to capture full runner/odds data.

    Args:
        minutes: How far ahead to look for upcoming races.
        log_callback: Optional callable(str) for GUI log streaming.
    """

    def log(msg: str) -> None:
        print(msg)
        if log_callback:
            log_callback(msg)

    racing_data: list[dict] = []
    all_responses: list[dict] = []
    ws_messages: list[dict] = []
    meetings_payload: list[dict] = []

    async def _capture_meetings(response: Response) -> None:
        """Extra listener to grab the meetings JSON for race-time filtering."""
        url = response.url
        if "tab-info-service/racing/dates" not in url or "meetings" not in url:
            return
        if "races/" in url:
            return  # individual race endpoint, skip
        try:
            body = await response.json()
            if isinstance(body, dict) and "meetings" in body:
                meetings_payload.append(body)
        except Exception:
            pass

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(**_launch_kwargs())

        stealth = Stealth()
        context = await browser.new_context(
            ignore_https_errors=True,
            viewport={"width": 1920, "height": 1080},
            locale="en-AU",
            timezone_id="Australia/Sydney",
            geolocation={"latitude": -33.8688, "longitude": 151.2093},
            permissions=["geolocation"],
        )
        await stealth.apply_stealth_async(context)

        page = await context.new_page()

        # Attach listeners
        page.on("response", lambda r: asyncio.ensure_future(
            handle_response(r, racing_data, all_responses)
        ))
        page.on("response", lambda r: asyncio.ensure_future(
            _capture_meetings(r)
        ))
        page.on("websocket", lambda ws: handle_ws_open(ws, ws_messages))

        # -- Land on TAB, pass any challenge ------------------------------------
        for attempt in range(1, MAX_CF_RETRIES + 1):
            try:
                log(f"[*] Attempt {attempt} -> {TARGET_URL}")
                resp = await page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
            except PlaywrightTimeout:
                log(f"[!] Timeout on attempt {attempt}")
                if attempt == MAX_CF_RETRIES:
                    log("[X] Max retries. Exiting.")
                    await browser.close()
                    return
                continue

            title = await page.title()
            status = resp.status if resp else 0
            log(f"    Status={status}, Title={title!r}")
            if status in (403, 503) or "just a moment" in title.lower():
                log("[!] Blocked. Waiting 8s before retry...")
                await page.wait_for_timeout(8_000)
                continue
            else:
                log("[OK] Page loaded.")
                break

        await page.wait_for_timeout(4_000)

        # -- Navigate to meetings page to trigger the meetings API call ---------
        log(f"\n[*] Navigating to meetings page -> {RACE_PAGE}")
        try:
            await page.goto(RACE_PAGE, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
        except PlaywrightTimeout:
            log("[!] Meetings page timed out, continuing anyway...")

        await page.wait_for_timeout(6_000)

        # -- Parse meetings payload to find upcoming races ----------------------
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(minutes=minutes)
        log(f"[*] Looking for races between now and +{minutes}min (UTC {now:%H:%M} - {cutoff:%H:%M})")

        race_urls: list[str] = []
        race_labels: list[str] = []

        for payload in meetings_payload:
            for meeting in payload.get("meetings", []):
                meet_name = meeting.get("meetingName", "?")
                race_type = meeting.get("raceType", "R")
                venue_mnemonic = meeting.get("venueMnemonic", "")
                meet_date = meeting.get("meetingDate", "")

                for race in meeting.get("races", []):
                    status = race.get("raceStatus", "")
                    if status in ("Paying", "Closed", "Abandoned", "Interim"):
                        continue

                    start_str = race.get("raceStartTime", "")
                    if not start_str:
                        continue

                    try:
                        start_time = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                    except ValueError:
                        continue

                    if now <= start_time <= cutoff:
                        race_num = race.get("raceNumber", "?")
                        # Build the TAB URL for this race
                        url = f"https://www.tab.com.au/racing/{meet_date}/{meet_name.replace(' ', '-').upper()}/{venue_mnemonic}/{race_type}/{race_num}"
                        race_urls.append(url)
                        label = f"R{race_num} {meet_name} ({start_time:%H:%M} UTC)"
                        race_labels.append(label)

        log(f"    Found {len(race_urls)} upcoming race(s) in the next {minutes} minutes")
        for label in race_labels:
            log(f"      - {label}")

        if not race_urls:
            log("[!] No upcoming races found in the time window.")
            await browser.close()
            # Still save whatever we captured
            output = {"racing_data": racing_data, "websocket_messages": ws_messages}
            OUTPUT_FILE.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
            return

        # -- Visit each race page to capture full runner data -------------------
        for i, url in enumerate(race_urls):
            log(f"\n[*] ({i+1}/{len(race_urls)}) Navigating to {race_labels[i]}")
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
            except PlaywrightTimeout:
                log(f"[!] Timeout loading {race_labels[i]}, skipping...")
                continue

            # Wait for race API data to arrive
            await page.wait_for_timeout(6_000)
            log(f"    Captured data for {race_labels[i]}")

        await browser.close()

    # -- Save results ----------------------------------------------------------
    log(f"\nAPI responses: {len(racing_data)} | WS messages: {len(ws_messages)}")

    output = {
        "racing_data": racing_data,
        "websocket_messages": ws_messages,
    }

    if racing_data or ws_messages:
        OUTPUT_FILE.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        log(f"[OK] Saved to {OUTPUT_FILE}")
    else:
        log("[!] No racing data captured.")


if __name__ == "__main__":
    url_arg = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        asyncio.run(run(target_url=url_arg))
    except KeyboardInterrupt:
        print("\n[!] Interrupted.")
        sys.exit(1)
