"""
agent.py - Ollama tool-calling orchestration for The Stable Boy.

Sends user messages to llama3.1, handles tool calls in a loop,
and returns the final assistant response.
"""

import json
from pathlib import Path
from typing import Callable

from ollama import Client

from tools import TOOL_SCHEMAS, TOOL_DISPATCH

HISTORY_FILE = Path(__file__).parent / "chat_history.json"

_ollama = Client(host="http://localhost:11434")

SYSTEM_PROMPT = """\
You are "The Stable Boy" — a seasoned trackside insider and horse racing \
analyst. Confident, casual, sharp Australian voice.

════════════════════════════════════════
ABSOLUTE RULES — never break these
════════════════════════════════════════
1. NEVER ask the user for a URL. You do not need one. The tools navigate \
   TAB.com.au on their own.
2. NEVER fabricate race data, runner names, odds, or predictions. \
   Only present what the tools return.
3. Do NOT call tools for general chat, greetings, or questions about your \
   capabilities — reply with words only.

════════════════════════════════════════
TOOLS
════════════════════════════════════════
scan_race()
  Opens TAB.com.au and captures data for a single race. Call with NO \
arguments — the interceptor navigates there automatically. Use whenever \
the user mentions a specific race, track, or just says "scan a race". \
Do not ask for or construct a URL.

scan_next_races(minutes=N)
  Scans every race starting within the next N minutes. Use when the user \
asks about upcoming races, "what's on soon", "best bet in the next hour", \
or wants a broad look at what's running. Pick a sensible N from context \
(default 30, up to 90 for "next hour").

parse_data()
  Parses the raw captured data into a structured race matrix CSV. \
Always call this after a scan before doing anything else with the data.

inspect_race_matrix(track=None, race_number=None)
  Returns the full parsed runner list — barriers, weights, jockeys, odds, \
form, and results. For finished races (race_status "Paying") it returns \
finishing_position, win_dividend, and place_dividend. Use this to answer \
"who won", "what were the results", "show me the runners", "who are the \
favourites", "what's the form like". No scan needed if a matrix already \
exists from a prior scan.

predict_winners()
  Runs the XGBoost model, compares model probability vs TAB implied odds, \
flags EDGE_FOUND value bets. Always call after parse_data.

════════════════════════════════════════
WORKFLOWS
════════════════════════════════════════
"Scan [any race / track / race name]"
  → scan_race() → parse_data() → predict_winners()

"Upcoming races / next 30 min / best bet right now"
  → scan_next_races(minutes=N) → parse_data() → predict_winners()

"Show me the runners / form / favourites"
  → inspect_race_matrix() — only scan first if no matrix exists yet

════════════════════════════════════════
PRESENTING RESULTS
════════════════════════════════════════
- Group by track and race number.
- For each runner: TAB odds, implied %, model %, edge %.
- Highlight EDGE_FOUND: "TAB has this bloke at $8.00 (12.5% implied). \
  Model reckons 18.2% — that's a +5.7% edge, worth a look."
- For multi-race scans, name the single best value bet across all races.
- Keep it sharp and trackside — no waffle.\
"""

DEFAULT_MODEL = "llama3.1"


def list_models() -> list[str]:
    """Return names of all models available in Ollama."""
    try:
        resp = _ollama.list()
        return sorted(m.model for m in resp.models)
    except Exception:
        return [DEFAULT_MODEL]


class Agent:
    """Manages conversation history and the Ollama tool-calling loop."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        status_callback: Callable[[str], None] | None = None,
    ):
        self.model = model
        self.history: list[dict] = self._load_history()
        self._status_cb = status_callback

    @staticmethod
    def _load_history() -> list[dict]:
        """Load conversation history from disk, or start fresh."""
        if HISTORY_FILE.exists():
            try:
                data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
                if isinstance(data, list) and data:
                    # Ensure system prompt is current
                    if data[0].get("role") == "system":
                        data[0]["content"] = SYSTEM_PROMPT
                    return data
            except (json.JSONDecodeError, KeyError):
                pass
        return [{"role": "system", "content": SYSTEM_PROMPT}]

    def _save_history(self) -> None:
        """Persist conversation history to disk.

        Only saves user and assistant messages (tool messages contain
        large payloads that bloat the file and aren't useful on reload).
        """
        saveable = [
            msg for msg in self.history
            if isinstance(msg, dict) and msg.get("role") in ("system", "user", "assistant")
        ]
        HISTORY_FILE.write_text(
            json.dumps(saveable, indent=2, default=str),
            encoding="utf-8",
        )

    def clear_history(self) -> None:
        """Reset conversation history."""
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink()

    def _status(self, msg: str) -> None:
        if self._status_cb:
            self._status_cb(msg)

    def chat(self, user_message: str) -> str:
        """Send a user message and run the tool-calling loop until the LLM
        produces a final text response. Returns the assistant's reply."""

        self.history.append({"role": "user", "content": user_message})

        while True:
            self._status("Thinking...")
            response = _ollama.chat(
                model=self.model,
                messages=self.history,
                tools=TOOL_SCHEMAS,
            )

            msg = response.message

            # If no tool calls, we have a final answer
            if not msg.tool_calls:
                assistant_text = msg.content or ""
                self.history.append({"role": "assistant", "content": assistant_text})
                self._save_history()
                self._status("Ready")
                return assistant_text

            # Process each tool call
            self.history.append(msg)

            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                func_args = tool_call.function.arguments or {}

                self._status(f"Running {func_name}...")

                func = TOOL_DISPATCH.get(func_name)
                if func is None:
                    result = {"error": f"Unknown tool: {func_name}"}
                else:
                    try:
                        result = func(**func_args)
                    except Exception as e:
                        result = {"error": str(e)}

                self.history.append({
                    "role": "tool",
                    "content": json.dumps(result, default=str),
                })
