"""
web_app.py - Mobile-friendly web front end for The Stable Boy agent.

Serves a single-page chat UI over HTTP so it can be used from a phone
browser (via a tunnel like ngrok), instead of the Tkinter desktop GUI.
Protected with HTTP Basic Auth since this is exposed to the internet.
"""

import os
import secrets
from functools import wraps

from flask import Flask, Response, jsonify, render_template_string, request

from agent import Agent, DEFAULT_MODEL

AUTH_USER = os.environ.get("STABLEBOY_USER", "admin")
AUTH_PASS = os.environ.get("STABLEBOY_PASS")
if not AUTH_PASS:
    raise SystemExit(
        "Set STABLEBOY_PASS before starting the server, e.g.\n"
        '  $env:STABLEBOY_PASS = "your-password"'
    )

app = Flask(__name__)
_agent = Agent(model=DEFAULT_MODEL)


def _check_auth(username: str, password: str) -> bool:
    return secrets.compare_digest(username, AUTH_USER) and secrets.compare_digest(
        password, AUTH_PASS
    )


def require_auth(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        auth = request.authorization
        if not auth or not _check_auth(auth.username, auth.password):
            return Response(
                "Authentication required", 401,
                {"WWW-Authenticate": 'Basic realm="Stable Boy"'},
            )
        return f(*args, **kwargs)

    return wrapped


PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>The Stable Boy</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  body {
    margin: 0; background: #0A0A0A; color: #E0E0E0;
    font-family: -apple-system, "Segoe UI", sans-serif;
    display: flex; flex-direction: column; height: 100dvh;
  }
  header {
    padding: 14px 16px; border-bottom: 1px solid #2A2A2A;
    font-weight: 700; font-size: 18px; color: #39FF14;
    letter-spacing: 0.5px;
  }
  #chat {
    flex: 1; overflow-y: auto; padding: 12px 14px;
    -webkit-overflow-scrolling: touch;
  }
  .msg { margin-bottom: 14px; }
  .who { font-size: 12px; color: #808080; margin-bottom: 2px; }
  .who.bot { color: #39FF14; }
  .bubble { white-space: pre-wrap; line-height: 1.4; font-size: 15px; }
  .status { color: #555; font-size: 13px; margin: 4px 0; }
  form {
    display: flex; gap: 8px; padding: 10px;
    border-top: 1px solid #2A2A2A; background: #0A0A0A;
  }
  input {
    flex: 1; background: #1E1E1E; border: 1px solid #2A2A2A;
    border-radius: 12px; color: #E0E0E0; padding: 12px 14px;
    font-size: 16px; outline: none;
  }
  button {
    background: #1E1E1E; color: #39FF14; border: 1px solid #39FF14;
    border-radius: 12px; padding: 0 18px; font-weight: 700; font-size: 15px;
  }
  button:disabled { opacity: 0.5; }
</style>
</head>
<body>
<header>THE STABLE BOY</header>
<div id="chat"></div>
<form id="form">
  <input id="input" autocomplete="off" placeholder="Type a message..." />
  <button id="send" type="submit">SEND</button>
</form>
<script>
const chat = document.getElementById('chat');
const form = document.getElementById('form');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');

function addMsg(who, text, isBot) {
  const wrap = document.createElement('div');
  wrap.className = 'msg';
  wrap.innerHTML = `<div class="who ${isBot ? 'bot' : ''}">${who}</div><div class="bubble"></div>`;
  wrap.querySelector('.bubble').textContent = text;
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  addMsg('You', text, false);
  sendBtn.disabled = true;

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text}),
    });
    const data = await resp.json();
    addMsg('The Stable Boy', data.reply || data.error || 'No response', true);
  } catch (err) {
    addMsg('The Stable Boy', 'Connection error: ' + err, true);
  } finally {
    sendBtn.disabled = false;
  }
});

(function restore() {
  const history = {{ history | tojson }};
  let shown = false;
  for (const m of history) {
    if (m.role === 'user') { addMsg('You', m.content, false); shown = true; }
    else if (m.role === 'assistant' && m.content) { addMsg('The Stable Boy', m.content, true); shown = true; }
  }
  if (!shown) {
    addMsg('The Stable Boy',
      "G'day! I'm The Stable Boy. Tell me which race you want to analyse " +
      "and I'll fire up the feeds, crunch the numbers, and find you some value.",
      true);
  }
})();
</script>
</body>
</html>
"""


@app.route("/")
@require_auth
def index():
    return render_template_string(PAGE, history=_agent.history)


@app.route("/api/chat", methods=["POST"])
@require_auth
def api_chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400
    try:
        reply = _agent.chat(message)
    except Exception as e:
        return jsonify({"error": f"Something went wrong: {e}"}), 500
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
