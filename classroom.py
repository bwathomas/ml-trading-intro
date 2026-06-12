"""Live classroom activities for the ML-for-trading lab.

Two activities, each hosted from a single Colab cell:

    from classroom import launch_wisdom, launch_arena

    launch_wisdom(teacher_key="...")     # Activity 1: Wisdom of the Crowd
    launch_arena(teacher_key="...")      # Activity 2: the Strategy Arena

Each launcher starts a small Flask server on the Colab VM plus a free Cloudflare
quick-tunnel, and prints two links: a STUDENT link to share with the class and a
TEACHER link to keep on the projector. No accounts or API keys needed.

Heads-up for the Strategy Arena: it executes code that students submit, in a
subprocess on *your* runtime with a hard timeout. That is fine for a classroom on a
throwaway Colab VM; do not host it anywhere you care about.
"""

import concurrent.futures
import io
import json
import logging
import multiprocessing
import os
import re
import secrets
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request
from werkzeug.serving import make_server

TRADING_DAYS = 252

# ----------------------------------------------------------------------------
# shared plumbing: servers and tunnels
# ----------------------------------------------------------------------------

_SERVERS = {}          # activity name -> (werkzeug server, thread)
_TUNNELS = {}          # activity name -> subprocess
CLOUDFLARED_URL = ("https://github.com/cloudflare/cloudflared/releases/latest/"
                   "download/cloudflared-linux-amd64")


def _free_port(preferred):
    for port in range(preferred, preferred + 20):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port found")


def _start_server(name, app, port):
    """(Re)start a named Flask server on a background thread."""
    logging.getLogger("werkzeug").setLevel(logging.ERROR)   # silence request spam
    if name in _SERVERS:
        try:
            _SERVERS[name][0].shutdown()
        except Exception:
            pass
    if name in _TUNNELS:
        try:
            _TUNNELS[name].terminate()
        except Exception:
            pass
    port = _free_port(port)
    server = make_server("0.0.0.0", port, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _SERVERS[name] = (server, thread)
    return port


def _cloudflared_path():
    path = "/tmp/cloudflared"
    if not os.path.exists(path):
        print("downloading cloudflared (one-time, ~40 MB)...")
        urllib.request.urlretrieve(CLOUDFLARED_URL, path)
        os.chmod(path, 0o755)
    return path


def _start_tunnel(name, port, timeout=45):
    """Open a Cloudflare quick-tunnel to the given local port; return its URL."""
    proc = subprocess.Popen(
        [_cloudflared_path(), "tunnel", "--url", f"http://127.0.0.1:{port}",
         "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    _TUNNELS[name] = proc
    url, deadline = None, time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if m:
            url = m.group(0)
            break
    # keep draining the pipe so cloudflared never blocks on a full buffer
    threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()
    return url


def _announce(activity, student_url, teacher_url):
    text = (f"\n{'=' * 70}\n{activity}\n{'=' * 70}\n"
            f"  STUDENT LINK (share this):   {student_url}\n"
            f"  TEACHER LINK (keep private): {teacher_url}\n{'=' * 70}\n")
    try:
        from IPython.display import HTML, display
        display(HTML(
            f"<div style='font-family:sans-serif;border:2px solid #4a6fa5;"
            f"border-radius:10px;padding:14px;margin:6px 0'>"
            f"<b style='font-size:1.2em'>{activity}</b><br><br>"
            f"📣 <b>Student link</b> (share with the class):<br>"
            f"<a href='{student_url}' target='_blank' style='font-size:1.25em'>"
            f"{student_url}</a><br><br>"
            f"🔑 <b>Teacher link</b> (open this yourself, don't share):<br>"
            f"<a href='{teacher_url}' target='_blank'>{teacher_url}</a></div>"))
    except Exception:
        print(text)


def _serve(name, app, port, tunnel):
    port = _start_server(name, app, port)
    base = f"http://127.0.0.1:{port}"
    if tunnel:
        url = _start_tunnel(name, port)
        if url is None:
            print("⚠️ tunnel did not come up — links below only work on this machine")
        else:
            base = url
    return base


# ----------------------------------------------------------------------------
# small shared HTML pieces
# ----------------------------------------------------------------------------

_BASE_CSS = """
:root { --blue:#4a6fa5; --bg:#f7f8fa; --card:#fff; --ink:#1d2733; }
* { box-sizing:border-box; }
body { font-family:system-ui,sans-serif; background:var(--bg); color:var(--ink);
       margin:0; padding:24px; }
.card { background:var(--card); border-radius:14px; padding:22px;
        box-shadow:0 2px 10px rgba(0,0,0,.08); max-width:680px; margin:0 auto 16px; }
h1 { font-size:1.4em; margin:0 0 10px; } h2 { font-size:1.1em; }
input,textarea { width:100%; padding:10px; font-size:1.05em; border:1px solid #ccd;
                 border-radius:8px; font-family:inherit; }
textarea { font-family:ui-monospace,monospace; font-size:.85em; }
button { background:var(--blue); color:#fff; border:0; border-radius:8px;
         padding:12px 22px; font-size:1.05em; cursor:pointer; margin:8px 4px 0 0; }
button:disabled { background:#aab; cursor:default; }
.big { font-size:1.5em; } .muted { color:#67737f; } .ok { color:#188038; }
.bad { color:#c5221f; }
.stat { display:inline-block; margin:8px 18px 0 0; }
.stat b { font-size:1.6em; display:block; }
.teacher body { background:#11151c; }
"""

_DARK_CSS = """
body { background:#11151c; color:#e8edf3; }
.card { background:#1a212c; box-shadow:none; max-width:1100px; }
.muted { color:#93a0ad; }
"""

CHARTJS = ("<script src='https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js' "
           "integrity='sha384-NrKB+u6Ts6AtkIhwPixiKTzgSKNblyhlk0Sohlgar9UHUBzai/sgnNNWWd291xqt' "
           "crossorigin='anonymous'></script>")


def _page(title, css, body, script):
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{title}</title>{CHARTJS}<style>{_BASE_CSS}{css}</style></head>"
            f"<body>{body}<script>{script}</script></body></html>")


# ============================================================================
# ACTIVITY 1 — Wisdom of the Crowd
# ============================================================================

DEFAULT_QUESTIONS = [
    ("In 1906, 787 visitors to an English county fair each guessed the weight "
     "of an ox. Francis Galton expected nonsense. Your turn: how much does a "
     "prize ox weigh, in pounds?", 1198, "lbs"),
    ("How many jellybeans fit in a classic one-gallon jar?", 930, "jellybeans"),
    ("How long is the Golden Gate Bridge (total, including approaches), in feet?",
     8980, "feet"),
    ("How far away is the Moon, on average, in miles?", 238855, "miles"),
    ("What is the population of Australia, in millions?", 27.2, "million"),
    ("How tall is Mount Everest, in feet?", 29032, "feet"),
]


class _WisdomState:
    def __init__(self, questions):
        self.questions = questions
        self.idx = 0
        self.phase = "open"               # open | revealed
        self.subs = {i: [] for i in range(len(questions))}
        self.lock = threading.Lock()

    def reveal_payload(self):
        _, answer, unit = self.questions[self.idx]
        vals = np.array(self.subs[self.idx], dtype=float)
        if len(vals) == 0:
            return {"answer": answer, "unit": unit, "n": 0}
        shown = vals
        if len(vals) >= 8:                # clip absurd outliers so the chart reads
            lo, hi = np.percentile(vals, [2, 98])
            shown = vals[(vals >= lo) & (vals <= hi)]
        nbins = int(min(15, max(5, len(shown) // 2 + 3)))
        counts, edges = np.histogram(shown, bins=nbins)
        labels = [f"{edges[i]:,.0f}–{edges[i+1]:,.0f}" for i in range(len(counts))]
        truth_bin = int(np.clip(np.digitize(answer, edges) - 1, 0, len(counts) - 1)) \
            if edges[0] <= answer <= edges[-1] else -1
        return {"answer": answer, "unit": unit, "n": int(len(vals)),
                "mean": float(vals.mean()), "median": float(np.median(vals)),
                "labels": labels, "counts": counts.tolist(), "truth_bin": truth_bin}


_WISDOM_STUDENT = """
<div class='card'>
  <h1>🔮 Wisdom of the Crowd</h1>
  <div id='qnum' class='muted'></div>
  <h2 id='prompt'>loading…</h2>
  <div id='entry'>
    <input id='guess' type='number' step='any' inputmode='decimal'
           placeholder='your guess'> <span id='unit' class='muted'></span>
    <button id='send' onclick='submitGuess()'>Submit my guess</button>
    <div id='msg'></div>
  </div>
  <div id='revealed' style='display:none'>
    <div class='stat'><span class='muted'>true answer</span><b id='truth' class='ok'></b></div>
    <div class='stat'><span class='muted'>crowd average</span><b id='mean'></b></div>
    <div class='stat'><span class='muted'>crowd median</span><b id='median'></b></div>
    <canvas id='hist' height='200'></canvas>
  </div>
</div>
"""

_WISDOM_STUDENT_JS = """
let lastIdx = -1, lastPhase = '', chart = null;
function key(i){ return 'wisdom_answered_' + i; }
async function poll(){
  const s = await fetch('state').then(r=>r.json()).catch(()=>null);
  if(!s) return;
  if(s.idx !== lastIdx){
    lastIdx = s.idx; lastPhase = '';
    document.getElementById('msg').textContent = '';
    document.getElementById('guess').value = '';
  }
  document.getElementById('qnum').textContent = 'Question ' + (s.idx+1) + ' of ' + s.n_questions;
  document.getElementById('prompt').textContent = s.prompt;
  document.getElementById('unit').textContent = s.unit;
  const answered = localStorage.getItem(key(s.idx));
  document.getElementById('send').disabled = (s.phase !== 'open') || !!answered;
  if(answered && s.phase === 'open')
    document.getElementById('msg').innerHTML = "<span class='ok'>✔ submitted — watch the screen!</span>";
  if(s.phase === 'revealed' && lastPhase !== 'revealed'){ showReveal(s.reveal); }
  if(s.phase === 'open'){ document.getElementById('revealed').style.display = 'none';
                          document.getElementById('entry').style.display = 'block'; }
  lastPhase = s.phase;
}
function fmt(x){ return Number(x).toLocaleString(undefined, {maximumFractionDigits: 1}); }
function showReveal(r){
  document.getElementById('entry').style.display = 'none';
  document.getElementById('revealed').style.display = 'block';
  document.getElementById('truth').textContent = fmt(r.answer) + ' ' + r.unit;
  document.getElementById('mean').textContent = r.n ? fmt(r.mean) : '—';
  document.getElementById('median').textContent = r.n ? fmt(r.median) : '—';
  if(chart) chart.destroy();
  if(!r.n) return;
  const colors = r.counts.map((_,i)=> i === r.truth_bin ? '#188038' : '#4a6fa5');
  chart = new Chart(document.getElementById('hist'), {type:'bar',
    data:{labels:r.labels, datasets:[{data:r.counts, backgroundColor:colors}]},
    options:{plugins:{legend:{display:false}},
             scales:{y:{ticks:{precision:0}}, x:{ticks:{maxRotation:60,minRotation:45}}}}});
}
async function submitGuess(){
  const v = document.getElementById('guess').value;
  if(v === ''){ return; }
  const res = await fetch('submit', {method:'POST',
    headers:{'Content-Type':'application/json'}, body:JSON.stringify({value:+v})});
  if(res.ok){ localStorage.setItem(key(lastIdx), '1'); }
  poll();
}
poll(); setInterval(poll, 2000);
"""

_WISDOM_TEACHER = """
<div class='card'>
  <h1>🔮 Wisdom of the Crowd — teacher console</h1>
  <div id='qnum' class='muted'></div>
  <h2 id='prompt' class='big'></h2>
  <div class='stat'><span class='muted'>submissions so far</span><b id='count'>0</b></div>
  <div id='stats' style='display:none'>
    <div class='stat'><span class='muted'>true answer</span><b id='truth' class='ok'></b></div>
    <div class='stat'><span class='muted'>crowd average</span><b id='mean'></b></div>
    <div class='stat'><span class='muted'>crowd median</span><b id='median'></b></div>
  </div>
  <canvas id='hist' height='110'></canvas>
  <div>
    <button onclick="act('reveal')">🎉 Reveal the answer</button>
    <button onclick="act('next')">Next question →</button>
    <button onclick="act('back')" style='background:#67737f'>← Back</button>
  </div>
</div>
"""

_WISDOM_TEACHER_JS = """
const KEY = new URLSearchParams(location.search).get('key') || '';
let chart = null, lastPhase = '', lastIdx = -1;
function fmt(x){ return Number(x).toLocaleString(undefined, {maximumFractionDigits: 1}); }
async function poll(){
  const s = await fetch('state').then(r=>r.json()).catch(()=>null);
  if(!s) return;
  document.getElementById('qnum').textContent = 'Question ' + (s.idx+1) + ' of ' + s.n_questions;
  document.getElementById('prompt').textContent = s.prompt;
  document.getElementById('count').textContent = s.n_subs;
  if(s.phase === 'revealed' && (lastPhase !== 'revealed' || lastIdx !== s.idx)) showReveal(s.reveal);
  if(s.phase === 'open'){ document.getElementById('stats').style.display='none';
                          if(chart){ chart.destroy(); chart = null; } }
  lastPhase = s.phase; lastIdx = s.idx;
}
function showReveal(r){
  document.getElementById('stats').style.display = 'block';
  document.getElementById('truth').textContent = fmt(r.answer) + ' ' + r.unit;
  document.getElementById('mean').textContent = r.n ? fmt(r.mean) : 'no submissions';
  document.getElementById('median').textContent = r.n ? fmt(r.median) : '—';
  if(chart) chart.destroy();
  if(!r.n) return;
  const colors = r.counts.map((_,i)=> i === r.truth_bin ? '#34a853' : '#7ba3d6');
  chart = new Chart(document.getElementById('hist'), {type:'bar',
    data:{labels:r.labels, datasets:[{data:r.counts, backgroundColor:colors,
          label:'guesses (green bar holds the truth)'}]},
    options:{plugins:{legend:{labels:{color:'#e8edf3'}}},
      scales:{y:{ticks:{color:'#93a0ad',precision:0},grid:{color:'#2a3340'}},
              x:{ticks:{color:'#93a0ad',maxRotation:60,minRotation:45},grid:{display:false}}}}});
}
async function act(a){
  await fetch('action', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({key: KEY, action: a})});
  poll();
}
poll(); setInterval(poll, 1500);
"""


def launch_wisdom(questions=None, teacher_key=None, port=8770, tunnel=True):
    """Host the Wisdom of the Crowd page. Returns (student_url, teacher_url)."""
    questions = [tuple(q) for q in (questions or DEFAULT_QUESTIONS)]
    teacher_key = teacher_key or secrets.token_hex(4)
    state = _WisdomState(questions)
    app = Flask("wisdom")

    @app.get("/")
    def student():
        return Response(_page("Wisdom of the Crowd", "", _WISDOM_STUDENT,
                              _WISDOM_STUDENT_JS), mimetype="text/html")

    @app.get("/teacher")
    def teacher():
        if request.args.get("key") != teacher_key:
            return Response("wrong key", status=403)
        return Response(_page("Teacher console", _DARK_CSS, _WISDOM_TEACHER,
                              _WISDOM_TEACHER_JS), mimetype="text/html")

    @app.get("/state")
    def get_state():
        with state.lock:
            prompt, _, unit = state.questions[state.idx]
            payload = {"idx": state.idx, "n_questions": len(state.questions),
                       "prompt": prompt, "unit": unit, "phase": state.phase,
                       "n_subs": len(state.subs[state.idx])}
            if state.phase == "revealed":
                payload["reveal"] = state.reveal_payload()
        return jsonify(payload)

    @app.post("/submit")
    def submit():
        try:
            value = float(request.get_json(force=True)["value"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "send a number"}), 400
        if not np.isfinite(value):
            return jsonify({"error": "send a finite number"}), 400
        with state.lock:
            if state.phase != "open":
                return jsonify({"error": "submissions are closed"}), 409
            state.subs[state.idx].append(value)
        return jsonify({"ok": True})

    @app.post("/action")
    def action():
        body = request.get_json(force=True)
        if body.get("key") != teacher_key:
            return jsonify({"error": "wrong key"}), 403
        with state.lock:
            act = body.get("action")
            if act == "reveal":
                state.phase = "revealed"
            elif act == "next" and state.idx < len(state.questions) - 1:
                state.idx += 1
                state.phase = "open"
            elif act == "back" and state.idx > 0:
                state.idx -= 1
                state.phase = "revealed" if state.subs[state.idx] else "open"
        return jsonify({"ok": True})

    base = _serve("wisdom", app, port, tunnel)
    student_url = f"{base}/"
    teacher_url = f"{base}/teacher?key={teacher_key}"
    _announce("🔮 Activity 1 — Wisdom of the Crowd", student_url, teacher_url)
    return student_url, teacher_url


# ============================================================================
# ACTIVITY 2 — the Strategy Arena
# ============================================================================

class Strategy:
    """Same interface as the lab notebook: see only the past, return weights."""

    name = "unnamed strategy"

    def allocate(self, past_returns):
        raise NotImplementedError


DOW30 = ["AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX",
         "DIS", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM",
         "MRK", "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT"]


def _sp500_tickers():
    import requests
    html = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0 (classroom notebook)"}, timeout=30).text
    table = pd.read_html(io.StringIO(html))[0]
    return sorted(table["Symbol"].str.replace(".", "-", regex=False))


def _resolve_tickers(universe, custom_tickers):
    if universe.lower().startswith("dow"):
        return DOW30
    if universe.lower().startswith("custom"):
        tickers = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
        if len(tickers) < 2:
            raise ValueError("give at least two tickers in custom_tickers")
        return tickers
    return _sp500_tickers()


def _load_window(tickers, start_ts, end_ts, warmup):
    """Open->close returns for [start_ts, end_ts) plus ~warmup days of run-in.

    Returns (rets, t0): rets includes the warmup run-in; t0 is the first row index
    on/after start_ts (i.e. the first tradable day of the window).
    """
    import yfinance as yf
    pad_start = start_ts - pd.Timedelta(days=int(warmup * 1.6) + 12)
    print(f"downloading {len(tickers)} tickers, {pad_start.date()} -> {end_ts.date()}...")
    raw = yf.download(tickers, start=str(pad_start.date()), end=str(end_ts.date()),
                      auto_adjust=True, progress=False)
    rets = (raw["Close"] / raw["Open"] - 1)
    rets = rets.dropna(axis=1, thresh=int(0.95 * len(rets))).fillna(0.0)
    t0 = int((rets.index < start_ts).sum())
    if len(rets) - t0 < 5:
        raise ValueError(f"window {start_ts.date()}..{end_ts.date()} has <5 trading days")
    return rets, t0


# ---------------------------------------------------------------------------
# LLM code generation (OpenAI chat completions, no SDK needed)
# ---------------------------------------------------------------------------

CODEGEN_SYSTEM = """You write trading strategies for a classroom competition.

The environment defines this base class:

    class Strategy:
        name = "unnamed strategy"
        def allocate(self, past_returns):
            ...

allocate() is called once per simulated trading day. `past_returns` is a pandas
DataFrame (rows = past trading days in chronological order, columns = stock tickers)
holding each stock's daily open-to-close return for every day BEFORE today. It must
return a pandas Series mapping tickers to weights: positive = long, negative = short.
If the absolute weights sum to more than 1 they get rescaled to 1 (the player bets $1
per day total); unallocated money sits in cash. A fee of 0.05% is charged on every
dollar of position changed between days, and the winner has the highest ROI.

Hard rules for the code you write:
- Use ONLY the `past_returns` argument plus numpy (as np) and pandas (as pd) — no
  other imports, no network, no files, no randomness.
- Keep allocate() fast: a few vectorized pandas operations.
- Set a short `name` class attribute.

Implement the user's strategy idea as a single Strategy subclass. Reply with ONLY the
Python code — no backticks, no explanations, no example usage."""


def _strip_fences(text):
    text = text.strip()
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return m.group(1).strip() if m else text


def _generate_code(prompt_text, api_key, model):
    """One OpenAI chat-completions call; returns generated code or raises."""
    body = json.dumps({"model": model, "messages": [
        {"role": "system", "content": CODEGEN_SYSTEM},
        {"role": "user", "content": f"My strategy idea: {prompt_text}"},
    ]}).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            out = json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        raise RuntimeError(f"LLM call failed (HTTP {e.code}): {detail}") from e
    return _strip_fences(out["choices"][0]["message"]["content"])


def _replay_in_child(code, which, conn):
    """Run one student strategy on the named window (market is fork-shared)."""
    try:
        returns, t0, t1 = _MARKET[which]
        fee = _MARKET["fee"]
        ns = {"np": np, "pd": pd, "numpy": np, "pandas": pd, "Strategy": Strategy,
              "run_competition": lambda *a, **k: None}
        exec(code, ns)
        strat_cls, strat = None, None
        for v in ns.values():
            if isinstance(v, type) and issubclass(v, Strategy) and v is not Strategy:
                strat_cls = v
            elif isinstance(v, Strategy):
                strat = v
        if strat_cls is not None:
            strat = strat_cls()
        if strat is None:
            raise ValueError("no Strategy subclass found in the submitted code")
        R = returns.to_numpy()
        w_prev = pd.Series(0.0, index=returns.columns)
        pnl = []
        for t in range(t0, t1):
            w = pd.Series(strat.allocate(returns.iloc[:t]), dtype=float)
            w = w.reindex(returns.columns).fillna(0.0)
            gross = float(w.abs().sum())
            if gross > 1:
                w = w / gross
            turnover = float((w - w_prev).abs().sum())
            pnl.append(float(w.to_numpy() @ R[t]) - fee * turnover)
            w_prev = w
        conn.send({"pnl": pnl})
    except Exception as e:                                   # noqa: BLE001
        conn.send({"error": f"{type(e).__name__}: {e}"})
    finally:
        conn.close()


_MARKET = None         # {"practice"/"competition": (rets, t0, t1), "fee": f} — set pre-fork


def _sharpe(pnl):
    pnl = np.asarray(pnl, dtype=float)
    if len(pnl) < 2 or pnl.std() == 0:
        return float("nan")
    return float(pnl.mean() / pnl.std() * np.sqrt(TRADING_DAYS))


_ARENA_STUDENT = """
<div class='card'>
  <h1>🏆 The Strategy Arena</h1>
  <div id='openview'>
    <input id='name' placeholder='your name / team name' maxlength='24'>
    <div id='llmbox'>
      <p><b>Describe your trading strategy</b> in plain English. An LLM turns it into
      code and test-drives it on a <i>practice window</i> — the real competition runs
      on dates you haven't seen scored yet.</p>
      <textarea id='prompt' rows='4' placeholder="e.g. Buy the 5 stocks that fell the most last week — they usually bounce back. Put a little extra weight on the biggest loser."></textarea>
      <button id='genbtn' onclick='generate()'>✨ Generate &amp; test-drive</button>
    </div>
    <details style='margin-top:8px'><summary class='muted'>…or paste Strategy code directly</summary>
      <textarea id='code' rows='10' placeholder='class MyStrategy(Strategy): ...'></textarea>
      <button onclick='tryout()'>🧪 Test-drive my code</button>
    </details>
    <div id='msg'></div>
    <div id='preview' style='display:none'>
      <h2>Practice run <span id='pstats' class='muted'></span></h2>
      <canvas id='pchart' height='160'></canvas>
      <details><summary class='muted'>the generated code</summary><pre id='pcode' style='font-size:.75em;overflow-x:auto'></pre></details>
      <button id='subbtn' onclick='submitEntry()'>🚀 Submit this strategy to the competition</button>
      <div id='submsg'></div>
    </div>
  </div>
  <div id='waitview' style='display:none'>
    <h2>🔒 Submissions are locked.</h2>
    <p class='muted'>Watch the big screen — the market is about to run.</p>
  </div>
  <div id='doneview' style='display:none'>
    <h2>Results are in!</h2>
    <div id='placement' class='big'></div>
    <div id='detail' class='muted'></div>
  </div>
</div>
"""

_ARENA_STUDENT_JS = """
let chart = null, busy = false;
function token(){ return localStorage.getItem('arena_token') || ''; }
function msg(id, html){ document.getElementById(id).innerHTML = html; }
async function callTest(endpoint, payload){
  if(busy) return; busy = true;
  const name = document.getElementById('name').value.trim();
  if(!name){ msg('msg', "<span class='bad'>enter a name first</span>"); busy = false; return; }
  msg('msg', "⏳ working — generating and replaying your strategy (10–30 s)…");
  document.getElementById('genbtn').disabled = true;
  try{
    const res = await fetch(endpoint, {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(Object.assign({name:name, token: token()}, payload))});
    const j = await res.json();
    if(!res.ok){ msg('msg', "<span class='bad'>" + (j.error||'error') + "</span>"); return; }
    localStorage.setItem('arena_token', j.token);
    if(j.run_error){
      msg('msg', "<span class='bad'>💥 the strategy crashed in testing: " + j.run_error + "</span>");
      document.getElementById('pcode').textContent = j.code || '';
      return;
    }
    msg('msg', "<span class='ok'>✔ test-drive complete</span>");
    showPreview(j);
  } finally { busy = false; document.getElementById('genbtn').disabled = false; }
}
function generate(){
  const p = document.getElementById('prompt').value.trim();
  if(!p){ msg('msg', "<span class='bad'>describe a strategy first</span>"); return; }
  callTest('generate', {prompt: p});
}
function tryout(){
  const c = document.getElementById('code').value;
  if(!c.trim()){ msg('msg', "<span class='bad'>paste some code first</span>"); return; }
  callTest('tryout', {code: c});
}
function showPreview(j){
  document.getElementById('preview').style.display = 'block';
  document.getElementById('pcode').textContent = j.code;
  document.getElementById('pstats').textContent =
    '— ROI ' + (100*j.preview.roi).toFixed(2) + '% · Sharpe ' + j.preview.sharpe.toFixed(2);
  msg('submsg', '');
  if(chart) chart.destroy();
  chart = new Chart(document.getElementById('pchart'), {type:'line',
    data:{labels:j.preview.dates, datasets:[{label:'your strategy (practice window)',
      data:j.preview.cum, borderColor:'#4a6fa5', borderWidth:2, pointRadius:0}]},
    options:{plugins:{legend:{display:true}},
      scales:{x:{ticks:{maxTicksLimit:8}}, y:{title:{display:true,text:'profit ($1/day)'}}}}});
}
async function submitEntry(){
  const res = await fetch('submit', {method:'POST',
    headers:{'Content-Type':'application/json'}, body:JSON.stringify({token: token()})});
  const j = await res.json();
  if(res.ok) msg('submsg', "<span class='ok'>✔ in the competition as <b>" + j.name +
    "</b> — generate again and resubmit any time before lock</span>");
  else msg('submsg', "<span class='bad'>" + (j.error||'error') + "</span>");
}
async function poll(){
  const s = await fetch('state').then(r=>r.json()).catch(()=>null);
  if(!s) return;
  if(!s.llm_enabled) document.getElementById('llmbox').style.display = 'none';
  const show = id => { for(const v of ['openview','waitview','doneview'])
      document.getElementById(v).style.display = (v===id) ? 'block':'none'; };
  if(s.phase === 'open') show('openview');
  else if(s.phase === 'done'){ show('doneview'); loadPlacement(); }
  else show('waitview');
}
let placed = false;
async function loadPlacement(){
  if(placed || !token()) return;
  const r = await fetch('me/' + token()).then(r=>r.json()).catch(()=>null);
  if(!r) return;
  placed = true;
  const p = document.getElementById('placement');
  if(r.error_msg){
    p.innerHTML = "💥 <b>" + r.name + "</b>: your strategy crashed";
    document.getElementById('detail').textContent = r.error_msg;
  } else {
    p.innerHTML = "<b>" + r.name + "</b> placed <b>#" + r.rank + "</b> of " + r.n;
    document.getElementById('detail').textContent =
      'ROI ' + (r.roi*100).toFixed(2) + '%  ·  Sharpe ' + r.sharpe.toFixed(2) +
      '  (only you can see this)';
  }
}
poll(); setInterval(poll, 2500);
"""

_ARENA_TEACHER = """
<div class='card'>
  <h1>🏆 The Strategy Arena — teacher console</h1>
  <div class='stat'><span class='muted'>entries</span><b id='count'>0</b></div>
  <div class='stat'><span class='muted'>status</span><b id='phase'>open</b></div>
  <div>
    <button onclick="act('lock')">🔒 Lock submissions</button>
    <button onclick="act('run')" id='runbtn'>🎬 RUN THE MARKET</button>
    <button onclick="act('reopen')" style='background:#67737f'>↺ Reopen</button>
  </div>
  <div id='entrants' class='muted' style='margin-top:8px'></div>
</div>
<div class='card'>
  <canvas id='race' height='130'></canvas>
  <div id='podium' style='display:none;text-align:center'></div>
  <details style='margin-top:14px'><summary class='muted'>full results (don't project this)</summary>
    <div id='full'></div></details>
</div>
"""

_ARENA_TEACHER_JS = """
const KEY = new URLSearchParams(location.search).get('key') || '';
let chart = null, animating = false, shown = false;
const PALETTE = ['#7ba3d6','#e8a87c','#85cdca','#e27d60','#c38d9e','#a8d672',
                 '#f4d35e','#9fa8da','#80cbc4','#ef9a9a','#ce93d8','#ffcc80'];
async function poll(){
  const s = await fetch('state').then(r=>r.json()).catch(()=>null);
  if(!s) return;
  document.getElementById('count').textContent = s.n_subs;
  document.getElementById('phase').textContent = s.phase;
  document.getElementById('entrants').textContent =
      s.names.length ? 'in the arena: ' + s.names.join(' · ') : '';
  document.getElementById('runbtn').disabled = (s.phase === 'running');
  if(s.phase === 'done' && !shown){ shown = true; revealRace(); }
  if(s.phase === 'open'){ shown = false; }
}
async function act(a){
  await fetch('action', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({key: KEY, action: a})});
  poll();
}
async function revealRace(){
  const r = await fetch('results?key=' + KEY).then(r=>r.json());
  if(chart) chart.destroy();
  document.getElementById('podium').style.display = 'none';
  const sets = r.series.map((s,i) => ({label: s.label, data: [],
      borderColor: PALETTE[i % PALETTE.length], borderWidth: 2.2, pointRadius: 0,
      borderDash: s.label.includes('baseline') ? [6,4] : []}));
  chart = new Chart(document.getElementById('race'), {type:'line',
    data:{labels: [], datasets: sets},
    options:{animation:false, plugins:{legend:{display:false}},
      scales:{y:{title:{display:true,text:'profit ($1/day)',color:'#93a0ad'},
                 ticks:{color:'#93a0ad'}, grid:{color:'#2a3340'}},
              x:{ticks:{color:'#93a0ad', maxTicksLimit:10}, grid:{display:false}}}}});
  // the dramatic part: extend every line across the chart together
  let t = 0; animating = true;
  const total = r.dates.length;
  const stepMs = Math.max(25, Math.min(120, 12000/total));
  const timer = setInterval(() => {
    if(t >= total){ clearInterval(timer); animating = false; podium(r); return; }
    chart.data.labels.push(r.dates[t]);
    r.series.forEach((s,i)=> chart.data.datasets[i].data.push(s.cum[t]));
    chart.update('none'); t++;
  }, stepMs);
}
function podium(r){
  const medals = ['🥇','🥈','🥉'];
  const rows = r.top3.map((s,i)=>
    "<div style='font-size:" + (1.9-0.3*i) + "em;margin:6px'>" + medals[i] + " <b>" +
    s.name + "</b> &nbsp; ROI " + (s.roi*100).toFixed(2) + "%</div>").join('');
  const pod = document.getElementById('podium');
  pod.innerHTML = "<h2>🏁 Final standings</h2>" + rows +
      "<div class='muted'>" + (r.window_note || '') + "</div>" +
      "<div class='muted'>everyone else: check your own phone for your private placement</div>";
  pod.style.display = 'block';
  document.getElementById('full').innerHTML = '<pre>' + r.full_table + '</pre>';
}
poll(); setInterval(poll, 2000);
"""


def launch_arena(universe="S&P 500", custom_tickers="",
                 random_years=True, year_min=2016, year_max=None,
                 replay_start="2026-01-02", replay_end="2026-06-10",
                 warmup=20, fee=0.0005,
                 teacher_key=None, port=8771, tunnel=True, strategy_timeout=60,
                 openai_api_key="", llm_model="gpt-5-mini", practice_days=60):
    """Host the Strategy Arena. Returns (student_url, teacher_url).

    Default mode (random_years=True): students test-drive on one random calendar
    year >= year_min, and the competition replays a *different* random year — so
    nobody can overfit the scoring window. Manual mode (random_years=False) uses
    [replay_start, replay_end] for the competition and the practice_days before it.
    """
    global _MARKET
    teacher_key = teacher_key or secrets.token_hex(4)
    tickers = _resolve_tickers(universe, custom_tickers)
    if random_years:
        year_max = year_max or pd.Timestamp.now().year - 1
        years = list(range(year_min, year_max + 1))
        prac_year = years[secrets.randbelow(len(years))]
        comp_year = prac_year
        while comp_year == prac_year and len(years) > 1:
            comp_year = years[secrets.randbelow(len(years))]
        print(f"🎲 practice year: {prac_year} | competition year: {comp_year}")
        prac_rets, p0 = _load_window(tickers, pd.Timestamp(f"{prac_year}-01-01"),
                                     pd.Timestamp(f"{prac_year + 1}-01-01"), warmup)
        comp_rets, c0 = _load_window(tickers, pd.Timestamp(f"{comp_year}-01-01"),
                                     pd.Timestamp(f"{comp_year + 1}-01-01"), warmup)
        _MARKET = {"fee": fee,
                   "practice": (prac_rets, p0, len(prac_rets)),
                   "competition": (comp_rets, c0, len(comp_rets))}
        window_note = (f"practice window: {prac_year} · "
                       f"competition window: {comp_year}")
    else:
        end_ts = pd.Timestamp(replay_end)
        start_ts = pd.Timestamp(replay_start)
        prac_start = start_ts - pd.Timedelta(days=int(practice_days * 1.6) + 12)
        full, p0 = _load_window(tickers, prac_start, end_ts, warmup)
        c0 = int((full.index < start_ts).sum())
        if len(full) - c0 < 5:
            raise ValueError("replay window has fewer than 5 trading days")
        _MARKET = {"fee": fee,
                   "practice": (full, p0, c0),
                   "competition": (full, c0, len(full))}
        window_note = f"competition window: {replay_start} -> {replay_end}"
    comp_rets_, comp_t0_, _ = _MARKET["competition"]
    prac_rets_, prac_t0_, prac_t1_ = _MARKET["practice"]
    replay_dates = [d.strftime("%Y-%m-%d") for d in comp_rets_.index[comp_t0_:]]
    practice_dates = [d.strftime("%Y-%m-%d")
                      for d in prac_rets_.index[prac_t0_:prac_t1_]]
    print(f"arena ready: {comp_rets_.shape[1]} stocks | "
          f"{len(practice_dates)} practice days, {len(replay_dates)} competition days")

    state = {"phase": "open", "players": {}, "results": None}
    lock = threading.Lock()
    app = Flask("arena")

    def _finite(x, fallback=0.0):
        return float(x) if np.isfinite(x) else fallback

    def _replay(code, which):
        ctx = multiprocessing.get_context("fork")
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=_replay_in_child, args=(code, which, child))
        proc.start()
        child.close()
        out = {"error": f"timed out ({strategy_timeout}s limit)"}
        if parent.poll(strategy_timeout):
            out = parent.recv()
        proc.join(1)
        if proc.is_alive():
            proc.terminate()
        return out

    def _test_drive(token_, name, code_text):
        """Replay on the practice window only; the competition window stays unseen."""
        out = _replay(code_text, "practice")
        if "error" in out:
            return {"token": token_, "name": name, "code": code_text,
                    "run_error": out["error"]}
        pnl = np.asarray(out["pnl"], dtype=float)
        return {"token": token_, "name": name, "code": code_text, "preview": {
            "dates": practice_dates,
            "cum": np.cumsum(pnl).round(5).tolist(),
            "roi": _finite(pnl.sum() / max(len(pnl), 1)),
            "sharpe": _finite(_sharpe(pnl)),
        }}

    def _register(body):
        """Resolve (token, player), creating and de-duplicating names. Lock held."""
        token_ = str(body.get("token") or "")
        name = str(body.get("name", "")).strip()[:24]
        if token_ not in state["players"]:
            token_ = secrets.token_hex(8)
            state["players"][token_] = {"name": name or "anon", "code": None,
                                        "entry_code": None, "last_try": 0.0}
        player = state["players"][token_]
        if name and name != player["name"]:
            taken = {p["name"] for t, p in state["players"].items() if t != token_}
            base, k = name, 2
            while name in taken:
                name = f"{base} ({k})"
                k += 1
            player["name"] = name
        return token_, player

    def _gate(body, cooldown):
        """Common open-phase + rate-limit gate. Returns (error, token, name)."""
        with lock:
            if state["phase"] != "open":
                return ({"error": "submissions are locked"}, 409), None, None
            token_, player = _register(body)
            now = time.time()
            if now - player["last_try"] < cooldown:
                return ({"error": "easy there — wait a few seconds and try again"},
                        429), None, None
            player["last_try"] = now
            return None, token_, player["name"]

    def _run_all():
        with lock:
            entries = [(tok, p["name"], p["entry_code"])
                       for tok, p in state["players"].items() if p["entry_code"]]
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_replay, code_text, "competition"): (tok, name)
                       for tok, name, code_text in entries}
            for fut in concurrent.futures.as_completed(futures):
                tok, name = futures[fut]
                results[tok] = {"name": name, **fut.result()}
        R = comp_rets_.to_numpy()[comp_t0_:]
        results["__baseline__"] = {"name": "1/N baseline",
                                   "pnl": R.mean(axis=1).tolist()}
        n_days = len(replay_dates)
        ranked = []
        for tok, r in results.items():
            if "pnl" in r:
                pnl = np.asarray(r["pnl"], dtype=float)
                r["roi"] = _finite(pnl.sum() / n_days)
                r["sharpe"] = _finite(_sharpe(pnl))
                r["cum"] = np.cumsum(pnl).round(5).tolist()
                if tok != "__baseline__":
                    ranked.append(tok)
        ranked.sort(key=lambda tok: results[tok]["roi"], reverse=True)
        for i, tok in enumerate(ranked):
            results[tok]["rank"] = i + 1
        with lock:
            state["results"] = {"by_token": results, "ranked": ranked}
            state["phase"] = "done"

    @app.get("/")
    def student():
        return Response(_page("Strategy Arena", "", _ARENA_STUDENT,
                              _ARENA_STUDENT_JS), mimetype="text/html")

    @app.get("/teacher")
    def teacher():
        if request.args.get("key") != teacher_key:
            return Response("wrong key", status=403)
        return Response(_page("Arena console", _DARK_CSS, _ARENA_TEACHER,
                              _ARENA_TEACHER_JS), mimetype="text/html")

    @app.get("/state")
    def get_state():
        with lock:
            entered = [p["name"] for p in state["players"].values() if p["entry_code"]]
            return jsonify({"phase": state["phase"], "n_subs": len(entered),
                            "names": entered, "llm_enabled": bool(openai_api_key)})

    @app.post("/generate")
    def generate():
        if not openai_api_key:
            return jsonify({"error": "AI generation is not enabled — paste code instead"}), 400
        body = request.get_json(force=True)
        prompt_text = str(body.get("prompt", "")).strip()
        if not prompt_text or len(prompt_text) > 2000:
            return jsonify({"error": "describe a strategy (under 2000 characters)"}), 400
        if not str(body.get("name", "")).strip():
            return jsonify({"error": "enter a name first"}), 400
        gate_err, token_, name = _gate(body, cooldown=8)
        if gate_err:
            payload, status = gate_err
            return jsonify(payload), status
        try:
            code_text = _generate_code(prompt_text, openai_api_key, llm_model)
        except Exception as e:                               # noqa: BLE001
            return jsonify({"error": str(e)}), 502
        result = _test_drive(token_, name, code_text)
        with lock:
            state["players"][token_]["code"] = code_text
        return jsonify(result)

    @app.post("/tryout")
    def tryout():
        body = request.get_json(force=True)
        code_text = str(body.get("code", ""))
        if not code_text.strip() or len(code_text) > 20000:
            return jsonify({"error": "paste some code (under 20k characters)"}), 400
        if not str(body.get("name", "")).strip():
            return jsonify({"error": "enter a name first"}), 400
        gate_err, token_, name = _gate(body, cooldown=3)
        if gate_err:
            payload, status = gate_err
            return jsonify(payload), status
        result = _test_drive(token_, name, code_text)
        with lock:
            state["players"][token_]["code"] = code_text
        return jsonify(result)

    @app.post("/submit")
    def submit():
        body = request.get_json(force=True)
        token_ = str(body.get("token") or "")
        with lock:
            if state["phase"] != "open":
                return jsonify({"error": "submissions are locked"}), 409
            player = state["players"].get(token_)
            if player is None or not player["code"]:
                return jsonify({"error": "test-drive a strategy first"}), 400
            player["entry_code"] = player["code"]
            name = player["name"]
        return jsonify({"token": token_, "name": name})

    @app.post("/action")
    def action():
        body = request.get_json(force=True)
        if body.get("key") != teacher_key:
            return jsonify({"error": "wrong key"}), 403
        with lock:
            act = body.get("action")
            if act == "lock" and state["phase"] == "open":
                state["phase"] = "locked"
            elif act == "reopen":
                state["phase"] = "open"
                state["results"] = None
            elif act == "run" and state["phase"] in ("open", "locked"):
                state["phase"] = "running"
                threading.Thread(target=_run_all, daemon=True).start()
        return jsonify({"ok": True, "phase": state["phase"]})

    @app.get("/results")
    def results_route():
        if request.args.get("key") != teacher_key:
            return jsonify({"error": "wrong key"}), 403
        with lock:
            res = state["results"]
        if res is None:
            return jsonify({"error": "not finished"}), 409
        by_tok, ranked = res["by_token"], res["ranked"]
        series = []
        for tok, r in by_tok.items():
            if "cum" not in r:
                continue
            rank = r.get("rank")
            label = r["name"] if (tok == "__baseline__" or (rank and rank <= 3)) \
                else f"contender {rank}"
            series.append({"label": label, "cum": r["cum"]})
        top3 = [{"name": by_tok[t]["name"], "roi": by_tok[t]["roi"]}
                for t in ranked[:3]]
        lines = [f"{'rank':>4}  {'name':<26} {'ROI %':>8} {'Sharpe':>7}"]
        for t in ranked:
            r = by_tok[t]
            lines.append(f"{r['rank']:>4}  {r['name']:<26} "
                         f"{100 * r['roi']:>8.2f} {r['sharpe']:>7.2f}")
        for t, r in by_tok.items():
            if "error" in r and t != "__baseline__":
                lines.append(f"  💥  {r['name']:<26} {r['error']}")
        return jsonify({"dates": replay_dates, "series": series, "top3": top3,
                        "window_note": window_note,
                        "full_table": "\n".join(lines)})

    @app.get("/me/<token>")
    def me(token):
        with lock:
            res = state["results"]
        if res is None or token not in res["by_token"]:
            return jsonify({"error": "no result"}), 404
        r = res["by_token"][token]
        return jsonify({"name": r["name"], "rank": r.get("rank"),
                        "n": len(res["ranked"]), "roi": r.get("roi"),
                        "sharpe": r.get("sharpe"), "error_msg": r.get("error")})

    base = _serve("arena", app, port, tunnel)
    student_url = f"{base}/"
    teacher_url = f"{base}/teacher?key={teacher_key}"
    _announce("🏆 Activity 2 — The Strategy Arena", student_url, teacher_url)
    return student_url, teacher_url


# ============================================================================
# ACTIVITY 3 — the lecture dashboard
# ============================================================================

def _dashboard_compute(tickers, start, cutoff):
    """Download data and precompute every panel's series (JSON-safe floats)."""
    import warnings as _warnings
    import yfinance as yf
    from statsmodels.tsa.arima.model import ARIMA
    names = [t.strip().upper() for t in str(tickers).split(",") if t.strip()]
    if not names:
        raise ValueError("give at least one ticker")
    print(f"dashboard: downloading {names} from {start}...")
    raw = yf.download(names, start=str(start), auto_adjust=True, progress=False)
    opens, closes = raw["Open"].dropna(), raw["Close"].dropna()
    opens, closes = opens.align(closes, join="inner", axis=0)
    if len(closes) < 80:
        raise ValueError("fewer than 80 shared trading days — check the tickers")

    dates = [d.strftime("%Y-%m-%d") for d in closes.index]
    growth = {t: (closes[t] / closes[t].iloc[0]).round(4).tolist()
              for t in closes.columns}
    day_ret = closes / opens - 1
    daycum = {t: day_ret[t].cumsum().round(4).tolist() for t in day_ret.columns}

    R = day_ret.to_numpy()
    n = len(R)
    ftl = np.zeros(n)
    if R.shape[1] > 1:
        ftl[1:] = R[np.arange(1, n), R[:-1].argmax(axis=1)]
    arena = {
        "1/N uniform": np.cumsum(R.mean(axis=1)).round(4).tolist(),
        "FTL (yesterday's best)": np.cumsum(ftl).round(4).tolist(),
        "oracle (sees today)": np.cumsum(R.max(axis=1)).round(4).tolist(),
    }

    cutoff_ts = pd.Timestamp(cutoff)
    forecast = {}
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        for t in closes.columns:
            price = closes[t]
            train = price[price.index <= cutoff_ts]
            test = price[price.index > cutoff_ts]
            if len(train) < 100 or len(test) < 5:
                continue
            res = ARIMA(train.to_numpy(), order=(1, 1, 1)).fit()
            fc = res.get_forecast(len(test))
            mean = np.asarray(fc.predicted_mean)
            ci = np.asarray(fc.conf_int())
            tail = int(min(250, len(train)))
            forecast[t] = {
                "labels": ([d.strftime("%Y-%m-%d") for d in train.index[-tail:]]
                           + [d.strftime("%Y-%m-%d") for d in test.index]),
                "n_train": tail,
                "train": train.iloc[-tail:].round(2).tolist(),
                "actual": test.round(2).tolist(),
                "mean": mean.round(2).tolist(),
                "lo": ci[:, 0].round(2).tolist(),
                "hi": ci[:, 1].round(2).tolist(),
            }
    print("dashboard data ready")
    return {"tickers": ", ".join(closes.columns), "start": str(start),
            "cutoff": str(cutoff), "dates": dates, "growth": growth,
            "daycum": daycum, "arena": arena, "forecast": forecast}


_DASH_HTML = """
<div class='card'>
  <h1>📊 Lecture Dashboard</h1>
  <div id='controls'>
    <input id='tickers' style='max-width:360px;display:inline-block'
           placeholder='^GSPC, BTC-USD, GME'>
    <input id='cutoff' type='date' style='max-width:180px;display:inline-block'>
    <button id='apply' onclick='applySettings()'>Apply &amp; rebuild</button>
    <span id='cmsg' class='muted'></span>
  </div>
</div>
<div class='card'><h2>1 — Growth of $1, buy &amp; hold <span class='muted'>(log scale)</span></h2>
  <button onclick="play('growth')">▶ Play</button>
  <button onclick="finish('growth')" style='background:#67737f'>⏭ End</button>
  <canvas id='growth' height='110'></canvas></div>
<div class='card'><h2>2 — $1 every day: in at the open, out at the close</h2>
  <button onclick="play('daycum')">▶ Play</button>
  <button onclick="finish('daycum')" style='background:#67737f'>⏭ End</button>
  <canvas id='daycum' height='110'></canvas></div>
<div class='card'><h2>3 — Online learning: uniform vs. FTL vs. the oracle</h2>
  <button onclick="play('arena')">▶ Play</button>
  <button onclick="finish('arena')" style='background:#67737f'>⏭ End</button>
  <canvas id='arena' height='110'></canvas></div>
<div class='card'><h2>4 — Forecasting past the cutoff (ARIMA)</h2>
  <select id='fticker' onchange='buildForecast()'
          style='padding:8px;border-radius:8px;background:#1a212c;color:#e8edf3'></select>
  <button onclick="play('forecast')">▶ Play</button>
  <button onclick="finish('forecast')" style='background:#67737f'>⏭ End</button>
  <canvas id='forecast' height='110'></canvas></div>
"""

_DASH_JS = """
const KEY = new URLSearchParams(location.search).get('key') || '';
const PALETTE = ['#7ba3d6','#e8a87c','#85cdca','#e27d60','#c38d9e','#a8d672',
                 '#f4d35e','#9fa8da','#80cbc4','#ef9a9a'];
let charts = {}, fulls = {}, timers = {}, DATA = null;
function mk(id, labels, datasets, yextra){
  if(charts[id]) charts[id].destroy();
  fulls[id] = {labels: labels, data: datasets.map(d => d.data)};
  const sets = datasets.map(d => Object.assign({}, d, {data: []}));
  charts[id] = new Chart(document.getElementById(id), {type:'line',
    data:{labels:[], datasets:sets},
    options:{animation:false, spanGaps:false,
      plugins:{legend:{labels:{color:'#e8edf3',
               filter:(it)=>!it.text.startsWith('_')}}},
      scales:{x:{ticks:{color:'#93a0ad', maxTicksLimit:10}, grid:{display:false}},
              y:Object.assign({ticks:{color:'#93a0ad'}, grid:{color:'#2a3340'}},
                              yextra || {})}}});
  finish(id);
}
function stopTimer(id){ if(timers[id]){ clearInterval(timers[id]); timers[id] = null; } }
function finish(id){ stopTimer(id);
  const c = charts[id], f = fulls[id];
  c.data.labels = f.labels.slice();
  c.data.datasets.forEach((d,i)=> d.data = f.data[i].slice());
  c.update('none');
}
function play(id){ stopTimer(id);
  const c = charts[id], f = fulls[id];
  c.data.labels = []; c.data.datasets.forEach(d => d.data = []); c.update('none');
  let t = 0; const total = f.labels.length;
  const step = Math.max(8, Math.min(80, 9000/total));
  timers[id] = setInterval(() => {
    if(t >= total){ stopTimer(id); return; }
    c.data.labels.push(f.labels[t]);
    c.data.datasets.forEach((d,i)=> d.data.push(f.data[i][t]));
    c.update('none'); t++;
  }, step);
}
function series(dict){
  return Object.keys(dict).map((t,i)=>({label:t, data:dict[t],
    borderColor:PALETTE[i % PALETTE.length], borderWidth:2.2, pointRadius:0}));
}
function buildForecast(){
  const t = document.getElementById('fticker').value;
  if(!t || !DATA.forecast[t]) return;
  const f = DATA.forecast[t], nt = f.n_train, nT = f.actual.length;
  const pad = arr => Array(nt).fill(null).concat(arr);
  mk('forecast', f.labels, [
    {label:'train (actual)', data: f.train.concat(Array(nT).fill(null)),
     borderColor:'#e8edf3', borderWidth:1.4, pointRadius:0},
    {label:'reality', data: pad(f.actual), borderColor:'#93a0ad', borderWidth:2, pointRadius:0},
    {label:'ARIMA(1,1,1) forecast', data: pad(f.mean), borderColor:'#7ba3d6',
     borderWidth:2.6, pointRadius:0},
    {label:'95% CI', data: pad(f.hi), borderColor:'rgba(0,0,0,0)',
     backgroundColor:'rgba(123,163,214,.16)', pointRadius:0, fill:'+1'},
    {label:'_lo', data: pad(f.lo), borderColor:'rgba(0,0,0,0)', pointRadius:0},
  ]);
}
async function load(){
  DATA = await fetch('data').then(r=>r.json());
  document.getElementById('tickers').value = DATA.tickers;
  document.getElementById('cutoff').value = DATA.cutoff;
  if(!KEY){ document.getElementById('apply').disabled = true;
            document.getElementById('cmsg').textContent = 'view-only (no key)'; }
  mk('growth', DATA.dates, series(DATA.growth), {type:'logarithmic'});
  mk('daycum', DATA.dates, series(DATA.daycum),
     {title:{display:true, text:'profit ($1/day)', color:'#93a0ad'}});
  mk('arena', DATA.dates, series(DATA.arena),
     {title:{display:true, text:'profit ($1/day)', color:'#93a0ad'}});
  const sel = document.getElementById('fticker');
  sel.innerHTML = Object.keys(DATA.forecast)
      .map(t => "<option>" + t + "</option>").join('');
  buildForecast();
}
async function applySettings(){
  document.getElementById('cmsg').textContent = 'rebuilding — downloads fresh data, ~30 s…';
  const res = await fetch('settings', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({key: KEY,
      tickers: document.getElementById('tickers').value,
      cutoff: document.getElementById('cutoff').value})});
  const j = await res.json().catch(()=>({}));
  document.getElementById('cmsg').textContent = res.ok ? '' : (j.error || 'error');
  if(res.ok) load();
}
load();
"""


def launch_dashboard(tickers="^GSPC, BTC-USD, GME", start="2019-01-01",
                     cutoff="2025-06-01", teacher_key=None, port=8772, tunnel=True):
    """Host the animated lecture dashboard. Returns (viewer_url, control_url)."""
    teacher_key = teacher_key or secrets.token_hex(4)
    payload = {"data": _dashboard_compute(tickers, start, cutoff)}
    lock = threading.Lock()
    app = Flask("dashboard")

    @app.get("/")
    def page():
        return Response(_page("Lecture Dashboard", _DARK_CSS, _DASH_HTML, _DASH_JS),
                        mimetype="text/html")

    @app.get("/data")
    def data():
        with lock:
            return jsonify(payload["data"])

    @app.post("/settings")
    def settings():
        body = request.get_json(force=True)
        if body.get("key") != teacher_key:
            return jsonify({"error": "wrong key"}), 403
        try:
            fresh = _dashboard_compute(body.get("tickers") or tickers,
                                       body.get("start") or start,
                                       body.get("cutoff") or cutoff)
        except Exception as e:                               # noqa: BLE001
            return jsonify({"error": str(e)}), 400
        with lock:
            payload["data"] = fresh
        return jsonify({"ok": True})

    base = _serve("dashboard", app, port, tunnel)
    viewer_url = f"{base}/"
    control_url = f"{base}/?key={teacher_key}"
    _announce("📊 Lecture Dashboard", viewer_url, control_url)
    return viewer_url, control_url
