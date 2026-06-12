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

import io
import multiprocessing
import os
import re
import secrets
import socket
import subprocess
import threading
import time
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


def _load_market(universe, custom_tickers, replay_start, replay_end, warmup):
    if universe.lower().startswith("dow"):
        tickers = DOW30
    elif universe.lower().startswith("custom"):
        tickers = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
        if len(tickers) < 2:
            raise ValueError("give at least two tickers in custom_tickers")
    else:
        tickers = _sp500_tickers()
    import yfinance as yf
    start = pd.Timestamp(replay_start) - pd.Timedelta(days=2 * warmup + 40)
    print(f"downloading {len(tickers)} tickers from {start.date()} to {replay_end}...")
    raw = yf.download(tickers, start=str(start.date()), end=str(replay_end),
                      auto_adjust=True, progress=False)
    rets = (raw["Close"] / raw["Open"] - 1)
    rets = rets.dropna(axis=1, thresh=int(0.95 * len(rets))).fillna(0.0)
    n_replay = int((rets.index >= pd.Timestamp(replay_start)).sum())
    if n_replay < 5:
        raise ValueError("replay window has fewer than 5 trading days")
    print(f"market ready: {rets.shape[1]} stocks, {n_replay} replay days "
          f"(+{len(rets) - n_replay} warmup days)")
    return rets, len(rets) - n_replay


def _replay_in_child(code, conn):
    """Run one student strategy against the module-global market (fork-shared)."""
    try:
        returns, warmup_t, fee = _MARKET
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
        for t in range(warmup_t, len(returns)):
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


_MARKET = None         # (returns_df, warmup_t, fee) — set before forking children


def _sharpe(pnl):
    pnl = np.asarray(pnl, dtype=float)
    if len(pnl) < 2 or pnl.std() == 0:
        return float("nan")
    return float(pnl.mean() / pnl.std() * np.sqrt(TRADING_DAYS))


_ARENA_STUDENT = """
<div class='card'>
  <h1>🏆 The Strategy Arena</h1>
  <div id='openview'>
    <p>Paste the <b>Strategy subclass</b> from your competition cell (Part 6 of the
    lab). One entry per team — resubmitting replaces your old entry.</p>
    <input id='name' placeholder='your name / team name' maxlength='24'>
    <br><br>
    <textarea id='code' rows='14' placeholder='class MyStrategy(Strategy):
    name = "my strategy"
    def allocate(self, past_returns):
        ...'></textarea>
    <button onclick='submitCode()'>🚀 Enter the arena</button>
    <div id='msg'></div>
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
function token(){ return localStorage.getItem('arena_token') || ''; }
async function submitCode(){
  const name = document.getElementById('name').value.trim();
  const code = document.getElementById('code').value;
  const msg = document.getElementById('msg');
  if(!name || !code.trim()){ msg.innerHTML = "<span class='bad'>need a name and some code</span>"; return; }
  const res = await fetch('submit', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({name:name, code:code, token: token()})});
  const j = await res.json();
  if(res.ok){ localStorage.setItem('arena_token', j.token);
    msg.innerHTML = "<span class='ok'>✔ entered as <b>" + j.name + "</b> — you can resubmit until the teacher locks it</span>";
  } else { msg.innerHTML = "<span class='bad'>" + (j.error || 'error') + "</span>"; }
}
async function poll(){
  const s = await fetch('state').then(r=>r.json()).catch(()=>null);
  if(!s) return;
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
      "<div class='muted'>everyone else: check your own phone for your private placement</div>";
  pod.style.display = 'block';
  document.getElementById('full').innerHTML = '<pre>' + r.full_table + '</pre>';
}
poll(); setInterval(poll, 2000);
"""


def launch_arena(universe="S&P 500", custom_tickers="", replay_start="2026-01-02",
                 replay_end="2026-06-10", warmup=20, fee=0.0005,
                 teacher_key=None, port=8771, tunnel=True, strategy_timeout=60):
    """Host the Strategy Arena. Returns (student_url, teacher_url)."""
    global _MARKET
    teacher_key = teacher_key or secrets.token_hex(4)
    returns, warmup_t = _load_market(universe, custom_tickers,
                                     replay_start, replay_end, warmup)
    _MARKET = (returns, warmup_t, fee)
    replay_dates = [d.strftime("%Y-%m-%d") for d in returns.index[warmup_t:]]

    state = {"phase": "open", "subs": {}, "results": None}
    lock = threading.Lock()
    app = Flask("arena")

    def _run_all():
        with lock:
            entries = [(tok, s["name"], s["code"]) for tok, s in state["subs"].items()]
        results = {}
        ctx = multiprocessing.get_context("fork")
        for tok, name, code in entries:
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_replay_in_child, args=(code, child))
            proc.start()
            child.close()
            out = {"error": f"timed out ({strategy_timeout}s limit)"}
            if parent.poll(strategy_timeout):
                out = parent.recv()
            proc.join(1)
            if proc.is_alive():
                proc.terminate()
            results[tok] = {"name": name, **out}
        # baseline: 1/N across the whole field
        R = returns.to_numpy()[warmup_t:]
        results["__baseline__"] = {"name": "1/N baseline", "pnl": R.mean(axis=1).tolist()}
        n_days = len(replay_dates)
        ranked = []
        for tok, r in results.items():
            if "pnl" in r:
                pnl = np.asarray(r["pnl"])
                r["roi"] = float(pnl.sum() / n_days)
                r["sharpe"] = _sharpe(pnl)
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
            return jsonify({"phase": state["phase"], "n_subs": len(state["subs"]),
                            "names": [s["name"] for s in state["subs"].values()]})

    @app.post("/submit")
    def submit():
        body = request.get_json(force=True)
        name = str(body.get("name", "")).strip()[:24]
        code_text = str(body.get("code", ""))
        if not name or not code_text.strip():
            return jsonify({"error": "need a name and some code"}), 400
        if len(code_text) > 20000:
            return jsonify({"error": "code too long"}), 400
        with lock:
            if state["phase"] != "open":
                return jsonify({"error": "submissions are locked"}), 409
            token = str(body.get("token") or "")
            if token not in state["subs"]:
                token = secrets.token_hex(8)
            taken = {s["name"] for t, s in state["subs"].items() if t != token}
            base_name, k = name, 2
            while name in taken:
                name = f"{base_name} ({k})"
                k += 1
            state["subs"][token] = {"name": name, "code": code_text}
        return jsonify({"token": token, "name": name})

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
