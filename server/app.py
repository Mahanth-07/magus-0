"""
Ludus minimal dashboard — FastAPI server.

Reads from runs/ on disk (decoupled from the loop).
Start: .venv/bin/python -m uvicorn server.app:app --port 8137
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

RUNS_DIR = Path(__file__).parent.parent / "runs"

app = FastAPI(title="Ludus Dashboard")


def _latest_run_dir() -> Path | None:
    """Return the most-recently-modified run directory under runs/."""
    dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime)


def _last_step(run_dir: Path) -> dict[str, Any] | None:
    jsonl = run_dir / "steps.jsonl"
    if not jsonl.exists():
        return None
    last: dict[str, Any] | None = None
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last


def _episode(run_dir: Path) -> dict[str, Any] | None:
    ep = run_dir / "episode.json"
    if not ep.exists():
        return None
    return json.loads(ep.read_text())


def _latest_png(run_dir: Path) -> Path | None:
    pngs = sorted(run_dir.glob("step_*.png"))
    return pngs[-1] if pngs else None


def _compare_episodes() -> dict[str, Any]:
    """Collect final_metrics + rules for all run dirs, keyed by dir name."""
    result: dict[str, Any] = {}
    if not RUNS_DIR.exists():
        return result
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        ep = _episode(d)
        if ep:
            result[d.name] = {
                "game": ep.get("game"),
                "mode": ep.get("mode"),
                "steps": ep.get("steps"),
                "final_metrics": ep.get("final_metrics", {}),
                "rules": ep.get("rules", []),
            }
    return result


@app.get("/api/state")
def api_state() -> dict[str, Any]:
    run_dir = _latest_run_dir()
    if run_dir is None:
        raise HTTPException(status_code=404, detail="No run directories found")

    step = _last_step(run_dir)
    episode = _episode(run_dir)
    rules: list[str] = episode.get("rules", []) if episode else []

    # Also surface rules found in step records (rule_added field)
    if not rules and run_dir.exists():
        jsonl = run_dir / "steps.jsonl"
        if jsonl.exists():
            with jsonl.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        r = rec.get("rule_added")
                        if r and r not in rules:
                            rules.append(r)

    return {
        "run_dir": run_dir.name,
        "last_step": step,
        "episode": episode,
        "rules": rules,
        "all_runs": _compare_episodes(),
    }


@app.get("/api/screenshot")
def api_screenshot() -> FileResponse:
    run_dir = _latest_run_dir()
    if run_dir is None:
        raise HTTPException(status_code=404, detail="No run directories found")
    png = _latest_png(run_dir)
    if png is None:
        raise HTTPException(status_code=404, detail="No screenshots in run dir")
    return FileResponse(str(png), media_type="image/png")


_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Ludus Dashboard</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:system-ui,sans-serif;background:#0f0f13;color:#e2e8f0;min-height:100vh;padding:1.5rem}
  h1{font-size:1.5rem;font-weight:700;margin-bottom:1.25rem;color:#a78bfa}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}
  .panel{background:#1a1a2e;border:1px solid #2d2d4e;border-radius:8px;padding:1rem}
  .panel h2{font-size:.85rem;text-transform:uppercase;letter-spacing:.08em;color:#7c7ca8;margin-bottom:.75rem}
  img#screenshot{width:100%;border-radius:4px;border:1px solid #2d2d4e}
  .kv{display:flex;justify-content:space-between;font-size:.82rem;padding:.2rem 0;border-bottom:1px solid #1e1e38}
  .kv:last-child{border-bottom:none}
  .val{color:#a78bfa;font-weight:600}
  .rule{font-size:.8rem;background:#12122a;border-radius:4px;padding:.35rem .5rem;margin:.2rem 0;color:#c4b5fd}
  .badge{display:inline-block;font-size:.7rem;padding:.15rem .4rem;border-radius:3px;font-weight:700;margin-left:.4rem}
  .badge-baseline{background:#1e3a5f;color:#7dd3fc}
  .badge-memory{background:#1e3d2f;color:#6ee7b7}
  .run-row{font-size:.78rem;padding:.3rem 0;border-bottom:1px solid #1e1e38}
  .run-row:last-child{border-bottom:none}
  .ts{font-size:.7rem;color:#555;margin-top:.4rem}
  #status{position:fixed;bottom:.75rem;right:1rem;font-size:.7rem;color:#555}
</style>
</head>
<body>
<h1>Ludus Dashboard</h1>
<div class="grid">
  <div class="panel">
    <h2>Latest Screenshot</h2>
    <img id="screenshot" src="/api/screenshot" alt="latest step"/>
  </div>

  <div class="panel">
    <h2>Current Decision</h2>
    <div id="decision-content"></div>
  </div>

  <div class="panel">
    <h2>Latest Metrics</h2>
    <div id="metrics-content"></div>
  </div>

  <div class="panel">
    <h2>Learned Rulebook</h2>
    <div id="rules-content"><em style="color:#555">No rules yet.</em></div>
  </div>

  <div class="panel" style="grid-column:1/-1">
    <h2>Baseline vs Memory (all runs)</h2>
    <div id="compare-content"></div>
  </div>
</div>

<div id="status">connecting…</div>

<script>
function kv(k,v){return `<div class="kv"><span>${k}</span><span class="val">${v}</span></div>`}
function badge(mode){return `<span class="badge badge-${mode}">${mode}</span>`}

async function refresh(){
  try{
    const r=await fetch('/api/state');
    if(!r.ok){document.getElementById('status').textContent='error '+r.status;return;}
    const d=await r.json();

    // screenshot
    document.getElementById('screenshot').src='/api/screenshot?t='+Date.now();

    // decision
    const dc=document.getElementById('decision-content');
    const step=d.last_step;
    if(step&&step.decision){
      const dec=step.decision;
      dc.innerHTML=[
        kv('Run',d.run_dir),
        kv('Step',step.step_index),
        kv('Mode',step.mode),
        kv('Action',dec.action),
        kv('Confidence',(dec.confidence*100).toFixed(0)+'%'),
        kv('Expected',dec.expected_result),
        `<div class="kv" style="flex-direction:column"><span style="color:#7c7ca8;font-size:.78rem">Reason</span><span style="color:#d1d5db;font-size:.78rem;margin-top:.2rem">${dec.reason}</span></div>`,
      ].join('');
    } else {
      dc.innerHTML='<span style="color:#555">No step data yet.</span>';
    }

    // metrics
    const mc=document.getElementById('metrics-content');
    if(step&&step.metric_delta){
      const rows=Object.entries(step.metric_delta).map(([k,v])=>kv(k,v>=0?'+'+v:v));
      if(step.primary_metric){
        rows.unshift(kv('Primary metric',step.primary_metric));
        rows.unshift(kv('Improved',step.improved?'yes':'no'));
      }
      mc.innerHTML=rows.join('');
    } else if(d.episode&&d.episode.final_metrics){
      mc.innerHTML=Object.entries(d.episode.final_metrics).map(([k,v])=>kv(k,v)).join('');
    } else {
      mc.innerHTML='<span style="color:#555">No metric data yet.</span>';
    }

    // rules
    const rc=document.getElementById('rules-content');
    if(d.rules&&d.rules.length){
      rc.innerHTML=d.rules.map(r=>`<div class="rule">${r}</div>`).join('');
    } else {
      rc.innerHTML='<em style="color:#555">No rules yet.</em>';
    }

    // compare
    const cc=document.getElementById('compare-content');
    const runs=d.all_runs||{};
    const names=Object.keys(runs);
    if(names.length===0){
      cc.innerHTML='<span style="color:#555">No completed episodes.</span>';
    } else {
      cc.innerHTML=names.map(name=>{
        const ep=runs[name];
        const metrics=Object.entries(ep.final_metrics||{}).map(([k,v])=>`${k}=${v}`).join(' · ');
        return `<div class="run-row"><strong>${ep.game}</strong>${badge(ep.mode)} — steps: ${ep.steps} &nbsp; ${metrics}</div>`;
      }).join('');
    }

    document.getElementById('status').textContent='last update: '+new Date().toLocaleTimeString();
  } catch(e){
    document.getElementById('status').textContent='poll error: '+e.message;
  }
}

refresh();
setInterval(refresh, 1200);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(_HTML)
