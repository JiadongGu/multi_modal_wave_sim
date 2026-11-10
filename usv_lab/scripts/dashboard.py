"""Live training dashboard: every skrl run, its progress and learning curves, and the GPU.

    ~/IsaacLab/_isaac_sim/python.sh <repo>/usv_lab/scripts/dashboard.py

then open http://localhost:8765 on this machine, or http://<tailscale ip>:8765 from another
device on the tailnet. Reads Isaac Lab's log folder and nvidia-smi; it changes nothing and has
no login, so it only listens on localhost and the tailnet address, never on the lab network.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = argparse.ArgumentParser()
parser.add_argument("--logs", default=os.path.expanduser("~/IsaacLab/logs/skrl"))
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--hosts", default="127.0.0.1,100.77.125.24", help="comma-separated addresses to listen on")
args = parser.parse_args()

GPU_FIELDS = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
gpu_history = deque(maxlen=1800)   # one sample every 2 s: the last hour


def sample_gpu():
    while True:
        try:
            line = subprocess.run(["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=5).stdout.splitlines()[0]
            util, used, total, temp, power = (float(v) for v in line.split(","))
            gpu_history.append({"t": time.time(), "util": util, "mem_gb": used / 1024, "total_gb": total / 1024,
                                "temp": temp, "power": power})
        except Exception:  # a missed sample is not worth stopping the dashboard for
            pass
        time.sleep(2)


def yaml_number(path, key):
    """First `key: number` in a params file. These files are flat enough that this beats a YAML dependency."""
    try:
        match = re.search(rf"^\s*{key}:\s*([0-9.eE+-]+)\s*$", open(path).read(), re.M)
        return float(match.group(1)) if match else None
    except OSError:
        return None


class Run:
    def __init__(self, path, run_id):
        self.path, self.id = path, run_id
        self.acc = EventAccumulator(path, size_guidance={"scalars": 0})
        self.seen_mtime = 0.0
        self.lock = threading.Lock()
        try:
            self.task = json.load(open(os.path.join(path, "run.json"))).get("task", "?")
        except (OSError, ValueError):
            self.task = "?"
        env, agent = os.path.join(path, "params/env.yaml"), os.path.join(path, "params/agent.yaml")
        self.num_envs = int(yaml_number(env, "num_envs") or 0)
        self.policy_dt = (yaml_number(env, "dt") or 0.0) * (yaml_number(env, "decimation") or 0.0)
        self.rollouts = int(yaml_number(agent, "rollouts") or 1)
        self.total_steps = int(yaml_number(agent, "timesteps") or 0)

    def refresh(self):
        with self.lock:
            files = glob.glob(os.path.join(self.path, "events*"))
            mtime = max((os.path.getmtime(f) for f in files), default=0.0)
            if mtime != self.seen_mtime:
                self.acc.Reload()
                self.seen_mtime = mtime

    def scalars(self, tag):
        return self.acc.Scalars(tag) if tag in self.acc.Tags()["scalars"] else []


runs: dict[str, Run] = {}


def training_process_running():
    out = subprocess.run(["pgrep", "-f", "reinforcement_learning/train"], capture_output=True, text=True).stdout
    return bool(out.strip())


def discover():
    for path in glob.glob(os.path.join(args.logs, "*", "*")):
        run_id = os.path.relpath(path, args.logs)
        if run_id not in runs and glob.glob(os.path.join(path, "events*")):
            runs[run_id] = Run(path, run_id)
    for run in runs.values():
        run.refresh()


def thin(points, limit=500):
    step = max(1, len(points) // limit)
    kept = points[::step]
    if points and kept[-1] is not points[-1]:
        kept.append(points[-1])
    return kept


def series(run, tag, scale=1.0):
    return [[p.step / run.rollouts, p.value * scale] for p in thin(run.scalars(tag))]


def throughput(run):
    """Environment steps per second, from when each iteration's log line was written (averaged over 5 iterations)."""
    pts = run.scalars("Loss / Policy loss")
    out = []
    for i in range(5, len(pts)):
        dt = pts[i].wall_time - pts[i - 5].wall_time
        if dt > 0:
            out.append([pts[i].step / run.rollouts, (pts[i].step - pts[i - 5].step) / dt * run.num_envs])
    return thin(out)


def summary(run, live_id):
    pts = run.scalars("Loss / Policy loss")
    step = pts[-1].step if pts else 0
    reward = run.scalars("Reward / Total reward (mean)")
    if run.id == live_id:
        status = "training"
    elif run.total_steps and step >= run.total_steps - 2 * run.rollouts:
        status = "finished"
    else:
        status = "stopped"
    return {"id": run.id, "task": run.task, "started": os.path.basename(run.id)[:19].replace("_", " "),
            "status": status, "iteration": step // run.rollouts, "total_iterations": run.total_steps // run.rollouts,
            "num_envs": run.num_envs, "reward": reward[-1].value if reward else None,
            "best_reward": max((p.value for p in reward), default=None)}


def run_list():
    discover()
    newest = max(runs, key=lambda r: os.path.basename(r), default=None)
    live_id = newest if newest and training_process_running() else None
    return sorted((summary(r, live_id) for r in runs.values()), key=lambda s: s["started"], reverse=True)


def run_detail(run_id):
    run = runs[run_id]
    info = next(s for s in run_list() if s["id"] == run_id)
    rate = throughput(run)
    info["env_steps_per_s"] = rate[-1][1] if rate else None
    info["sim_seconds_per_s"] = rate[-1][1] * run.policy_dt if rate else None
    remaining = info["total_iterations"] - info["iteration"]
    info["eta_s"] = remaining * run.rollouts * run.num_envs / rate[-1][1] if rate and info["status"] == "training" else None
    info["charts"] = {
        "reward": series(run, "Reward / Total reward (mean)"),
        "distance": series(run, "Nav/mean_distance_m") or series(run, "Station/mean_distance_m"),
        "within": series(run, "Nav/success_rate", 100.0) or series(run, "Station/within_2m_fraction", 100.0),
        "episode": series(run, "Episode / Total timesteps (mean)", run.policy_dt),
        "rate": rate,
        "stepping": series(run, "Stats / Env stepping time (ms)"),
        "update": series(run, "Stats / Algorithm update time (ms)"),
    }
    return info


def gpu_detail():
    now = time.time()
    samples = list(gpu_history)
    return {"latest": samples[-1] if samples else None,
            "util": [[(s["t"] - now) / 60, s["util"]] for s in samples],
            "mem": [[(s["t"] - now) / 60, s["mem_gb"]] for s in samples]}


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>USV training</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{color-scheme:light;--bg:#f4f4f1;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--ink3:#77766f;--line:#e4e3de;--series:#2a78d6;--good:#008300;--idle:#77766f}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#111110;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--ink3:#96958b;--line:#30302d;--series:#3987e5;--good:#3fae4a;--idle:#96958b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1320px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:20px;margin:0}h2{font-size:13px;font-weight:600;color:var(--ink2);margin:26px 0 10px}
header{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}#sub{color:var(--ink2)}
.card{background:var(--surface);border:1px solid var(--line);border-radius:8px}
table{width:100%;border-collapse:collapse}th{font-weight:500;color:var(--ink3);font-size:12px;text-align:left;padding:8px 12px;border-bottom:1px solid var(--line)}
td{padding:8px 12px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}tr:last-child td{border-bottom:0}
tbody tr{cursor:pointer}tbody tr:hover{background:color-mix(in srgb,var(--series) 8%,transparent)}tr.sel{box-shadow:inset 3px 0 0 var(--series)}
.wrap{overflow-x:auto}.status::before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;background:var(--idle)}.status.training::before{background:var(--good)}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(165px,1fr));gap:10px}.tile{padding:12px 14px}.tile b{display:block;font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}.tile span{color:var(--ink3);font-size:12px}
.bar{height:6px;border-radius:3px;background:var(--line);margin-top:12px;overflow:hidden}.bar i{display:block;height:100%;background:var(--series)}
.charts{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}.chart{padding:12px 14px;position:relative}.chart h3{margin:0;font-size:13px;font-weight:600}.chart p{margin:0 0 6px;color:var(--ink3);font-size:12px}
svg{display:block;width:100%;height:150px;touch-action:none}svg text{fill:var(--ink3);font-size:11px}.tip{position:absolute;pointer-events:none;background:var(--ink);color:var(--surface);padding:3px 7px;border-radius:4px;font-size:12px;white-space:nowrap;display:none;z-index:2;font-variant-numeric:tabular-nums}
.empty{height:150px;display:flex;align-items:center;justify-content:center;color:var(--ink3)}
</style></head><body><main>
<header><h1>USV training</h1><span id="sub">loading</span></header>
<h2>Runs</h2><div class="card wrap"><table><thead><tr><th>Started</th><th>Task</th><th>Status</th><th>Iteration</th><th>Boats</th><th>Latest reward</th><th>Best reward</th></tr></thead><tbody id="runs"></tbody></table></div>
<h2 id="runhead">Selected run</h2><div class="tiles" id="tiles"></div><div class="bar"><i id="prog" style="width:0"></i></div>
<h2>Learning</h2><div class="charts" id="learn"></div>
<h2>Speed</h2><div class="charts" id="speed"></div>
<h2>GPU, last hour</h2><div class="tiles" id="gputiles" style="margin-bottom:10px"></div><div class="charts" id="gpu"></div>
</main><script>
const LEARN=[["reward","Episode reward","mean total per finished episode"],["distance","Distance to goal","mean across boats, metres"],["within","Success","percent: reached the pillar, or within 2 m of centre"],["episode","Episode length","mean, simulated seconds"]];
const SPEED=[["rate","Environment steps per second","all boats, 5-iteration average"],["stepping","Simulation step time","milliseconds per step"],["update","Learning update time","milliseconds"]];
const GPU=[["util","GPU utilisation","percent"],["mem","GPU memory","GB"]];
let selected=null;
const q=s=>document.querySelector(s);
const fmt=v=>v==null||!isFinite(v)?"-":Math.abs(v)>=1000?Math.round(v).toLocaleString():Math.abs(v)>=10?v.toFixed(1):v.toFixed(2);
const dur=s=>s==null?"-":s>=3600?`${Math.floor(s/3600)}h ${Math.round(s%3600/60)}m`:s>=60?`${Math.round(s/60)} min`:`${Math.round(s)} s`;
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function makeCards(root,defs){q(root).innerHTML=defs.map(([k,t,s])=>`<div class="card chart" data-k="${k}"><h3>${t}</h3><p>${s}</p><div class="plot"></div><div class="tip"></div></div>`).join("")}
function tiles(root,items){q(root).innerHTML=items.map(([v,l])=>`<div class="card tile"><b>${v}</b><span>${l}</span></div>`).join("")}
function draw(card,pts,xlabel){
  const plot=card.querySelector(".plot"),tip=card.querySelector(".tip");
  if(!pts||pts.length<2){plot.innerHTML='<div class="empty">No data yet</div>';plot.onpointermove=null;return}
  const W=Math.max(plot.clientWidth,240),H=150,L=46,R=8,T=8,B=20;
  const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  if(x1===x0)x1=x0+1;const pad=(y1-y0||Math.abs(y1)||1)*.08;y0-=pad;y1+=pad;
  const X=x=>L+(x-x0)/(x1-x0)*(W-L-R),Y=y=>T+(1-(y-y0)/(y1-y0))*(H-T-B);
  const ticks=[y0+pad,(y0+y1)/2,y1-pad];
  plot.innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(card.querySelector("h3").textContent)}">`+
   ticks.map(t=>`<line x1="${L}" x2="${W-R}" y1="${Y(t)}" y2="${Y(t)}" stroke="var(--line)"/><text x="${L-6}" y="${Y(t)+4}" text-anchor="end">${fmt(t)}</text>`).join("")+
   `<text x="${L}" y="${H-5}">${fmt(x0)}</text><text x="${W-R}" y="${H-5}" text-anchor="end">${fmt(x1)} ${xlabel}</text>`+
   `<polyline fill="none" stroke="var(--series)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="${pts.map(p=>X(p[0]).toFixed(1)+","+Y(p[1]).toFixed(1)).join(" ")}"/>`+
   `<line class="cross" y1="${T}" y2="${H-B}" stroke="var(--ink3)" visibility="hidden"/><circle class="dot" r="4" fill="var(--series)" stroke="var(--surface)" stroke-width="2" visibility="hidden"/></svg>`;
  const svg=plot.querySelector("svg"),cross=svg.querySelector(".cross"),dot=svg.querySelector(".dot");
  svg.onpointermove=e=>{const r=svg.getBoundingClientRect(),mx=(e.clientX-r.left)/r.width*W;let b=pts[0];for(const p of pts)if(Math.abs(X(p[0])-mx)<Math.abs(X(b[0])-mx))b=p;
    cross.setAttribute("x1",X(b[0]));cross.setAttribute("x2",X(b[0]));cross.setAttribute("visibility","visible");dot.setAttribute("cx",X(b[0]));dot.setAttribute("cy",Y(b[1]));dot.setAttribute("visibility","visible");
    tip.textContent=`${fmt(b[1])} at ${fmt(b[0])} ${xlabel}`;tip.style.display="block";const px=X(b[0])/W*r.width+14;tip.style.left=Math.min(px,card.clientWidth-tip.offsetWidth-6)+"px";tip.style.top="44px"};
  svg.onpointerleave=()=>{cross.setAttribute("visibility","hidden");dot.setAttribute("visibility","hidden");tip.style.display="none"};
}
const hovering=root=>q(root).querySelector(".chart:hover");
function drawAll(root,data,xl){if(hovering(root))return;q(root).querySelectorAll(".chart").forEach(c=>draw(c,data[c.dataset.k],xl))}
async function tick(){
  try{
    const [list,gpu]=await Promise.all([fetch("/api/runs").then(r=>r.json()),fetch("/api/gpu").then(r=>r.json())]);
    if(!selected||!list.some(r=>r.id===selected))selected=list[0]?.id??null;
    q("#runs").innerHTML=list.length?list.map(r=>`<tr data-id="${esc(r.id)}" class="${r.id===selected?"sel":""}"><td>${esc(r.started)}</td><td>${esc(r.task)}</td><td><span class="status ${r.status}">${r.status}</span></td><td>${r.iteration} / ${r.total_iterations}</td><td>${r.num_envs.toLocaleString()}</td><td>${fmt(r.reward)}</td><td>${fmt(r.best_reward)}</td></tr>`).join(""):'<tr><td colspan="7">No runs yet</td></tr>';
    q("#runs").querySelectorAll("tr[data-id]").forEach(tr=>tr.onclick=()=>{selected=tr.dataset.id;tick()});
    const live=list.filter(r=>r.status==="training").length;q("#sub").textContent=live?"training now":"nothing training";
    if(selected){const d=await fetch("/api/run/"+encodeURIComponent(selected)).then(r=>r.json()),c=d.charts,last=a=>a&&a.length?a[a.length-1][1]:null;
      q("#runhead").textContent=`${d.task}, started ${d.started}`;
      tiles("#tiles",[[`${d.iteration} / ${d.total_iterations}`,"Iteration"],[d.status==="training"?dur(d.eta_s):d.status,d.status==="training"?"Time left":"Status"],[fmt(d.reward),"Episode reward"],[fmt(last(c.distance)),"Mean distance to goal, m"],[fmt(last(c.within))+"%","Success"],[fmt(d.env_steps_per_s),"Env steps per second"],[fmt(d.sim_seconds_per_s),"Simulated s per real s"]]);
      q("#prog").style.width=(d.total_iterations?Math.min(100,d.iteration/d.total_iterations*100):0)+"%";
      drawAll("#learn",c,"iter");drawAll("#speed",c,"iter")}
    const g=gpu.latest;if(g)tiles("#gputiles",[[fmt(g.util)+"%","Utilisation"],[`${fmt(g.mem_gb)} / ${fmt(g.total_gb)}`,"Memory, GB"],[fmt(g.temp)+" C","Temperature"],[fmt(g.power)+" W","Power"]]);
    drawAll("#gpu",gpu,"min ago");
  }catch(e){q("#sub").textContent="dashboard lost contact, retrying"}
}
makeCards("#learn",LEARN);makeCards("#speed",SPEED);makeCards("#gpu",GPU);tick();setInterval(tick,3000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):  # keep the terminal quiet
        pass

    def reply(self, body, kind="application/json", code=200):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            if self.path == "/":
                self.reply(PAGE, "text/html; charset=utf-8")
            elif self.path == "/api/runs":
                self.reply(json.dumps(run_list()))
            elif self.path == "/api/gpu":
                self.reply(json.dumps(gpu_detail()))
            elif self.path.startswith("/api/run/"):
                run_id = unquote(self.path[len("/api/run/"):])
                discover()
                if run_id in runs:   # only ids found on disk, never a path taken from the request
                    self.reply(json.dumps(run_detail(run_id)))
                else:
                    self.reply('{"error": "unknown run"}', code=404)
            else:
                self.reply('{"error": "not found"}', code=404)
        except BrokenPipeError:
            pass


threading.Thread(target=sample_gpu, daemon=True).start()
servers = []
for host in args.hosts.split(","):
    try:
        servers.append(ThreadingHTTPServer((host, args.port), Handler))
        print(f"dashboard on http://{host}:{args.port}", flush=True)
    except OSError as err:
        print(f"could not listen on {host}: {err}", flush=True)
if not servers:
    raise SystemExit("no address to listen on")
for server in servers[1:]:
    threading.Thread(target=server.serve_forever, daemon=True).start()
servers[0].serve_forever()
