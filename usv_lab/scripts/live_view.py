"""Live viewer: run metrics, GPU, and camera + lidar for a few boats. Plain python3, standard library only.

    python3 <repo>/usv_lab/scripts/live_view.py [--tensorboard http://100.77.125.24:6006]

then open http://localhost:8765 on the lab machine, or http://100.77.125.24:8765 from another device on
the tailnet. Reads what running environments write with `live_feed = True` (see tasks/live_feed.py)
and samples nvidia-smi. Runs on the CPU in its own process; it never touches a simulation. No login,
so it listens only on localhost and the tailnet address, never on the lab network.
"""

import argparse
import json
import os
import re
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

parser = argparse.ArgumentParser()
parser.add_argument("--feed", default="~/usv_feed", help="folder the environments write their feed to")
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--hosts", default="127.0.0.1,100.77.125.24", help="comma-separated addresses to listen on")
parser.add_argument("--tensorboard", default="", help="link to a TensorBoard showing training curves, if one runs")
args = parser.parse_args()
FEED = Path(args.feed).expanduser()
FEED.mkdir(parents=True, exist_ok=True)

gpu_history = deque(maxlen=1800)                 # one sample every 2 s: the last hour
run_history: dict[str, deque] = {}              # pid -> metric snapshots, one per feed update: the last hour
seen_mtime: dict[str, float] = {}
lock = threading.Lock()


def sample_gpu():
    fields = "utilization.gpu,memory.used,memory.total"
    while True:
        try:
            line = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=5).stdout.splitlines()[0]
            util, used, total = (float(v) for v in line.split(","))
            with lock:
                gpu_history.append([time.time(), util, used / 1024, total / 1024])
        except Exception:                        # a missed sample is not worth stopping the viewer for
            pass
        time.sleep(2)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def sample_runs():
    keys = ("env_steps_per_s", "sim_s_per_real_s", "resets", "mean_surge", "max_tilt_deg")
    while True:
        for d in FEED.iterdir() if FEED.exists() else []:
            latest = d / "latest.json"
            try:
                mtime = latest.stat().st_mtime
            except OSError:
                continue
            if seen_mtime.get(d.name) == mtime:
                continue
            seen_mtime[d.name] = mtime
            snap = read_json(latest)
            if snap:
                with lock:
                    run_history.setdefault(d.name, deque(maxlen=3600)).append([snap["time"]] + [snap.get(k) for k in keys])
        time.sleep(1)


def alive(pid: str) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def runs():
    out = []
    for d in FEED.iterdir() if FEED.exists() else []:
        info = read_json(d / "run.json")
        if not info:
            continue
        try:
            age = time.time() - (d / "latest.json").stat().st_mtime
        except OSError:
            age = None
        out.append({"pid": d.name, "task": info.get("task"), "num_envs": info.get("num_envs"),
                    "started": info.get("started"), "command": info.get("command"),
                    "live": alive(d.name) and age is not None and age < 10})
    return sorted(out, key=lambda r: r["started"] or 0, reverse=True)


def state(pid):
    with lock:
        gpu = list(gpu_history)
        hist = list(run_history.get(pid, []))
    latest = read_json(FEED / pid / "latest.json") if pid else None
    return {"now": time.time(), "gpu": gpu, "runs": runs(), "latest": latest, "history": hist,
            "tensorboard": args.tensorboard}


FEED_FILE = re.compile(r"^/feed/(\d+)/(run\.json|latest\.json|camera_\d+\.png)$")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):                   # quiet
        pass

    def send(self, code, body: bytes, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path == "/":
            return self.send(200, PAGE.encode(), "text/html; charset=utf-8")
        if url.path == "/api/state":
            pid = (parse_qs(url.query).get("run") or [""])[0]
            pid = pid if pid.isdigit() else ""
            return self.send(200, json.dumps(state(pid)).encode(), "application/json")
        m = FEED_FILE.match(url.path)
        if m:
            try:
                body = (FEED / m.group(1) / m.group(2)).read_bytes()
            except OSError:
                return self.send(404, b"not found", "text/plain")
            return self.send(200, body, "image/png" if m.group(2).endswith(".png") else "application/json")
        self.send(404, b"not found", "text/plain")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>USV live view</title>
<style>
:root{--surface:#fcfcfb;--surface-2:#f3f2ef;--grid:#e4e3df;--text:#0b0b0b;--text-2:#52514e;--series:#2a78d6;
      --good:#0ca30c;--muted:#8a8984}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--surface-2:#242423;--grid:#383835;--text:#f0efec;
      --text-2:#c3c2b7;--series:#3987e5;--muted:#8a8984}}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--text);font:14px/1.4 system-ui,-apple-system,sans-serif}
main{max-width:1200px;margin:0 auto;padding:16px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:16px}
h1{font-size:18px;margin:0 12px 0 0} h2{font-size:14px;margin:0 0 8px;color:var(--text-2);font-weight:600}
select{font:inherit;padding:4px 8px;background:var(--surface-2);color:var(--text);border:1px solid var(--grid);border-radius:6px;max-width:100%}
.status{font-size:13px;color:var(--text-2)} .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;background:var(--muted)}
.dot.live{background:var(--good)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin-bottom:16px}
.tile{background:var(--surface-2);border-radius:8px;padding:10px 12px}
.tile .v{font-size:22px;font-variant-numeric:tabular-nums} .tile .k{font-size:12px;color:var(--text-2)}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;margin-bottom:16px}
.chart{background:var(--surface-2);border-radius:8px;padding:10px 12px;position:relative}
.chart canvas{width:100%;height:140px;display:block}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--grid);border-radius:6px;
     padding:4px 8px;font-size:12px;display:none;white-space:nowrap}
.boats{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.boat{background:var(--surface-2);border-radius:8px;padding:10px 12px}
.boat img{width:100%;image-rendering:pixelated;border-radius:4px;display:block;background:var(--grid);aspect-ratio:4/3}
.boat canvas{width:100%;aspect-ratio:1;display:block;margin-top:8px}
.boat .s{font-size:12px;color:var(--text-2);margin-top:6px;font-variant-numeric:tabular-nums}
.note{font-size:12px;color:var(--text-2);margin:4px 0 16px} a{color:var(--series)}
</style></head><body><main>
<header><h1>USV live view</h1><select id="run"></select><span class="status" id="status"></span><span id="tb"></span></header>
<div class="tiles" id="tiles"></div>
<div class="charts">
  <div class="chart"><h2>Env-steps per second</h2><canvas id="c_steps"></canvas><div class="tip"></div></div>
  <div class="chart"><h2>GPU utilisation (%)</h2><canvas id="c_util"></canvas><div class="tip"></div></div>
  <div class="chart"><h2>GPU memory used (GB)</h2><canvas id="c_mem"></canvas><div class="tip"></div></div>
</div>
<div class="note" id="note"></div>
<div class="boats" id="boats"></div>
</main><script>
const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let selected = null, runInfo = {}, data = null;
const WINDOW = 300;   // seconds of history shown

function fmt(v, d=1){ if(v==null||!isFinite(v)) return "–"; return Math.abs(v)>=10000? Math.round(v).toLocaleString() : v.toFixed(d); }
function niceMax(v){ if(!(v>0)) return 1; const p=10**Math.floor(Math.log10(v)); return [1,2,2.5,5,10].map(m=>m*p).find(m=>m>=v); }

function line(canvas, pts, now, unit, digits){
  const dpr = window.devicePixelRatio||1, W = canvas.clientWidth, H = canvas.clientHeight;
  canvas.width = W*dpr; canvas.height = H*dpr;
  const g = canvas.getContext("2d"); g.scale(dpr,dpr); g.clearRect(0,0,W,H);
  const L=44, R=56, T=6, B=18, w=W-L-R, h=H-T-B;
  pts = pts.filter(p=>p[1]!=null && now-p[0]<=WINDOW);
  const ymax = niceMax(Math.max(...pts.map(p=>p[1]), 0));
  const X = t => L + w*(1-(now-t)/WINDOW), Y = v => T + h*(1-v/ymax);
  g.font="11px system-ui"; g.fillStyle=css("--text-2"); g.strokeStyle=css("--grid"); g.lineWidth=1;
  for(let i=0;i<=2;i++){ const v=ymax*i/2, y=Math.round(Y(v))+.5; g.beginPath(); g.moveTo(L,y); g.lineTo(L+w,y); g.stroke();
    g.textAlign="right"; g.fillText(v>=10000? (v/1000)+"K" : String(+v.toFixed(2)), L-6, y+4); }
  g.textAlign="left"; g.fillText("-5 min", L, H-4); g.textAlign="right"; g.fillText("now", L+w, H-4);
  canvas._pts = pts; canvas._X = X; canvas._Y = Y; canvas._unit = unit; canvas._d = digits;
  if(pts.length<1) return;
  const s = css("--series");
  g.beginPath(); pts.forEach((p,i)=> i? g.lineTo(X(p[0]),Y(p[1])) : g.moveTo(X(p[0]),Y(p[1])));
  g.lineTo(X(pts[pts.length-1][0]), Y(0)); g.lineTo(X(pts[0][0]), Y(0)); g.closePath();
  g.globalAlpha=.1; g.fillStyle=s; g.fill(); g.globalAlpha=1;
  g.beginPath(); pts.forEach((p,i)=> i? g.lineTo(X(p[0]),Y(p[1])) : g.moveTo(X(p[0]),Y(p[1])));
  g.strokeStyle=s; g.lineWidth=2; g.lineJoin=g.lineCap="round"; g.stroke();
  const last = pts[pts.length-1], lx=X(last[0]), ly=Y(last[1]);
  g.beginPath(); g.arc(lx,ly,4,0,7); g.fillStyle=s; g.fill(); g.lineWidth=2; g.strokeStyle=css("--surface-2"); g.stroke();
  g.fillStyle=css("--text"); g.textAlign="left"; g.fillText(fmt(last[1],digits), lx+8, ly+4);
}

function hover(canvas){
  const tip = canvas.parentElement.querySelector(".tip");
  canvas.addEventListener("mousemove", e=>{
    const pts = canvas._pts||[]; if(!pts.length) return;
    const r = canvas.getBoundingClientRect(), x = e.clientX-r.left;
    let best = pts[0]; for(const p of pts) if(Math.abs(canvas._X(p[0])-x) < Math.abs(canvas._X(best[0])-x)) best=p;
    tip.style.display="block"; tip.textContent = `${fmt(best[1],canvas._d)} ${canvas._unit} · ${Math.round(data.now-best[0])} s ago`;
    tip.style.left = Math.min(canvas._X(best[0])+16, r.width-100)+"px"; tip.style.top = (canvas._Y(best[1])+20)+"px";
  });
  canvas.addEventListener("mouseleave", ()=> tip.style.display="none");
}

function lidarPlot(canvas, ranges, info){
  const dpr = window.devicePixelRatio||1, S = canvas.clientWidth;
  canvas.width = S*dpr; canvas.height = S*dpr;
  const g = canvas.getContext("2d"); g.scale(dpr,dpr); g.clearRect(0,0,S,S);
  const az = info.lidar_azimuth, el = info.lidar_elevation, xs=[], ys=[];
  let hits = 0;
  ranges.forEach((r,i)=>{ if(r<0) return; hits++; const h=r*Math.cos(el[i]); xs.push(h*Math.cos(az[i])); ys.push(h*Math.sin(az[i])); });
  const far = xs.map((x,i)=>Math.hypot(x,ys[i])).sort((a,b)=>a-b);
  const reach = niceMax(far.length? far[Math.floor(far.length*.95)] : 10), c = S/2, k = (S/2-14)/reach;
  g.strokeStyle=css("--grid"); g.lineWidth=1; g.fillStyle=css("--text-2"); g.font="11px system-ui"; g.textAlign="left";
  for(const f of [.5,1]){ g.beginPath(); g.arc(c,c,reach*f*k,0,7); g.stroke(); g.fillText(`${reach*f} m`, c+4, c-reach*f*k+12); }
  g.fillStyle=css("--series");
  xs.forEach((x,i)=>{ g.fillRect(c - ys[i]*k - 1, c - x*k - 1, 2, 2); });   // forward = up, port = left
  g.fillStyle=css("--text"); g.beginPath(); g.moveTo(c,c-7); g.lineTo(c-5,c+5); g.lineTo(c+5,c+5); g.closePath(); g.fill();
  return ranges.length? hits/ranges.length : 0;
}

async function loadRunInfo(pid){
  if(!runInfo[pid]) runInfo[pid] = await fetch(`/feed/${pid}/run.json`).then(r=>r.ok?r.json():{}).catch(()=>({}));
  return runInfo[pid];
}

async function refresh(){
  try{ data = await fetch("/api/state"+(selected?`?run=${selected}`:"")).then(r=>r.json()); }catch(e){ $("status").textContent="viewer not reachable"; return; }
  const sel = $("run"), runs = data.runs;
  if(!selected && runs.length){ selected = (runs.find(r=>r.live)||runs[0]).pid; return refresh(); }
  sel.innerHTML = runs.length? runs.map(r=>`<option value="${r.pid}" ${r.pid===selected?"selected":""}>${r.live?"● ":""}${r.task} · ${r.num_envs} boats · pid ${r.pid} · ${new Date(r.started*1000).toLocaleTimeString()}</option>`).join("")
                             : "<option>no runs yet: start one with live_feed on</option>";
  const run = runs.find(r=>r.pid===selected), L = data.latest, gpu = data.gpu, lastGpu = gpu[gpu.length-1];
  $("status").innerHTML = run? `<span class="dot ${run.live?"live":""}"></span>${run.live?"running":"stopped"}` : "";
  $("tb").innerHTML = data.tensorboard? `<a href="${data.tensorboard}" target="_blank">Training curves (TensorBoard)</a>` : "";
  const tiles = [
    ["Env-steps / s", fmt(L?.env_steps_per_s,0)], ["Sim s per real s, per boat", fmt(L?.sim_s_per_real_s,2)],
    ["Sim time", L? fmt(L.sim_time,0)+" s" : "–"], ["Boats", L?.num_envs ?? "–"], ["Resets", L?.resets ?? "–"],
    ["Mean surge", L? fmt(L.mean_surge,2)+" m/s" : "–"], ["Max tilt", L? fmt(L.max_tilt_deg,1)+"°" : "–"],
    ["GPU", lastGpu? `${fmt(lastGpu[1],0)}% · ${fmt(lastGpu[2],1)}/${fmt(lastGpu[3],0)} GB` : "–"]];
  $("tiles").innerHTML = tiles.map(([k,v])=>`<div class="tile"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");
  line($("c_steps"), data.history.map(h=>[h[0],h[1]]), data.now, "env-steps/s", 0);
  line($("c_util"), gpu.map(x=>[x[0],x[1]]), data.now, "%", 0);
  line($("c_mem"), gpu.map(x=>[x[0],x[2]]), data.now, "GB", 1);
  const age = L? data.now - L.time : null;
  $("note").textContent = L? `Updated ${fmt(age,0)} s ago. Pictures and numbers trail the simulation by 1–2 s (they are copied without making it wait).` : "";
  if(!run || !L){ $("boats").innerHTML=""; return; }
  const info = await loadRunInfo(selected), n = L.boats_shown||0;
  if($("boats").children.length !== n) $("boats").innerHTML = Array.from({length:n},(_,i)=>
     `<div class="boat"><h2>Boat ${i}</h2>${info.camera?`<img id="cam${i}" alt="camera, boat ${i}">`:""}${info.lidar?`<canvas id="lid${i}"></canvas>`:""}<div class="s" id="st${i}"></div></div>`).join("");
  for(let i=0;i<n;i++){
    if(info.camera && L.camera_time) $("cam"+i).src = `/feed/${selected}/camera_${i}.png?t=${L.camera_time}`;
    let ret = "";
    if(info.lidar && L.lidar_ranges) ret = ` · lidar returns ${Math.round(100*lidarPlot($("lid"+i), L.lidar_ranges[i], info))}%`;
    const b = (L.boat||[])[i];
    $("st"+i).textContent = b? `surge ${b.surge} m/s · tilt ${b.tilt_deg}° · heave ${b.heave} m · Hs ${b.hs} m · Tp ${b.tp} s${ret}` : "";
  }
  if(!info.camera && !info.lidar) $("boats").innerHTML = "<div class='note'>This run has no camera or lidar.</div>";
}
$("run").addEventListener("change", e=>{ selected = e.target.value; $("boats").innerHTML=""; refresh(); });
["c_steps","c_util","c_mem"].forEach(id=>hover($(id)));
refresh(); setInterval(refresh, 1000);
</script></body></html>
"""

if __name__ == "__main__":
    threading.Thread(target=sample_gpu, daemon=True).start()
    threading.Thread(target=sample_runs, daemon=True).start()
    servers = []
    for host in args.hosts.split(","):
        try:
            servers.append(ThreadingHTTPServer((host.strip(), args.port), Handler))
        except OSError as err:
            print(f"live_view: cannot listen on {host}:{args.port}: {err}", flush=True)
    for s in servers[1:]:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    print("live_view: " + ", ".join(f"http://{s.server_address[0]}:{s.server_address[1]}" for s in servers), flush=True)
    servers[0].serve_forever()
