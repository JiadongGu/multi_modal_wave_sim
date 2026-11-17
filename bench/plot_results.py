"""Charts of the benchmark results, as PNGs for slides and reports.

    python3 bench/plot_results.py        # reads bench/results/*/results.csv, writes bench/plots/*.png

All runs: lab RTX 4090 (24 GB), Isaac Sim 6.2 + Isaac Lab 3.0, PhysX; waves unless noted;
camera 128x96 RGB + depth with a drawn sea on every boat (full sweep).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
RES, OUT = ROOT / "results", ROOT / "plots"
OUT.mkdir(exist_ok=True)

INK, MUTED, GRID = "#13212b", "#5b6871", "#e3e7ea"
SERIES = {  # fixed order, validated palette (light)
    "none": ("#2a78d6", "o", "No sensors"),
    "imu": ("#eb6834", "s", "IMU"),
    "lidar": ("#1baf7a", "^", "Lidar"),
    "camera": ("#eda100", "D", "Camera"),
    "all": ("#e87ba4", "v", "IMU + camera + lidar"),
}
PARTS = [  # (column(s), label, colour)
    (["physics_ms"], "Physics (PhysX)", "#2a78d6"),
    (["forces_ms"], "Force models (usv_core)", "#1baf7a"),
    (["lidar_ms"], "Lidar", "#eda100"),
    (["read_sensors_ms"], "Camera read-out (rendering)", "#6250d6"),
    (["water_ms"], "Water drawing", "#e87ba4"),
    (["other_ms", "write_sim_ms", "scene_update_ms", "treadmill_ms", "render_ms", "reset_ms"], "Other", "#b9c4cb"),
]
BOATS = [128, 256, 512, 1024, 2048, 4096]
FOOT = "Lab RTX 4090 (24 GB) · Isaac Sim 6.2 / Isaac Lab 3.0 · measured 11–16 Nov 2026"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1,
    "axes.axisbelow": True, "figure.dpi": 100, "savefig.dpi": 200, "text.color": INK,
})


def finish(fig, ax, title, subtitle, name, foot=FOOT):
    fig.text(0.06, 0.965, title, fontsize=18, fontweight="bold", ha="left", va="top")
    fig.text(0.06, 0.915, subtitle, fontsize=12, color=MUTED, ha="left", va="top")
    fig.text(0.06, 0.02, foot, fontsize=9.5, color=MUTED, ha="left", va="bottom")
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


def boats_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(BOATS)
    ax.set_xticklabels([f"{b:,}" for b in BOATS])
    ax.set_xlabel("Boats simulated at once")
    ax.set_xlim(100, 6200)


full = pd.read_csv(RES / "full" / "results.csv")
ok = full[full.status == "ok"].copy()
fails = full[full.status != "ok"]


# ---------------------------------------------------------------- 1. throughput
def line_chart(col, ylabel, name, title, subtitle, sea="waves", logy=True, ref=None, fmt=lambda v: f"{v:,.0f}"):
    fig, ax = plt.subplots(figsize=(11, 6.6))
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.1, right=0.8)
    import math

    labels = []
    for key, (colour, marker, label) in SERIES.items():
        d = ok[(ok.sensors == key) & (ok.sea == sea)].sort_values("num_envs")
        if d.empty:
            continue
        ax.plot(d.num_envs, d[col], color=colour, lw=2.2, marker=marker, ms=8, mec="white", mew=1.5, zorder=3)
        last = d.iloc[-1]
        f = fails[(fails.sensors == key) & (fails.sea == sea)]
        text = f"{label}  {fmt(last[col])}"
        x = last.num_envs
        if not f.empty:                                  # mark the first boat count that did not fit
            fx = int(f.num_envs.min())
            ax.plot([last.num_envs, fx], [last[col], last[col]], color=colour, lw=1.2, ls=":", zorder=2)
            ax.scatter([fx], [last[col]], marker="x", s=90, color=colour, lw=2.5, zorder=4)
            text += f"  (max {int(last.num_envs):,} boats)"
            x = fx
        labels.append([last[col], x, text])
    # push apart labels that would overlap (in log space when the axis is logarithmic)
    pos = lambda v: math.log10(v) if logy else v
    labels.sort(key=lambda l: pos(l[0]))
    gap = 0.075 if logy else 0.05 * max(l[0] for l in labels)
    ys = [pos(l[0]) for l in labels]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < gap:
            ys[i] = ys[i - 1] + gap
    for (v, x, text), y in zip(labels, ys):
        ax.annotate(text, (x, 10 ** y if logy else y), xytext=(12, 0), textcoords="offset points", va="center",
                    fontsize=11.5, color=INK, annotation_clip=False)
    boats_axis(ax)
    if logy:
        ax.set_yscale("log")
    if ref is not None:
        ax.axhline(ref[0], color=INK, lw=1.2, ls="--", zorder=2)
        ax.annotate(ref[1], (2900, ref[0]), xytext=(0, 6), textcoords="offset points", fontsize=10.5, color=INK, ha="center")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    finish(fig, ax, title, subtitle, name)


line_chart("env_steps_per_s", "Env-steps per second (log scale)", "01_throughput.png",
           "Throughput: without sensors it scales with the boats; with sensors it levels off",
           "Waves. × = the next boat count did not fit in GPU memory. Camera 128×96 with a drawn sea on every boat.")

line_chart("sim_s_per_real_s_per_boat", "Simulated seconds per real second, per boat (log scale)", "02_realtime_per_boat.png",
           "How fast each boat's world runs compared with real time",
           "Above the dashed line each boat's simulation runs faster than real time. Waves.",
           ref=(1.0, "real time"), fmt=lambda v: f"{v:.2f}×")

line_chart("sim_s_per_real_s_total", "Simulated seconds per real second, all boats together (log scale)",
           "03_sim_seconds_total.png",
           "Total simulated time produced per real second",
           "Summed over every boat: the amount of experience a training run gets per second of wall time. Waves.",
           fmt=lambda v: f"{v:,.0f} s")


# ---------------------------------------------------------------- 4. breakdown
def breakdown(n, name):
    rows = []
    for key in ["none", "imu", "lidar", "camera", "all"]:
        d = ok[(ok.sensors == key) & (ok.sea == "waves") & (ok.num_envs == n)]
        if not d.empty:
            rows.append((SERIES[key][2], d.iloc[0]))
    fig, ax = plt.subplots(figsize=(12, 5.8))
    fig.subplots_adjust(top=0.76, bottom=0.14, left=0.2, right=0.97)
    ax.grid(axis="y", visible=False)
    ymax = 0
    for y, (label, r) in enumerate(reversed(rows)):
        left = 0.0
        for cols, plabel, colour in PARTS:
            v = float(sum(r[c] for c in cols))
            if v <= 0:
                continue
            ax.barh(y, v, left=left, height=0.62, color=colour, edgecolor="white", linewidth=2)
            if v > 14:
                ax.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center", fontsize=10.5,
                        color="white" if colour in ("#2a78d6", "#6250d6") else INK, fontweight="bold")
            left += v
        ax.text(left + 4, y, f"{left:.0f} ms", va="center", fontsize=11, color=INK)
        ymax = max(ymax, left)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lbl for lbl, _ in reversed(rows)])
    ax.set_xlim(0, ymax * 1.12)
    ax.set_xlabel(f"Milliseconds per policy step (1/30 s of simulated time), {n:,} boats")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in PARTS]
    fig.legend(handles, [p for _, p, _ in PARTS], ncol=6, frameon=False, loc="upper left", bbox_to_anchor=(0.055, 0.875),
               fontsize=10.5, handlelength=1.2, columnspacing=1.2)
    finish(fig, ax, f"Where the time goes, {n:,} boats",
           "Each part timed separately (a GPU sync around each). Waves. Numbers inside segments are ms.", name,
           foot=FOOT + " · timed run; syncs add a few %")


breakdown(1024, "04_time_breakdown_1024.png")
breakdown(2048, "05_time_breakdown_2048.png")

# ---------------------------------------------------------------- 6. memory
fig, ax = plt.subplots(figsize=(11, 6.6))
fig.subplots_adjust(top=0.82, bottom=0.14, left=0.1, right=0.8)
for key, (colour, marker, label) in SERIES.items():
    d = ok[(ok.sensors == key) & (ok.sea == "waves")].sort_values("num_envs")
    ax.plot(d.num_envs, d.gpu_mem_process_gb, color=colour, lw=2.2, marker=marker, ms=8, mec="white", mew=1.5, zorder=3)
    last = d.iloc[-1]
    if key == "imu":                                     # same memory as no sensors: one shared label
        continue
    text = "No sensors, IMU" if key == "none" else label
    ax.annotate(f"{text}  {last.gpu_mem_process_gb:.1f} GB", (last.num_envs, last.gpu_mem_process_gb), xytext=(10, 0),
                textcoords="offset points", va="center", fontsize=11.5)
ax.axhline(24, color=INK, lw=1.2, ls="--")
ax.annotate("24 GB card", (110, 24), xytext=(0, 5), textcoords="offset points", fontsize=10.5)
boats_axis(ax)
ax.set_ylim(0, 27)
ax.set_ylabel("GPU memory used by the run (GB)")
finish(fig, ax, "GPU memory: cameras set the ceiling at about 2,048 boats",
       "Without cameras the whole sim fits in under 10 GB even at 4,096 boats. Waves.", "06_gpu_memory.png")

# ---------------------------------------------------------------- 7. lidar waves vs flat
fig, ax = plt.subplots(figsize=(11, 6.2))
fig.subplots_adjust(top=0.8, bottom=0.14, left=0.1, right=0.8)
for sea, colour, marker, label in [("waves", "#1baf7a", "^", "Lidar on waves"), ("flat", "#2a78d6", "o", "Lidar on flat water")]:
    d = ok[(ok.sensors == "lidar") & (ok.sea == sea)].sort_values("num_envs")
    ax.plot(d.num_envs, d.lidar_ms, color=colour, lw=2.2, marker=marker, ms=8, mec="white", mew=1.5)
    last = d.iloc[-1]
    ax.annotate(f"{label}  {last.lidar_ms:.0f} ms", (last.num_envs, last.lidar_ms), xytext=(10, 0),
                textcoords="offset points", va="center", fontsize=11.5)
boats_axis(ax)
ax.set_ylabel("Lidar, ms per policy step")
finish(fig, ax, "Lidar cost comes from searching the moving sea",
       "2,880 rays per boat, 10 scans per second. On flat water the hit point is a simple formula; on waves each ray searches.",
       "07_lidar_waves_vs_flat.png")

# ---------------------------------------------------------------- 8. ROM cost
rom = pd.read_csv(RES / "romcost" / "results.csv")
rom["rom"] = rom["rom"].fillna("").astype(str).replace({"": "none", "nan": "none"})
sizes = ["none", "2x64", "3x256", "4x512", "3x256x10", "6x1024"]
nice = {"none": "No ROM", "2x64": "2 × 64", "3x256": "3 × 256", "4x512": "4 × 512", "3x256x10": "3 × 256\n+10-step history",
        "6x1024": "6 × 1,024"}
fig, ax = plt.subplots(figsize=(11, 6.2))
fig.subplots_adjust(top=0.8, bottom=0.16, left=0.1, right=0.97)
ax.grid(axis="x", visible=False)
w = 0.36
for i, (n, colour) in enumerate([(1024, "#2a78d6"), (4096, "#eb6834")]):
    vals = [float(rom[(rom.num_envs == n) & (rom.rom == s)].ms_per_step_clean.iloc[0]) for s in sizes]
    xs = [k + (i - 0.5) * w for k in range(len(sizes))]
    ax.bar(xs, vals, width=w - 0.04, color=colour, label=f"{n:,} boats")
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.4, f"{v:.1f}", ha="center", fontsize=10, color=INK)
ax.set_xticks(range(len(sizes)))
ax.set_xticklabels([nice[s] for s in sizes])
ax.set_xlabel("Stand-in network size (hidden layers × width)")
ax.set_ylabel("ms per policy step, no sensors")
ax.set_ylim(0, 36)
ax.legend(frameon=False, loc="upper left")
finish(fig, ax, "A ROM adds about 7.5 ms per step, almost regardless of its size",
       "Untrained stand-in networks run every physics substep on every boat, output discarded. Most of the cost is gathering inputs.",
       "08_rom_cost.png")

# ---------------------------------------------------------------- 9. camera resolution ceilings
first = pd.read_csv(RES / "first" / "results.csv")
res_rows = [("128 × 96", 2048, "full sweep, with drawn water"), ("256 × 192", 512, "first sweep, no drawn water"),
            ("640 × 480", 128, "first sweep, no drawn water")]
fig, ax = plt.subplots(figsize=(11, 5))
fig.subplots_adjust(top=0.76, bottom=0.18, left=0.16, right=0.95)
ax.grid(axis="y", visible=False)
for y, (lbl, n, note) in enumerate(reversed(res_rows)):
    ax.barh(y, n, height=0.55, color="#eda100")
    ax.text(n * 1.08, y, f"{n:,} boats  ({note})", va="center", fontsize=11)
ax.set_xscale("log", base=2)
ax.set_xticks(BOATS)
ax.set_xticklabels([f"{b:,}" for b in BOATS])
ax.set_xlim(64, 30000)
ax.set_yticks(range(len(res_rows)))
ax.set_yticklabels([r[0] for r in reversed(res_rows)])
ax.set_xlabel("Largest boat count that fit in 24 GB")
finish(fig, ax, "Camera resolution sets how many boats fit on one GPU",
       "Each camera renders colour + depth every policy step. Doubling width and height roughly quarters the boats.",
       "09_camera_resolution_ceiling.png")

# ---------------------------------------------------------------- 10. improvements made
ff = pd.read_csv(RES / "forcefix" / "results.csv")
gw = pd.read_csv(RES / "gpuwater" / "results.csv")
fig, axes = plt.subplots(1, 3, figsize=(14, 5.4))
fig.subplots_adjust(top=0.74, bottom=0.2, left=0.06, right=0.98, wspace=0.42)
for ax in axes:
    ax.grid(axis="x", visible=False)
# force fix
before = float(first[(first.num_envs == 1024) & (first.sensors == "none") & (first.sea == "waves")].env_steps_per_s.iloc[0])
after = float(ff[(ff.num_envs == 1024) & (ff.sensors == "none")].env_steps_per_s.iloc[0])
axes[0].bar(["Before", "After"], [before, after], color=["#b9c4cb", "#2a78d6"], width=0.55)
for x, v in enumerate([before, after]):
    axes[0].text(x, v * 1.02, f"{v:,.0f}", ha="center", fontsize=11)
axes[0].set_ylabel("Env-steps per second")
axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
axes[0].set_title("Force code: fewer CPU↔GPU copies\n1,024 boats, no sensors (+19%)", fontsize=12, loc="left")
axes[0].set_ylim(0, after * 1.2)
# GPU water
cpu16 = float(gw[gw.tag == "cpu16"].water_ms.iloc[0])
gpu16 = float(gw[gw.tag == "gpu16"].water_ms.iloc[0])
gpuall = float(gw[gw.tag == "gpuall"].water_ms.iloc[0])
labels = ["16 seas\nCPU (old)", "16 seas\nGPU", "1,024 seas\nGPU"]
axes[1].bar(labels, [cpu16, gpu16, gpuall], color=["#b9c4cb", "#e87ba4", "#e87ba4"], width=0.55)
for x, v in enumerate([cpu16, gpu16, gpuall]):
    axes[1].text(x, v * 1.03, f"{v:.2f} ms", ha="center", fontsize=11)
axes[1].set_ylabel("Water drawing, ms per step")
axes[1].set_title("Drawn sea: moved to the GPU\nnow every boat's camera sees waves", fontsize=12, loc="left")
axes[1].set_ylim(0, gpuall * 1.3)
# live viewer
feed = pd.read_csv(RES / "feed" / "results.csv")
base = feed[feed.tag == "feed-off"].ms_per_step_clean.mean()
pct = [0.0, (feed[feed.tag == "feed-metrics"].ms_per_step_clean.mean() / base - 1) * 100,
       (feed[feed.tag == "feed-sensors"].ms_per_step_clean.mean() / base - 1) * 100]
axes[2].bar(["Viewer off", "Metrics", "+ camera,\nlidar"], pct, color=["#b9c4cb", "#1baf7a", "#1baf7a"], width=0.55)
for x, v in enumerate(pct):
    axes[2].text(x, v + 0.03, f"+{v:.1f}%", ha="center", fontsize=11)
axes[2].set_ylabel("Slowdown vs viewer off (%)")
axes[2].set_ylim(0, 1.2)
axes[2].set_title("Live viewer: cost of watching\n1,024 boats, all sensors (target < 1%)", fontsize=12, loc="left")
fig.text(0.06, 0.965, "Improvements made this week, measured", fontsize=18, fontweight="bold", ha="left", va="top")
fig.text(0.06, 0.915, "Each change measured on the same machine before and after.", fontsize=12, color=MUTED, ha="left", va="top")
fig.text(0.06, 0.02, FOOT, fontsize=9.5, color=MUTED, ha="left", va="bottom")
fig.savefig(OUT / "10_improvements.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote", OUT / "10_improvements.png")
