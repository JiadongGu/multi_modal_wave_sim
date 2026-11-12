"""Live feed for the viewer (scripts/live_view.py): run metrics, plus camera and lidar for a few boats.

Off unless `live_feed` is set in the config. Built so the simulation never waits for it:
  - it acts at most once per `live_feed_period_s` of wall-clock time; other steps cost one clock read;
  - GPU data is copied to pinned CPU memory without waiting (non_blocking), and a CUDA event marks
    when the copy is done. The copy is only picked up on a later tick, once the event says it finished;
  - files are written by a background thread, replaced atomically so the viewer never reads half a file.
Each run writes to its own folder, <live_feed_dir>/<pid>/, so several runs never collide.

Cost that remains: a few tiny GPU kernels and a ~0.4 MB copy per tick, and, when nothing else reads
the camera, one camera read-out per tick (Isaac Lab only fetches images when asked).
Shows only sensors the run already has: it never switches a sensor on.
"""

from __future__ import annotations

import json
import os
import queue
import struct
import sys
import threading
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from usv_core import world_to_body


def png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an H x W x 3 uint8 image as PNG (no image library needed)."""
    height, width, _ = rgb.shape

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))


def replace_file(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


class LiveFeed:
    def __init__(self, env):
        self.env = env
        cfg = env.cfg
        self.period = cfg.live_feed_period_s
        self.boats = min(cfg.live_feed_boats, env.num_envs)
        self.dir = Path(cfg.live_feed_dir).expanduser() / str(os.getpid())
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pending = None                              # (event, meta, pinned buffers) of the copy in flight
        self.buffers: dict[str, torch.Tensor] = {}
        self.last_tick = time.perf_counter()
        self.last_steps = 0
        self.steps = 0
        self.writes: queue.Queue = queue.Queue(maxsize=1)
        threading.Thread(target=self._writer, daemon=True).start()

        run = {"pid": os.getpid(), "task": type(env).__name__, "num_envs": env.num_envs, "started": time.time(),
               "command": " ".join(sys.argv), "step_dt": env.step_dt, "boats_shown": self.boats,
               "camera": env.camera is not None, "lidar": env.lidar is not None}
        if env.lidar is not None:
            d = env.lidar.directions.cpu().numpy()       # once, at start-up: fixed ray directions in the body frame
            run["lidar_azimuth"] = np.round(np.arctan2(d[:, 1], d[:, 0]), 4).tolist()
            run["lidar_elevation"] = np.round(np.arcsin(np.clip(d[:, 2], -1, 1)), 4).tolist()
            run["lidar_max_range"] = env.cfg.lidar.max_range
        replace_file(self.dir / "run.json", json.dumps(run).encode())

    def _pinned(self, name: str, src: torch.Tensor) -> torch.Tensor:
        buf = self.buffers.get(name)
        if buf is None or buf.shape != src.shape or buf.dtype != src.dtype:
            buf = self.buffers[name] = torch.empty(src.shape, dtype=src.dtype, pin_memory=True)
        buf.copy_(src, non_blocking=True)
        return buf

    def tick(self) -> None:
        """Call once per policy step."""
        self.steps += 1
        now = time.perf_counter()
        if now - self.last_tick < self.period:
            return
        # 1. hand over the previous copy if the GPU has finished it; otherwise drop it (never wait)
        if self.pending is not None:
            event, meta, bufs = self.pending
            if event.query():
                try:
                    self.writes.put_nowait((meta, {k: v.numpy().copy() for k, v in bufs.items()}))
                except queue.Full:
                    pass                                 # writer still busy: skip this one
            self.pending = None
        # 2. start the next copy
        env = self.env
        k = self.boats
        s = env.boat_state()
        tilt = torch.rad2deg(torch.arccos((-world_to_body(s.quat, env._down)[:, 2]).clamp(-1, 1)))
        sea = env.sea.sea_state()
        gpu = {
            # all boats: mean surge [m/s], max tilt [deg], mean heave [m]; then per shown boat: surge, tilt, heave, Hs, Tp
            "stats": torch.cat([
                torch.stack([s.lin_vel_body[:, 0].mean(), tilt.max(), s.pos[:, 2].mean()]),
                torch.stack([s.lin_vel_body[:k, 0], tilt[:k], s.pos[:k, 2], sea["hs"][:k], sea["tp"][:k]], -1).flatten(),
            ]).float(),
        }
        if env.camera is not None:
            gpu["rgb"] = env.camera.data.output["rgb"].torch[:k, ..., :3].clamp(0, 255).to(torch.uint8)
        if env.lidar is not None:
            gpu["lidar"] = env.lidar_ranges[:k].float()
        bufs = {name: self._pinned(name, t) for name, t in gpu.items()}
        event = torch.cuda.Event()
        event.record()
        wall = now - self.last_tick
        meta = {
            "time": time.time(), "sim_time": env.sim_time, "steps": self.steps, "num_envs": env.num_envs,
            "env_steps_per_s": (self.steps - self.last_steps) * env.num_envs / wall,
            "sim_s_per_real_s": (self.steps - self.last_steps) * env.step_dt / wall,
            "resets": env.reset_count, "gpu_mem_torch_gb": torch.cuda.memory_allocated() / 2**30, "boats_shown": k,
        }
        self.pending = (event, meta, bufs)
        self.last_tick, self.last_steps = now, self.steps

    def _writer(self) -> None:
        while True:
            meta, arrays = self.writes.get()
            try:
                stats = arrays.pop("stats")
                meta["mean_surge"], meta["max_tilt_deg"], meta["mean_heave"] = (round(float(v), 3) for v in stats[:3])
                meta["boat"] = [{"surge": round(float(a), 2), "tilt_deg": round(float(b), 1), "heave": round(float(c), 2),
                                 "hs": round(float(d), 2), "tp": round(float(e), 1)} for a, b, c, d, e in stats[3:].reshape(-1, 5)]
                if "rgb" in arrays:
                    for i, img in enumerate(arrays.pop("rgb")):
                        replace_file(self.dir / f"camera_{i}.png", png_bytes(np.ascontiguousarray(img)))
                    meta["camera_time"] = meta["time"]
                if "lidar" in arrays:
                    r = arrays.pop("lidar")
                    meta["lidar_ranges"] = np.where(np.isfinite(r), np.round(r, 1), -1.0).tolist()   # -1: no return
                replace_file(self.dir / "latest.json", json.dumps(meta).encode())
            except OSError as err:                       # a viewing aid must never stop a simulation
                print(f"[live feed] skipped an update: {err}", flush=True)
